"""Modernized dense decoder-only transformer.

Block recipe (pre-norm): x = x + attn(rmsnorm(x)); x = x + swiglu(rmsnorm(x)).
Modernizations baked in from the start (validated free on Colab before any
paid run): RoPE, RMSNorm, SwiGLU, GQA, QK-Norm, weight tying, residual-scaled
init. This is the architecture the long A100 run uses.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """RMSNorm with the reduction done in fp32 for mixed-precision safety."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        # keep the scale multiply in fp32, then cast back once
        return ((x * rms) * self.weight.float()).to(dtype)


def build_rope_cache(head_dim: int, max_seq_len: int, theta: float,
                     device=None, dtype=torch.float32):
    """Precompute (cos, sin) of shape (max_seq_len, head_dim).

    Half-rotation (Llama) convention: freqs are computed for head_dim/2 pairs
    and concatenated with themselves so they line up with rotate_half.
    """
    i = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (i / head_dim))             # (head_dim/2,)
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                        # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                 # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim) -> broadcast
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


class Attention(nn.Module):
    """Causal grouped-query attention with optional QK-Norm and RoPE."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.dropout = cfg.dropout
        self.softcap = cfg.attn_logit_softcap

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, cos, sin).to(v.dtype)
        k = apply_rope(k, cos, sin).to(v.dtype)

        # expand KV heads to match Q heads (GQA). repeat_interleave on head dim.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if self.softcap > 0:
            out = self._attn_softcap(q, k, v, T)        # eager path, no Flash
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

    def _attn_softcap(self, q, k, v, T):
        # tanh logit soft-cap (Gemma-2 style). Disables the fused SDPA kernel,
        # so only used for the qk_norm=False stability ablation.
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = self.softcap * torch.tanh(scores / self.softcap)
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.d_model)
        m = cfg.mlp_multiple_of
        hidden = ((hidden + m - 1) // m) * m
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.embed.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Residual-projection scaling (GPT-2 trick): keep residual-stream growth
        # bounded with depth by shrinking the layers that write back into it.
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("down.weight"):
                with torch.no_grad():
                    p.mul_(scale)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, "sequence longer than rope cache"
        x = self.embed(idx)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # tied: lm_head shares embed, so subtract the one embedding table
            n -= self.embed.weight.numel()
        return n
