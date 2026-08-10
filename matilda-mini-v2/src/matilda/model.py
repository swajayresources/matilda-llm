"""Modernized dense decoder-only transformer.

Block recipe (pre-norm): x = x + attn(rmsnorm(x)); x = x + swiglu(rmsnorm(x)).
Modernizations baked in from the start (validated free on Colab before any
paid run): RoPE, RMSNorm, SwiGLU, GQA, QK-Norm, weight tying, residual-scaled
init. This is the architecture the long A100 run uses.

Optional knobs for the 350M hero run, all default-off so the CPU/v1 path is
unchanged:
  - z_loss_coef > 0: PaLM-style log-Z penalty, applied alongside CE
  - use_liger: swap RMSNorm/RoPE/(linear+CE) for Liger Triton kernels. The
    fused linear+CE skips materializing the (B*T, vocab) logit tensor, which
    is the dominant memory cost at 350M with vocab=50257.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def _import_liger():
    """Lazy import so CPU tests don't pay for liger-kernel install.
    Raises ImportError with a helpful message if use_liger=True without it."""
    # Liger 0.8.0 RMSNorm calls torch.distributed.tensor.DTensor at runtime;
    # in torch 2.6 that submodule isn't auto-loaded as an attribute, so the
    # first forward dies with AttributeError under eager or a CUDA illegal
    # memory access under torch.compile. Importing the submodule registers it.
    import torch.distributed.tensor  # noqa: F401
    try:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.rope import liger_rotary_pos_emb
        from liger_kernel.transformers.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyLoss,
        )
    except ImportError as e:
        raise ImportError(
            "use_liger=True requires liger-kernel. "
            "Install: `pip install liger-kernel` (GPU only)."
        ) from e
    return LigerRMSNorm, liger_rotary_pos_emb, LigerFusedLinearCrossEntropyLoss


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

    def __init__(self, cfg: ModelConfig, liger=None):
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
            norm_cls = liger[0] if liger else RMSNorm
            self.q_norm = norm_cls(self.head_dim, cfg.norm_eps)
            self.k_norm = norm_cls(self.head_dim, cfg.norm_eps)
        self.dropout = cfg.dropout
        self.softcap = cfg.attn_logit_softcap
        self._liger_rope = liger[1] if liger else None

    def _apply_rope(self, q, k, cos, sin):
        # liger_rotary_pos_emb (Triton) crashes torch.compile autotune with
        # CUDA illegal memory access on liger-kernel 0.8.0 + torch 2.6,
        # independent of head_dim (reproduced at head_dim=80 and head_dim=64).
        # Eager apply_rope is < 1% of step time; LigerRMSNorm + LigerFusedLinearCE
        # stay engaged and carry the throughput claim.
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)

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

        q, k = self._apply_rope(q, k, cos, sin)

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


def _ffn_hidden(cfg: ModelConfig) -> int:
    """Hidden dim, rounded up to mlp_multiple_of for kernel friendliness."""
    h = int(cfg.mlp_ratio * cfg.d_model)
    m = cfg.mlp_multiple_of
    return ((h + m - 1) // m) * m


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = _ffn_hidden(cfg)
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ReLU2FFN(nn.Module):
    """Primer / modded-nanogpt FFN: down(relu(up(x))**2). Two matmuls instead of
    SwiGLU's three; at mlp_ratio=4.0 it param-matches SwiGLU at 8/3 and has the
    same total FLOPs. Empirically competitive with SwiGLU at sub-1B scale and
    simpler (one fewer kernel launch per FFN)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = _ffn_hidden(cfg)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.up(x)).square())


def _build_ffn(cfg: ModelConfig) -> nn.Module:
    if cfg.mlp_activation == "relu2":
        return ReLU2FFN(cfg)
    return SwiGLU(cfg)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, liger=None):
        super().__init__()
        norm_cls = liger[0] if liger else RMSNorm
        self.norm1 = norm_cls(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, liger=liger)
        self.norm2 = norm_cls(cfg.d_model, cfg.norm_eps)
        self.mlp = _build_ffn(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Liger bundle: (RMSNorm class, rope fn, FLCE class) or None.
        # Built once and threaded through Block/Attention so every layer uses
        # the same kernel family — avoids per-layer import cost.
        liger = _import_liger() if cfg.use_liger else None
        self._liger = liger
        self._flce = None
        if liger is not None:
            # Liger FLCE supports softcap natively (Gemma-2 style); pass None
            # when off so we don't pay the extra tanh kernel.
            softcap = (cfg.final_logit_softcap
                       if cfg.final_logit_softcap > 0 else None)
            self._flce = liger[2](
                ignore_index=-1,
                lse_square_scale=cfg.z_loss_coef,    # z-loss handled in-kernel
                label_smoothing=0.0,
                reduction="mean",
                softcap=softcap,
            )

        norm_cls = liger[0] if liger else RMSNorm
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg, liger=liger) for _ in range(cfg.n_layers)])
        self.norm_f = norm_cls(cfg.d_model, cfg.norm_eps)
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

        # Fused linear+CE path: skip logit materialization (the 350M+vocab=50k
        # memory hot-spot). Returns logits=None — training never uses them.
        if self._flce is not None and targets is not None:
            loss = self._flce(
                self.lm_head.weight,
                x.view(-1, x.size(-1)),
                targets.view(-1),
            )
            return None, loss

        logits = self.lm_head(x)
        if self.cfg.final_logit_softcap > 0:
            # Gemma-2 style: tanh-cap logits to bound magnitude. Eager path
            # only — the Liger fused path applies softcap inside the kernel.
            cap = self.cfg.final_logit_softcap
            logits = cap * torch.tanh(logits / cap)
        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        if self.cfg.z_loss_coef > 0:
            # PaLM/Pythia-style: penalize log(Z) magnitude to keep logits bounded.
            log_z = logits.logsumexp(dim=-1)
            loss = loss + self.cfg.z_loss_coef * (log_z ** 2).mean()
        return logits, loss

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # tied: lm_head shares embed, so subtract the one embedding table
            n -= self.embed.weight.numel()
        return n
