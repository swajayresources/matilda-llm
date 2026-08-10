"""MatildaForCausalLM — HF-compatible wrapper for the matilda Transformer.

Bridges `matilda.model.Transformer` semantics into the HuggingFace
`PreTrainedModel` API so the checkpoint can be loaded by
`AutoModelForCausalLM.from_pretrained("prometheus04/...", trust_remote_code=True)`
and serves with `.generate()`, `transformers` pipelines, vllm, etc.

What this implements:
  - GQA decoder-only transformer with RoPE (theta=10000), RMSNorm, QK-Norm,
    ReLU² FFN, Gemma-2-style final-logit softcap, residual-projection init
    scaling (baked into the saved weights — no runtime effect).
  - Loads its own LigerRMSNorm via the same {torch.distributed.tensor import +
    eager RoPE} workarounds documented in docs/instance-run-2026-05-30/.
    If liger-kernel isn't installed we fall back to a pure-PyTorch RMSNorm
    so the model still loads on CPU / Colab without GPU + Triton.

What this DOES NOT do:
  - Training loss (z_loss, FLCE) — inference-only. Anyone wanting to fine-tune
    should use matilda.model.Transformer directly.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_matilda import MatildaConfig


# ---------- norms ----------

def _try_liger_rms_norm(dim: int, eps: float):
    """Try to use LigerRMSNorm; fall back to plain PyTorch RMSNorm."""
    try:
        import torch.distributed.tensor  # noqa: F401  (Liger 0.8.0 needs this)
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        return LigerRMSNorm(dim, eps=eps)
    except Exception:
        return _RMSNorm(dim, eps)


class _RMSNorm(nn.Module):
    """RMSNorm with the reduction in fp32 for mixed-precision safety."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return ((x * rms) * self.weight.float()).to(dtype)


# ---------- rope (eager half-rotation, Llama convention) ----------

def _build_rope_cache(head_dim: int, max_seq_len: int, theta: float,
                      device=None, dtype=torch.float32):
    i = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (i / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Cast cos/sin to x.dtype so the result keeps x's dtype. The RoPE buffers
    # are stored in fp32 for precision; without this cast, x*cos broadcasts up
    # to fp32 and downstream SDPA sees mismatched q/k vs v dtypes when the
    # caller isn't inside torch.autocast.
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


# ---------- attention ----------

class MatildaAttention(nn.Module):
    def __init__(self, cfg: MatildaConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.dropout_p = cfg.dropout
        self.softcap = cfg.attn_logit_softcap

        d = cfg.d_model
        self.wq = nn.Linear(d, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, d, bias=False)

        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = _try_liger_rms_norm(self.head_dim, cfg.norm_eps)
            self.k_norm = _try_liger_rms_norm(self.head_dim, cfg.norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if self.softcap > 0:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            scores = self.softcap * torch.tanh(scores / self.softcap)
            mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------- MLPs ----------

def _ffn_hidden(cfg: MatildaConfig) -> int:
    h = int(cfg.mlp_ratio * cfg.d_model)
    m = cfg.mlp_multiple_of
    return ((h + m - 1) // m) * m


class MatildaReLU2FFN(nn.Module):
    """down(relu(up(x))**2). Two matmuls; matches matilda training."""

    def __init__(self, cfg: MatildaConfig):
        super().__init__()
        hidden = _ffn_hidden(cfg)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.up(x)).square())


class MatildaSwiGLU(nn.Module):
    """SwiGLU — present for completeness; matilda 152M uses ReLU²."""

    def __init__(self, cfg: MatildaConfig):
        super().__init__()
        hidden = _ffn_hidden(cfg)
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _build_ffn(cfg: MatildaConfig) -> nn.Module:
    if cfg.mlp_activation == "relu2":
        return MatildaReLU2FFN(cfg)
    return MatildaSwiGLU(cfg)


# ---------- block ----------

class MatildaBlock(nn.Module):
    def __init__(self, cfg: MatildaConfig):
        super().__init__()
        self.norm1 = _try_liger_rms_norm(cfg.d_model, cfg.norm_eps)
        self.attn = MatildaAttention(cfg)
        self.norm2 = _try_liger_rms_norm(cfg.d_model, cfg.norm_eps)
        self.mlp = _build_ffn(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------- backbone ----------

class MatildaModel(PreTrainedModel):
    """Bare transformer trunk — returns last_hidden_state. Mostly used through
    MatildaForCausalLM below; kept separate so AutoModel works too."""

    config_class = MatildaConfig
    supports_gradient_checkpointing = False
    _no_split_modules = ["MatildaBlock"]

    def __init__(self, config: MatildaConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([MatildaBlock(config) for _ in range(config.n_layers)])
        self.norm_f = _try_liger_rms_norm(config.d_model, config.norm_eps)
        cos, sin = _build_rope_cache(config.head_dim, config.max_seq_len, config.rope_theta)
        # PERSISTENT so they survive `from_pretrained()` — which builds the
        # model under HF's empty-weights context and leaves non-persistent
        # buffers as uninitialized garbage. ~1 MB total at max_seq_len=2048.
        self.register_buffer("rope_cos", cos, persistent=True)
        self.register_buffer("rope_sin", sin, persistent=True)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def forward(self, input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, "sequence longer than rope cache"
        x = self.embed(input_ids)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm_f(x)
        return x


# ---------- causal-LM head ----------

class MatildaForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MatildaConfig
    _no_split_modules = ["MatildaBlock"]

    def __init__(self, config: MatildaConfig):
        super().__init__(config)
        self.model = MatildaModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.model.embed.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed

    def set_input_embeddings(self, value):
        self.model.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden)

        # Gemma-2-style final softcap (applied to logits, not pre-softmax in
        # attention — matches matilda training).
        cap = self.config.final_logit_softcap
        if cap and cap > 0:
            logits = cap * torch.tanh(logits / cap)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=None,
            hidden_states=None, attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Cap context at max_seq_len (no KV cache yet).
        if input_ids.size(1) > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len:]
        return {"input_ids": input_ids}
