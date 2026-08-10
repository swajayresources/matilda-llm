"""Model + run configuration.

One frozen dataclass is the single source of truth for a run. It gets logged
verbatim (alongside the git SHA) so any run is exactly reproducible. Nothing in
the training stack reads a hyperparameter from anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class ModelConfig:
    # --- vocab / sequence ---
    vocab_size: int = 50257          # GPT-2 (tiktoken "gpt2"); fits uint16
    max_seq_len: int = 1024

    # --- transformer shape ---
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4              # GQA: n_heads must be divisible by this
    mlp_ratio: float = 8 / 3         # SwiGLU keeps params ~= 4x dense MLP
    mlp_multiple_of: int = 256       # round hidden dim up to this for kernels

    # --- numerics / regularization ---
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    init_std: float = 0.02
    dropout: float = 0.0             # pretraining: keep at 0

    # --- structural switches ---
    tie_weights: bool = True         # share embedding <-> lm_head
    qk_norm: bool = True             # RMSNorm on Q,K before attention
    attn_logit_softcap: float = 0.0  # >0 caps scores via tanh; needed for the
                                     # qk_norm=False ablation so it can't NaN.
                                     # 0 keeps the fast fused SDPA path.

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, \
            "n_heads must be divisible by n_kv_heads (GQA grouping)"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["head_dim"] = self.head_dim
        return d


# Tiny config for free correctness validation on Colab T4 / RTX 3060.
# Real vocab kept so tokenizer wiring is exercised; dims shrunk so it runs fast.
DEV_TINY = ModelConfig(
    vocab_size=50257,
    max_seq_len=256,
    d_model=128,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
)

# ~124M params (GPT-2 base footprint). The intended A100 long-run config.
BASE_124M = ModelConfig()
