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

    # --- loss / kernel knobs (no-op at v1 defaults) ---
    z_loss_coef: float = 0.0         # penalize log-partition (PaLM/Pythia); 0 = off
    use_liger: bool = False          # swap RMSNorm / RoPE / linear+CE for Liger
                                     # fused kernels on GPU. Requires liger-kernel.
    final_logit_softcap: float = 0.0 # Gemma-2 style tanh-cap on lm_head logits.
                                     # 0 = off. ~30 is a typical value. Helps
                                     # WSD's flat-LR phase stay stable.

    # --- FFN choice ---
    mlp_activation: str = "swiglu"   # "swiglu" (v1) or "relu2" (Primer/modded-nanogpt:
                                     # 2 matmuls instead of 3, ReLU(x)^2 activation).
                                     # For "relu2", set mlp_ratio=4.0 to param-match
                                     # SwiGLU at 8/3 (same total FLOPs and params).

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, \
            "n_heads must be divisible by n_kv_heads (GQA grouping)"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert self.mlp_activation in ("swiglu", "relu2"), \
            f"mlp_activation must be 'swiglu' or 'relu2', got {self.mlp_activation!r}"

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

# ~124M params (GPT-2 base footprint). The original A100 long-run config.
BASE_124M = ModelConfig()


# ~363M params total / ~315M non-embed. Original 350M-family target.
# Pivoted away from for the actual hero run (activation memory at 32L × seq=2048
# without grad checkpointing didn't fit on 40GB A100; calibration capped at
# BS=12 / 30% MFU, ~110h wall clock). Kept here for reference / future revisit.
BASE_350M = ModelConfig(
    vocab_size=50257,
    max_seq_len=2048,
    d_model=960,
    n_layers=32,
    n_heads=12,             # head_dim = 80; 12 % 4 (n_kv_heads) == 0 ✓
    n_kv_heads=4,           # GQA: 3 query heads per KV head
    mlp_ratio=8 / 3,        # SwiGLU; 8/3 * 960 = 2560 lands on the 256 grid
    mlp_multiple_of=256,
    tie_weights=True,
    qk_norm=True,
    z_loss_coef=1e-4,
    use_liger=True,
)


# ~152M params total / ~113M non-embed. The actual hero run (v1.5).
# Shape rationale: 18L × 768d directly matches Pythia-160M's width with 50%
# more depth — gives clean token-matched comparison to Pythia-160M's published
# checkpoints, and still beats-above-weight vs Pythia-410M (2.7× larger).
# head_dim = 64 → clean Flash / Liger RoPE path (no head_dim 80 bug).
# ReLU² FFN + final logit soft-cap added vs v1 baseline. SmolLM data + Muon
# + WSD + Liger preserved from the v2 design.
BASE_152M = ModelConfig(
    vocab_size=50257,
    max_seq_len=2048,
    d_model=768,
    n_layers=18,
    n_heads=12,             # head_dim = 64 ✓ (Flash + Liger RoPE friendly)
    n_kv_heads=4,           # GQA: 3 query heads per KV head
    mlp_ratio=4.0,          # ReLU² wants 4× to param-match SwiGLU at 8/3
    mlp_multiple_of=256,    # 4 × 768 = 3072; 3072 / 256 = 12 ✓
    tie_weights=True,
    qk_norm=True,
    z_loss_coef=1e-4,
    use_liger=True,
    mlp_activation="relu2",
    final_logit_softcap=30.0,
)
