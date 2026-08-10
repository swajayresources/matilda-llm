"""MatildaConfig — HF PretrainedConfig mirror of matilda.config.ModelConfig.

Custom fields (not in stock LlamaConfig):
  - `mlp_activation`: "relu2" (Primer / modded-nanogpt FFN) or "swiglu"
  - `final_logit_softcap`: tanh-cap applied to the lm_head output (Gemma-2 style)
  - `qk_norm`: per-head RMSNorm on Q and K before SDPA
  - `z_loss_coef`: kept for archival — applied only at training, not at inference
  - `attn_logit_softcap`: per-attention softcap (defaults to 0 = off, uses SDPA)

Architecture in one sentence: GQA decoder-only transformer with RoPE,
RMSNorm pre-norm, residual-projection init scaling, optional QK-Norm,
optional Gemma-2-style final-logit softcap, ReLU² FFN by default.
"""

from __future__ import annotations
from transformers import PretrainedConfig


class MatildaConfig(PretrainedConfig):
    model_type = "matilda"

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 2048,
        d_model: int = 768,
        n_layers: int = 18,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        mlp_ratio: float = 4.0,
        mlp_multiple_of: int = 256,
        norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        init_std: float = 0.02,
        dropout: float = 0.0,
        tie_weights: bool = True,
        qk_norm: bool = True,
        attn_logit_softcap: float = 0.0,
        z_loss_coef: float = 0.0,
        use_liger: bool = False,
        final_logit_softcap: float = 0.0,
        mlp_activation: str = "relu2",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.mlp_ratio = mlp_ratio
        self.mlp_multiple_of = mlp_multiple_of
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.init_std = init_std
        self.dropout = dropout
        self.tie_weights = tie_weights
        self.qk_norm = qk_norm
        self.attn_logit_softcap = attn_logit_softcap
        self.z_loss_coef = z_loss_coef
        self.use_liger = use_liger
        self.final_logit_softcap = final_logit_softcap
        self.mlp_activation = mlp_activation
        # Aliases HF utilities look for
        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        self.num_key_value_heads = n_kv_heads
        self.num_hidden_layers = n_layers
        self.max_position_embeddings = max_seq_len
        # tie_word_embeddings may already be in kwargs from a saved config
        # round-trip — let our `tie_weights` value win.
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(tie_word_embeddings=tie_weights, **kwargs)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
