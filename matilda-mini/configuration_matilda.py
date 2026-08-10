"""Matilda model configuration."""
from transformers import PretrainedConfig

class MatildaConfig(PretrainedConfig):
    model_type = "matilda"

    def __init__(self, vocab_size=50257, max_seq_len=1024, d_model=768,
                 n_layers=12, n_heads=12, n_kv_heads=4, qk_norm=True,
                 rope_theta=10000.0, tie_weights=True, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.qk_norm = qk_norm
        self.rope_theta = rope_theta
        self.tie_weights = tie_weights
