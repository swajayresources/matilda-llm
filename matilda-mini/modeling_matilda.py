"""Matilda causal LM wrapper for HuggingFace AutoModel."""
from __future__ import annotations
import sys, math
from pathlib import Path
import torch, torch.nn as nn
from transformers import PreTrainedModel
from .configuration_matilda import MatildaConfig

# import the real model — try several paths to handle local and HF cache loads
_HAS_SRC = False
for _candidate in [
    Path(__file__).resolve().parent / "src",
    Path(__file__).resolve().parent.parent / "src",
    Path("/workspace/matilda-mini/src"),
]:
    if (_candidate / "matilda").exists():
        sys.path.insert(0, str(_candidate))
        try:
            from matilda.model import Transformer
            from matilda.config import ModelConfig
            _HAS_SRC = True
        except ImportError:
            pass
        break


class MatildaForCausalLM(PreTrainedModel):
    config_class = MatildaConfig
    supports_gradient_checkpointing = False
    _tied_weights_keys = ["model.lm_head.weight", "model.tok_emb.weight"]
    all_tied_weights_keys = {}  # transformers 5.x compat

    def __init__(self, config: MatildaConfig):
        super().__init__(config)
        if not _HAS_SRC:
            raise RuntimeError("src/matilda not found alongside exported dir")
        mcfg = ModelConfig(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            qk_norm=config.qk_norm,
            rope_theta=config.rope_theta,
            tie_weights=config.tie_weights,
        )
        self.model = Transformer(mcfg)

    def get_input_embeddings(self):
        return self.model.tok_emb

    def forward(self, input_ids, labels=None, **kwargs):
        logits, loss = self.model(input_ids, labels)
        from transformers.modeling_outputs import CausalLMOutput
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
