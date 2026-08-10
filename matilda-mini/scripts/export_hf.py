"""Export matilda checkpoint to a HuggingFace-compatible directory.

Creates a self-contained directory with:
  - modeling_matilda.py  (thin HF PreTrainedModel wrapper)
  - configuration_matilda.py
  - config.json
  - pytorch_model.bin
  - tokenizer files (GPT-2, same vocab)

Load with:
  AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True)

Usage:
  python scripts/export_hf.py --ckpt checkpoints/base_124m/ckpt_5639.pt \
                               --out  exported/matilda-mini-124m
"""
from __future__ import annotations
import argparse, sys, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
from matilda.config import BASE_124M
from matilda.model import Transformer


MODELING_CODE = '''"""Matilda causal LM wrapper for HuggingFace AutoModel."""
from __future__ import annotations
import sys, math
from pathlib import Path
import torch, torch.nn as nn
from transformers import PreTrainedModel
from .configuration_matilda import MatildaConfig

# import the real model from src/matilda if available, else inline
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from matilda.model import Transformer
    from matilda.config import ModelConfig
    _HAS_SRC = True
except ImportError:
    _HAS_SRC = False


class MatildaForCausalLM(PreTrainedModel):
    config_class = MatildaConfig
    supports_gradient_checkpointing = False

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
'''

CONFIGURATION_CODE = '''"""Matilda model configuration."""
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
'''

AUTO_MAP = {
    "AutoConfig": "configuration_matilda.MatildaConfig",
    "AutoModelForCausalLM": "modeling_matilda.MatildaForCausalLM",
}


def export(ckpt_path: str, out_dir: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- load checkpoint ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = raw.get("model", raw)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    mcfg = BASE_124M

    # --- write model code files ---
    (out / "modeling_matilda.py").write_text(MODELING_CODE)
    (out / "configuration_matilda.py").write_text(CONFIGURATION_CODE)

    # --- config.json ---
    cfg = {
        "model_type": "matilda",
        "architectures": ["MatildaForCausalLM"],
        "auto_map": AUTO_MAP,
        "vocab_size": mcfg.vocab_size,
        "max_seq_len": mcfg.max_seq_len,
        "d_model": mcfg.d_model,
        "n_layers": mcfg.n_layers,
        "n_heads": mcfg.n_heads,
        "n_kv_heads": mcfg.n_kv_heads,
        "qk_norm": mcfg.qk_norm,
        "rope_theta": mcfg.rope_theta,
        "tie_weights": mcfg.tie_weights,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "torch_dtype": "bfloat16",
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    # --- weights: save with HF-compatible key names ---
    # Wrap temporarily to get the right state_dict keys
    from matilda.config import ModelConfig
    model = Transformer(mcfg)
    model.load_state_dict(state, strict=True)
    # Save under "model." prefix to match MatildaForCausalLM.model
    hf_state = {"model." + k: v for k, v in model.state_dict().items()}
    torch.save(hf_state, out / "pytorch_model.bin")

    # --- tokenizer ---
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.save_pretrained(out)

    # --- copy src so the model code can import it ---
    src_dst = out / "src"
    if not src_dst.exists():
        shutil.copytree(ROOT / "src", src_dst)

    size_gb = (out / "pytorch_model.bin").stat().st_size / 1e9
    print(f"\nExported to {out}/")
    print(f"  pytorch_model.bin  {size_gb:.2f} GB")
    print(f"  config.json + modeling_matilda.py + tokenizer")
    print(f"\nTest load:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  m = AutoModelForCausalLM.from_pretrained('{out}', trust_remote_code=True)")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/base_124m/ckpt_5639.pt")
    ap.add_argument("--out", default="exported/matilda-mini-124m")
    args = ap.parse_args()
    export(args.ckpt, args.out)
