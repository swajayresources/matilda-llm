"""Convert a matilda .pt checkpoint into a HuggingFace-loadable folder.

Output folder contains:
  - config.json              (MatildaConfig + auto_map pointing at the
                              modeling_matilda / configuration_matilda files)
  - model.safetensors        (state_dict with `_orig_mod.` prefix stripped and
                              keys rewritten to match the HF Matilda module tree:
                              `embed` → `model.embed`, `blocks` → `model.blocks`,
                              `norm_f` → `model.norm_f`, and `lm_head` added
                              from the tied embedding)
  - modeling_matilda.py      (copied from eval/hf/, with relative import patched)
  - configuration_matilda.py (copied)
  - tokenizer files          (gpt2 fast tokenizer files saved here so the
                              loader doesn't need a network round-trip)
  - generation_config.json   (sensible defaults)
  - README.md                (model card)

Usage:
    python eval/hf/convert_to_hf.py \\
        --ckpt checkpoints/base_152m_v2/ckpt_7150.pt \\
        --out hf-matilda-152m-v15-7.5B
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from configuration_matilda import MatildaConfig  # noqa: E402


# State-dict key rewrites: matilda Transformer -> HF MatildaForCausalLM
# matilda layout                   HF MatildaForCausalLM layout
# ─────────────────                ────────────────────────────
# embed.weight                  -> model.embed.weight  (+ lm_head.weight tied)
# blocks.{i}.norm1.weight       -> model.blocks.{i}.norm1.weight
# blocks.{i}.attn.wq.weight     -> model.blocks.{i}.attn.wq.weight
# blocks.{i}.attn.wk.weight     -> model.blocks.{i}.attn.wk.weight
# blocks.{i}.attn.wv.weight     -> model.blocks.{i}.attn.wv.weight
# blocks.{i}.attn.wo.weight     -> model.blocks.{i}.attn.wo.weight
# blocks.{i}.attn.q_norm.weight -> model.blocks.{i}.attn.q_norm.weight
# blocks.{i}.attn.k_norm.weight -> model.blocks.{i}.attn.k_norm.weight
# blocks.{i}.norm2.weight       -> model.blocks.{i}.norm2.weight
# blocks.{i}.mlp.up.weight      -> model.blocks.{i}.mlp.up.weight
# blocks.{i}.mlp.down.weight    -> model.blocks.{i}.mlp.down.weight
# norm_f.weight                 -> model.norm_f.weight


def remap_state_dict(sd: dict, tie_weights: bool) -> dict:
    out = {}
    embed_weight = None
    for k, v in sd.items():
        k = k.removeprefix("_orig_mod.")
        if k == "lm_head.weight":
            # in the native matilda checkpoint this is identical to (and
            # shares storage with) embed.weight via tying. Skip here — we'll
            # re-derive it below from the embedding so the safetensors file
            # is self-contained.
            continue
        if k == "embed.weight":
            embed_weight = v
        out["model." + k] = v
    # HF expects lm_head.weight as an explicit key (it'll be re-tied to
    # model.embed.weight at construction via config.tie_word_embeddings=True,
    # but load_state_dict still wants the key present). Save a clone so
    # safetensors doesn't refuse on shared storage.
    if embed_weight is None:
        raise KeyError("embed.weight not found in checkpoint")
    out["lm_head.weight"] = embed_weight.clone() if tie_weights else out.get("lm_head.weight")
    return out


def save_tokenizer(out_dir: Path):
    """Save the GPT-2 fast tokenizer files into the HF model dir."""
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.save_pretrained(out_dir)


def write_config_json(cfg: MatildaConfig, out_dir: Path):
    # auto_map directs `AutoModelForCausalLM.from_pretrained(... trust_remote_code=True)`
    # to load the classes from the files we ship alongside the weights.
    cfg.auto_map = {
        "AutoConfig": "configuration_matilda.MatildaConfig",
        "AutoModel": "modeling_matilda.MatildaModel",
        "AutoModelForCausalLM": "modeling_matilda.MatildaForCausalLM",
    }
    cfg.architectures = ["MatildaForCausalLM"]
    cfg.torch_dtype = "bfloat16"
    cfg.save_pretrained(out_dir)


def copy_modeling_files(out_dir: Path):
    # Ship verbatim. trust_remote_code loads these via transformers'
    # dynamic-module loader which preserves package context (so the
    # `from .configuration_matilda import ...` relative form keeps working).
    for name in ("configuration_matilda.py", "modeling_matilda.py"):
        shutil.copyfile(_HERE / name, out_dir / name)


def write_generation_config(out_dir: Path, eos_token_id: int):
    gen_cfg = {
        "bos_token_id": eos_token_id,    # GPT-2 has no BOS; reuse EOT
        "eos_token_id": eos_token_id,
        "pad_token_id": eos_token_id,
        "max_new_tokens": 256,
        "temperature": 0.8,
        "top_p": 0.95,
    }
    (out_dir / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2))


def write_model_card(out_dir: Path, ckpt_step: int, tokens_seen: int):
    card = f"""---
license: odc-by
language:
- en
pipeline_tag: text-generation
library_name: transformers
tags:
- causal-lm
- gqa
- relu2
- muon
- wsd
base_model: prometheus04/matilda-152m-v15-7.5B
---

# matilda-152m-v15

152M-parameter decoder-only transformer trained from scratch on a single A100
SXM4 40GB. Token-matched **beats Pythia-160M on PIQA, ARC-easy, HellaSwag and
LAMBADA at every checkpoint** (avg +0.05 / +0.15 / +0.05 / +0.09); ties on
Winogrande.

This is the **HuggingFace-format mirror** of checkpoint step {ckpt_step} (~{tokens_seen/1e9:.2f}B
training tokens) from `prometheus04/matilda-152m-v15-7.5B`. Use it with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "prometheus04/matilda-152m-v15-7.5B",
    trust_remote_code=True,
    torch_dtype="bfloat16",
).to("cuda")
tok = AutoTokenizer.from_pretrained("prometheus04/matilda-152m-v15-7.5B")
out = model.generate(**tok("The capital of France is", return_tensors="pt").to("cuda"),
                     max_new_tokens=40, do_sample=True, temperature=0.8)
print(tok.decode(out[0]))
```

## Architecture

- 18 layers × 768 hidden × 12 attention heads / 4 KV heads (GQA, head_dim 64)
- RoPE (theta 10000), RMSNorm pre-norm, **QK-Norm** per head
- **ReLU² FFN** (Primer / modded-nanogpt; 2 matmuls instead of SwiGLU's 3)
- **Final logit soft-cap 30.0** (Gemma-2 style)
- Tied input/output embeddings, GPT-2 BPE vocab (50 257)

## Training recipe

- **Muon hybrid optimizer** (Muon for 2-D weights at lr=0.02, AdamW for 1-D
  params and embeddings at lr=3e-4 — no weight decay on norms/biases/embeddings)
- **WSD schedule**: 700 warmup, ~5160 stable at peak, ~1290 linear decay to
  min_lr_ratio=0.1 of peak
- 7150 optimizer steps × 1.05M tokens/step ≈ **7.5B total tokens**, ~23h on
  A100 SXM4 40GB at 29.6 % avg MFU
- Liger-kernel RMSNorm + FusedLinearCrossEntropy at training (RoPE Triton
  kernel bypassed — see `prometheus04/matilda-mini-v2` `docs/instance-run-2026-05-30/`)
- Data: 15B-token mix of FineWeb-Edu-dedup (83.3 %) + Cosmopedia v2 (16.7 %)
  from `prometheus04/matilda-smollm-mix-15b-gpt2`

## Eval (lm-eval-harness, 0-shot, full splits)

Token-matched against published EleutherAI/pythia-160m revisions
(closest-≥ matilda tokens):

| Task | Matilda (7.50B) | Pythia step4000 (8.39B) | Δ |
|------|-----------------|-------------------------|---|
| PIQA acc | 0.649 | 0.592 | **+0.057** |
| ARC-easy acc | 0.554 | 0.419 | **+0.135** |
| HellaSwag acc_norm | 0.347 | 0.275 | **+0.072** |
| Winogrande acc | 0.526 | 0.529 | -0.003 (tie) |
| LAMBADA acc | 0.287 | 0.248 | **+0.040** |

Across all 10 saved checkpoints × 5 tasks: **41 wins, 3 ties, 2 losses (Winogrande only)**.
Full per-checkpoint curve in `prometheus04/matilda-mini-v2` `eval/matilda_results/`.

## Reproduce

The training code, custom lm-eval wrapper, conversion script, and operating
journal live in [`prometheus04/matilda-mini-v2`](https://huggingface.co/prometheus04/matilda-mini-v2).
The 15 B-token pre-tokenized corpus is in
[`prometheus04/matilda-smollm-mix-15b-gpt2`](https://huggingface.co/datasets/prometheus04/matilda-smollm-mix-15b-gpt2).
"""
    (out_dir / "README.md").write_text(card)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to matilda .pt file")
    ap.add_argument("--out", required=True, help="output HF-format directory")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[convert] loading {args.ckpt}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ck["config"]
    if hasattr(cfg_dict, "to_dict"):
        cfg_dict = cfg_dict.to_dict()
    # Drop liger flag from the saved config — at HF load time we let
    # _try_liger_rms_norm decide based on what's available.
    cfg_dict.pop("use_liger", None)
    cfg = MatildaConfig(**cfg_dict)

    sd = remap_state_dict(ck["model"], tie_weights=cfg.tie_weights)

    # Add the RoPE cos/sin buffers. They aren't in the native checkpoint
    # because matilda registers them non-persistent and recomputes per
    # __init__. In the HF wrapper we make them persistent so they survive
    # `from_pretrained(init_empty_weights)` — so they need to be in the file.
    # Inlined here so convert_to_hf doesn't need to import the modeling file
    # (which uses package-relative imports for the HF runtime layout).
    head_dim = cfg.head_dim
    i = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (cfg.rope_theta ** (i / head_dim))
    t = torch.arange(cfg.max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    sd["model.rope_cos"] = emb.cos().contiguous()
    sd["model.rope_sin"] = emb.sin().contiguous()

    print(f"[convert] state_dict entries: {len(sd)}")

    safetensors_path = out_dir / "model.safetensors"
    # safetensors disallows shared storage; tied embeddings would normally
    # share storage, but here we excluded `lm_head.weight` from the dump,
    # so there are no shared tensors left.
    save_file(sd, str(safetensors_path), metadata={"format": "pt"})
    print(f"[convert] wrote {safetensors_path} ({safetensors_path.stat().st_size/1e6:.1f} MB)")

    write_config_json(cfg, out_dir)
    print(f"[convert] wrote config.json")

    copy_modeling_files(out_dir)
    print(f"[convert] copied modeling_matilda.py + configuration_matilda.py")

    save_tokenizer(out_dir)
    print(f"[convert] saved gpt2 tokenizer files")

    eos = 50256  # GPT-2 EOT
    write_generation_config(out_dir, eos)
    print(f"[convert] wrote generation_config.json")

    write_model_card(out_dir, ckpt_step=ck.get("step", 0),
                     tokens_seen=ck.get("step", 0) * 16 * 32 * 2048)
    print(f"[convert] wrote README.md")
    print(f"[convert] done -> {out_dir}")
    print(f"[convert] files: {sorted(p.name for p in out_dir.iterdir())}")


if __name__ == "__main__":
    main()
