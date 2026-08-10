---
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

This is the **HuggingFace-format mirror** of checkpoint step 7150 (~7.50B
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
Full per-checkpoint curve: `eval/matilda_results/` (10 JSON + `summary.csv`) and
`eval/pythia_results/` (5 JSON) in this repo.

## What's in this repo

- `model.safetensors`, `ckpt_750.pt … ckpt_7150.pt` — final + 10 intermediate checkpoints
- `configs/base_152m_v2.json` — exact training config
- `eval/` — `run_eval.py`, `run_pythia.py`, `matilda_lm.py` and the raw per-checkpoint JSON for both Matilda and Pythia-160M
- `tools/convert_to_hf.py`, `tools/verify_parity.py` — checkpoint → HF-format conversion and parity check
- `metrics.jsonl` — training-time loss / MFU / grad-norm log

## Reproduce

The full training stack, operating journal, and DeepSeek review bundle live in
[`prometheus04/matilda-mini-v2`](https://huggingface.co/prometheus04/matilda-mini-v2).
The 15B-token pre-tokenized corpus is
[`prometheus04/matilda-smollm-mix-15b-gpt2`](https://huggingface.co/datasets/prometheus04/matilda-smollm-mix-15b-gpt2).
