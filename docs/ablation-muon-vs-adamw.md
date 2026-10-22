# Muon vs AdamW — 124M model, 300M tokens

**Date:** 2026-05-30
**Status:** completed A/B run; artifacts were gitignored on the rented instance and not recovered — figures below are from the run log recorded at the time.

## Setup

| | |
|---|---|
| Model | Matilda-Mini v1 — 124M params, 12L × 768d, GQA 12/4, RoPE, QK-Norm, SwiGLU, tied embeddings |
| Tokenizer | GPT-2 BPE (50,257) |
| Data | FineWeb-Edu sample-10BT, identical stream for both arms |
| Budget | **300M tokens per arm** (matched tokens, not matched loss) |
| Batching | batch size 24 × grad-accum 8 |
| Hardware | 1× RTX 3090, bf16 |
| Muon | Newton–Schulz⁵ orthogonalised momentum on 2-D interior weights, `muon_lr=0.02`; AdamW on embeddings, norms and biases |
| AdamW | `lr=6e-4`-class stack, same schedule and data order |

## Result

| Arm | Final train loss | NaNs | MFU |
|---|---:|---:|---|
| AdamW | **3.7236** | 0 | 31–34% |
| Muon | **3.5570** | 0 | 31–34% |

**Δ = −0.167 nats for Muon (≈15% lower perplexity) at equal tokens.**

## Caveats

- The gap was judged large enough not to need a seed repeat; no multi-seed arm was run.
- Both arms thermally throttled at 84–85 °C, so this is a fair A/B on the same box but **not** a peak-MFU measurement.
- Comparison is at **matched tokens** (equal compute). A matched-loss comparison answers a different question — "how much earlier does Muon reach a given loss" — and needs intermediate checkpoints.
- Muon had only been unit-tested before this run; this was its first real training run on this stack.

## Independent follow-up (published 11 days later)

Maincode, *"Same Dictionary, Different Geometry"*, **2026-06-10** — a deeper 124M study on a matched-loss protocol:

- 124M GPT-2-class model (12L, d=768, **RoPE, QK-norm, ReLU²**), FineWeb, 3 seeds per arm, 8×B200.
- Final losses at 5,100 steps: **Muon 3.292 ± 0.001 vs AdamW 3.350 ± 0.002** — the loss gap reproduces.
- Their question is different from this one: *at matched loss*, are the learned features different? Answer: **the feature set is the same up to seed noise**, but the packaging differs — Muon spreads weights across 2–10× more directions (higher stable rank), runs the residual stream 4–7× hotter, produces fewer and rarer SAE features packed closer together, and is **more reproducible across seeds**.
- Code: <https://github.com/maincodeMax/muon-feature-study>
- Post: <https://maincode.com/blog/same-dictionary-different-geometry>

This run measured *loss at equal tokens*; Maincode measured *feature geometry at equal loss*. The two are complementary: Muon wins the compute-efficiency question here, and their result shows the win is not achieved by learning a different feature dictionary.
