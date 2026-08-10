<p align="center">
  <img src="assets/banner.svg" width="100%"/>
</p>

# Matilda â€” LLMs Trained From Scratch

Decoder-only transformers trained from the ground up in **pure PyTorch â€” no training frameworks** â€” with a production-grade training stack: crash-safe checkpoints, NaN/Inf fault tolerance, SIGTERM handling, SHA-256-verified data pipelines, and MFU observability.

> Hugging Face: [prometheus04/matilda-mini](https://huggingface.co/prometheus04/matilda-mini) Â· [matilda-mini-v2](https://huggingface.co/prometheus04/matilda-mini-v2) Â· [matilda-152m-v15-7.5B](https://huggingface.co/prometheus04/matilda-152m-v15-7.5B)

## The three generations

| | **v1 â€” Matilda-Mini** | **v1.5 â€” Matilda-152M** | **v2 â€” Matilda-mini-v2 (infra)** |
|---|---|---|---|
| Params | 114M | 152M | 152M hero config |
| Architecture | 12LÃ—768, SwiGLU, GQA 12/4, QK-Norm, tied embeddings | 18L, GQA, RoPE, QK-Norm, ReLUÂ² FFN, logit soft-cap | 18LÃ—768Ã—12 heads, 4 KV heads, seq 2048, ReLUÂ², soft-cap 30.0, z-loss 1e-4 |
| Training | ~3B FineWeb-Edu tokens | **7.5B tokens in ~23h on 1Ã—A100** | hybrid **Muon + AdamW**, **WSD** 700/5160/1290, Liger kernels |
| Efficiency | **53.4% MFU** on RTX 3090 (torch.compile) | 29.6% avg MFU | â€” |
| Tests | 30/30 GPU correctness + resume | â€” | 35 tests, bit-for-bit resume (<1e-6) |

## vs Pythia-160M (token-matched, 0-shot)

| Task | Matilda-152M | Pythia-160M | Î” |
|---|---|---|---|
| PIQA | **0.649** | 0.592 | **+5.7 pt** |
| ARC-easy | **0.554** | 0.419 | **+13.5 pt** |
| HellaSwag (acc_norm) | **0.347** | 0.275 | **+7.2 pt** |
| LAMBADA | **0.287** | 0.248 | **+4.0 pt** |
| Winogrande | 0.526 | 0.529 | âˆ’0.3 (tie) |

**41 wins / 3 ties / 2 losses** across 10 checkpoints Ã— 5 tasks (lm-eval-harness).

## Infrastructure highlights

- **Crash-safe training**: bit-for-bit checkpoint resume (<1e-6), NaN/Inf guards, SIGTERM spot-instance handling â€” multi-day runs never restart from zero
- **Data integrity**: mmap'd pipeline over FineWeb-Edu + Cosmopedia, SHA-256 manifest, EOT token control
- **Optimization**: hybrid Muon (2D, lr 0.02) + AdamW with WSD schedule; Liger fused RMSNorm + linear+CE; MFU monitor
- **Eval rigor**: per-checkpoint curves, 10 checkpoints Ã— 5 tasks, conversion + parity tooling

## Repo layout

```
matilda-mini/            v1 code + training stack (114M)
matilda-mini-v2/         v2 infra + 152M hero config + tests
matilda-152m-v15-7.5B/   the hero run: checkpoints index, metrics.jsonl, eval/ (10 JSON + summary)
```

Model weights (large) live on Hugging Face â€” this repo carries code, configs, metrics and eval results.

