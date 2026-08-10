# Matilda-Mini

A sub-400M-parameter language model **trained from scratch** — and, more to the
point, the **training infrastructure** around it: distributed-ready training loop,
crash-safe checkpoint/resume, fault tolerance, observability, and a verifiable
data pipeline. Built from first principles in PyTorch, no training frameworks.

> Not a fine-tune. Not a wrapper. Random init → a working LM, trained by code in
> this repo. The model is standard-modern; the **systems work is the point**.

## Three configs, one stack

| Config | Params | Tokens | Optimizer | Data | Schedule | Purpose |
|---|---|---|---|---|---|---|
| `configs/base_124m.json` | 114M | ~3B | AdamW | FineWeb-Edu | warmup+cosine | v1 — original portfolio run |
| **`configs/base_152m_v2.json`** | **152M** | **~15B** | **Muon+AdamW hybrid** | **SmolLM-corpus 83/17** | **WSD (MiniCPM)** | **v1.5 — actual hero run (18L × 768d, ReLU², soft-cap, Liger)** |
| `configs/base_350m.json` | 363M | (15B) | Muon+AdamW | SmolLM-corpus | WSD | reference only — pivoted away from (activation-memory ceiling on 40GB) |

The **v1.5 hero run** (152M) is designed around two claims: (1) **direct token-
matched comparison vs Pythia-160M** (same width, 50% more depth, better data,
better optimizer); (2) **approaching Pythia-410M at the same token count
despite being 2.7× smaller** — the "beats above weight" framing. Eight things
differ from v1: deeper architecture (18L vs 12L), seq_len 2048 (vs 1024), Muon
hybrid (vs AdamW), WSD schedule (vs cosine), Liger fused linear+CE, z-loss,
ReLU² FFN (vs SwiGLU), final logit soft-cap.

## Why this exists

This is a portfolio project for an LLM **training-infrastructure** role. The
interesting problems in training large models aren't the architecture (well
understood) — they're the systems: making multi-day runs reliable, resumable,
observable, and fast on the hardware you have. So this repo is deliberately
weighted toward operational excellence over architectural novelty.

## Architecture (`src/matilda/model.py`)

A modern dense decoder-only transformer — the same recipe as Llama/Qwen-class
models. Shape is a runtime knob:

| Component | v1 (114M) | **v1.5 (152M, hero)** |
|-----------|-----------|------------------------|
| Layers × d_model × n_heads | 12 × 768 × 12 | **18 × 768 × 12** (50% deeper) |
| KV heads (GQA) | 4 | 4 |
| Head dim | 64 | 64 (Flash + Liger RoPE friendly) |
| Seq length | 1024 | **2048** |
| Positions | RoPE | RoPE (Liger fused) |
| Normalization | RMSNorm (fp32 reduction) | RMSNorm (Liger fused) |
| MLP | SwiGLU (8/3 sizing) | **ReLU² (4.0 sizing, param-matched)** |
| QK-Norm | on | on |
| Embedding tying | on | on |
| Loss | CE | CE **+ z-loss (1e-4)** **+ final logit soft-cap 30** |
| Vocab projection | `nn.Linear` + `F.cross_entropy` | **Liger fused linear+CE** (skips logit materialization, supports softcap natively) |
| Optimizer | AdamW | **Hybrid Muon (2D) + AdamW (embed/norm)** |
| LR schedule | warmup + cosine | **warmup + 80% stable + 20% linear decay (WSD)** |

## Training infrastructure (the actual deliverable)

| Capability | Where | What it does |
|-----------|-------|--------------|
| **Bit-for-bit resume** | `checkpoint.py` | atomic writes; saves model+opt+sched+step+**RNG+dataloader position**; a killed run resumes to a loss curve identical to the uninterrupted one (`< 1e-6`, tested) |
| **Fault tolerance** | `train.py` | NaN/Inf guard (skip+log+abort-after-N); SIGTERM → checkpoint-and-exit for spot-instance death |
| **Observability** | `monitor.py` | MFU (incl. attention FLOPs), tokens/s, rolling step-time (catches throttling), grad-norm, peak GPU mem → always-on `metrics.jsonl` + optional W&B |
| **Throughput** | `train.py` | bf16 autocast, Flash-SDPA, `torch.compile`, fused AdamW, TF32, pinned/non-blocking H2D, grad-accum with DDP `no_sync` |
| **Data pipeline** | `data.py`, `scripts/prepare_data.py` | streams FineWeb-Edu → tokenizes → `uint16` shards with **SHA-256 manifest**; mmap'd, resumable `BinStream` |
| **Optimizers** | `optim.py` | AdamW (correct param-group decay) + **Muon** (Newton-Schulz orthogonalization, hybrid with AdamW) |
| **Reproducibility** | `train.py` | full config + **git SHA** logged per run; deterministic seeding |

## Results

**Validated (RTX 3090):** 30/30 tests pass on GPU, smoke + bit-for-bit resume
clean, **53.4% MFU** at batch_size=24 with `torch.compile` (BS≥28 OOMs on the
vocab projection — the expected memory hotspot).

**Training run + ablations:** pending the A100 run. The ablation harness
(`scripts/ablate.py`) emits `docs/ABLATIONS.md` — a controlled comparison, one
change per row:

| Variant | What it isolates |
|---------|------------------|
| baseline | full modern stack |
| no_qk_norm | QK-Norm's stability contribution |
| mha / mqa | GQA ratio vs full multi-head / multi-query |
| muon | Muon vs AdamW convergence |

Target (124M, ~3B tokens, vs Pythia-160M): HellaSwag ~30-35%, ARC-easy ~40-45%,
PIQA ~60%.

## Quickstart

```bash
pip install -r requirements.txt          # GPU: install torch from cu124 first (see runbook)
pytest tests/ -q                          # 35 tests: correctness, resume, NaN guard, data integrity, WSD, z-loss

# train (synthetic dry run, no data needed)
python run.py --config configs/calibration.json --dry-run \
  --set model.d_model=128 model.n_layers=2 train.total_steps=20 train.device=cpu train.compile=false

# v1 run (124M / FineWeb-Edu / ~3B tokens)
python scripts/prepare_data.py --out-dir data/fwedu --target-tokens 3000000000
python run.py --config configs/base_124m.json --data-dir data/fwedu

# v1.5 hero run (152M / SmolLM 83/17 / ~15B tokens) — A100 + liger-kernel required
pip install liger-kernel
python scripts/prepare_smollm_data.py --out-dir data/smollm_mix --target-tokens 15000000000
python run.py --config configs/base_152m_v2.json --data-dir data/smollm_mix
```

Full GPU procedure (validate → calibrate → ablate → train → eval) is in
[`docs/INSTANCE_RUNBOOK.md`](docs/INSTANCE_RUNBOOK.md).

## Repository layout

```
src/matilda/    config, model, optim, checkpoint, monitor, data, train
scripts/        prepare_data.py (FineWeb-Edu), prepare_smollm_data.py (SmolLM 75/15/10 mix),
                ablate.py (experiments), launch_vast.sh
configs/        calibration.json (MFU tuning), base_124m.json (v1), base_350m.json (v2 hero)
tests/          35 tests — model, checkpoint, train loop, data, optim, ablation, run, WSD, z-loss
docs/           INSTANCE_RUNBOOK.md (operating manual)
run.py          training entrypoint (--config + --set overrides)
```

## Testing

35 tests run on CPU in ~2 min. Highlights: overfit-single-batch (the model can
learn), causal-mask-no-leak (no future-token leakage), bit-for-bit resume,
NaN-skip-then-recover, shard checksum corruption detection, Muon overfit, WSD
three-phase shape, z-loss equals CE + lse² when on. BASE_350M shape test skips
on CPU dev boxes without `liger-kernel`.

```bash
pytest tests/ -q
```
