# Matilda-Mini

A sub-200M-parameter language model **trained from scratch** — and, more to the
point, the **training infrastructure** around it: distributed-ready training loop,
crash-safe checkpoint/resume, fault tolerance, observability, and a verifiable
data pipeline. Built from first principles in PyTorch, no training frameworks.

> Not a fine-tune. Not a wrapper. Random init → a working LM, trained by code in
> this repo. The model is standard-modern; the **systems work is the point**.

## Why this exists

This is a portfolio project for an LLM **training-infrastructure** role. The
interesting problems in training large models aren't the architecture (well
understood) — they're the systems: making multi-day runs reliable, resumable,
observable, and fast on the hardware you have. So this repo is deliberately
weighted toward operational excellence over architectural novelty.

## Architecture (`src/matilda/model.py`)

A modern dense decoder-only transformer — the same recipe as Llama/Qwen-class
models, at ~124M params:

| Component | Choice |
|-----------|--------|
| Positions | **RoPE** (rotary, half-rotation convention) |
| Normalization | **RMSNorm**, pre-norm, fp32 reduction |
| MLP | **SwiGLU** (2/3 sizing) |
| Attention | **GQA** (12 query / 4 KV heads) + **QK-Norm** |
| Stability | residual-projection init scaled 1/√(2·n_layers); optional attn logit soft-cap |
| Tying | embedding ↔ LM head |
| Size | 114M total / 75.5M non-embedding · d=768 · 12 layers · seq 1024 · GPT-2 50k vocab |

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
pytest tests/ -q                          # 30 tests: correctness, resume, NaN guard, data integrity

# train (synthetic dry run, no data needed)
python run.py --config configs/calibration.json --dry-run \
  --set model.d_model=128 model.n_layers=2 train.total_steps=20 train.device=cpu train.compile=false

# real run (after tokenizing data — see docs/INSTANCE_RUNBOOK.md)
python scripts/prepare_data.py --out-dir data/fwedu --target-tokens 3000000000
python run.py --config configs/base_124m.json --data-dir data/fwedu
```

Full GPU procedure (validate → calibrate → ablate → train → eval) is in
[`docs/INSTANCE_RUNBOOK.md`](docs/INSTANCE_RUNBOOK.md).

## Repository layout

```
src/matilda/    config, model, optim, checkpoint, monitor, data, train
scripts/        prepare_data.py (tokenize), ablate.py (experiments), launch_vast.sh
configs/        calibration.json (MFU tuning), base_124m.json (the run)
tests/          30 tests — model, checkpoint, train loop, data, optim, ablation, run
docs/           INSTANCE_RUNBOOK.md (operating manual)
run.py          training entrypoint (--config + --set overrides)
```

## Testing

30 tests run on CPU in ~2 min. Highlights: overfit-single-batch (the model can
learn), causal-mask-no-leak (no future-token leakage), bit-for-bit resume,
NaN-skip-then-recover, shard checksum corruption detection, Muon overfit.

```bash
pytest tests/ -q
```
