# Calibration data — 2026-05-30

Two sweeps were run. The 363M sweep led to the v1.5 pivot. The 152M sweep
chose the hero run's BS=16.

All runs: `seq_len=2048`, `compile=true`, bf16, Liger RMSNorm + FLCE on,
Liger RoPE bypassed (eager), 50 calibration steps unless noted.

## 363M sweep (configs/base_350m.json)

| BS | grad_accum | env       | mem peak | MFU   | tok/s   | result |
|----|------------|-----------|----------|-------|---------|--------|
| 32 | 2          | default   | 39.30 GB at OOM | – | – | OOM @ compile autotune |
| 32 | 2          | expandable| 38.74 GB at OOM | – | – | OOM @ compile autotune |
| 16 | 2          | default   | 39.24 GB at OOM | – | – | OOM @ FLCE forward |
| 16 | 2          | expandable| 39.27 GB at OOM | – | – | OOM @ FLCE forward |
| 14 | 2          | expandable| 39.41 GB at OOM | – | – | OOM @ FLCE forward |
| **12** | 2      | expandable| **40.40 GB** | **32.4 %** | 38.3 k | ✓ |
| 10 | 2          | default   | 35.27 GB | 31.8 % | 37.5 k | ✓ |
|  8 | 2          | default   | 29.80 GB | 30.2 % | 35.6 k | ✓ |
|  4 | 1          | default no-compile | 17.21 GB | 15.0 % | 17.5 k | baseline |

Verdict: 363M runs at BS=12 with no mem headroom, MFU 32 %, hero ETA
~4.5 days at observed throughput. Decision: pivot to 152M.

## 152M sweep (configs/base_152m_v2.json)

| BS | grad_accum | env       | mem peak | MFU   | tok/s   | gnorm | result |
|----|------------|-----------|----------|-------|---------|-------|--------|
| 56 | 2          | expandable| 38.97 GB at OOM | – | – | – | OOM |
| 48 | 2          | expandable| 39.18 GB at OOM | – | – | – | OOM |
| 40 | 2          | expandable| 39.28 GB at OOM | – | – | – | OOM (runbook's "shipped default") |
| 32 | 2          | expandable| 39.42 GB at OOM | – | – | – | OOM |
| 32 | 2          | default   | 1.14 GB fragmented | – | – | – | OOM, ES helps but not enough |
| 24 | 2          | expandable| – | – | – | – | **NaN abort** (20× nan_skip from step 0) |
| 20 | 2          | expandable| 30.54 GB | 33.3 % | 101 k | 477 772 | broken grads, loss flat |
| **16** | 2      | expandable| **25.96 GB** | **27.2 %** | 83 k | 9.3 | ✓ healthy |
|  8 | 2          | expandable| 16.29 GB | 22.6 % | 69 k | 9.8 | ✓ healthy |

Verdict at the 50-step measurement: **BS=16** is the working ceiling. Mem
peak 26 GB → 34 % headroom on the 40 GB card. MFU 27 %, tok/s 83 k. Beyond
BS=16, Liger FLCE's `chunk_size` jumps from 512 → 1024 and produces
40 000×-too-large gradients at BS=20, all-NaN at BS≥24 (see
`bugs-and-fixes.md` §4). The BS=32-56 OOMs are unrelated — those would also
fail on the FLCE bug if memory weren't first to die.

## Hero-run extrapolation

From calibration's BS=16 numbers:
- tokens/step = 16 × 32 × 2048 = 1 048 576
- 7150 steps × 1.05 M = 7.51 B tokens
- at 83 k tok/s: 7.51 B / 83 k = 25.1 h

Actual at-launch throughput came in higher than calibration measured:
~90.7 k tok/s at step 50 → ETA ~23 h. Likely cause: calibration's grad_accum=2
under-utilizes the FLCE backward pass, which amortizes its workspace setup
across accumulation steps. At grad_accum=32 the per-step cost is dominated
by the matmuls, not setup.

## Memory headroom rule

Calibration ran 50 steps. Long runs occasionally see late-run mem creep
from CUDA cache + Liger workspace re-allocation. Targeting **≥ 15 %
headroom** on the 40 GB card means peak ≤ 33.6 GB. BS=16 at 26.0 GB
clears this easily (~34 % headroom). The hero run holds peak in the
25-26 GB band at step 2000 — no creep observed so far.
