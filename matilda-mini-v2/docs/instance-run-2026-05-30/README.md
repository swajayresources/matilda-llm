# Instance run journal — 2026-05-30 → 2026-05-31

Operating notes from the A100-SXM4-40GB run that produced the 152M v1.5 hero
checkpoint. Captures every blocker hit between `git clone` and the hero
training launch, the diagnoses, the patches actually applied to the working
tree, and the calibration data that drove the final BS / grad_accum / steps
decision.

Audience: the next person (or future-me) doing a repeat run on a fresh
instance — so they can skip the 3-4h of debugging that the runbook didn't
anticipate.

## TL;DR

The runbook expected the v2 363M shape to fit at BS=32-48 on 40GB with the
full Liger stack engaged. Empirically, on this exact toolchain
(torch 2.6.0+cu124, liger-kernel 0.8.0, triton 3.2.0), it didn't:

1. `liger_rotary_pos_emb` crashes torch.compile autotune with a CUDA illegal
   memory access — **at any head_dim** (363M head_dim=80 and 152M head_dim=64
   both reproduce). Worked around by bypassing Liger RoPE; eager `apply_rope`
   is < 1 % of step time.
2. `LigerRMSNorm` calls `torch.distributed.tensor.DTensor` at runtime, but
   torch 2.6 doesn't auto-load that submodule. One-line fix: explicit
   `import torch.distributed.tensor` in `_import_liger()`.
3. `LigerFusedLinearCrossEntropy`'s chunked Triton kernel produces broken
   gradients (gnorm ≈ 5×10⁵ at BS=20, all-NaN at BS≥24) once the auto-picked
   `chunk_size` crosses 512 → 1024. Limits BS to ≤ 16 on this stack.

Also: `HuggingFaceTB/smollm-corpus` `python-edu` subset is `blob_id`
pointers to Software Heritage S3, not inline text. `prepare_smollm_data.py`
crashed with `KeyError: 'text'`. We dropped it and rebalanced the mix to
83.33 % fineweb-edu-dedup / 16.67 % cosmopedia-v2 (preserves the 5:1 ratio).

After the workarounds, the **363M** plan was infeasible (BS ceiling 12 on
this stack, 4-5 day ETA at observed MFU 32%). The repo pivoted to **152M**
(commit `7a66951`), reusing the same 15B-token corpus, and calibration found
BS=16 as the working ceiling there too — but at much healthier mem headroom
(34%) and a runbook-aligned ~25h ETA.

The hero run launched at `2026-05-30 ~16:53 UTC` with:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python run.py --config configs/base_152m_v2.json --data-dir data/smollm_mix \
  --set train.batch_size=16 train.grad_accum=32 train.total_steps=7150 \
        train.warmup_steps=700 train.ckpt_every=750
```

7150 steps × 1.05 M tokens/step = **7.5 B tokens** (half the runbook target,
chosen to keep total wall-clock under one day).

## Files in this folder

- `README.md` (this file)
- `bugs-and-fixes.md` — the three Liger bugs, the python-edu data bug, full
  diagnoses + applied patches.
- `calibration-data.md` — the BS sweep table, with mem and MFU per BS.
- `model.py.patch` — the actual model.py diff carrying the Liger workarounds.
- `hf-conversion.md` — how `matilda.model.Transformer` got wrapped as a
  HuggingFace `AutoModelForCausalLM` (added 2026-05-31 post-eval): three
  landmines hit (`tie_word_embeddings` round-trip, `_tied_weights_keys`
  format change, **non-persistent RoPE buffers destroyed by
  `from_pretrained`**), bit-exact parity verification, and the
  `trust_remote_code=True` one-liner load.

## Run-time progress (live as of writing)

- Step 50: loss 8.61, gnorm 3.65, MFU 29.7 %, 90.7 k tok/s
- Step 200: loss 5.83 (already in runbook's "step 500" range)
- Step 500: loss 4.04 (runbook expected 6-7)
- Step 1000: loss 3.37 (runbook expected ~5; at/below their *final* projection)
- Step 2000: loss 3.13

The pace at the start was much faster than the runbook AdamW+cosine baseline.
Most of that is the Muon hybrid getting a clean orthogonalized step on a fresh
init — Muon equalizes singular-value scales, so every weight direction gets
the same effective learning rate, and the residual stream organizes faster
than Adam's curvature-following ramp would allow.

Final loss / eval numbers go in a follow-up commit after the run completes
and lm-eval-harness lands on the saved checkpoints.
