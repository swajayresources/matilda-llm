# Bugs and fixes — 2026-05-30

Each entry: symptom → diagnosis → fix → why this fix and not another.

---

## 1. `prepare_smollm_data.py` — `KeyError: 'text'` on `python-edu`

**Symptom.** Script crashed after streaming the first python-edu document:

```
File "scripts/prepare_smollm_data.py", line 93, in main
  ids = enc.encode_ordinary(doc[TEXT_KEY])
KeyError: 'text'
```

**Diagnosis.** `HuggingFaceTB/smollm-corpus` ships `python-edu` rows as
metadata only — `blob_id`, `repo_name`, `path`, `length_bytes`, `score`,
`int_score`. The actual code lives in Software Heritage S3 and must be
fetched per-row using the `blob_id`. The data prep script assumed `text` was
present in every subset, which is true for `fineweb-edu-dedup` and
`cosmopedia-v2` but not `python-edu`.

**Fix (applied for this run).** Drop python-edu, rebalance the remaining two
to preserve the 5:1 fineweb-edu : cosmopedia ratio:

```diff
 SOURCES = (
-    ("fineweb-edu-dedup", 0.75),
-    ("cosmopedia-v2",     0.15),
-    ("python-edu",        0.10),
+    ("fineweb-edu-dedup", 0.8333333333333333),  # 0.75 / 0.90
+    ("cosmopedia-v2",     0.1666666666666667),  # 0.15 / 0.90
 )
```

This patch is **not** in the committed `prepare_smollm_data.py` — the file
was reset to upstream form after the data was prepared. The actual data on
disk (and uploaded to HF as a dataset repo) carries the 83.33/16.67 mix.
Future runs that want to re-tokenize from scratch with python-edu should
add a Software Heritage S3 fetcher; example call:

```python
import boto3
from botocore import UNSIGNED
from botocore.config import Config
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
import gzip
obj = s3.get_object(Bucket="softwareheritage",
                    Key=f"content/{blob_id}")
text = gzip.decompress(obj["Body"].read()).decode("utf-8", errors="replace")
```

**Throughput note.** The upstream tokenizer is single-threaded tiktoken —
extrapolated to ~7 h for 15B tokens. For this run we patched
`scripts/prepare_smollm_data.py` to use `enc.encode_ordinary_batch(num_threads=32)`
in 512-document batches; throughput jumped to ~120 M tokens/min and the full
15 B run finished in ~2 h. That patch is also not committed; restore it if
re-tokenizing.

---

## 2. `liger_rotary_pos_emb` — CUDA illegal memory access in Triton autotune

**Symptom.** First step of any 363M calibration crashed with:

```
File "liger_kernel/ops/rope.py", line 217, in forward
  ...
File "triton/testing.py", line 118, in do_bench
  di.synchronize()
RuntimeError: CUDA error: an illegal memory access was encountered
terminate called after throwing an instance of 'c10::Error'
```

with `compile=True`. With `compile=False`, the same path produced an
illegal access at the next stage (`F.scaled_dot_product_attention`),
because Liger RoPE had returned corrupt q/k.

The pivot to 152M (head_dim=64 instead of 80) **did not fix this** — the
same crash reproduced at head_dim=64. The original hypothesis ("head_dim 80
is the trigger") was wrong; the actual trigger is the liger-kernel 0.8.0
Triton RoPE kernel × torch 2.6 inductor autotune, independent of head_dim.

**Diagnosis.** liger-kernel 0.8.0 ships a Triton kernel for the rotary
embedding. Inductor's autotune phase runs `benchmark_gpu` against that
kernel and one of the autotune configs allocates invalid memory. The actual
runtime kernel is fine — the bug is in the autotune-time benchmark, but
because it fires inside the compiled forward, the whole forward crashes.

**Fix (applied, in committed model.py patch).** Bypass Liger RoPE. Use the
eager `apply_rope` defined in `model.py`. Keep `LigerRMSNorm` and
`LigerFusedLinearCrossEntropy` engaged — those carry the throughput claim.
RoPE itself is < 1 % of step time, so the perf cost is negligible.

```python
def _apply_rope(self, q, k, cos, sin):
    # liger_rotary_pos_emb (Triton) crashes torch.compile autotune with
    # CUDA illegal memory access on liger-kernel 0.8.0 + torch 2.6,
    # independent of head_dim (reproduced at head_dim=80 and head_dim=64).
    # Eager apply_rope is < 1% of step time; LigerRMSNorm + LigerFusedLinearCE
    # stay engaged and carry the throughput claim.
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin)
```

**Alternatives considered.**
- Downgrade liger-kernel to 0.5.x or 0.4.x. Possible but a deeper bisect; not
  attempted this run.
- Disable `torch.compile`. Drops MFU by 10-20 %; loses Inductor's fused
  kernels. Too expensive.
- Switch attention dim. Tried head_dim=64 (pivot to 152M) — same crash.
- File a liger-kernel upstream issue: TODO. Reproducer in `bugs-and-fixes.md`
  + this folder.

---

## 3. `LigerRMSNorm` — `AttributeError: module 'torch.distributed' has no attribute 'tensor'`

**Symptom.** After the Liger RoPE bypass, the next 152M run failed
during first forward:

```
File "liger_kernel/ops/rms_norm.py", line 612, in forward
  if isinstance(X, torch.distributed.tensor.DTensor):
AttributeError: module 'torch.distributed' has no attribute 'tensor'
```

Under `torch.compile`, the same failure surfaced as a CUDA illegal memory
access (dynamo recovers from the AttributeError during graph capture but the
runtime path is then incoherent).

**Diagnosis.** liger-kernel 0.8.0's `LigerRMSNormFunction.forward` runs:

```python
if isinstance(X, torch.distributed.tensor.DTensor):
    ...
```

In torch 2.6, `torch.distributed.tensor` exists as a submodule but is
**not** auto-imported as an attribute of `torch.distributed`. Accessing
`torch.distributed.tensor` without first `import torch.distributed.tensor`
raises AttributeError.

Test suite (41 passing) didn't catch this because the BASE_152M shape test
runs LigerRMSNorm forward only after importing `liger_kernel.transformers.rope`,
which transitively imports `torch.distributed.tensor`. In `run.py`, the
import path differs and the submodule never gets touched, so production
hits the bug that tests don't.

**Fix (applied, in committed model.py patch).** Import the submodule
explicitly inside `_import_liger()`, scoped so CPU dev boxes that don't use
Liger don't pay any side effect:

```python
def _import_liger():
    # Liger 0.8.0 RMSNorm calls torch.distributed.tensor.DTensor at runtime;
    # in torch 2.6 that submodule isn't auto-loaded as an attribute, so the
    # first forward dies with AttributeError under eager or a CUDA illegal
    # memory access under torch.compile. Importing the submodule registers it.
    import torch.distributed.tensor  # noqa: F401
    try:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        ...
```

**Why this and not other approaches.**
- Adding `import torch.distributed.tensor` at the top of `model.py` would
  side-effect even on CPU runs without Liger. Scoping inside `_import_liger`
  matches existing lazy-import policy.
- Monkey-patching Liger is brittle and ages badly across upgrades.
- Pinning liger-kernel < 0.8.0 ducks the bug but would also revert any 0.8.x
  improvements.

---

## 4. `LigerFusedLinearCrossEntropy` — broken gradients at chunk_size ≥ 1024

**Symptom.** During the 152M BS sweep, calibration runs above BS = 16 either
NaN-aborted from step 0 (BS = 24, 32+) or ran with absurd grad norms
(BS = 20: gnorm = 477 772 at step 30, loss flat at 10.96 ± 0.02).

```
[nan-guard] step 0 skipped (1/20) near data_pos=2
[nan-guard] step 0 skipped (2/20) near data_pos=3
...
[nan-guard] step 0 skipped (20/20) near data_pos=21
RuntimeError: too many non-finite steps; aborting run
```

The same data positions ran fine at BS = 8 and BS = 16 — so it isn't a
data bug.

**Diagnosis.** `LigerFusedLinearCrossEntropy` chunks the (B·T, V) logit
matrix to bound memory. The chunk_size is auto-picked:

```
BT          = batch * seq_len
inc_factor  = ceil(V / H)              # 50257 / 768 = 66
chunk_size  = next_power_of_2(BT / inc_factor)
```

| BS | BT     | BT/66 | chunk_size |
|----|--------|-------|------------|
|  8 | 16384  |   248 |        256 |
| 16 | 32768  |   496 |        512 |
| 20 | 40960  |   620 |       1024 |
| 24 | 49152  |   744 |       1024 |
| 32 | 65536  |   993 |       1024 |
| 40 | 81920  |  1241 |       2048 |

The discontinuity at chunk_size = 1024 produces fp32 gradients that are
~40 000× too large at BS=20 (gnorm 477 772 vs ~10 at smaller BS). At
BS≥24, those grads overflow to inf, every batch trips the NaN guard, and
training aborts after `max_skips` consecutive skips.

**Fix (none applied).** Cap BS at 16. This is what limited the hero run
to BS=16, grad_accum=32 — not a memory ceiling. Real memory ceiling on
this stack is around BS=24 (BS=24 NaN-aborts with peak mem only ~32 GB,
plenty of room).

**Alternatives considered.**
- Force `chunk_size = 512` via kwarg — Liger 0.8.0 doesn't expose a chunk
  override; would need a fork.
- Downgrade Liger to a version with a different chunking algorithm — not
  attempted this run.
- Use the non-fused `F.cross_entropy` path — OOMs at BS=8 already at this
  scale (logits tensor (8 × 2048, 50257) = 3.3 GB just for forward).

---

## 5. Memory ceiling lower than runbook predicted

**Symptom.** Runbook's pre-computed table claimed BS=32-48 would fit on
40 GB with Liger. Empirically:

- 363M: BS=12 was the ceiling (with `expandable_segments`)
- 152M: BS=16 was the ceiling (Liger FLCE bug, not memory — see #4)

The 363M memory ceiling was real (40.4 GB peak at BS=12); BS=14+ genuinely
OOMed.

**Diagnosis.** Two compounding factors the runbook authors didn't have:
1. Liger 0.8.0 carries more workspace than the version they validated on
   (Liger 0.4.x baseline per `requirements.txt`).
2. The deep+thin 32-layer 363M shape stacks more activations than typical
   "350M" configs (which are usually wider+shallower).

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` recovers 1-2 GB of
fragmentation at the 363M ceiling — necessary to fit BS=12 there. At 152M
it's belt-and-braces; BS=16 fits without it too, but the hero run uses it
since that's the configuration calibration verified.

---

## Summary of working-tree state at hero-run launch

Local modifications vs `origin/main` commit `7a66951`:

- `src/matilda/model.py`: two patches
  - `_import_liger()` adds `import torch.distributed.tensor`
  - `_apply_rope()` bypasses `liger_rotary_pos_emb`, calls eager `apply_rope`

`scripts/prepare_smollm_data.py` is back to upstream — the python-edu drop
+ rebalance and the batched tokenization were applied during the 363M
attempt only. The resulting data was reused as-is for the 152M run.
