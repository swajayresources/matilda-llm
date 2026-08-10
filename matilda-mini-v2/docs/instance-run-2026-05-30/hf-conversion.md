# HuggingFace-format conversion — 2026-05-31

Adapter that turns a `matilda.model.Transformer` checkpoint into a folder
that `transformers.AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
loads as a one-liner. End-to-end goal:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "prometheus04/matilda-152m-v15-7.5B",
    trust_remote_code=True, dtype="bfloat16",
).cuda().eval()
tok = AutoTokenizer.from_pretrained("prometheus04/matilda-152m-v15-7.5B")
print(tok.decode(model.generate(**tok("Once upon a time", return_tensors="pt").to("cuda"),
                                max_new_tokens=30)[0]))
# -> "Once upon a time, there was a wise old owl named Oliver who lived in a cozy treehouse..."
```

The end-user touches none of our code. `trust_remote_code` makes
`transformers` fetch the modeling/config files from the HF repo, load them
into a dynamic module, and instantiate the class.

## Files shipped to the HF model repo

- `model.safetensors` (763 MB) — bf16 weights, HF naming convention
- `config.json` — `MatildaConfig` serialized + `auto_map` for
  AutoConfig/AutoModel/AutoModelForCausalLM
- `configuration_matilda.py` — `MatildaConfig(PretrainedConfig)`
- `modeling_matilda.py` — `MatildaForCausalLM(PreTrainedModel, GenerationMixin)`
- `generation_config.json` — default sampling settings
- `tokenizer.json`, `tokenizer_config.json` — GPT-2 fast tokenizer files
- `README.md` — model card with eval headline

## Architecture preserved

Same as the native model:

- 18 layers × 768 hidden × 12 heads / 4 KV heads (GQA, head_dim 64)
- RoPE (theta 10000), RMSNorm pre-norm
- **QK-Norm** per head — not in stock LlamaConfig; preserved via `MatildaConfig.qk_norm`
- **ReLU² FFN** — two matmuls (`up`, `down`), `down(relu(up(x))²)`. Not Llama's three-matmul SwiGLU.
- **Final logit soft-cap 30.0** (Gemma-2 style) applied in `MatildaForCausalLM.forward`
- Tied input/output embeddings

LigerRMSNorm is used when liger-kernel + torch.distributed.tensor are
importable (Triton GPU available); falls back to a pure-PyTorch RMSNorm
otherwise so the model also loads on CPU / Colab without contortions.

## State-dict key remap

The native checkpoint has its tensors at top level
(`embed.weight`, `blocks.0.attn.wq.weight`, …). HF convention wraps the
trunk inside a `model.*` namespace and adds an explicit `lm_head.weight`
even when tied. Rewrite at conversion time:

| Native (after `_orig_mod.` strip)   | HF                                         |
|-------------------------------------|--------------------------------------------|
| `embed.weight`                      | `model.embed.weight`                       |
| `blocks.{i}.norm1.weight`           | `model.blocks.{i}.norm1.weight`            |
| `blocks.{i}.attn.{wq,wk,wv,wo}.weight` | `model.blocks.{i}.attn.{wq,wk,wv,wo}.weight` |
| `blocks.{i}.attn.{q,k}_norm.weight` | `model.blocks.{i}.attn.{q,k}_norm.weight`  |
| `blocks.{i}.norm2.weight`           | `model.blocks.{i}.norm2.weight`            |
| `blocks.{i}.mlp.{up,down}.weight`   | `model.blocks.{i}.mlp.{up,down}.weight`    |
| `norm_f.weight`                     | `model.norm_f.weight`                      |
| —                                   | `lm_head.weight` (clone of `embed.weight`) |
| —                                   | `model.rope_cos`, `model.rope_sin` (see below) |

## Three landmines hit (and how each got fixed)

### 1. `tie_word_embeddings` double-pass on config round-trip

`MatildaConfig.__init__` originally did
`super().__init__(tie_word_embeddings=tie_weights, **kwargs)`. On
`from_pretrained`, the saved `config.json` includes
`tie_word_embeddings=True` in the kwargs already — `TypeError: got multiple
values for keyword argument 'tie_word_embeddings'`.

Fix: pop it from `kwargs` before delegating. The locally-passed
`tie_weights` wins.

### 2. `_tied_weights_keys` format changed in newer transformers

Setting `_tied_weights_keys = ["lm_head.weight"]` (the conventional list)
crashed `post_init` with
`AttributeError: 'list' object has no attribute 'keys'` — the field is now
treated as a `{output_name: input_name}` dict. Easiest fix: drop the
declaration entirely. `config.tie_word_embeddings=True` is what actually
drives the tying via `PreTrainedModel.tie_weights()`, no manual override
needed.

### 3. RoPE buffers vanish under `from_pretrained(init_empty_weights)`

This was the big one. Native `Transformer` registers `rope_cos` / `rope_sin`
with `persistent=False`. That's fine for `torch.save`/`load_state_dict` —
they get recomputed in `__init__` each time. But HF's `from_pretrained`
constructs the model under an "empty weights" context where every
`torch.empty`, `torch.zeros`, etc. returns garbage or zeros, and that
includes the cos/sin tables computed in `__init__`. The buffers stay garbage
because non-persistent buffers aren't in the state_dict that load fills.

Result: bit-identical weight values across native and HF
(`max_abs_diff = 0.0`), but per-layer output diverging starting at block 0
because RoPE was multiplying queries/keys against NaN-adjacent numbers.

Fix: make `rope_cos` and `rope_sin` **persistent buffers**, and inject them
into the safetensors during conversion. Total cost: ~1 MB extra on disk
(2048 × 64 × fp32 × 2). Now they ride through `from_pretrained` exactly
like any other tensor.

### 4. (Generation) RoPE buffer dtype mismatch under no-autocast inference

Native training wraps the forward in `torch.autocast(dtype=bf16)`, which
silently casts the fp32 cos/sin buffers to bf16 before they meet bf16 q/k.
HF inference (e.g. `model.generate()`) calls forward outside autocast — so
`q * cos` broadcasts up to fp32, then SDPA sees fp32 q/k vs bf16 v:

```
RuntimeError: Expected query, key, and value to have the same dtype,
but got query.dtype: float key.dtype: float and value.dtype: c10::BFloat16
```

Fix: inside `_apply_rope`, cast `cos` and `sin` to `x.dtype` before the
multiply. Costs nothing in autocast (the cast is a no-op when dtypes match)
and unblocks plain `.eval()` inference.

### 5. (Cosmetic) GenerationMixin warning

`prepare_inputs_for_generation` is defined but the class didn't directly
inherit `GenerationMixin`. In transformers v4.50+ `PreTrainedModel` will
stop inheriting that mixin, and the model would lose `.generate()`. Fixed
by adding the explicit inheritance:

```python
class MatildaForCausalLM(PreTrainedModel, GenerationMixin):
    ...
```

(Order matters — `PreTrainedModel` first.)

## Verification — bit-exact parity

`eval/hf/verify_parity.py` runs the same input through
`matilda.model.Transformer` (native) and
`MatildaForCausalLM.from_pretrained(...)` (HF) in bf16 on the same GPU.
Final state at the end of the trail of fixes above:

```
[parity] logits_native shape = (2, 64, 50257) dtype=torch.bfloat16
[parity] logits_hf     shape = (2, 64, 50257) dtype=torch.bfloat16
[parity] max abs diff    = 0.00000e+00
[parity] mean abs diff   = 0.00000e+00
[parity] argmax agreement= 100.0000%
```

A greedy 30-token generation on three prompts:

```
'Once upon a time' ->
  'Once upon a time, there was a wise old owl named Oliver who lived in a cozy
   treehouse in the forest. He loved to listen to stories about the past and'

'The capital of France is' ->
  'The capital of France is the capital of the country. The capital of the
   country is the capital of the country...'  (loops — small models often do)

'Photosynthesis is the process' ->
  'Photosynthesis is the process by which plants convert sunlight into energy.'
```

## Where the files live now

| Artefact                                       | Repo                                                         |
|------------------------------------------------|--------------------------------------------------------------|
| `eval/hf/configuration_matilda.py`             | `prometheus04/matilda-mini-v2` (this repo)                   |
| `eval/hf/modeling_matilda.py`                  | same                                                         |
| `eval/hf/convert_to_hf.py`                     | same                                                         |
| `eval/hf/verify_parity.py`                     | same                                                         |
| HF-format weights + modeling files (live copy) | `prometheus04/matilda-152m-v15-7.5B` (model repo)            |
| Native `.pt` checkpoints (10 ckpts + metrics)  | same model repo (mirrored 2026-05-31 ~16:00 UTC)             |
