"""lm-eval-harness adapter for matilda checkpoints.

Wraps `matilda.model.Transformer` in an `lm_eval.api.model.TemplateLM` so
checkpoints can be evaluated without converting to HuggingFace format.

Usage:

    PYTHONPATH=src lm_eval --model matilda \
        --model_args ckpt_path=checkpoints/base_152m_v2/ckpt_7150.pt \
        --tasks piqa --batch_size 16

The `register_model("matilda")` decorator wires this into lm-eval's model
registry. Import this module before `lm_eval.simple_evaluate` runs so the
registration sticks.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Make matilda.* importable when running from the eval/ folder
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model

from matilda.config import ModelConfig          # noqa: E402
from matilda.model import Transformer           # noqa: E402


def _load_ckpt_to_model(ckpt_path: str, device: torch.device, dtype: torch.dtype):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ck["config"]
    if hasattr(cfg_dict, "to_dict"):
        cfg_dict = cfg_dict.to_dict()
    mcfg = ModelConfig(**cfg_dict)

    # Force-disable Liger FLCE for eval. We need raw logits for loglikelihood,
    # and FLCE only fires when targets are passed — which we never do at eval —
    # so leaving use_liger=True is fine in principle, but RMSNorm still goes
    # through Liger. Keep use_liger as-is to match training-time numerics.
    model = Transformer(mcfg)

    # Strip torch.compile _orig_mod. prefix from state_dict keys
    sd = ck["model"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, mcfg, ck.get("step")


@register_model("matilda")
class MatildaLM(TemplateLM):
    """lm-eval LM wrapping a matilda Transformer checkpoint."""

    def __init__(
        self,
        ckpt_path: str,
        batch_size: int | str = 16,
        max_length: int = 2048,
        device: str = "cuda",
        dtype: str = "bfloat16",
        tokenizer: str = "gpt2",
    ):
        super().__init__()
        self._device = torch.device(device)
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        self.model, self.mcfg, self.ckpt_step = _load_ckpt_to_model(
            ckpt_path, self._device, torch_dtype)

        # GPT-2 BPE — matches the tokenizer the training data was packed with.
        from transformers import GPT2TokenizerFast
        self._tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer)
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._batch_size = int(batch_size) if str(batch_size).isdigit() else 16
        self._max_length = int(max_length)

    # ---- properties lm-eval probes ---------------------------------------

    @property
    def eot_token_id(self) -> int:
        return self._tokenizer.eos_token_id  # GPT-2 = 50256

    @property
    def prefix_token_id(self) -> int:
        return self.eot_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def tokenizer_name(self) -> str:
        return "gpt2"

    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    # ---- tokenization ----------------------------------------------------

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self._tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens, **kwargs) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=False)

    # ---- model call ------------------------------------------------------

    @torch.no_grad()
    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids (B, T) long -> logits (B, T, V) float."""
        with torch.autocast(device_type=self._device.type, dtype=torch.bfloat16):
            logits, _ = self.model(input_ids.to(self._device), targets=None)
        return logits.float()

    # ---- _loglikelihood_tokens (multiple choice) -------------------------
    # TemplateLM.loglikelihood handles encoding via tok_encode/_encode_pair
    # and calls into _loglikelihood_tokens with (req_args, context_enc,
    # continuation_enc) triples.

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **kwargs):
        """For HellaSwag / ARC / PIQA / Winogrande — score pre-tokenized
        (context_enc, continuation_enc) pairs."""
        encoded = [(ctx_enc, cont_enc) for (_args, ctx_enc, cont_enc) in requests]

        # Sort longest first → if it OOMs, it fails fast
        order = sorted(range(len(encoded)),
                       key=lambda i: -(len(encoded[i][0]) + len(encoded[i][1])))

        results: list[tuple[float, bool] | None] = [None] * len(encoded)
        pbar = tqdm(total=len(encoded), disable=disable_tqdm,
                    desc=f"loglikelihood (ckpt step {self.ckpt_step})")

        for batch_start in range(0, len(order), self._batch_size):
            batch_idxs = order[batch_start:batch_start + self._batch_size]
            seqs, cont_lens = [], []
            for i in batch_idxs:
                ctx, cont = encoded[i]
                # Cap at max_length; need one extra position because the
                # logit at index t-1 predicts token t, so length-T sequence
                # gives T-1 prediction positions.
                full = (ctx + cont)[-(self._max_length):]
                seqs.append(full)
                # Continuation length might have been clipped if ctx+cont was
                # longer than max_length; recover actual cont length post-clip.
                cont_lens.append(min(len(cont), len(full) - 1))

            max_len = max(len(s) for s in seqs)
            inps = torch.full((len(seqs), max_len),
                              self.eot_token_id, dtype=torch.long,
                              device=self._device)
            for k, s in enumerate(seqs):
                inps[k, -len(s):] = torch.tensor(s, dtype=torch.long,
                                                 device=self._device)

            logits = self._model_call(inps)        # (B, T, V)

            for k, i in enumerate(batch_idxs):
                cont_len = cont_lens[k]
                seq_len = len(seqs[k])
                offset = max_len - seq_len         # left-padding offset
                cont_start = offset + seq_len - cont_len
                cont_end = offset + seq_len
                target_positions = torch.arange(cont_start, cont_end,
                                                device=self._device)
                target_tokens = inps[k, target_positions]
                logit_positions = target_positions - 1
                logprobs = torch.log_softmax(logits[k, logit_positions], dim=-1)
                tgt_lp = logprobs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
                ll = tgt_lp.sum().item()
                pred = logits[k, logit_positions].argmax(dim=-1)
                is_greedy = bool((pred == target_tokens).all().item())
                results[i] = (ll, is_greedy)

            pbar.update(len(batch_idxs))

        pbar.close()
        return results

    # ---- loglikelihood_rolling (LAMBADA, perplexity tasks) ---------------

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        """Sliding-window perplexity. Returns one float per request."""
        results: list[float] = []
        pbar = tqdm(total=len(requests), disable=disable_tqdm,
                    desc=f"loglikelihood_rolling (ckpt step {self.ckpt_step})")
        for req in requests:
            (text,) = req.args
            tokens = [self.eot_token_id] + self.tok_encode(text)
            if len(tokens) <= 1:
                results.append(0.0)
                pbar.update(1)
                continue

            total_ll = 0.0
            stride = self._max_length // 2
            # First window scores positions [1, len), later windows score
            # only the new region [max_length-stride, max_length) so we don't
            # double-count.
            for start in range(0, len(tokens) - 1, stride):
                chunk = tokens[start:start + self._max_length]
                if len(chunk) < 2:
                    continue
                inp = torch.tensor([chunk], dtype=torch.long,
                                   device=self._device)
                logits = self._model_call(inp)     # (1, T, V)
                if start == 0:
                    score_start = 1
                else:
                    score_start = max(1, self._max_length - stride)
                if score_start >= len(chunk):
                    continue
                target_tokens = inp[0, score_start:]
                logp = torch.log_softmax(
                    logits[0, score_start - 1:len(chunk) - 1], dim=-1)
                ll = logp.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1).sum().item()
                total_ll += ll
            results.append(total_ll)
            pbar.update(1)
        pbar.close()
        return results

    # ---- generate_until (not implemented; only needed for generation tasks) ---

    def generate_until(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "MatildaLM does not implement generate_until. Use multiple-choice "
            "and perplexity tasks only (loglikelihood / loglikelihood_rolling).")
