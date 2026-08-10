"""Sanity-check: HF MatildaForCausalLM and matilda.model.Transformer produce
the same logits on the same input (within float tolerance).

Usage:
    python eval/hf/verify_parity.py --ckpt checkpoints/base_152m_v2/ckpt_7150.pt \\
        --hf-dir hf-export/matilda-152m-v15-7.5B
"""

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def load_matilda(ckpt_path: str, device: torch.device):
    from matilda.config import ModelConfig
    from matilda.model import Transformer
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = Transformer(cfg)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    model.load_state_dict(sd, strict=True)
    return model.to(device=device, dtype=torch.bfloat16).eval()


def load_hf(hf_dir: str, device: torch.device):
    # Load by pointing trust_remote_code at the local folder
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        hf_dir, trust_remote_code=True, dtype=torch.bfloat16)
    return model.to(device).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hf-dir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[parity] device = {device}")

    print(f"[parity] loading matilda native ...")
    m_native = load_matilda(args.ckpt, device)

    print(f"[parity] loading HF wrapper ...")
    m_hf = load_hf(args.hf_dir, device)

    # Same input through both
    torch.manual_seed(0)
    input_ids = torch.randint(0, 50257, (2, 64), device=device, dtype=torch.long)

    # native: returns (logits, loss)
    logits_native, _ = m_native(input_ids, targets=None)
    # HF: returns CausalLMOutputWithPast
    out_hf = m_hf(input_ids=input_ids)
    logits_hf = out_hf.logits

    print(f"[parity] logits_native shape = {tuple(logits_native.shape)} dtype={logits_native.dtype}")
    print(f"[parity] logits_hf     shape = {tuple(logits_hf.shape)} dtype={logits_hf.dtype}")
    assert logits_native.shape == logits_hf.shape

    # bf16 numerics: allow tolerance ~ 1e-2 in abs diff (norm/softcap stack adds noise)
    diff = (logits_native.float() - logits_hf.float()).abs()
    print(f"[parity] max abs diff    = {diff.max().item():.5e}")
    print(f"[parity] mean abs diff   = {diff.mean().item():.5e}")
    print(f"[parity] argmax agreement= {(logits_native.argmax(-1) == logits_hf.argmax(-1)).float().mean().item():.4%}")

    # Generate a sample (short) to verify end-to-end pipeline works
    print()
    print(f"[parity] sample generation (HF .generate) ...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    prompt = "Once upon a time"
    enc = tok(prompt, return_tensors="pt").to(device)
    out = m_hf.generate(**enc, max_new_tokens=40, do_sample=True,
                        temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
    print(f"  prompt    : {prompt!r}")
    print(f"  completion: {tok.decode(out[0])!r}")


if __name__ == "__main__":
    main()
