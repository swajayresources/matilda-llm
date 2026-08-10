"""Architecture ablation harness.

Trains each variant for the SAME fixed token budget on the SAME data, then emits
a markdown results table + JSON. The point of an ablation is a controlled
comparison: one thing changes per row, everything else held fixed.

    python scripts/ablate.py --data-dir data/fwedu --tokens 300000000
    python scripts/ablate.py --dry-run            # tiny synthetic, for CI/sanity
    python scripts/ablate.py --data-dir data/fwedu --only baseline no_qk_norm

Each variant only overrides what it's testing; the rest inherits from BASE below.
Results land in results/ablations/<name>/ (checkpoints + metrics.jsonl) and a
combined docs/ABLATIONS.md + results/ablations.json.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from matilda.config import ModelConfig                          # noqa: E402
from matilda.train import Trainer, TrainConfig                  # noqa: E402
from matilda.model import Transformer                           # noqa: E402
from matilda.data import SyntheticStream, BinStream, shard_paths  # noqa: E402

# Base config every variant inherits. Real architecture, short token budget.
BASE_MODEL = dict(vocab_size=50257, max_seq_len=1024, d_model=768,
                  n_layers=12, n_heads=12, n_kv_heads=4)
import torch  # noqa: E402

_HAS_CUDA = torch.cuda.is_available()
BASE_TRAIN = dict(seq_len=1024, batch_size=24, grad_accum=8, warmup_steps=80,
                  lr=6e-4, log_every=20, ckpt_every=100000,
                  device="cuda" if _HAS_CUDA else "cpu",
                  dtype="bfloat16" if _HAS_CUDA else "float32",
                  compile=_HAS_CUDA)

# One change per row. Inherits BASE; only the listed keys differ.
VARIANTS = [
    {"name": "baseline",   "model": {}},
    {"name": "no_qk_norm", "model": {"qk_norm": False, "attn_logit_softcap": 20.0}},
    {"name": "mha",        "model": {"n_kv_heads": 12}},   # full multi-head (no GQA)
    {"name": "mqa",        "model": {"n_kv_heads": 1}},    # multi-query (extreme GQA)
    {"name": "muon",       "model": {}, "train": {"optimizer": "muon"}},
]


def steps_for_tokens(tokens, tcfg: TrainConfig) -> int:
    return max(1, round(tokens / (tcfg.batch_size * tcfg.grad_accum * tcfg.seq_len)))


def final_metrics(ckpt_dir, tail=5) -> dict:
    """Average the last `tail` logged steps from metrics.jsonl."""
    path = os.path.join(ckpt_dir, "metrics.jsonl")
    try:
        with open(path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return {"loss": None, "mfu": None}     # crashed before logging a step
    steps = [r for r in rows if r.get("event") == "step"]
    if not steps:
        return {"loss": None, "mfu": None}
    last = steps[-tail:]
    avg = lambda k: sum(s[k] for s in last if s.get(k) is not None) / len(last)
    return {"loss": round(avg("loss"), 4), "mfu": round(avg("mfu"), 4),
            "tokens_per_s": round(avg("tokens_per_s"))}


def run_variant(v, tokens, data_dir, dry_run, results_root):
    model = {**BASE_MODEL, **v.get("model", {})}
    train = {**BASE_TRAIN, **v.get("train", {})}
    if dry_run:                                   # shrink for CPU/CI
        model.update(d_model=64, n_layers=2, n_heads=4,
                     n_kv_heads=min(4, model.get("n_kv_heads", 4)), max_seq_len=64)
        if v["name"] == "mha":
            model["n_kv_heads"] = 4
        train.update(seq_len=64, batch_size=4, grad_accum=1, warmup_steps=2,
                     device="cpu", dtype="float32", compile=False)
    ckpt_dir = os.path.join(results_root, v["name"])
    train["ckpt_dir"] = ckpt_dir

    mcfg = ModelConfig(**model)
    tcfg = TrainConfig(total_steps=steps_for_tokens(tokens, TrainConfig(**train)),
                       **train)
    active = Transformer(mcfg).num_params(non_embedding=True)

    if dry_run or not data_dir:
        stream = SyntheticStream(mcfg.vocab_size, tcfg.batch_size, tcfg.seq_len,
                                 seed=0, device=tcfg.device)
    else:
        stream = BinStream(shard_paths(data_dir), tcfg.batch_size, tcfg.seq_len,
                           seed=0, device=tcfg.device)

    print(f"\n=== variant: {v['name']} | steps={tcfg.total_steps} "
          f"active={active/1e6:.1f}M kv_heads={mcfg.n_kv_heads} "
          f"qk_norm={mcfg.qk_norm} ===")
    Trainer(mcfg, tcfg, stream).train()
    m = final_metrics(ckpt_dir)
    return {"name": v["name"], "active_params_m": round(active / 1e6, 1),
            "n_kv_heads": mcfg.n_kv_heads, "qk_norm": mcfg.qk_norm,
            "optimizer": tcfg.optimizer,
            "final_loss": m["loss"], "mfu": m["mfu"],
            "tokens_per_s": m.get("tokens_per_s")}


def write_table(rows, tokens, out_md, out_json):
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump({"tokens_per_variant": tokens, "rows": rows},
              open(out_json, "w"), indent=2)
    lines = [
        "# Architecture Ablations",
        "",
        f"Each variant trained on **{tokens/1e6:.0f}M tokens** of the same data, "
        "identical except for the column under test.",
        "",
        "| Variant | Active params | n_kv_heads | QK-norm | Optimizer | Final loss | MFU | tok/s |",
        "|---------|--------------|-----------|---------|-----------|-----------|-----|-------|",
    ]
    for r in rows:
        mfu = f"{r['mfu']*100:.1f}%" if r["mfu"] is not None else "—"
        tps = f"{r['tokens_per_s']:,}" if r.get("tokens_per_s") else "—"
        loss = r["final_loss"] if r["final_loss"] is not None else "—"
        lines.append(f"| {r['name']} | {r['active_params_m']}M | {r['n_kv_heads']} "
                     f"| {r['qk_norm']} | {r.get('optimizer','adamw')} | {loss} "
                     f"| {mfu} | {tps} |")
    open(out_md, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {out_md} and {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--tokens", type=int, default=300_000_000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of variant names to run")
    ap.add_argument("--results", default=str(ROOT / "results" / "ablations"))
    args = ap.parse_args()

    tokens = 50_000 if args.dry_run else args.tokens
    variants = [v for v in VARIANTS
                if args.only is None or v["name"] in args.only]
    rows = [run_variant(v, tokens, args.data_dir, args.dry_run, args.results)
            for v in variants]
    write_table(rows, tokens,
                out_md=str(ROOT / "docs" / "ABLATIONS.md"),
                out_json=str(Path(args.results) / "ablations.json"))
    print("\n".join(f"  {r['name']:12} loss={r['final_loss']} mfu={r['mfu']}"
                    for r in rows))


if __name__ == "__main__":
    main()
