"""Driver: evaluate every saved checkpoint on a fixed task set.

Usage:

    python eval/run_eval.py \
        --ckpt-dir checkpoints/base_152m_v2 \
        --tasks piqa,arc_easy,hellaswag,winogrande,lambada_openai \
        --out-dir eval/results \
        --batch-size 16 \
        --limit 0   # 0 = full eval; >0 = first N items per task (smoke test)

Per-checkpoint result writes to `<out-dir>/ckpt_<step>.json` and a summary
table writes to `<out-dir>/summary.csv` at the end.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import torch

# Make eval/matilda_lm.py importable for the side-effect of register_model
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matilda_lm  # noqa: F401 — registers "matilda" with lm-eval

from lm_eval import simple_evaluate


CKPT_RE = re.compile(r"ckpt_(\d+)\.pt$")


def discover_ckpts(ckpt_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in sorted(ckpt_dir.glob("ckpt_*.pt")):
        m = CKPT_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort()
    return out


def extract_metric(task_results: dict, task_name: str) -> dict[str, float]:
    """Pull the headline metrics for known tasks. Falls back to acc if present."""
    out = {}
    for k, v in task_results.items():
        if k in ("alias", "stderr"):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--tasks", required=True,
                    help="comma-separated lm-eval task names")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = full; >0 = first N items per task")
    ap.add_argument("--ckpts", default="",
                    help="optional comma-separated list of ckpt step numbers "
                         "to restrict to (e.g. 750,7150). default: all.")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    all_ckpts = discover_ckpts(ckpt_dir)
    if args.ckpts:
        wanted = {int(s) for s in args.ckpts.split(",")}
        all_ckpts = [(s, p) for (s, p) in all_ckpts if s in wanted]

    print(f"[eval] {len(all_ckpts)} checkpoints x {len(tasks)} tasks  -> {out_dir}")
    for step, path in all_ckpts:
        print(f"        ckpt_{step}: {path}")

    summary_rows = []
    for step, ckpt_path in all_ckpts:
        out_path = out_dir / f"ckpt_{step}.json"
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            with open(out_path) as f:
                results = json.load(f)
        else:
            print(f"[eval] ckpt step {step} -> {ckpt_path}")
            # NB: do NOT include batch_size here — simple_evaluate passes it
            # through its own kwarg, and duplicating triggers a TypeError.
            model_args = (
                f"ckpt_path={ckpt_path},"
                f"max_length={args.max_length},device=cuda,dtype=bfloat16"
            )
            results = simple_evaluate(
                model="matilda",
                model_args=model_args,
                tasks=tasks,
                batch_size=args.batch_size,
                limit=args.limit if args.limit > 0 else None,
                bootstrap_iters=1000,
                cache_requests=True,
            )
            # Strip non-serializable bits before saving
            results_to_save = {
                "results": results["results"],
                "config": {
                    "ckpt_step": step,
                    "ckpt_path": str(ckpt_path),
                    "tokens_seen": step * 16 * 32 * 2048,
                    "tasks": tasks,
                    "batch_size": args.batch_size,
                    "limit": args.limit,
                },
            }
            with open(out_path, "w") as f:
                json.dump(results_to_save, f, indent=2)
            print(f"[eval] wrote {out_path}")
            # Release GPU between checkpoints
            torch.cuda.empty_cache()

        row = {"ckpt_step": step,
               "tokens_B": round(step * 16 * 32 * 2048 / 1e9, 3)}
        for tname, tmetrics in (results.get("results") or {}).items():
            for k, v in tmetrics.items():
                if isinstance(v, (int, float)):
                    row[f"{tname}/{k}"] = round(float(v), 4)
        summary_rows.append(row)

    # Write CSV summary
    summary_path = out_dir / "summary.csv"
    if summary_rows:
        cols = sorted({k for r in summary_rows for k in r.keys()})
        # Put step + tokens first
        for first in ("tokens_B", "ckpt_step"):
            if first in cols:
                cols.remove(first)
                cols.insert(0, first)
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)
        print(f"[eval] summary -> {summary_path}")
        # Print summary to stdout
        print()
        print("step  tokens_B  " + "  ".join(c for c in cols if c not in ("ckpt_step", "tokens_B")))
        for row in summary_rows:
            print(f"{row['ckpt_step']:>5}  {row['tokens_B']:>8.3f}  " +
                  "  ".join(f"{row.get(c, ''):.4f}" if isinstance(row.get(c), float) else str(row.get(c, ''))
                            for c in cols if c not in ("ckpt_step", "tokens_B")))


if __name__ == "__main__":
    main()
