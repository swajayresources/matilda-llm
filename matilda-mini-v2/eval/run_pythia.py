"""Evaluate EleutherAI/pythia-160m at token-matched revisions against
matilda's saved checkpoints.

Pythia-160m batch is 1024 sequences × 2048 tokens = 2.097 M tokens/step.
Published revisions on HF: step1, step2, step4, step8, ..., step512,
step1000, step2000, step3000, ..., step143000. We pick the published
revision closest (and not exceeding too far) to each matilda token count.

Usage:
    python eval/run_pythia.py --tasks piqa,arc_easy,winogrande,hellaswag,lambada_openai \
        --out-dir eval/pythia_results --batch-size 16
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lm_eval import simple_evaluate


PYTHIA_MODEL = "EleutherAI/pythia-160m"
PYTHIA_TOKENS_PER_STEP = 1024 * 2048  # 2,097,152

# Map: matilda step -> Pythia revision (closest available not too far above target)
# Token target per matilda step is step * 16 * 32 * 2048 = step * 1_048_576.
# Pythia revisions available: step1, step2, step4, ..., step512, step1000, step2000, ..., step143000.
MATILDA_TO_PYTHIA = [
    # Pythia early revisions are powers of two (step1 ... step512), then
    # linear (step1000, step2000, ...). step500/step1500 don't exist.
    ( 750, 0.79e9, "step512"),     # Pythia step512 = 1.07B tokens
    (1500, 1.57e9, "step1000"),    # 2.10B
    (2250, 2.36e9, "step1000"),    # 2.10B (closer than step2000 = 4.19B)
    (3000, 3.15e9, "step2000"),    # 4.19B
    (3750, 3.93e9, "step2000"),    # 4.19B
    (4500, 4.72e9, "step2000"),    # 4.19B
    (5250, 5.51e9, "step3000"),    # 6.29B
    (6000, 6.29e9, "step3000"),    # 6.29B (exact match)
    (6750, 7.08e9, "step3000"),    # 6.29B
    (7150, 7.50e9, "step4000"),    # 8.39B
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dedup revisions — we only need to evaluate each Pythia revision once
    unique_revs = sorted({rev for (_, _, rev) in MATILDA_TO_PYTHIA},
                        key=lambda s: int(s.replace("step", "")))
    print(f"[pythia] {len(unique_revs)} unique revisions x {len(tasks)} tasks  -> {out_dir}")
    for rev in unique_revs:
        print(f"        {rev}")

    for rev in unique_revs:
        out_path = out_dir / f"pythia_{rev}.json"
        if out_path.exists():
            print(f"[skip] {out_path}")
            continue
        print(f"[pythia] {rev}")
        model_args = (
            f"pretrained={PYTHIA_MODEL},revision={rev},"
            f"dtype=bfloat16"
        )
        results = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=tasks,
            batch_size=args.batch_size,
            limit=args.limit if args.limit > 0 else None,
            bootstrap_iters=1000,
            cache_requests=True,
        )
        results_to_save = {
            "results": results["results"],
            "config": {
                "revision": rev,
                "pythia_step": int(rev.replace("step", "")),
                "tokens_seen": int(rev.replace("step", "")) * PYTHIA_TOKENS_PER_STEP,
                "tasks": tasks,
                "batch_size": args.batch_size,
            },
        }
        with open(out_path, "w") as f:
            json.dump(results_to_save, f, indent=2)
        print(f"[pythia] wrote {out_path}")
        torch.cuda.empty_cache()

    print(f"[pythia] done. {len(unique_revs)} revisions saved to {out_dir}")


if __name__ == "__main__":
    main()
