"""Launch entrypoint: build configs + data stream and train.

    python run.py --config configs/base_124m.json --data-dir data/fwedu
    python run.py --config configs/calibration.json --data-dir data/fwedu
    python run.py --config configs/calibration.json --dry-run   # synthetic, no data

A config JSON has two objects, "model" and "train", whose keys map directly onto
ModelConfig / TrainConfig fields. CLI --set k=v applies last-mile overrides
(e.g. --set train.batch_size=48) so calibration sweeps don't need new files.
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from matilda.config import ModelConfig                    # noqa: E402
from matilda.train import Trainer, TrainConfig            # noqa: E402
from matilda.data import SyntheticStream, BinStream, shard_paths  # noqa: E402


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """--set train.batch_size=48 -> cfg['train']['batch_size'] = 48"""
    for item in overrides:
        path, _, raw = item.partition("=")
        section, _, key = path.partition(".")
        cfg.setdefault(section, {})[key] = _coerce(raw)
    return cfg


def build(config: dict):
    mcfg = ModelConfig(**config.get("model", {}))
    tcfg = TrainConfig(**config.get("train", {}))
    return mcfg, tcfg


def build_stream(mcfg, tcfg, data_dir: str | None, dry_run: bool):
    if dry_run or not data_dir:
        print("[data] synthetic stream (dry run, no real data)")
        return SyntheticStream(mcfg.vocab_size, tcfg.batch_size, tcfg.seq_len,
                               seed=tcfg.seed, device=tcfg.device)
    paths = shard_paths(data_dir)
    print(f"[data] {len(paths)} shards from {data_dir}")
    return BinStream(paths, tcfg.batch_size, tcfg.seq_len, seed=tcfg.seed,
                     device=tcfg.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text())
    config = apply_overrides(config, args.overrides)
    mcfg, tcfg = build(config)
    stream = build_stream(mcfg, tcfg, args.data_dir, args.dry_run)

    print(f"[run] model={mcfg.d_model}d/{mcfg.n_layers}L "
          f"steps={tcfg.total_steps} bs={tcfg.batch_size}x{tcfg.grad_accum} "
          f"seq={tcfg.seq_len} compile={tcfg.compile} ckpt={tcfg.ckpt_dir}")
    Trainer(mcfg, tcfg, stream).train()


if __name__ == "__main__":
    main()
