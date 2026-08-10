"""The launch entrypoint: configs parse, build, override, and run."""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run  # noqa: E402


def _load(name):
    return json.loads((ROOT / "configs" / name).read_text())


def test_shipped_configs_build():
    for name in ("calibration.json", "base_124m.json"):
        mcfg, tcfg = run.build(_load(name))
        assert mcfg.d_model == 768 and mcfg.n_layers == 12
        assert tcfg.seq_len == 1024


def test_base_config_targets_about_3B_tokens():
    _, tcfg = run.build(_load("base_124m.json"))
    tokens = tcfg.total_steps * tcfg.batch_size * tcfg.grad_accum * tcfg.seq_len
    assert 2.5e9 < tokens < 3.5e9


def test_overrides_coerce_types():
    cfg = {"train": {"batch_size": 24}}
    out = run.apply_overrides(cfg, ["train.batch_size=48", "train.compile=true",
                                    "train.lr=0.0006"])
    assert out["train"]["batch_size"] == 48 and isinstance(
        out["train"]["batch_size"], int)
    assert out["train"]["compile"] is True
    assert out["train"]["lr"] == 0.0006


def test_dry_run_builds_synthetic_and_steps(tmp_path):
    cfg = _load("calibration.json")
    cfg = run.apply_overrides(cfg, [
        "model.d_model=64", "model.n_layers=2", "model.n_heads=4",
        "model.n_kv_heads=2", "model.vocab_size=256", "model.max_seq_len=64",
        "train.total_steps=4", "train.warmup_steps=1", "train.batch_size=4",
        "train.seq_len=64", "train.device=cpu", "train.dtype=float32",
        "train.compile=false", f"train.ckpt_dir={tmp_path.as_posix()}",
    ])
    mcfg, tcfg = run.build(cfg)
    stream = run.build_stream(mcfg, tcfg, data_dir=None, dry_run=True)
    from matilda.train import Trainer
    assert Trainer(mcfg, tcfg, stream).train() == 4
