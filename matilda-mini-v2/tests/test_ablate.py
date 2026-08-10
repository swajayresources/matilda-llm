"""Ablation harness: variants run, metrics captured, table emitted (dry CPU)."""

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ablate  # noqa: E402


def test_steps_for_tokens():
    from matilda.train import TrainConfig
    tc = TrainConfig(batch_size=24, grad_accum=8, seq_len=1024)
    # 300M / (24*8*1024) ~= 1526
    assert ablate.steps_for_tokens(300_000_000, tc) == 1526


def test_run_variant_dry_and_table(tmp_path):
    results = str(tmp_path / "abl")
    rows = []
    for vname in ("baseline", "mqa"):
        v = next(x for x in ablate.VARIANTS if x["name"] == vname)
        row = ablate.run_variant(v, tokens=50_000, data_dir=None,
                                 dry_run=True, results_root=results)
        rows.append(row)
        assert row["final_loss"] is not None      # metrics captured
        assert row["mfu"] is not None
    assert rows[1]["n_kv_heads"] == 1              # mqa override applied

    md = tmp_path / "ABLATIONS.md"
    js = tmp_path / "ablations.json"
    ablate.write_table(rows, 50_000, str(md), str(js))
    assert md.exists() and js.exists()
    text = md.read_text()
    assert "baseline" in text and "mqa" in text and "MFU" in text
    data = json.loads(js.read_text())
    assert len(data["rows"]) == 2
