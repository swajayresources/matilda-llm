"""Training-loop reliability guards (all CPU-testable)."""

import os
import glob

import torch
import pytest

from matilda import ModelConfig
from matilda.data import SyntheticStream
from matilda.train import Trainer, TrainConfig
from matilda.monitor import mfu, peak_tflops

MCFG = ModelConfig(vocab_size=128, max_seq_len=32, d_model=64,
                  n_layers=2, n_heads=4, n_kv_heads=2)


def _make(tmp_path, **over):
    kw = dict(total_steps=20, warmup_steps=2, batch_size=4, seq_len=16,
              log_every=5, ckpt_every=10, keep_last=2, device="cpu",
              dtype="float32", ckpt_dir=str(tmp_path))
    kw.update(over)
    tc = TrainConfig(**kw)
    stream = SyntheticStream(MCFG.vocab_size, tc.batch_size, tc.seq_len, seed=0)
    return Trainer(MCFG, tc, stream)


def test_loop_runs_and_checkpoints(tmp_path):
    t = _make(tmp_path)
    final = t.train()
    assert final == 20
    # final checkpoint on clean completion + rotation kept <= keep_last
    ckpts = glob.glob(os.path.join(tmp_path, "ckpt_*.pt"))
    assert len(ckpts) <= 2
    assert os.path.exists(os.path.join(tmp_path, "ckpt_20.pt"))


def test_loop_resume_continues(tmp_path):
    # run only 10 steps, leaving a checkpoint at step 10
    t1 = _make(tmp_path, total_steps=10, ckpt_every=10)
    assert t1.train() == 10
    # a fresh trainer in the same dir must resume from step 10, not restart
    t2 = _make(tmp_path, total_steps=15, ckpt_every=10)
    assert t2.maybe_resume() is True
    assert t2.step == 10
    assert t2.train() == 15


def test_nan_guard_aborts_after_max_skips(tmp_path):
    t = _make(tmp_path, total_steps=50, max_skips=3)
    nan = torch.tensor(float("nan"))
    t.model.forward = lambda x, targets=None: (None, nan)  # shadow bound method
    with pytest.raises(RuntimeError, match="non-finite"):
        t.train()
    assert t.step == 0  # never advanced past a bad batch


def test_nan_guard_skips_then_recovers(tmp_path):
    t = _make(tmp_path, total_steps=5, max_skips=10)
    real_forward = t.model.forward
    calls = {"n": 0}

    def flaky(x, targets=None):
        calls["n"] += 1
        if calls["n"] <= 2:                     # first two micro-batches are bad
            return None, torch.tensor(float("inf"))
        return real_forward(x, targets)

    t.model.forward = flaky
    assert t.train() == 5                        # recovered and finished
    assert t.consecutive_skips == 0


def test_mfu_sanity():
    # 100M params, 500k tokens/step, 2.0s, A100 -> ~0.48 MFU
    val = mfu(100_000_000, 500_000, 2.0, peak_tflops("A100") * 1e12)
    assert 0.0 < val < 1.0
    assert peak_tflops("A100") == 312.0
    assert peak_tflops("totally-unknown-gpu") == 312.0  # default
