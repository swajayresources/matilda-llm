"""Resume must continue *identically* to an uninterrupted run.

Strategy: run a short reference training. Run it again but checkpoint halfway,
throw the objects away, build *fresh* model/optimizer/scheduler/data-stream,
restore from the checkpoint, and continue. The post-resume per-step losses must
match the reference run's second half exactly. If any piece of state is missing
(opt moments, scheduler, RNG, data position) the curves diverge and this fails.
"""

import os
import random

import numpy as np
import torch

from matilda import Transformer, ModelConfig
from matilda.data import SyntheticStream
from matilda.optim import build_adamw, cosine_warmup_scheduler
from matilda.checkpoint import (
    save_checkpoint, load_checkpoint, latest_checkpoint, rotate_checkpoints,
)

CFG = ModelConfig(vocab_size=128, max_seq_len=32, d_model=64,
                  n_layers=2, n_heads=4, n_kv_heads=2)
TOTAL, WARMUP, HALF = 12, 2, 6


def _stream(seed):
    return SyntheticStream(CFG.vocab_size, batch_size=4, seq_len=16, seed=seed)


def _seed_all(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build():
    model = Transformer(CFG).train()
    opt = build_adamw(model, lr=1e-3)
    sched = cosine_warmup_scheduler(opt, WARMUP, TOTAL)
    return model, opt, sched


def _train_steps(model, opt, sched, stream, n):
    losses = []
    for _ in range(n):
        x, y = stream.next()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    return losses


def test_resume_is_bit_for_bit(tmp_path):
    # --- reference: uninterrupted run ---
    _seed_all()
    m, o, s = _build()
    ref_stream = _stream(42)
    ref_losses = _train_steps(m, o, s, ref_stream, TOTAL)

    # --- interrupted run: train HALF, checkpoint, drop everything ---
    _seed_all()
    m, o, s = _build()
    stream = _stream(42)
    first_half = _train_steps(m, o, s, stream, HALF)
    assert first_half == ref_losses[:HALF], "training is not deterministic"

    ckpt = os.path.join(tmp_path, f"ckpt_{HALF}.pt")
    save_checkpoint(ckpt, model=m, optimizer=o, scheduler=s, step=HALF,
                    config=CFG, data_state=stream.state_dict())

    # --- fresh objects, restore, continue ---
    m2, o2, s2 = _build()                    # fresh random init + zeroed moments
    stream2 = _stream(999)      # deliberately wrong seed...
    ck = load_checkpoint(ckpt, model=m2, optimizer=o2, scheduler=s2)
    stream2.load_state_dict(ck["data_state"])  # ...corrected by restore
    assert ck["step"] == HALF

    resumed = _train_steps(m2, o2, s2, stream2, TOTAL - HALF)

    for i, (a, b) in enumerate(zip(resumed, ref_losses[HALF:])):
        assert abs(a - b) < 1e-6, f"resume diverged at step {HALF+i}: {a} vs {b}"


def test_latest_and_rotation(tmp_path):
    for step in (10, 20, 30, 40):
        p = os.path.join(tmp_path, f"ckpt_{step}.pt")
        torch.save({"step": step}, p)
    assert latest_checkpoint(tmp_path).endswith("ckpt_40.pt")

    rotate_checkpoints(tmp_path, keep_last=2, protect={"ckpt_10.pt"})
    remaining = sorted(os.path.basename(p) for p in
                       __import__("glob").glob(os.path.join(tmp_path, "ckpt_*.pt")))
    # keep last 2 (30, 40) + protected 10; drop 20
    assert remaining == ["ckpt_10.pt", "ckpt_30.pt", "ckpt_40.pt"]
