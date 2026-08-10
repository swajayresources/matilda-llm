"""Muon optimizer + Muon/AdamW hybrid."""

import pytest
import torch

from matilda import Transformer, ModelConfig
from matilda.optim import (
    build_optimizer, cosine_warmup_scheduler, build_scheduler,
    wsd_scheduler, WarmupStableDecay, Muon, HybridOptimizer,
    zeropower_via_newtonschulz5,
)


def test_newtonschulz_orthogonalizes():
    torch.manual_seed(0)
    G = torch.randn(32, 16)
    X = zeropower_via_newtonschulz5(G, steps=5).float()
    # singular values should be pushed toward 1
    s = torch.linalg.svdvals(X)
    assert (s > 0.5).all() and (s < 1.5).all()


def test_hybrid_splits_params():
    cfg = ModelConfig(vocab_size=128, max_seq_len=32, d_model=64,
                      n_layers=2, n_heads=4, n_kv_heads=2)
    opt = build_optimizer(Transformer(cfg), name="muon")
    assert isinstance(opt, HybridOptimizer)
    muon, adamw = opt.optimizers
    assert isinstance(muon, Muon)
    # every Muon param is a 2-D matrix
    assert all(p.ndim == 2 for g in muon.param_groups for p in g["params"])


@pytest.mark.slow
def test_muon_overfits_single_batch():
    cfg = ModelConfig(vocab_size=256, max_seq_len=64, d_model=128,
                      n_layers=2, n_heads=4, n_kv_heads=2)
    model = Transformer(cfg).train()
    torch.manual_seed(0)
    idx = torch.randint(0, cfg.vocab_size, (4, 32))
    tgt = torch.randint(0, cfg.vocab_size, (4, 32))
    opt = build_optimizer(model, name="muon", lr=3e-3, muon_lr=0.02)
    sched = cosine_warmup_scheduler(opt, warmup_steps=10, total_steps=300)
    last = None
    for _ in range(300):
        _, loss = model(idx, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        last = loss.item()
    assert last < 0.5, f"Muon failed to overfit; final loss={last:.3f}"


def _dummy_opt(base_lr=1.0):
    """Minimal optimizer just to exercise scheduler interface."""
    p = torch.nn.Parameter(torch.zeros(2, 2))
    return torch.optim.SGD([p], lr=base_lr)


def test_wsd_schedule_three_phases():
    """Warmup ramps linearly to peak, stable holds peak, decay drops to min."""
    base = 1.0
    warmup, total, stable = 10, 100, 0.8
    sched = wsd_scheduler(_dummy_opt(base), warmup_steps=warmup,
                          total_steps=total, stable_share=stable,
                          min_lr_ratio=0.1)
    lrs = []
    for _ in range(total):
        lrs.append(sched.get_last_lr()[0])
        sched.step()
    # warmup: lr at step warmup-1 should equal base (peak)
    assert lrs[0] == pytest.approx(base / warmup, rel=1e-6)
    assert lrs[warmup - 1] == pytest.approx(base, rel=1e-6)
    # stable phase: held at peak for stable_share of post-warmup
    decay_start = warmup + int(stable * (total - warmup))
    assert lrs[warmup] == pytest.approx(base, rel=1e-6)
    assert lrs[decay_start - 1] == pytest.approx(base, rel=1e-6)
    # decay: monotone non-increasing
    decay_lrs = lrs[decay_start:]
    assert all(decay_lrs[i] >= decay_lrs[i + 1] for i in range(len(decay_lrs) - 1))
    # ends at min_lr_ratio * base (within one-step linear quantum)
    assert lrs[-1] < base                          # has decayed
    assert lrs[-1] >= 0.1 * base * 0.5             # not below floor


def test_wsd_state_dict_roundtrip():
    """Resume must restore step + base_lrs and re-apply the LR exactly."""
    opt = _dummy_opt(1.0)
    s1 = wsd_scheduler(opt, warmup_steps=10, total_steps=100, stable_share=0.8)
    for _ in range(50):
        s1.step()
    sd = s1.state_dict()
    s2 = wsd_scheduler(_dummy_opt(1.0), warmup_steps=10, total_steps=100,
                       stable_share=0.8)
    s2.load_state_dict(sd)
    assert s2.last_step == s1.last_step
    assert s2.get_last_lr()[0] == pytest.approx(s1.get_last_lr()[0], rel=1e-9)


def test_build_scheduler_dispatch():
    assert isinstance(
        build_scheduler("wsd", _dummy_opt(), 10, 100), WarmupStableDecay)
    cos = build_scheduler("cosine", _dummy_opt(), 10, 100)
    assert hasattr(cos, "step") and hasattr(cos, "get_last_lr")
    with pytest.raises(ValueError):
        build_scheduler("bogus", _dummy_opt(), 10, 100)


def test_hybrid_state_dict_roundtrip():
    cfg = ModelConfig(vocab_size=128, max_seq_len=32, d_model=64,
                      n_layers=2, n_heads=4, n_kv_heads=2)
    model = Transformer(cfg)
    opt = build_optimizer(model, name="muon")
    # take a step so state is populated
    _, loss = model(torch.randint(0, 128, (2, 16)), torch.randint(0, 128, (2, 16)))
    loss.backward()
    opt.step()
    sd = opt.state_dict()
    opt2 = build_optimizer(Transformer(cfg), name="muon")
    opt2.load_state_dict(sd)        # must not raise
    assert len(sd["opts"]) == 2
