"""Muon optimizer + Muon/AdamW hybrid."""

import pytest
import torch

from matilda import Transformer, ModelConfig
from matilda.optim import (
    build_optimizer, cosine_warmup_scheduler, Muon, HybridOptimizer,
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
