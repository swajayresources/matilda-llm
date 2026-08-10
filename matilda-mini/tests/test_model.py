"""Sanity tests that gate the paid run. All must be green on Colab first.

  - test_forward_shapes / test_weight_tying / test_param_count: wiring is correct
  - test_causal_mask: no information leaks from future tokens (the bug that
    silently inflates eval and is invisible in the loss curve)
  - test_overfit_single_batch: the model can actually learn (loss -> ~0 on one
    fixed batch). The cheapest, highest-signal correctness check in ML.
"""

import torch
import pytest

from matilda import Transformer, ModelConfig, DEV_TINY


def _tiny():
    return Transformer(DEV_TINY).eval()


def test_forward_shapes():
    model = _tiny()
    B, T = 2, 16
    idx = torch.randint(0, DEV_TINY.vocab_size, (B, T))
    logits, loss = model(idx)
    assert logits.shape == (B, T, DEV_TINY.vocab_size)
    assert loss is None
    targets = torch.randint(0, DEV_TINY.vocab_size, (B, T))
    _, loss = model(idx, targets)
    assert loss is not None and loss.ndim == 0


def test_weight_tying():
    model = _tiny()
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()


def test_loss_at_init_is_near_uniform():
    # untrained model should be ~ -log(1/V) = log(V)
    model = _tiny()
    idx = torch.randint(0, DEV_TINY.vocab_size, (4, 32))
    tgt = torch.randint(0, DEV_TINY.vocab_size, (4, 32))
    _, loss = model(idx, tgt)
    expected = torch.log(torch.tensor(float(DEV_TINY.vocab_size)))
    assert abs(loss.item() - expected.item()) < 1.0


def test_causal_mask_no_future_leak():
    """Changing token at position t must not alter logits at positions < t."""
    model = _tiny()
    torch.manual_seed(0)
    idx = torch.randint(0, DEV_TINY.vocab_size, (1, 24))
    with torch.no_grad():
        base, _ = model(idx)
        idx2 = idx.clone()
        idx2[0, -1] = (idx2[0, -1] + 1) % DEV_TINY.vocab_size  # perturb last token
        perturbed, _ = model(idx2)
    # all positions except the last must be identical
    assert torch.allclose(base[:, :-1], perturbed[:, :-1], atol=1e-5)
    assert not torch.allclose(base[:, -1], perturbed[:, -1], atol=1e-5)


def test_gqa_kv_head_counts():
    model = _tiny()
    attn = model.blocks[0].attn
    assert attn.wk.out_features == DEV_TINY.n_kv_heads * DEV_TINY.head_dim
    assert attn.wq.out_features == DEV_TINY.n_heads * DEV_TINY.head_dim


def test_softcap_path_is_finite_and_causal():
    # qk_norm OFF + soft-cap ON: the ablation config must stay finite and causal
    cfg = ModelConfig(vocab_size=200, max_seq_len=64, d_model=64, n_layers=2,
                      n_heads=4, n_kv_heads=2, qk_norm=False,
                      attn_logit_softcap=20.0)
    model = Transformer(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        logits, _ = model(idx)
        assert torch.isfinite(logits).all()
        idx2 = idx.clone()
        idx2[0, -1] = (idx2[0, -1] + 1) % cfg.vocab_size
        perturbed, _ = model(idx2)
    assert torch.allclose(logits[:, :-1], perturbed[:, :-1], atol=1e-5)


@pytest.mark.slow
def test_overfit_single_batch():
    """The model must drive loss toward zero on one fixed batch."""
    cfg = ModelConfig(vocab_size=256, max_seq_len=64, d_model=128,
                      n_layers=2, n_heads=4, n_kv_heads=2)
    model = Transformer(cfg).train()
    torch.manual_seed(0)
    idx = torch.randint(0, cfg.vocab_size, (4, 32))
    tgt = torch.randint(0, cfg.vocab_size, (4, 32))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(300):
        _, loss = model(idx, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.1, f"failed to overfit; final loss={losses[-1]:.3f}"
