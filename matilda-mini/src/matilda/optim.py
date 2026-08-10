"""Optimizer + LR schedule construction.

Two details that materially affect quality and that most tutorials get wrong:
  1. No weight decay on 1-D params (norms, biases) or the embedding table.
     Decaying norm/embedding weights quietly hurts.
  2. The LR schedule includes linear warmup; starting at peak LR on random
     weights diverges.
"""

from __future__ import annotations

import math

import torch


def _adamw(groups, lr, betas, eps):
    """AdamW with fused kernel when available, falling back if unsupported."""
    fused = torch.cuda.is_available()
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def build_adamw(model, lr=3e-4, weight_decay=0.1,
                betas=(0.9, 0.95), eps=1e-8) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 2-D matmul weights get decay; norms/biases (1-D) and embeddings don't.
        if p.ndim < 2 or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return _adamw(groups, lr, betas, eps)


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Orthogonalize a 2-D gradient via the quintic Newton-Schulz iteration
    (Keller Jordan). Pushes all singular values toward 1 in ~5 matmuls, so the
    update weights every direction equally instead of being dominated by large
    singular directions. Runs in bf16, as in the reference implementation.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon: momentum + orthogonalized update for 2-D parameters only.

    Use ONLY for interior matrices (attn/MLP weights). Embeddings, norms, biases
    must stay on AdamW (see build_optimizer). Muon's natural LR is ~0.02, much
    higher than Adam's, because the orthogonalized update has ~unit scale.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, nesterov = group["lr"], group["momentum"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(g)
                buf = state["m"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if nesterov else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # scale so the update RMS matches across differently-shaped matrices
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(g, alpha=-lr * scale)


class HybridOptimizer:
    """Presents several optimizers as one (unified step/zero_grad/state_dict and
    a concatenated param_groups so a single LR scheduler drives all of them)."""

    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return {"opts": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, sd):
        for o, s in zip(self.optimizers, sd["opts"]):
            o.load_state_dict(s)


def build_optimizer(model, name="adamw", lr=3e-4, weight_decay=0.1,
                    betas=(0.9, 0.95), eps=1e-8, muon_lr=0.02):
    """Dispatch: 'adamw' (default) or 'muon' (Muon on 2-D interior weights +
    AdamW on embeddings/norms/biases)."""
    if name == "adamw":
        return build_adamw(model, lr, weight_decay, betas, eps)
    if name != "muon":
        raise ValueError(f"unknown optimizer: {name}")

    muon_p, adamw_decay, adamw_nodecay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in n:     # interior matrices -> Muon
            muon_p.append(p)
        elif p.ndim < 2 or "embed" in n:          # norms/biases/embeddings -> AdamW
            adamw_nodecay.append(p)
        else:
            adamw_decay.append(p)
    adamw = _adamw(
        [{"params": adamw_decay, "weight_decay": weight_decay},
         {"params": adamw_nodecay, "weight_decay": 0.0}],
        lr, betas, eps)
    return HybridOptimizer([Muon(muon_p, lr=muon_lr), adamw])


class WarmupCosine:
    """Linear warmup then cosine decay to min_lr_ratio * peak.

    Works on any object exposing `param_groups` (torch optimizers AND our
    HybridOptimizer, which torch's LambdaLR rejects). Applies one multiplier to
    every group, scaling each group's own base lr (so Muon's 0.02 and AdamW's
    3e-4 warm up/decay together). state_dict captures the step for exact resume.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.opt = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_step = -1
        self.step()                       # apply step 0

    def _scale(self, step):
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        if step >= self.total_steps:
            return self.min_lr_ratio
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine

    def step(self):
        self.last_step += 1
        s = self._scale(self.last_step)
        for group, base in zip(self.opt.param_groups, self.base_lrs):
            group["lr"] = base * s

    def get_last_lr(self):
        return [g["lr"] for g in self.opt.param_groups]

    def state_dict(self):
        return {"last_step": self.last_step, "base_lrs": self.base_lrs}

    def load_state_dict(self, sd):
        self.last_step = sd["last_step"]
        self.base_lrs = sd["base_lrs"]
        # re-apply so lrs match the restored step
        s = self._scale(self.last_step)
        for group, base in zip(self.opt.param_groups, self.base_lrs):
            group["lr"] = base * s


def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    return WarmupCosine(optimizer, warmup_steps, total_steps, min_lr_ratio)
