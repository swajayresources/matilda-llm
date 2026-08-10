"""Crash-safe checkpointing.

A resumed run must continue *identically*, not just "without an obvious spike".
That requires saving everything that influences the next step:
  model, optimizer, scheduler, step, AND all RNG states AND the dataloader
  position. Dropping the RNG or data position is the classic cause of a loss
  blip on resume (you re-see data / re-sample dropout differently).

Writes are atomic (write tmp -> os.replace) so an instance dying mid-save can
never leave a half-written checkpoint that fails to load.
"""

from __future__ import annotations

import os
import glob
import random
from dataclasses import asdict

import numpy as np
import torch


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # RNG states must be CPU ByteTensors. A checkpoint loaded with
    # map_location="cuda" moves them to GPU, which set_rng_state rejects.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, *, model, optimizer, scheduler, step,
                    config=None, data_state=None, extra=None) -> None:
    """Atomically write a complete checkpoint to `path`.

    `extra` carries run provenance (TrainConfig, git SHA) so a resume can detect
    a changed schedule rather than silently corrupting the LR curve.
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "rng": _rng_state(),
        "data_state": data_state,
        "config": asdict(config) if hasattr(config, "__dataclass_fields__") else config,
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on POSIX and Windows


def load_checkpoint(path, *, model, optimizer=None, scheduler=None,
                    map_location="cpu", restore_rng=True) -> dict:
    """Restore in-place. Returns the raw payload (for step, data_state, config).

    weights_only=False is required to unpickle optimizer state. Safe here (we only
    load our own checkpoints); never point this at an untrusted checkpoint.
    """
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer") is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler") is not None:
        scheduler.load_state_dict(ck["scheduler"])
    if restore_rng and ck.get("rng") is not None:
        _set_rng_state(ck["rng"])
    return ck


def latest_checkpoint(directory) -> str | None:
    """Highest-step checkpoint matching ckpt_*.pt, or None."""
    paths = glob.glob(os.path.join(directory, "ckpt_*.pt"))
    if not paths:
        return None
    return max(paths, key=lambda p: int(os.path.basename(p)[5:-3]))


def rotate_checkpoints(directory, keep_last: int, protect: set[str] | None = None) -> None:
    """Delete oldest ckpt_*.pt beyond keep_last. `protect` = basenames to keep."""
    protect = protect or set()
    paths = sorted(
        glob.glob(os.path.join(directory, "ckpt_*.pt")),
        key=lambda p: int(os.path.basename(p)[5:-3]),
    )
    deletable = [p for p in paths if os.path.basename(p) not in protect]
    for p in deletable[:-keep_last] if keep_last > 0 else deletable:
        try:
            os.remove(p)
        except OSError as e:
            print(f"[warn] checkpoint rotation could not delete {p}: {e}")
