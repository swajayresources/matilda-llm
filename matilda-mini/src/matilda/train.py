"""Training loop with the reliability guards a long run needs.

Designed so an overnight Vast run survives the things that actually kill runs:
  - one bad batch (NaN/Inf loss or grad) -> skip it, log loudly, keep going
  - spot-instance SIGTERM -> checkpoint before exiting
  - process death -> resume bit-for-bit from latest checkpoint on restart

Throughput (MFU, tokens/s, step-time) is logged every `log_every` steps. The
data source is any object with .next()/.state_dict()/.load_state_dict() (see
data.py), so swapping synthetic data for real FineWeb shards changes nothing.
"""

from __future__ import annotations

import os
import json
import signal
import subprocess
from dataclasses import dataclass, asdict

import torch

from .model import Transformer
from .config import ModelConfig
from .optim import build_optimizer, cosine_warmup_scheduler
from .monitor import Throughput, peak_tflops, flops_per_token
from .checkpoint import (
    save_checkpoint, load_checkpoint, latest_checkpoint, rotate_checkpoints,
)


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def setup_backends():
    """Free A100 throughput: TF32 matmuls + cuDNN autotuner."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


@dataclass
class TrainConfig:
    total_steps: int = 1000
    warmup_steps: int = 100
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    min_lr_ratio: float = 0.1
    optimizer: str = "adamw"          # "adamw" or "muon" (Muon+AdamW hybrid)
    muon_lr: float = 0.02

    batch_size: int = 8
    seq_len: int = 1024

    log_every: int = 10
    ckpt_every: int = 500
    keep_last: int = 3
    ckpt_dir: str = "checkpoints"

    device: str = "cuda"
    dtype: str = "bfloat16"            # bf16 on Ampere+, fp32 fallback on CPU
    compile: bool = False
    seed: int = 1234
    max_skips: int = 20               # abort if too many bad batches in a row
    wandb_project: str | None = None


class Trainer:
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig, stream):
        self.mcfg = model_cfg
        self.cfg = train_cfg
        self.stream = stream
        self.step = 0
        self.consecutive_skips = 0
        self._interrupted = False

        setup_backends()
        self.git_sha = get_git_sha()
        torch.manual_seed(train_cfg.seed)
        self.device = torch.device(
            train_cfg.device if torch.cuda.is_available()
            or train_cfg.device == "cpu" else "cpu")

        self.model = Transformer(model_cfg).to(self.device)
        if train_cfg.compile:
            self.model = torch.compile(self.model, dynamic=True)
        self.opt = build_optimizer(self.model, name=train_cfg.optimizer,
                                   lr=train_cfg.lr,
                                   weight_decay=train_cfg.weight_decay,
                                   muon_lr=train_cfg.muon_lr)
        self.sched = cosine_warmup_scheduler(
            self.opt, train_cfg.warmup_steps, train_cfg.total_steps,
            train_cfg.min_lr_ratio)

        # bf16 only where supported; CPU/T4 fall back to fp32 for correctness.
        want_bf16 = (train_cfg.dtype == "bfloat16"
                     and self.device.type == "cuda"
                     and torch.cuda.is_bf16_supported())
        self.amp_dtype = torch.bfloat16 if want_bf16 else None

        tokens_per_step = (train_cfg.batch_size * train_cfg.seq_len
                           * train_cfg.grad_accum)
        dev_name = (torch.cuda.get_device_name(self.device)
                    if self.device.type == "cuda" else "cpu")
        fps = flops_per_token(self._active_params(), model_cfg.n_layers,
                              model_cfg.d_model, train_cfg.seq_len) * tokens_per_step
        self.monitor = Throughput(
            flops_per_step=fps,
            tokens_per_step=tokens_per_step,
            peak_flops_per_sec=peak_tflops(dev_name) * 1e12,
        )
        self.metrics_path = os.path.join(train_cfg.ckpt_dir, "metrics.jsonl")
        self.wandb = self._init_wandb()
        self._install_signal_handlers()

    # --- helpers -----------------------------------------------------------
    def _unwrap(self):
        m = self.model
        if hasattr(m, "_orig_mod"):          # torch.compile
            m = m._orig_mod
        if hasattr(m, "module"):             # DDP
            m = m.module
        return m

    def _active_params(self):
        return self._unwrap().num_params(non_embedding=True)

    def _init_wandb(self):
        if not self.cfg.wandb_project:
            return None
        try:
            import wandb
        except ImportError:
            print("[warn] wandb not installed; using metrics.jsonl only")
            return None
        try:
            wandb.init(project=self.cfg.wandb_project,
                       config={**self.mcfg.to_dict(), **vars(self.cfg)})
            return wandb
        except Exception as e:                 # network/auth/quota — non-fatal
            print(f"[warn] wandb init failed: {e}")
            return None

    def _install_signal_handlers(self):
        def handler(signum, frame):
            print(f"[signal] {signum} received -> checkpoint and exit")
            self._interrupted = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError) as e:
                # not in main thread (e.g. under pytest) -> can't install
                print(f"[warn] could not install handler for signal {sig}: {e}")

    def _autocast(self):
        if self.amp_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return torch.autocast(device_type="cpu", enabled=False)

    # --- checkpoint --------------------------------------------------------
    def _ckpt_path(self):
        return os.path.join(self.cfg.ckpt_dir, f"ckpt_{self.step}.pt")

    def save(self):
        save_checkpoint(self._ckpt_path(), model=self.model, optimizer=self.opt,
                        scheduler=self.sched, step=self.step, config=self.mcfg,
                        data_state=self.stream.state_dict(),
                        extra={"train_config": asdict(self.cfg),
                               "git_sha": self.git_sha})
        rotate_checkpoints(self.cfg.ckpt_dir, self.cfg.keep_last)

    def maybe_resume(self):
        path = latest_checkpoint(self.cfg.ckpt_dir)
        if path is None:
            return False
        ck = load_checkpoint(path, model=self.model, optimizer=self.opt,
                             scheduler=self.sched,
                             map_location=self.device)
        self.step = ck["step"]
        if ck.get("data_state") is not None:
            self.stream.load_state_dict(ck["data_state"])
        # warn if the schedule changed since the checkpoint (silent LR corruption)
        prev = (ck.get("extra") or {}).get("train_config", {})
        for k in ("total_steps", "warmup_steps", "lr"):
            if k in prev and prev[k] != getattr(self.cfg, k):
                print(f"[warn] {k} changed on resume: {prev[k]} -> "
                      f"{getattr(self.cfg, k)}; LR schedule will differ")
        print(f"[resume] from {path} at step {self.step}")
        return True

    # --- one optimizer step (grad accum + guards) --------------------------
    def _step(self):
        self.opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for micro in range(self.cfg.grad_accum):
            x, y = self.stream.next()
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            sync = (micro == self.cfg.grad_accum - 1)
            ctx = (self.model.no_sync()
                   if (not sync and hasattr(self.model, "no_sync"))
                   else _nullcontext())
            with ctx, self._autocast():
                _, loss = self.model(x, y)
                loss = loss / self.cfg.grad_accum
            if not torch.isfinite(loss):
                return None  # bad micro-batch -> abort this step
            loss.backward()
            total_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip)
        if not torch.isfinite(grad_norm):
            return None  # non-finite grad -> skip update

        self.opt.step()
        self.sched.step()
        return total_loss, grad_norm.item()

    # --- main loop ---------------------------------------------------------
    def train(self):
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        self.maybe_resume()
        self._record({"event": "start", "git_sha": self.git_sha,
                      "model": self.mcfg.to_dict(), "train": asdict(self.cfg),
                      "step": self.step})
        self.model.train()
        while self.step < self.cfg.total_steps:
            result = self._step()
            if result is None:
                self.consecutive_skips += 1
                data_pos = self.stream.state_dict().get("pos")
                print(f"[nan-guard] step {self.step} skipped "
                      f"({self.consecutive_skips}/{self.cfg.max_skips}) "
                      f"near data_pos={data_pos}")
                self._record({"event": "nan_skip", "step": self.step,
                              "data_pos": data_pos})
                self.opt.zero_grad(set_to_none=True)
                if self.consecutive_skips >= self.cfg.max_skips:
                    raise RuntimeError("too many non-finite steps; aborting run")
                continue
            self.consecutive_skips = 0
            loss, grad_norm = result
            self.step += 1

            stats = self.monitor.tick()
            if self.step % self.cfg.log_every == 0:
                self._log(loss, grad_norm, stats)
            if self.step % self.cfg.ckpt_every == 0:
                self.save()
            if self._interrupted:
                self.save()
                print(f"[exit] checkpointed at step {self.step}")
                break
        else:
            self.save()  # final checkpoint on clean completion
        return self.step

    def _record(self, row: dict):
        """Always-on JSONL metric log (survives wandb outage)."""
        try:
            with open(self.metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as e:
            print(f"[warn] could not write metrics: {e}")

    def _log(self, loss, grad_norm, stats):
        lr = self.sched.get_last_lr()[0]
        row = {"event": "step", "step": self.step, "loss": loss, "lr": lr,
               "grad_norm": grad_norm}
        if stats:
            row.update({"tokens_per_s": stats["tokens_per_s"],
                        "mfu": stats["mfu_avg"], "step_time_s": stats["dt_avg_s"]})
        if self.device.type == "cuda":
            row["gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        self._record(row)

        msg = (f"step {self.step:>6} | loss {loss:7.4f} | lr {lr:.2e} "
               f"| gnorm {grad_norm:5.2f}")
        if stats:
            msg += (f" | {stats['tokens_per_s']:8.0f} tok/s "
                    f"| mfu {stats['mfu_avg']*100:4.1f}%")
        print(msg)
        if self.wandb:
            self.wandb.log({k: v for k, v in row.items()
                            if k not in ("event",)}, step=self.step)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
