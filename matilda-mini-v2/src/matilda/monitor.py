"""Throughput / utilization observability.

MFU (Model FLOPs Utilization) is the headline number a training engineer lives
in: what fraction of the GPU's theoretical bf16 throughput you actually extract.
We also track a rolling step-time window so a *degradation* mid-run (thermal
throttling, a noisy Vast neighbour) is visible, not just the instantaneous rate.
"""

from __future__ import annotations

import time
from collections import deque

# Peak bf16 *dense* TFLOPs (no 2:4 sparsity). Substring-matched on device name.
PEAK_TFLOPS_BF16 = {
    "A100": 312.0,
    "H100": 494.0,
    "H800": 494.0,
    "4090": 165.0,
    "3090": 71.0,
    "3060": 51.0,
    "T4": 65.0,    # fp16; T4 has no native bf16 (Turing) -> emulated, ignore MFU
    "A10": 125.0,
    "L4": 121.0,
}


def peak_tflops(device_name: str, default: float = 312.0) -> float:
    for key, val in PEAK_TFLOPS_BF16.items():
        if key in device_name:
            return val
    return default


def mfu(n_params_active: int, tokens_per_step: int, dt_seconds: float,
        peak_flops_per_sec: float) -> float:
    """Raw 6N-flops/token MFU (no attention term). Kept for quick estimates."""
    if dt_seconds <= 0:
        return 0.0
    achieved = 6 * n_params_active * tokens_per_step / dt_seconds
    return achieved / peak_flops_per_sec


def flops_per_token(n_params, n_layers, d_model, seq_len) -> float:
    """Karpathy/PaLM estimate: 6N (all matmuls incl. QKVO projections) plus the
    attention score+value matmuls that scale with sequence length and are NOT
    captured by the parameter count: 12 * L * d_model * T per token.
    """
    return 6 * n_params + 12 * n_layers * d_model * seq_len


class Throughput:
    """Rolling step-time tracker; reports tokens/sec and MFU.

    `warmup` ticks are excluded from the running average so torch.compile /
    cuDNN autotuning on the first steps doesn't depress the reported MFU.
    """

    def __init__(self, flops_per_step, tokens_per_step, peak_flops_per_sec,
                 window=50, warmup=10):
        self.fps = flops_per_step
        self.tps = tokens_per_step
        self.peak = peak_flops_per_sec
        self.times = deque(maxlen=window)
        self.warmup = warmup
        self._n = 0
        self._t0 = None

    def _mfu(self, dt):
        return (self.fps / dt) / self.peak if dt > 0 else 0.0

    def tick(self) -> dict | None:
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
            return None
        dt = now - self._t0
        self._t0 = now
        self._n += 1
        if self._n > self.warmup:
            self.times.append(dt)
        avg = (sum(self.times) / len(self.times)) if self.times else dt
        return {
            "dt_s": dt,
            "dt_avg_s": avg,
            "tokens_per_s": self.tps / dt,
            "mfu": self._mfu(dt),
            "mfu_avg": self._mfu(avg),
        }
