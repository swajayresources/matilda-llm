"""Data streams + tokenized-shard tooling.

All streams expose the same interface so the training loop never changes when
we swap synthetic data for real tokenized FineWeb shards:

    x, y = stream.next()            # (B, T) input, (B, T) targets
    state = stream.state_dict()     # resumable position / RNG
    stream.load_state_dict(state)

On-disk format: token ids as a flat `uint16` `.bin` (valid for vocab <= 65535,
which GPT-2's 50257 satisfies). A `manifest.json` records per-shard token counts
and SHA-256 so the prepared data is verifiable and reproducible.
"""

from __future__ import annotations

import os
import json
import glob
import hashlib

import numpy as np
import torch

DTYPE = np.uint16


# --------------------------------------------------------------------------
# shard writing / verification (used by scripts/prepare_data.py, unit-tested)
# --------------------------------------------------------------------------
def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class ShardWriter:
    """Accumulate token ids and flush fixed-size `.bin` shards + a manifest."""

    def __init__(self, out_dir, shard_tokens=100_000_000, prefix="shard"):
        self.out_dir = out_dir
        self.shard_tokens = shard_tokens
        self.prefix = prefix
        os.makedirs(out_dir, exist_ok=True)
        self._buf: list[np.ndarray] = []
        self._buf_len = 0
        self._shard_idx = 0
        self.total_tokens = 0
        self.manifest_entries: list[dict] = []

    def add(self, tokens) -> None:
        arr = np.asarray(tokens, dtype=DTYPE)
        self._buf.append(arr)
        self._buf_len += arr.size
        self.total_tokens += arr.size
        while self._buf_len >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def _flush(self, n) -> None:
        flat = np.concatenate(self._buf) if len(self._buf) > 1 else self._buf[0]
        out, rest = flat[:n], flat[n:]
        path = os.path.join(self.out_dir, f"{self.prefix}_{self._shard_idx:05d}.bin")
        out.tofile(path)
        self.manifest_entries.append({
            "file": os.path.basename(path),
            "tokens": int(out.size),
            "sha256": sha256_file(path),
        })
        self._shard_idx += 1
        self._buf = [rest] if rest.size else []
        self._buf_len = rest.size

    def close(self, meta: dict | None = None) -> dict:
        if self._buf_len > 0:
            self._flush(self._buf_len)        # final partial shard
        manifest = {
            "total_tokens": int(self.total_tokens),
            "shards": self.manifest_entries,
            **(meta or {}),
        }
        with open(os.path.join(self.out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest


def verify_manifest(out_dir) -> bool:
    """Re-checksum every shard against the manifest. Raises on mismatch."""
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)
    for entry in manifest["shards"]:
        path = os.path.join(out_dir, entry["file"])
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {entry['file']}")
        if os.path.getsize(path) != entry["tokens"] * 2:  # uint16 = 2 bytes
            raise ValueError(f"size mismatch for {entry['file']}")
    return True


def shard_paths(out_dir) -> list[str]:
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)
    return [os.path.join(out_dir, e["file"]) for e in manifest["shards"]]


# --------------------------------------------------------------------------
# streams
# --------------------------------------------------------------------------
class SyntheticStream:
    """Deterministic random tokens. For dev/CI and loop testing on CPU."""

    def __init__(self, vocab_size, batch_size, seq_len, seed=0, device="cpu"):
        self.vocab_size = vocab_size
        self.B = batch_size
        self.T = seq_len
        self.device = device
        self.gen = torch.Generator().manual_seed(seed)
        self.pos = 0

    def next(self):
        seq = torch.randint(0, self.vocab_size, (self.B, self.T + 1),
                            generator=self.gen)
        x = seq[:, :-1].contiguous().to(self.device)
        y = seq[:, 1:].contiguous().to(self.device)
        self.pos += 1
        return x, y

    def state_dict(self):
        return {"pos": self.pos, "gen": self.gen.get_state()}

    def load_state_dict(self, s):
        self.pos = s["pos"]
        # set_state needs a CPU ByteTensor; map_location="cuda" can move it.
        self.gen.set_state(s["gen"].cpu())


class BinStream:
    """Packed windows sampled from memory-mapped uint16 token shards.

    Each batch element is a random T+1 window from a shard chosen with
    probability proportional to its size; x/y are the next-token shift. Random
    sampling (rather than a sequential cursor) means epochs are implicit and
    resume only needs the RNG state.
    """

    def __init__(self, bin_paths, batch_size, seq_len, seed=0, device="cpu"):
        assert bin_paths, "no shards provided"
        self.paths = list(bin_paths)
        self.B = batch_size
        self.T = seq_len
        self.device = device
        # Drop shards too small to sample a T+1 window (e.g. the small final
        # overflow shard prepare_data emits). Filter by on-disk size FIRST so we
        # never open mmaps we'll discard (those hold a file lock on Windows).
        min_bytes = (seq_len + 2) * 2          # uint16 = 2 bytes/token
        keep = [p for p in self.paths if os.path.getsize(p) >= min_bytes]
        dropped = len(self.paths) - len(keep)
        if dropped:
            print(f"[data] skipping {dropped} shard(s) smaller than seq_len+1")
        assert keep, "no shard is large enough for seq_len+1"
        self.paths = keep
        self.arrays = [np.memmap(p, dtype=DTYPE, mode="r") for p in keep]
        self.sizes = torch.tensor([float(a.size) for a in self.arrays])
        self.weights = self.sizes / self.sizes.sum()
        self.gen = torch.Generator().manual_seed(seed)
        self.pos = 0
        # int32 halves H2D bytes vs int64; cast to long happens on the GPU.
        # uint16 max (50257 vocab) exceeds int16 range, so int32 is the floor.
        pin = (str(device) != "cpu")
        self._xbuf = torch.empty((batch_size, seq_len), dtype=torch.int32,
                                 pin_memory=pin)
        self._ybuf = torch.empty((batch_size, seq_len), dtype=torch.int32,
                                 pin_memory=pin)

    def next(self):
        # vectorized sampling: one multinomial + one rand call, no per-element sync
        shard_ids = torch.multinomial(self.weights, self.B, replacement=True,
                                      generator=self.gen)
        u = torch.rand(self.B, generator=self.gen)
        sizes = self.sizes[shard_ids]
        starts = (u * (sizes - (self.T + 1)).clamp(min=0)).long()
        xb = np.empty((self.B, self.T), dtype=np.int32)
        yb = np.empty((self.B, self.T), dtype=np.int32)
        for i in range(self.B):
            arr = self.arrays[int(shard_ids[i])]
            s = int(starts[i])
            window = arr[s:s + self.T + 1].astype(np.int32)
            xb[i] = window[:-1]
            yb[i] = window[1:]
        self.pos += 1
        self._xbuf.copy_(torch.from_numpy(xb))
        self._ybuf.copy_(torch.from_numpy(yb))
        x = self._xbuf.to(self.device, non_blocking=True).long()
        y = self._ybuf.to(self.device, non_blocking=True).long()
        return x, y

    def state_dict(self):
        return {"pos": self.pos, "gen": self.gen.get_state()}

    def load_state_dict(self, s):
        self.pos = s["pos"]
        # set_state needs a CPU ByteTensor; map_location="cuda" can move it.
        self.gen.set_state(s["gen"].cpu())
