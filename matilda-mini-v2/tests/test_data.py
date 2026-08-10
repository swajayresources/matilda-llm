"""Data pipeline: shard writing/verification + BinStream correctness."""

import os
import json

import numpy as np
import pytest
import torch

from matilda.data import (
    ShardWriter, verify_manifest, shard_paths, BinStream, DTYPE,
)


def test_shardwriter_roundtrip_and_manifest(tmp_path):
    w = ShardWriter(str(tmp_path), shard_tokens=100)
    # 250 tokens -> two full shards (100) + one partial (50)
    w.add(list(range(0, 120)))
    w.add(list(range(120, 250)))
    manifest = w.close(meta={"tokenizer": "gpt2"})

    assert manifest["total_tokens"] == 250
    assert [s["tokens"] for s in manifest["shards"]] == [100, 100, 50]
    assert manifest["tokenizer"] == "gpt2"
    assert verify_manifest(str(tmp_path)) is True

    # reconstructed token stream matches the input exactly
    rebuilt = np.concatenate(
        [np.fromfile(p, dtype=DTYPE) for p in shard_paths(str(tmp_path))])
    assert rebuilt.tolist() == list(range(250))


def test_verify_detects_corruption(tmp_path):
    w = ShardWriter(str(tmp_path), shard_tokens=100)
    w.add(list(range(150)))
    w.close()
    # corrupt the first shard's bytes without changing its size
    p = shard_paths(str(tmp_path))[0]
    data = bytearray(open(p, "rb").read())
    data[0] ^= 0xFF
    open(p, "wb").write(data)
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_manifest(str(tmp_path))


def _write_ramp_bin(path, n):
    # values 0..n-1 so a correct next-token window always has y == x + 1
    np.arange(n, dtype=DTYPE).tofile(path)


def test_binstream_shift_is_next_token(tmp_path):
    p = os.path.join(tmp_path, "ramp.bin")
    _write_ramp_bin(p, 5000)
    s = BinStream([p], batch_size=8, seq_len=32, seed=0)
    x, y = s.next()
    assert x.shape == (8, 32) and y.shape == (8, 32)
    assert torch.equal(y, x + 1)              # packed next-token target


def test_binstream_resume_is_deterministic(tmp_path):
    p = os.path.join(tmp_path, "ramp.bin")
    _write_ramp_bin(p, 5000)
    s = BinStream([p], batch_size=4, seq_len=16, seed=7)
    snap = s.state_dict()
    x1, y1 = s.next()
    s.load_state_dict(snap)
    x2, y2 = s.next()
    assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_binstream_skips_too_small_shards(tmp_path):
    # a tiny final shard (< seq_len+1) must be skipped, not crash (regression)
    big = os.path.join(tmp_path, "big.bin")
    tiny = os.path.join(tmp_path, "tiny.bin")
    _write_ramp_bin(big, 5000)
    _write_ramp_bin(tiny, 10)                  # smaller than seq_len+1
    s = BinStream([big, tiny], batch_size=4, seq_len=32, seed=0)
    assert len(s.arrays) == 1                  # tiny dropped
    x, y = s.next()
    assert torch.equal(y, x + 1)


def test_binstream_multishard_weighting(tmp_path):
    p1 = os.path.join(tmp_path, "a.bin")
    p2 = os.path.join(tmp_path, "b.bin")
    _write_ramp_bin(p1, 2000)
    _write_ramp_bin(p2, 2000)
    s = BinStream([p1, p2], batch_size=4, seq_len=8, seed=1)
    x, y = s.next()
    assert torch.equal(y, x + 1)              # still correct across shards
