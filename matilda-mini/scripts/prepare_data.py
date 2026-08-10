"""Tokenize a HuggingFace text dataset into verifiable uint16 shards.

Default: FineWeb-Edu sample-10BT (best small-scale benchmark movement, max
comparability to Pythia/SmolLM). Dataset-agnostic by design so DCLM /
Nemotron-CC become drop-in data-ablation rows:

    python scripts/prepare_data.py --target-tokens 3_000_000_000 \
        --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --out-dir data/fwedu

    # ablation rows (same flags, different source):
    --dataset mlfoundations/dclm-baseline-1.0 --out-dir data/dclm
    --dataset nvidia/Nemotron-CC            --out-dir data/nemotron

Streams the source (no full download), so disk holds only the tokenized output.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from matilda.data import ShardWriter, verify_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--target-tokens", type=int, default=3_000_000_000)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--out-dir", default="data/fwedu")
    args = ap.parse_args()

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding(args.tokenizer)
    eot = enc.eot_token
    assert enc.n_vocab <= 65535, "vocab > uint16; use a smaller tokenizer"

    ds = load_dataset(args.dataset, name=args.name, split=args.split,
                      streaming=True)
    writer = ShardWriter(args.out_dir, shard_tokens=args.shard_tokens)

    n_docs = 0
    for doc in ds:
        ids = enc.encode_ordinary(doc[args.text_key])
        ids.append(eot)                       # document boundary
        writer.add(ids)
        n_docs += 1
        if n_docs % 1000 == 0:
            print(f"\rdocs={n_docs:,} tokens={writer.total_tokens:,}", end="")
        if writer.total_tokens >= args.target_tokens:
            break

    manifest = writer.close(meta={
        "dataset": args.dataset, "name": args.name, "split": args.split,
        "tokenizer": args.tokenizer, "eot_token": eot, "n_docs": n_docs,
    })
    print(f"\nwrote {manifest['total_tokens']:,} tokens in "
          f"{len(manifest['shards'])} shards -> {args.out_dir}")
    verify_manifest(args.out_dir)
    print("manifest verified (checksums + sizes OK)")


if __name__ == "__main__":
    main()
