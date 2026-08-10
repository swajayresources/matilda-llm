"""Build the SmolLM-corpus 75/15/10 mix into verifiable uint16 shards.

Adopts the SmolLM data recipe (proven sub-1B winner over Pythia-410M /
MobileLLM-350M / Qwen2-500M) but re-tokenized through our locked GPT-2 BPE so
that loader, shard format, and tokenizer-comparability with v1 are unchanged.

Mix targets are token-based, not document-based: documents have wildly
different lengths, so picking-by-doc-count would drift far from the intended
mix. We pick from whichever source is most under-represented in *tokens*
emitted so far, which converges to exactly the configured weights.

Default 15B-token target matches configs/base_350m.json. Streams the three
HuggingFaceTB/smollm-corpus subsets so disk only holds the tokenized output.

Usage (on the GPU instance):

    python scripts/prepare_smollm_data.py \
        --target-tokens 15_000_000_000 \
        --out-dir data/smollm_mix
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from matilda.data import ShardWriter, verify_manifest  # noqa: E402

log = logging.getLogger("prepare_smollm_data")


SOURCES: tuple[tuple[str, float], ...] = (
    ("fineweb-edu-dedup", 0.75),   # bulk: web reading-comprehension signal
    ("cosmopedia-v2",     0.15),   # the Cosmopedia premium (synthetic textbooks)
    ("python-edu",        0.10),   # code: HumanEval signal, modest cost
)
DATASET = "HuggingFaceTB/smollm-corpus"
TEXT_KEY = "text"


def _open_stream(name: str):
    from datasets import load_dataset
    return iter(load_dataset(DATASET, name=name, split="train", streaming=True))


def _pick_source(accumulated: dict[str, int]) -> str:
    """Return the source whose token deficit relative to its weight is largest.
    Equivalent to argmin(accumulated[s] / weight[s]) over s in SOURCES."""
    return min(
        (name for name, _ in SOURCES),
        key=lambda n: accumulated[n] / next(w for s, w in SOURCES if s == n),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=15_000_000_000)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--out-dir", default="data/smollm_mix")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import tiktoken

    enc = tiktoken.get_encoding(args.tokenizer)
    eot = enc.eot_token
    assert enc.n_vocab <= 65535, "vocab > uint16; loader assumes uint16 shards"

    streams = {name: _open_stream(name) for name, _ in SOURCES}
    accumulated = {name: 0 for name, _ in SOURCES}
    n_docs = {name: 0 for name, _ in SOURCES}
    writer = ShardWriter(args.out_dir, shard_tokens=args.shard_tokens)

    while writer.total_tokens < args.target_tokens:
        pick = _pick_source(accumulated)
        try:
            doc = next(streams[pick])
        except StopIteration:
            # Defensive: at 15B target with smollm-corpus sizes, this branch
            # shouldn't trigger. If it does, restart the source so the mix
            # ratio is preserved rather than silently rebalancing.
            log.warning("source %s exhausted at %d tokens; restarting stream",
                        pick, writer.total_tokens)
            streams[pick] = _open_stream(pick)
            doc = next(streams[pick])

        ids = enc.encode_ordinary(doc[TEXT_KEY])
        ids.append(eot)                                    # document boundary
        writer.add(ids)
        accumulated[pick] += len(ids)
        n_docs[pick] += 1

        total_docs = sum(n_docs.values())
        if total_docs % args.log_every == 0:
            mix = {n: f"{accumulated[n] / max(1, writer.total_tokens) * 100:5.2f}%"
                   for n in accumulated}
            log.info("docs=%d tokens=%d mix=%s",
                     total_docs, writer.total_tokens, mix)

    manifest = writer.close(meta={
        "dataset": DATASET,
        "splits": [name for name, _ in SOURCES],
        "weights": {name: weight for name, weight in SOURCES},
        "tokenizer": args.tokenizer,
        "eot_token": eot,
        "n_docs_per_source": n_docs,
        "tokens_per_source": accumulated,
    })
    log.info("wrote %d tokens in %d shards -> %s",
             manifest["total_tokens"], len(manifest["shards"]), args.out_dir)
    verify_manifest(args.out_dir)
    log.info("manifest verified (checksums + sizes OK)")


if __name__ == "__main__":
    main()
