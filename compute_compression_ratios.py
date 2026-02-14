from __future__ import annotations

import argparse
import random
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer

EOT = b"<|endoftext|>"


def reservoir_sample_docs(path: Path, k: int, seed: int) -> list[str]:
    """
    Reservoir-sample k documents from a corpus where docs are delimited by <|endoftext|>.
    """
    rng = random.Random(seed)
    samples: list[bytes] = []
    seen = 0

    with open(path, "rb") as f:
        carry = b""
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break

            data = carry + chunk
            parts = data.split(EOT)
            carry = parts.pop()

            for doc in parts:
                doc = doc.strip(b"\r\n")
                if not doc:
                    continue
                seen += 1
                if len(samples) < k:
                    samples.append(doc)
                else:
                    j = rng.randrange(seen)
                    if j < k:
                        samples[j] = doc

        tail = carry.strip(b"\r\n")
        if tail:
            seen += 1
            if len(samples) < k:
                samples.append(tail)
            else:
                j = rng.randrange(seen)
                if j < k:
                    samples[j] = tail

    return [d.decode("utf-8", errors="ignore") for d in samples]


def bytes_per_token(tokenizer: Tokenizer, docs: list[str]) -> tuple[float, int, int]:
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))
    if total_tokens == 0:
        return float("inf"), total_bytes, total_tokens
    return total_bytes / total_tokens, total_bytes, total_tokens


def load_tinystories_tokenizer(root: Path) -> Tokenizer:
    # Preferred artifact location.
    vocab_path = root / "artifacts" / "tinystories_bpe_10k" / "vocab.json"
    merges_path = root / "artifacts" / "tinystories_bpe_10k" / "merges.txt"
    if vocab_path.exists() and merges_path.exists():
        return Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

    # Backward-compatible fallback artifact names.
    return Tokenizer.from_files(
        root / "artifacts" / "tinystories_bpe_vocab_10k.json",
        root / "artifacts" / "tinystories_bpe_merges_10k.txt",
        special_tokens=["<|endoftext|>"],
    )


def load_openwebtext_tokenizer(root: Path) -> Tokenizer:
    return Tokenizer.from_files(
        root / "artifacts" / "owt_bpe_32k" / "vocab.json",
        root / "artifacts" / "owt_bpe_32k" / "merges.txt",
        special_tokens=["<|endoftext|>"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute bytes/token compression ratios on sampled docs.")
    parser.add_argument("--k", type=int, default=10, help="Number of docs to sample from each corpus.")
    parser.add_argument("--seed", type=int, default=336, help="Random seed for reservoir sampling.")
    parser.add_argument("--show-cross-domain", action="store_true", help="Also print cross-domain ratios.")
    args = parser.parse_args()

    root = Path(".")
    tiny_corpus = root / "data" / "TinyStoriesV2-GPT4-train.txt"
    owt_corpus = root / "data" / "owt_train.txt"

    tiny_docs = reservoir_sample_docs(tiny_corpus, args.k, args.seed)
    owt_docs = reservoir_sample_docs(owt_corpus, args.k, args.seed)

    tiny_tokenizer = load_tinystories_tokenizer(root)
    owt_tokenizer = load_openwebtext_tokenizer(root)

    tiny_ratio, tiny_bytes, tiny_tokens = bytes_per_token(tiny_tokenizer, tiny_docs)
    owt_ratio, owt_bytes, owt_tokens = bytes_per_token(owt_tokenizer, owt_docs)

    print("In-domain compression ratio (bytes/token):")
    print(
        f"TinyStories tokenizer (10k) on TinyStories sample: "
        f"{tiny_ratio:.4f} (bytes={tiny_bytes}, tokens={tiny_tokens})"
    )
    print(
        f"OpenWebText tokenizer (32k) on OpenWebText sample: "
        f"{owt_ratio:.4f} (bytes={owt_bytes}, tokens={owt_tokens})"
    )

    if args.show_cross_domain:
        tiny_on_owt, b1, t1 = bytes_per_token(tiny_tokenizer, owt_docs)
        owt_on_tiny, b2, t2 = bytes_per_token(owt_tokenizer, tiny_docs)
        print()
        print("Cross-domain compression ratio (bytes/token):")
        print(
            f"TinyStories tokenizer (10k) on OpenWebText sample: "
            f"{tiny_on_owt:.4f} (bytes={b1}, tokens={t1})"
        )
        print(
            f"OpenWebText tokenizer (32k) on TinyStories sample: "
            f"{owt_on_tiny:.4f} (bytes={b2}, tokens={t2})"
        )


if __name__ == "__main__":
    main()
