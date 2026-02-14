from __future__ import annotations

import argparse
import random
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer

EOT = b"<|endoftext|>"


def reservoir_sample_docs(path: Path, k: int, seed: int) -> list[str]:
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


def load_tinystories_tokenizer(root: Path) -> Tokenizer:
    vocab_path = root / "artifacts" / "tinystories_bpe_10k" / "vocab.json"
    merges_path = root / "artifacts" / "tinystories_bpe_10k" / "merges.txt"
    if vocab_path.exists() and merges_path.exists():
        return Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

    return Tokenizer.from_files(
        root / "artifacts" / "tinystories_bpe_vocab_10k.json",
        root / "artifacts" / "tinystories_bpe_merges_10k.txt",
        special_tokens=["<|endoftext|>"],
    )


def load_owt_tokenizer(root: Path) -> Tokenizer:
    return Tokenizer.from_files(
        root / "artifacts" / "owt_bpe_32k" / "vocab.json",
        root / "artifacts" / "owt_bpe_32k" / "merges.txt",
        special_tokens=["<|endoftext|>"],
    )


def bytes_per_token(tokenizer: Tokenizer, docs: list[str]) -> tuple[float, int, int]:
    total_bytes = 0
    total_tokens = 0
    for d in docs:
        total_bytes += len(d.encode("utf-8"))
        total_tokens += len(tokenizer.encode(d))
    return (total_bytes / total_tokens), total_bytes, total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OWT compression with TinyStories vs OWT tokenizers.")
    parser.add_argument("--k", type=int, default=10, help="Number of OWT documents to sample.")
    parser.add_argument("--seed", type=int, default=336, help="Sampling seed.")
    args = parser.parse_args()

    root = Path(".")
    owt_path = root / "data" / "owt_train.txt"
    docs = reservoir_sample_docs(owt_path, args.k, args.seed)

    tiny_tok = load_tinystories_tokenizer(root)
    owt_tok = load_owt_tokenizer(root)

    tiny_ratio, tiny_bytes, tiny_tokens = bytes_per_token(tiny_tok, docs)
    owt_ratio, owt_bytes, owt_tokens = bytes_per_token(owt_tok, docs)

    print(f"Sampled OWT docs: {len(docs)}")
    print()
    print("OpenWebText sample compression (bytes/token):")
    print(f"TinyStories tokenizer (10k): {tiny_ratio:.4f} (bytes={tiny_bytes}, tokens={tiny_tokens})")
    print(f"OpenWebText tokenizer (32k): {owt_ratio:.4f} (bytes={owt_bytes}, tokens={owt_tokens})")
    print()

    if owt_ratio > tiny_ratio:
        improvement = (owt_ratio / tiny_ratio - 1.0) * 100.0
        print(
            "Qualitative summary: OWT tokenizer compresses better on OWT, "
            f"using ~{improvement:.2f}% more bytes per token. "
            "TinyStories tokenizer tends to split web-specific patterns into more tokens."
        )
    elif tiny_ratio > owt_ratio:
        improvement = (tiny_ratio / owt_ratio - 1.0) * 100.0
        print(
            "Qualitative summary: TinyStories tokenizer compresses better on this OWT sample "
            f"(~{improvement:.2f}% more bytes per token), which is unusual and worth re-checking artifacts."
        )
    else:
        print("Qualitative summary: both tokenizers have identical compression on this sample.")


if __name__ == "__main__":
    main()
