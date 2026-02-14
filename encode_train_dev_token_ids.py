from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

EOT_TOKEN = "<|endoftext|>"
EOT_BYTES = EOT_TOKEN.encode("utf-8")


def _load_tokenizer_class() -> Any:
    try:
        from cs336_basics.tokenizer import Tokenizer

        return Tokenizer
    except Exception:
        tokenizer_path = Path(__file__).resolve().parent / "cs336_basics" / "tokenizer.py"
        spec = importlib.util.spec_from_file_location("cs336_tokenizer_module", tokenizer_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load tokenizer module from: {tokenizer_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Tokenizer


Tokenizer = _load_tokenizer_class()


@dataclass(frozen=True)
class EncodeJob:
    name: str
    input_path: Path
    output_path: Path
    tokenizer: Any


def load_tinystories_tokenizer(root: Path) -> Any:
    preferred_vocab = root / "artifacts" / "tinystories_bpe_10k" / "vocab.json"
    preferred_merges = root / "artifacts" / "tinystories_bpe_10k" / "merges.txt"
    if preferred_vocab.exists() and preferred_merges.exists():
        return Tokenizer.from_files(preferred_vocab, preferred_merges, special_tokens=[EOT_TOKEN])

    fallback_vocab = root / "artifacts" / "tinystories_bpe_vocab_10k.json"
    fallback_merges = root / "artifacts" / "tinystories_bpe_merges_10k.txt"
    return Tokenizer.from_files(fallback_vocab, fallback_merges, special_tokens=[EOT_TOKEN])


def load_openwebtext_tokenizer(root: Path) -> Any:
    return Tokenizer.from_files(
        root / "artifacts" / "owt_bpe_32k" / "vocab.json",
        root / "artifacts" / "owt_bpe_32k" / "merges.txt",
        special_tokens=[EOT_TOKEN],
    )


def _get_eot_id(tokenizer: Any) -> int:
    if hasattr(tokenizer, "special_token_to_id") and EOT_TOKEN in tokenizer.special_token_to_id:
        return int(tokenizer.special_token_to_id[EOT_TOKEN])

    encoded = tokenizer.encode(EOT_TOKEN)
    if len(encoded) != 1:
        raise ValueError("Tokenizer did not encode <|endoftext|> as a single token ID.")
    return int(encoded[0])


def _assert_uint16_safe(tokenizer: Any) -> None:
    max_token_id = max(int(token_id) for token_id in tokenizer.vocab)
    if max_token_id > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Max token id ({max_token_id}) exceeds uint16 range; use uint32 instead."
        )


def _count_tokens(tokenizer: Any, input_path: Path, chunk_size: int, eot_id: int) -> tuple[int, int]:
    del eot_id
    total_tokens = 0
    total_bytes = input_path.stat().st_size

    with open(input_path, "rb") as f, tqdm(
        total=total_bytes,
        desc=f"Count {input_path.name}",
        unit="B",
        unit_scale=True,
    ) as pbar:
        carry = b""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            pbar.update(len(chunk))
            data = carry + chunk
            parts = data.split(EOT_BYTES)
            carry = parts.pop()

            for part in parts:
                if part:
                    total_tokens += len(tokenizer.encode(part.decode("utf-8", errors="replace")))
                total_tokens += 1

        if carry:
            total_tokens += len(tokenizer.encode(carry.decode("utf-8", errors="replace")))

    return total_tokens, total_bytes


def _write_tokens(
    tokenizer: Any,
    input_path: Path,
    output_path: Path,
    total_tokens: int,
    chunk_size: int,
    eot_id: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(total_tokens,))

    total_bytes = input_path.stat().st_size
    offset = 0

    with open(input_path, "rb") as f, tqdm(
        total=total_bytes,
        desc=f"Write {input_path.name}",
        unit="B",
        unit_scale=True,
    ) as pbar:
        carry = b""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            pbar.update(len(chunk))
            data = carry + chunk
            parts = data.split(EOT_BYTES)
            carry = parts.pop()

            for part in parts:
                if part:
                    ids = tokenizer.encode(part.decode("utf-8", errors="replace"))
                    n = len(ids)
                    if n:
                        arr[offset : offset + n] = np.asarray(ids, dtype=np.uint16)
                        offset += n
                arr[offset] = eot_id
                offset += 1

        if carry:
            ids = tokenizer.encode(carry.decode("utf-8", errors="replace"))
            n = len(ids)
            if n:
                arr[offset : offset + n] = np.asarray(ids, dtype=np.uint16)
                offset += n

    if offset != total_tokens:
        raise RuntimeError(f"Token count mismatch for {input_path}: expected {total_tokens}, wrote {offset}.")

    arr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode TinyStories and OpenWebText train/valid splits into uint16 token-id arrays."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing dataset txt files.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts") / "tokenized",
        help="Output directory for .npy files.",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=16,
        help="Streaming chunk size in MB when reading corpora.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_dir = args.data_dir
    out_dir = args.out_dir
    chunk_size = args.chunk_size_mb * 1024 * 1024

    tiny_tok = load_tinystories_tokenizer(root)
    owt_tok = load_openwebtext_tokenizer(root)
    _assert_uint16_safe(tiny_tok)
    _assert_uint16_safe(owt_tok)

    tiny_eot_id = _get_eot_id(tiny_tok)
    owt_eot_id = _get_eot_id(owt_tok)

    jobs = [
        EncodeJob(
            name="tinystories_train",
            input_path=data_dir / "TinyStoriesV2-GPT4-train.txt",
            output_path=out_dir / "tinystories_train_uint16.npy",
            tokenizer=tiny_tok,
        ),
        EncodeJob(
            name="tinystories_valid",
            input_path=data_dir / "TinyStoriesV2-GPT4-valid.txt",
            output_path=out_dir / "tinystories_valid_uint16.npy",
            tokenizer=tiny_tok,
        ),
        EncodeJob(
            name="owt_train",
            input_path=data_dir / "owt_train.txt",
            output_path=out_dir / "owt_train_uint16.npy",
            tokenizer=owt_tok,
        ),
        EncodeJob(
            name="owt_valid",
            input_path=data_dir / "owt_valid.txt",
            output_path=out_dir / "owt_valid_uint16.npy",
            tokenizer=owt_tok,
        ),
    ]

    summary: dict[str, dict[str, object]] = {}

    for job in jobs:
        if not job.input_path.exists():
            raise FileNotFoundError(f"Missing input file: {job.input_path}")

        print(f"\n[{job.name}] {job.input_path} -> {job.output_path}")
        eot_id = tiny_eot_id if job.tokenizer is tiny_tok else owt_eot_id
        total_tokens, total_bytes = _count_tokens(job.tokenizer, job.input_path, chunk_size, eot_id)
        _write_tokens(job.tokenizer, job.input_path, job.output_path, total_tokens, chunk_size, eot_id)

        summary[job.name] = {
            "input_path": str(job.input_path),
            "output_path": str(job.output_path),
            "dtype": "uint16",
            "num_tokens": int(total_tokens),
            "input_bytes": int(total_bytes),
            "bytes_per_token": (float(total_bytes) / float(total_tokens)) if total_tokens else None,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "tokenization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone. Saved files:")
    for name, info in summary.items():
        print(f"- {name}: {info['output_path']} (tokens={info['num_tokens']})")
    print(f"- summary: {summary_path}")


if __name__ == "__main__":
    main()
