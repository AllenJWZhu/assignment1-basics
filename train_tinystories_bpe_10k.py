from __future__ import annotations

import json
import os
import pathlib
import threading
import time

import psutil

from tests.adapters import run_train_bpe_tinystories
from tests.common import gpt2_bytes_to_unicode


def main() -> None:
    out_dir = pathlib.Path("artifacts/tinystories_bpe_10k")
    out_dir.mkdir(parents=True, exist_ok=True)
    num_processes = min(8, max(1, os.cpu_count() or 1))
    num_chunks = max(num_processes * 8, num_processes)

    peak_rss = {"value": 0}
    stop = {"flag": False}
    proc = psutil.Process(os.getpid())

    def monitor_memory() -> None:
        while not stop["flag"]:
            try:
                rss = proc.memory_info().rss
                if rss > peak_rss["value"]:
                    peak_rss["value"] = rss
            except Exception:
                pass
            time.sleep(0.2)

    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()

    t0 = time.time()
    vocab, merges = run_train_bpe_tinystories(
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
        num_processes=num_processes,
        num_chunks=num_chunks,
        verbose=True,
    )
    t1 = time.time()

    stop["flag"] = True
    monitor_thread.join(timeout=1)

    byte_encoder = gpt2_bytes_to_unicode()

    def to_gpt2_string(token_bytes: bytes) -> str:
        return "".join(byte_encoder[b] for b in token_bytes)

    # Save vocab as GPT-2 printable token strings -> token id.
    vocab_json = {to_gpt2_string(token_bytes): token_id for token_id, token_bytes in vocab.items()}
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False)

    # Save merges in GPT-2 merges format: "<token1> <token2>" per line.
    with open(out_dir / "merges.txt", "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{to_gpt2_string(left)} {to_gpt2_string(right)}\n")

    longest = max(vocab.values(), key=len)
    metrics = {
        "vocab_size_final": len(vocab),
        "num_merges": len(merges),
        "wall_hours": (t1 - t0) / 3600.0,
        "peak_rss_gb": peak_rss["value"] / (1024**3),
        "longest_token_num_bytes": len(longest),
        "longest_token_utf8": longest.decode("utf-8", errors="replace"),
        "longest_token_bytes_repr": repr(longest),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Saved:", out_dir / "vocab.json")
    print("Saved:", out_dir / "merges.txt")
    print("Saved:", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
