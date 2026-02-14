from __future__ import annotations

import json
import os
import pathlib
import threading
import time

import psutil

from tests.adapters import run_train_bpe_openwebtext
from tests.common import gpt2_bytes_to_unicode


def main() -> None:
    input_path = pathlib.Path("data/owt_train.txt")
    out_dir = pathlib.Path("artifacts/owt_bpe_32k")
    out_dir.mkdir(parents=True, exist_ok=True)

    peak_rss = {"value": 0}
    stop = {"flag": False}
    proc = psutil.Process(os.getpid())

    def monitor() -> None:
        while not stop["flag"]:
            try:
                rss = proc.memory_info().rss
                if rss > peak_rss["value"]:
                    peak_rss["value"] = rss
            except Exception:
                pass
            time.sleep(0.5)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()

    t0 = time.time()
    vocab, merges = run_train_bpe_openwebtext(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )
    t1 = time.time()

    stop["flag"] = True
    t.join(timeout=2)

    byte_encoder = gpt2_bytes_to_unicode()

    def enc(bs: bytes) -> str:
        return "".join(byte_encoder[b] for b in bs)

    # vocab.json format: token_string -> token_id (same style as GPT-2 fixtures)
    vocab_json = {enc(token_bytes): token_id for token_id, token_bytes in vocab.items()}
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False)

    # merges.txt format: "token1 token2" per line
    with open(out_dir / "merges.txt", "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{enc(left)} {enc(right)}\n")

    longest = max(vocab.values(), key=len)
    top_longest = sorted(vocab.values(), key=len, reverse=True)[:10]

    metrics = {
        "input_path": str(input_path),
        "vocab_size_final": len(vocab),
        "num_merges": len(merges),
        "wall_seconds": t1 - t0,
        "wall_hours": (t1 - t0) / 3600.0,
        "peak_rss_gb": peak_rss["value"] / (1024**3),
        "longest_token_num_bytes": len(longest),
        "longest_token_utf8": longest.decode("utf-8", errors="replace"),
        "longest_token_bytes_repr": repr(longest),
        "top_10_longest_tokens_utf8": [x.decode("utf-8", errors="replace") for x in top_longest],
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Saved:", out_dir / "vocab.json")
    print("Saved:", out_dir / "merges.txt")
    print("Saved:", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
