from __future__ import annotations

import argparse
import cProfile
import inspect
import io
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

from tests.adapters import run_train_bpe


def run_once(input_path: Path, vocab_size: int, special_tokens: list[str]) -> None:
    run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )


def profile_cprofile(
    input_path: Path, vocab_size: int, special_tokens: list[str], top: int, sort_by: str
) -> None:
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    run_once(input_path, vocab_size, special_tokens)
    pr.disable()
    elapsed = time.perf_counter() - t0

    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sort_by).print_stats(top)

    print(f"\n=== cProfile (wall time: {elapsed:.3f}s) ===")
    print(s.getvalue())


def profile_lines(
    input_path: Path, vocab_size: int, special_tokens: list[str], top: int
) -> None:
    target_file = Path(run_train_bpe.__code__.co_filename).resolve()
    target_names = {"run_train_bpe", "merge_pair_in_word"}

    line_time = defaultdict(float)
    line_hits = defaultdict(int)
    frame_state = {}

    def is_target(frame) -> bool:
        code = frame.f_code
        return (
            Path(code.co_filename).resolve() == target_file
            and code.co_name in target_names
        )

    def tracer(frame, event, arg):
        now = time.perf_counter()

        if event == "call":
            if is_target(frame):
                frame_state[frame] = (frame.f_lineno, now, frame.f_code.co_name)
            return tracer

        if frame in frame_state:
            last_line, last_time, func_name = frame_state[frame]
            dt = now - last_time
            key = (func_name, last_line)
            line_time[key] += dt
            line_hits[key] += 1

            if event == "line":
                frame_state[frame] = (frame.f_lineno, now, func_name)
            elif event == "return":
                frame_state.pop(frame, None)
            elif event == "exception":
                frame_state[frame] = (frame.f_lineno, now, func_name)

        return tracer

    t0 = time.perf_counter()
    sys.settrace(tracer)
    try:
        run_once(input_path, vocab_size, special_tokens)
    finally:
        sys.settrace(None)
    elapsed = time.perf_counter() - t0

    src_lines, start_line = inspect.getsourcelines(run_train_bpe)
    line_lookup = {
        start_line + i: text.rstrip("\n") for i, text in enumerate(src_lines)
    }

    print(f"\n=== Line Hotspots (trace wall time: {elapsed:.3f}s, includes tracing overhead) ===")
    for (func, lineno), secs in sorted(line_time.items(), key=lambda kv: kv[1], reverse=True)[:top]:
        code = line_lookup.get(lineno, "") if func == "run_train_bpe" else "merge_pair_in_word"
        print(f"{func}:{lineno:4d}  {secs:8.4f}s  hits={line_hits[(func, lineno)]}  {code.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("tests/fixtures/tinystories_sample_5M.txt"))
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--special-token", action="append")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--sort", default="cumtime")
    parser.add_argument("--line", action="store_true", help="Also show line-level hotspots")
    args = parser.parse_args()

    special_tokens = args.special_token or ["<|endoftext|>"]

    profile_cprofile(args.input, args.vocab_size, special_tokens, args.top, args.sort)
    if args.line:
        profile_lines(args.input, args.vocab_size, special_tokens, args.top)


if __name__ == "__main__":
    main()
