from __future__ import annotations

import heapq
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from typing import BinaryIO

import regex as re
from tqdm import tqdm

GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

_WORKER_PRETOKEN_PATTERN: re.Pattern[str] | None = None
_WORKER_SPECIAL_SPLIT_PATTERN: re.Pattern[str] | None = None


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk a byte file into independent ranges whose boundaries align with the
    start of `split_special_token`.
    """
    assert isinstance(split_special_token, bytes), "split_special_token must be bytes."

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _build_special_split_pattern(special_tokens: list[str]) -> re.Pattern[str] | None:
    if not special_tokens:
        return None
    escaped = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    return re.compile("|".join(escaped))


def _count_pretokens_in_text(
    text: str,
    pretoken_pattern: re.Pattern[str],
    special_split_pattern: re.Pattern[str] | None,
) -> Counter[bytes]:
    segments = special_split_pattern.split(text) if special_split_pattern else [text]
    counts: Counter[bytes] = Counter()
    for segment in segments:
        for match in pretoken_pattern.finditer(segment):
            counts[match.group(0).encode("utf-8")] += 1
    return counts


def _init_pretoken_worker(special_tokens: list[str]) -> None:
    global _WORKER_PRETOKEN_PATTERN, _WORKER_SPECIAL_SPLIT_PATTERN
    _WORKER_PRETOKEN_PATTERN = re.compile(GPT2_PRETOKEN_PATTERN)
    _WORKER_SPECIAL_SPLIT_PATTERN = _build_special_split_pattern(special_tokens)


def _count_pretokens_in_chunk(args: tuple[str, int, int]) -> Counter[bytes]:
    input_path, start, end = args
    assert _WORKER_PRETOKEN_PATTERN is not None
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
    chunk_text = chunk.decode("utf-8", errors="ignore")
    return _count_pretokens_in_text(
        chunk_text,
        _WORKER_PRETOKEN_PATTERN,
        _WORKER_SPECIAL_SPLIT_PATTERN,
    )


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer and return (vocab, merges).
    """
    pretoken_pattern = re.compile(GPT2_PRETOKEN_PATTERN)
    special_split_pattern = _build_special_split_pattern(special_tokens)
    num_processes = int(kwargs.get("num_processes", 1))
    num_chunks = int(kwargs.get("num_chunks", max(num_processes * 8, num_processes)))
    verbose = bool(kwargs.get("verbose", False))

    def merge_pair_in_word(word: tuple[int, ...], pair: tuple[int, int], new_token_id: int) -> tuple[int, ...]:
        if len(word) < 2:
            return word
        merged: list[int] = []
        i = 0
        first, second = pair
        while i < len(word):
            if i + 1 < len(word) and word[i] == first and word[i + 1] == second:
                merged.append(new_token_id)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        return tuple(merged)

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    if vocab_size <= len(vocab):
        return vocab, []

    if num_processes > 1 and special_tokens:
        desired_num_chunks = max(1, max(num_processes, num_chunks))
        split_token = "<|endoftext|>" if "<|endoftext|>" in special_tokens else max(special_tokens, key=len)
        with open(input_path, "rb") as f:
            boundaries = _find_chunk_boundaries(f, desired_num_chunks, split_token.encode("utf-8"))
        chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))

        if len(chunk_ranges) > 1:
            pretoken_counts: Counter[bytes] = Counter()
            with mp.Pool(
                processes=min(num_processes, len(chunk_ranges)),
                initializer=_init_pretoken_worker,
                initargs=(special_tokens,),
            ) as pool:
                chunk_iter = pool.imap_unordered(
                    _count_pretokens_in_chunk,
                    [(os.fspath(input_path), start, end) for start, end in chunk_ranges],
                    chunksize=1,
                )
                if verbose:
                    chunk_iter = tqdm(chunk_iter, total=len(chunk_ranges), desc="Pretokenize chunks", unit="chunk")
                for c in chunk_iter:
                    pretoken_counts.update(c)
        else:
            with open(input_path, encoding="utf-8") as f:
                corpus = f.read()
            pretoken_counts = _count_pretokens_in_text(corpus, pretoken_pattern, special_split_pattern)
    else:
        with open(input_path, encoding="utf-8") as f:
            corpus = f.read()
        pretoken_counts = _count_pretokens_in_text(corpus, pretoken_pattern, special_split_pattern)

    word_by_id: dict[int, tuple[int, ...]] = {}
    word_freq_by_id: dict[int, int] = {}
    for word_id, (word_bytes, freq) in enumerate(pretoken_counts.items()):
        word_by_id[word_id] = tuple(word_bytes)
        word_freq_by_id[word_id] = freq

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for word_id, word in word_by_id.items():
        if len(word) < 2:
            continue
        word_freq = word_freq_by_id[word_id]
        local_pair_counts = Counter(zip(word, word[1:]))
        for pair, local_count in local_pair_counts.items():
            pair_counts[pair] += local_count * word_freq
            pair_to_words[pair].add(word_id)

    pair_heap: list[tuple[int, tuple[int, int]]] = [(-count, pair) for pair, count in pair_counts.items() if count > 0]
    heapq.heapify(pair_heap)

    merges: list[tuple[bytes, bytes]] = []
    target_merges = max(0, vocab_size - len(vocab))
    merge_pbar = tqdm(total=target_merges, desc="BPE merges", unit="merge", disable=not verbose)
    merges_since_heap_rebuild = 0

    while len(vocab) < vocab_size and pair_counts and pair_heap:
        best_count: int | None = None
        candidate_pairs: list[tuple[int, int]] = []

        while pair_heap:
            neg_count, pair = heapq.heappop(pair_heap)
            count = -neg_count
            if pair_counts.get(pair, 0) != count:
                continue
            best_count = count
            candidate_pairs.append(pair)
            while pair_heap and -pair_heap[0][0] == count:
                neg_count2, pair2 = heapq.heappop(pair_heap)
                count2 = -neg_count2
                if pair_counts.get(pair2, 0) == count2:
                    candidate_pairs.append(pair2)
            break

        if best_count is None or not candidate_pairs:
            break

        best_pair = max(candidate_pairs, key=lambda p: (vocab[p[0]], vocab[p[1]]))
        for pair in candidate_pairs:
            if pair != best_pair:
                heapq.heappush(pair_heap, (-best_count, pair))

        best_pair_bytes = (vocab[best_pair[0]], vocab[best_pair[1]])
        new_token_id = len(vocab)
        vocab[new_token_id] = best_pair_bytes[0] + best_pair_bytes[1]
        merges.append(best_pair_bytes)
        merge_pbar.update(1)

        affected_word_ids = pair_to_words.pop(best_pair, set())
        # Aggregate pair-count deltas across all affected words, then apply once.
        pair_count_deltas: dict[tuple[int, int], int] = defaultdict(int)
        for word_id in affected_word_ids:
            old_word = word_by_id[word_id]
            word_freq = word_freq_by_id[word_id]

            old_local_pair_counts = Counter(zip(old_word, old_word[1:]))
            for pair, local_count in old_local_pair_counts.items():
                pair_count_deltas[pair] -= local_count * word_freq

                word_set = pair_to_words.get(pair)
                if word_set is not None:
                    word_set.discard(word_id)
                    if not word_set:
                        pair_to_words.pop(pair, None)

            new_word = merge_pair_in_word(old_word, best_pair, new_token_id)
            word_by_id[word_id] = new_word

            new_local_pair_counts = Counter(zip(new_word, new_word[1:]))
            for pair, local_count in new_local_pair_counts.items():
                pair_count_deltas[pair] += local_count * word_freq
                pair_to_words[pair].add(word_id)

        for pair, delta in pair_count_deltas.items():
            if delta == 0:
                continue
            new_count = pair_counts.get(pair, 0) + delta
            if new_count > 0:
                pair_counts[pair] = new_count
                heapq.heappush(pair_heap, (-new_count, pair))
            else:
                pair_counts.pop(pair, None)
                if pair in pair_to_words and not pair_to_words[pair]:
                    pair_to_words.pop(pair, None)

        merges_since_heap_rebuild += 1
        # Periodically rebuild heap to drop stale entries from lazy updates.
        if merges_since_heap_rebuild >= 50 and len(pair_heap) > (len(pair_counts) * 3 + 2048):
            pair_heap = [(-count, pair) for pair, count in pair_counts.items() if count > 0]
            heapq.heapify(pair_heap)
            merges_since_heap_rebuild = 0

    merge_pbar.close()
    return vocab, merges
