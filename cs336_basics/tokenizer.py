from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator

import regex as re

GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Byte-to-unicode mapping used by GPT-2 serialization.
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(x) for x in cs), strict=False))


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab)

        # Reverse map: keep first ID for each byte sequence.
        self.bytes_to_id: dict[bytes, int] = {}
        for token_id in sorted(self.vocab):
            token_bytes = self.vocab[token_id]
            self.bytes_to_id.setdefault(token_bytes, token_id)

        self.special_tokens = list(special_tokens or [])
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                next_id = max(self.vocab.keys(), default=-1) + 1
                self.vocab[next_id] = token_bytes
                self.bytes_to_id[token_bytes] = next_id

        self.special_token_to_id = {token: self.bytes_to_id[token.encode("utf-8")] for token in self.special_tokens}

        # Only keep merges that actually correspond to an existing merged token.
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {}
        for rank, pair in enumerate(merges):
            if pair[0] + pair[1] in self.bytes_to_id:
                self.merge_ranks[pair] = rank

        self.pretoken_pattern = re.compile(GPT2_PRETOKEN_PATTERN)
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            self.special_split_pattern = re.compile("(" + "|".join(re.escape(token) for token in sorted_special) + ")")
        else:
            self.special_split_pattern = None

        self._pretoken_cache: dict[bytes, tuple[int, ...]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        byte_decoder = {u: b for b, u in gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath, encoding="utf-8") as f:
            serialized_vocab = json.load(f)

        vocab: dict[int, bytes] = {}
        for token_str, token_id in serialized_vocab.items():
            try:
                token_bytes = bytes(byte_decoder[ch] for ch in token_str)
            except KeyError:
                token_bytes = token_str.encode("utf-8")
            vocab[int(token_id)] = token_bytes

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned = line.rstrip()
                if not cleaned:
                    continue
                parts = cleaned.split(" ")
                if len(parts) != 2:
                    continue
                left, right = parts
                try:
                    left_bytes = bytes(byte_decoder[ch] for ch in left)
                    right_bytes = bytes(byte_decoder[ch] for ch in right)
                except KeyError:
                    left_bytes = left.encode("utf-8")
                    right_bytes = right.encode("utf-8")
                merges.append((left_bytes, right_bytes))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _encode_pretoken(self, pretoken: bytes) -> tuple[int, ...]:
        cached = self._pretoken_cache.get(pretoken)
        if cached is not None:
            return cached

        symbols: list[bytes] = [bytes([b]) for b in pretoken]
        if not symbols:
            return ()

        while len(symbols) >= 2:
            best_rank: int | None = None
            best_pair: tuple[bytes, bytes] | None = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            first, second = best_pair
            merged_symbols: list[bytes] = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and symbols[i] == first and symbols[i + 1] == second:
                    merged_symbols.append(first + second)
                    i += 2
                else:
                    merged_symbols.append(symbols[i])
                    i += 1
            symbols = merged_symbols

        encoded = tuple(self.bytes_to_id[symbol] for symbol in symbols)
        self._pretoken_cache[pretoken] = encoded
        return encoded

    def _iter_encode_text(self, text: str) -> Iterator[int]:
        if not text:
            return

        if self.special_split_pattern is None:
            parts = [text]
        else:
            parts = self.special_split_pattern.split(text)

        for part in parts:
            if not part:
                continue

            special_id = self.special_token_to_id.get(part)
            if special_id is not None:
                yield special_id
                continue

            for match in self.pretoken_pattern.finditer(part):
                pretoken = match.group(0).encode("utf-8")
                yield from self._encode_pretoken(pretoken)

    def encode(self, text: str) -> list[int]:
        return list(self._iter_encode_text(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self._iter_encode_text(text)

    def decode(self, ids: list[int]) -> str:
        # Replacement bytes for unknown token IDs.
        replacement = b"\xef\xbf\xbd"
        output = bytearray()
        for token_id in ids:
            token_bytes = self.vocab.get(token_id, replacement)
            output.extend(token_bytes)
        return bytes(output).decode("utf-8", errors="replace")
