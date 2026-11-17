"""Turn text into integer ids a neural network can look up. Week 3, Part D.

Each word becomes: one id for the whole word + one id for each 3-letter piece ("char trigram").
    "anna" -> word:anna, #an, ann, nna, na#
    "ana"  -> word:ana,  #an, ana, na#        <- shares #an and na# with "anna": spelling-robust

Ids come from hashing the piece into one of N_BUCKETS slots ("hashing trick"): no vocabulary file,
any new word still gets ids. Two pieces can collide in one slot; with 262,144 slots that is rare.
"""
from __future__ import annotations

import zlib

import torch

from shoprank.retrieval.bm25 import tokenize

N_BUCKETS = 1 << 18  # 262,144
EMPTY_ID = 0         # reserved for "no text", so every text has at least one id


def _bucket(piece: str) -> int:
    # crc32 is stable across runs and machines (Python's hash() is randomized per process).
    return zlib.crc32(piece.encode("utf-8")) % (N_BUCKETS - 1) + 1


def text_ids(text: str | None) -> list[int]:
    ids = []
    for word in tokenize(text):
        ids.append(_bucket("w:" + word))
        padded = f"#{word}#"
        ids.extend(_bucket("c:" + padded[i:i + 3]) for i in range(len(padded) - 2))
    return ids or [EMPTY_ID]


def pack(id_lists: list[list[int]], device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """EmbeddingBag input format: one flat list of all ids + where each text starts (offsets)."""
    lengths = torch.tensor([len(ids) for ids in id_lists])
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]])
    flat = torch.tensor([i for ids in id_lists for i in ids], dtype=torch.long)
    return flat.to(device), offsets.to(device)
