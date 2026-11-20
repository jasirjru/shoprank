"""Synthetic spelling mistakes for typo-robust training (augmentation) and typo evaluation (section 18).

Four common keyboard mistakes, applied to ONE word of 4+ letters, never to its first letter (people rarely
mistype the first letter, and it carries the most information):
  - transpose: two neighbouring letters swapped     cable -> cbale
  - delete:    one letter missing                    cable -> cale
  - double:    one letter typed twice                cable -> cabble
  - neighbour: a letter replaced by a QWERTY neighbour  cable -> cabke

Training (`augment_queries`) draws the word and the mistake at random. Evaluation uses two FIXED perturbations
of the longest word (`eval_typo`): "transpose" (the 21 Nov audit's test) and "neighbour". Both families also
occur in training, so a gain here shows robustness to these synthetic typos, not to every real misspelling.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

KINDS = ("transpose", "delete", "double", "neighbour")
MIN_WORD = 4
QWERTY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def _neighbours() -> dict[str, str]:
    """Letters next to each letter on the same QWERTY row (left and right)."""
    out = {}
    for row in QWERTY_ROWS:
        for i, ch in enumerate(row):
            out[ch] = row[max(0, i - 1):i] + row[i + 1:i + 2]
    return out


NEIGHBOURS = _neighbours()


def misspell(word: str, kind: str, pos: int, pick: int = 0) -> str:
    """Apply one mistake of `kind` at character `pos` (1 <= pos < len(word)); `pick` chooses the neighbour key."""
    if kind not in KINDS:
        raise ValueError(f"unknown typo kind {kind!r}")
    if not 1 <= pos < len(word):
        raise ValueError(f"position {pos} outside 1..{len(word) - 1} for {word!r}")
    if kind == "transpose":
        pos = min(pos, len(word) - 2)
        return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
    if kind == "delete":
        return word[:pos] + word[pos + 1:]
    if kind == "double":
        return word[:pos + 1] + word[pos] + word[pos + 1:]
    options = NEIGHBOURS.get(word[pos].lower(), "")
    if not options:  # digit or symbol: no keyboard neighbour on the letter rows, fall back to a swap
        return misspell(word, "transpose", pos)
    return word[:pos] + options[pick % len(options)] + word[pos + 1:]


def _eligible(words: Sequence[str]) -> list[int]:
    return [i for i, w in enumerate(words) if len(w) >= MIN_WORD and any(c.isalpha() for c in w)]


MAX_TRIES = 20  # swapping two equal letters ("oo") changes nothing: draw again until the word changes


def random_typo(query: str, rng: np.random.Generator) -> str:
    """One random mistake in one random eligible word; the query is unchanged if no word has 4+ characters."""
    words = query.split()
    idx = _eligible(words)
    if not idx:
        return query
    i = idx[int(rng.integers(len(idx)))]
    w = words[i]
    for _ in range(MAX_TRIES):
        new = misspell(w, KINDS[int(rng.integers(len(KINDS)))], int(rng.integers(1, len(w))), int(rng.integers(2)))
        if new != w:
            words[i] = new
            break
    return " ".join(words)


def _stable_seed(key: object, seed: int) -> int:
    """A seed that depends only on (key, seed), not on row order or Python's hash randomisation."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "little")


def augment_queries(queries: Sequence[str], query_ids: Sequence[object], rate: float, seed: int) -> list[str]:
    """Give a share `rate` of QUERIES (not pairs) one random typo; every pair of a chosen query gets the same
    misspelled text, as one user's search would. Deterministic for a given (query_id, seed)."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    if len(queries) != len(query_ids):
        raise ValueError("one query id per query")
    cache: dict[object, str] = {}
    out = []
    for q, qid in zip(queries, query_ids, strict=True):
        if qid not in cache:
            rng = np.random.default_rng(_stable_seed(qid, seed))
            cache[qid] = random_typo(q, rng) if rng.random() < rate else q
        out.append(cache[qid])
    return out


def eval_typo(query: str, kind: str, key: object, seed: int = 0) -> str:
    """Fixed evaluation typo: mistake `kind` in the LONGEST eligible word, at a position drawn from (key, seed)."""
    words = query.split()
    idx = _eligible(words)
    if not idx:
        return query
    i = max(idx, key=lambda k: len(words[k]))
    rng = np.random.default_rng(_stable_seed(key, seed))
    w = words[i]
    for _ in range(MAX_TRIES):
        new = misspell(w, kind, int(rng.integers(1, len(w) - 1)), int(rng.integers(2)))
        if new != w:
            words[i] = new
            break
    return " ".join(words)
