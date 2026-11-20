"""Typo generator: each mistake kind, first letter kept, per-query consistency, determinism, the augmentation rate."""
from __future__ import annotations

import numpy as np
import pytest

from shoprank.cross.typos import (
    KINDS,
    augment_queries,
    eval_typo,
    misspell,
    random_typo,
)


@pytest.mark.parametrize(("kind", "expected"), [
    ("transpose", "cbale"), ("delete", "cble"), ("double", "caable"), ("neighbour", "csble"),
])
def test_each_kind_at_position_one(kind, expected):
    assert misspell("cable", kind, 1) == expected


def test_neighbour_keys_come_from_the_same_qwerty_row():
    assert misspell("cable", "neighbour", 2, pick=0) == "cavle"  # b -> v
    assert misspell("cable", "neighbour", 2, pick=1) == "canle"  # b -> n
    assert misspell("cable", "neighbour", 1, pick=1) == "csble"  # a sits at the row end: only s
    assert misspell("tv65", "neighbour", 2) == "tv56"  # a digit has no letter neighbour: swap instead


def test_equal_letters_still_produce_a_visible_typo():
    assert eval_typo("coffee", "transpose", key=3) != "coffee"


def test_first_letter_is_never_touched_and_bad_input_is_rejected():
    rng = np.random.default_rng(0)
    for _ in range(200):
        out = random_typo("wireless headphones", rng)
        assert out[0] == "w" and out.split()[1][0] == "h"
    with pytest.raises(ValueError):
        misspell("cable", "transpose", 0)
    with pytest.raises(ValueError):
        misspell("cable", "smudge", 1)


def test_short_words_and_numbers_only_are_left_alone():
    rng = np.random.default_rng(0)
    assert random_typo("usb c 6 ft", rng) == "usb c 6 ft"
    assert eval_typo("tv 4k", "transpose", key=1) == "tv 4k"


def test_random_typo_changes_exactly_one_word():
    rng = np.random.default_rng(1)
    q = "bluetooth speaker waterproof"
    for _ in range(50):
        out = random_typo(q, rng).split()
        assert sum(a != b for a, b in zip(out, q.split(), strict=False)) == 1


def test_augment_rate_counts_queries_and_keeps_pairs_of_a_query_identical():
    queries = [f"product number{i}" for i in range(2000) for _ in range(3)]
    ids = [i for i in range(2000) for _ in range(3)]
    out = augment_queries(queries, ids, rate=0.3, seed=7)
    changed = {qid for qid, a, b in zip(ids, queries, out, strict=True) if a != b}
    assert 0.27 < len(changed) / 2000 < 0.33
    for i in range(0, len(out), 3):
        assert out[i] == out[i + 1] == out[i + 2]
    assert augment_queries(queries, ids, 0.3, 7) == out  # deterministic
    assert augment_queries(queries, ids, 0.3, 8) != out  # seed matters
    assert augment_queries(queries, ids, 0.0, 7) == queries


def test_eval_typo_hits_the_longest_word_deterministically():
    out = eval_typo("red ceramic mug", "transpose", key=42)
    assert out.startswith("red ") and out.endswith(" mug") and out != "red ceramic mug"
    assert out == eval_typo("red ceramic mug", "transpose", key=42)
    assert set(KINDS) == {"transpose", "delete", "double", "neighbour"}
