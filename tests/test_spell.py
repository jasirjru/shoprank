"""Catalogue spell correction: one-edit candidates, frequency rules, words that must never change."""
from __future__ import annotations

import pandas as pd
import pytest

from shoprank.serve.spell import SpellCorrector, build_vocab, one_edit_apart

VOCAB = {"cable": 500, "table": 300, "charger": 400, "anker": 50, "wireless": 200, "mouse": 150,
         "cbale": 3, "sony": 80, "headphones": 120}


@pytest.mark.parametrize(("a", "b", "expected"), [
    ("cable", "cbale", True),   # swap
    ("cable", "cale", True),    # delete
    ("cable", "caable", True),  # insert
    ("cable", "cavle", True),   # replace
    ("cable", "table", True),   # replace at the first letter
    ("cable", "cable", False),
    ("cable", "bcale", False),  # two edits (two swaps)
    ("cable", "ca", False),
])
def test_one_edit_apart(a, b, expected):
    assert one_edit_apart(a, b) is expected


def test_fixes_each_typo_kind():
    sc = SpellCorrector(VOCAB)
    assert sc.correct("usb c cbale 6ft") == "usb c cable 6ft"  # 'cbale' is in the catalogue 3 times: < 5
    assert sc.correct("wireles mouse") == "wireless mouse"
    assert sc.correct("chargger") == "charger"
    assert sc.correct("headphonez") == "headphones"


def test_known_words_numbers_and_short_words_are_kept():
    sc = SpellCorrector(VOCAB)
    assert sc.correct("anker cable") == "anker cable"   # known words
    assert sc.correct("usb-c 6ft tv") == "usb-c 6ft tv"   # symbols, digits, short words
    assert sc.correct("zzzzzz") == "zzzzzz"               # no candidate: unchanged
    assert sc.correct("niño") == "niño"                    # non-ASCII words are left alone


def test_most_frequent_candidate_wins_and_ratio_rule_blocks_weak_replacements():
    sc = SpellCorrector(VOCAB)
    assert sc.correct("xable") == "cable"  # cable (500) beats table (300); both one edit away
    rare = SpellCorrector({"sony": 40, "sonya": 2})  # 'sonx': candidate 'sony' 40 >= 10 x 1
    assert rare.correct("sonx") == "sony"
    weak = SpellCorrector({"sony": 30, "sonx": 4})  # typed word seen 4 times: needs a candidate >= 40
    assert weak.correct("sonx") == "sonx"


def test_build_vocab_counts_title_brand_colour(tmp_path):
    path = tmp_path / "p.parquet"
    pd.DataFrame({"title": ["USB Cable 6ft", "HDMI cable", "cable"], "brand": ["Anker", None, "Anker"],
                  "color": ["Black", "black", None], "bullets": ["ignored words", "x", "y"]}).to_parquet(path)
    vocab = build_vocab(path, tmp_path / "v.json")
    assert vocab["cable"] == 3 and vocab["anker"] == 2 and vocab["black"] == 2
    assert "ignored" not in vocab and "usb" not in vocab  # bullets unused; words seen once are dropped


def test_spell_eval_query_sets_and_summary():
    import numpy as np

    from shoprank.eval.spell_eval import SPELL, query_sets, summarise

    pairs = pd.DataFrame({"query_id": [1, 1, 2], "query": ["wireless mouse", "wireless mouse", "sony headphones"]})
    sets, stats = query_sets(pairs, SpellCorrector(VOCAB))
    assert sets["clean" + SPELL] == sets["clean"]  # known words: nothing to fix
    assert stats["clean"]["changed_by_spell"] == 0
    assert stats["typo_transpose"]["typo_queries"] == 2
    assert 0 <= stats["typo_transpose"]["restored_share"] <= 1
    q = 300
    base = np.linspace(0.6, 1.0, q)
    table = {"m": {"clean": base, "clean" + SPELL: base, "typo_transpose": base - 0.04,
                   "typo_transpose" + SPELL: base - 0.01, "typo_neighbour": base - 0.04,
                   "typo_neighbour" + SPELL: base - 0.04}}
    out = summarise(table, ["m"])
    assert out["typo_transpose"]["recovered_share"] == pytest.approx(0.75, abs=1e-3)
    assert out["typo_neighbour"]["recovered_share"] == pytest.approx(0.0, abs=1e-3)
    assert out["clean"]["spell_minus_typed"][0] == pytest.approx(0.0)


def test_query_with_nothing_to_fix_is_returned_unchanged_including_spacing():
    sc = SpellCorrector(VOCAB)
    assert sc.correct("wireless  mouse") == "wireless  mouse"
    assert sc.correct("wireless  mosue") == "wireless mouse"  # corrected: spacing normalised
