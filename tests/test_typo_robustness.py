"""Typo-robustness comparison logic, without models: query perturbation, seed averaging, gap recovery."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shoprank.eval.typo_robustness import (
    CONDITIONS,
    arm_means,
    compare,
    condition_queries,
)


def test_typo_conditions_change_each_query_once_and_consistently():
    pairs = pd.DataFrame({"query_id": [1, 1, 2], "query": ["wireless mouse", "wireless mouse", "tv"]})
    assert condition_queries(pairs, "clean") == ["wireless mouse", "wireless mouse", "tv"]
    typo = condition_queries(pairs, "typo_transpose")
    assert typo[0] == typo[1] != "wireless mouse"  # same text for every pair of a query
    assert typo[2] == "tv"  # no word of 4+ letters: unchanged


def test_compare_reports_gap_recovery_and_seed_spread():
    q = 400
    rng = np.random.default_rng(0)
    clean = rng.uniform(0.7, 1.0, q)

    def model(clean_shift, typo_drop):
        return {"clean": clean + clean_shift, "typo_transpose": clean - typo_drop, "typo_neighbour": clean - typo_drop}

    table = {"b1": model(0.0, 0.03), "b2": model(0.002, 0.03), "t1": model(-0.001, 0.01), "t2": model(0.001, 0.01)}
    out = compare(table, ["b1", "b2"], ["t1", "t2"])
    assert out["clean"]["base_seed_spread"] == pytest.approx(0.002)
    assert out["typo_transpose"]["base_gap"] == pytest.approx(0.031)
    assert out["typo_transpose"]["recovered_share"] == pytest.approx(0.02 / 0.031, abs=1e-3)
    delta, lo, hi = out["typo_transpose"]["treat_minus_base"]
    assert delta == pytest.approx(0.02) and lo <= delta <= hi
    assert set(arm_means(table, ["b1"])) == set(CONDITIONS)
