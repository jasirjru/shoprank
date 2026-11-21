"""Label audit statistics: Wilson interval, agreement, likely-error definition."""
from __future__ import annotations

import pandas as pd
import pytest

from shoprank.eval.label_audit import summarise, wilson


def test_wilson_matches_known_values_and_stays_in_range():
    lo, hi = wilson(10, 100)
    assert lo == pytest.approx(0.0552, abs=1e-3) and hi == pytest.approx(0.1744, abs=1e-3)
    assert wilson(0, 20)[0] == 0.0 and wilson(20, 20)[1] == 1.0


def test_likely_error_needs_both_judges_to_agree_against_the_label():
    idx = pd.Index([1, 2, 3, 4], name="item")
    gold = pd.Series(["E", "E", "S", "I"], index=idx)
    a = pd.Series(["E", "S", "S", "E"], index=idx)
    b = pd.Series(["E", "S", "E", "E"], index=idx)
    out = summarise(gold, a, b)
    assert out["likely_label_error"]["count"] == 2  # items 2 and 4: judges agree, label differs
    assert out["judges_agree"]["count"] == 3
    assert out["judge_a"]["agree_4class"]["count"] == 2
    assert out["likely_error_by_label"] == {"E": 1, "S": 0, "C": 0, "I": 1}


def test_mismatched_items_are_rejected():
    gold = pd.Series(["E"], index=pd.Index([1], name="item"))
    with pytest.raises(ValueError):
        summarise(gold, pd.Series(["E"], index=pd.Index([2])), pd.Series(["E"], index=pd.Index([1])))
