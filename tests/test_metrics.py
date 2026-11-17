"""Spec for your metrics. Expected numbers were computed by hand from the formulas in the docstrings."""
import math

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from shoprank.eval.metrics import dcg_at_k, ndcg_at_k, recall_at_k, reciprocal_rank

RANKED = [1.0, 0.1, 0.0, 0.01]  # E, S, I, C in ranked order


def test_dcg_matches_hand_calculation():
    # 1/log2(2) + 0.1/log2(3) + 0/log2(4) + 0.01/log2(5)
    assert dcg_at_k(RANKED, 4) == pytest.approx(1.0673997409378797)


def test_dcg_short_list_and_bad_k():
    assert dcg_at_k(RANKED, 10) == pytest.approx(dcg_at_k(RANKED, 4))
    assert dcg_at_k([], 5) == 0.0
    with pytest.raises(ValueError):
        dcg_at_k(RANKED, 0)


def test_ndcg_matches_hand_calculation():
    assert ndcg_at_k(RANKED, 4) == pytest.approx(0.9993509606043105)
    assert ndcg_at_k(RANKED, 2) == pytest.approx(1.0)  # top 2 already ideal


def test_ndcg_edge_cases():
    assert ndcg_at_k([0.0, 0.0], 10) == 0.0
    assert ndcg_at_k([0.0, 1.0], 1) == 0.0
    assert ndcg_at_k([1.0], 1) == 1.0


@pytest.mark.parametrize("seed", range(5))
def test_ndcg_agrees_with_sklearn(seed):
    rng = np.random.default_rng(seed)
    gains = rng.choice([1.0, 0.1, 0.01, 0.0], size=12)
    scores = rng.permutation(12).astype(float)  # distinct scores: no ties
    ranked = gains[np.argsort(-scores)]
    expected = ndcg_score([gains], [scores], k=10)
    assert ndcg_at_k(list(ranked), 10) == pytest.approx(expected)


def test_recall():
    flags = [True, False, True, False]
    assert recall_at_k(flags, total_relevant=3, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(flags, total_relevant=3, k=10) == pytest.approx(2 / 3)
    assert recall_at_k(flags, total_relevant=0, k=5) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(flags, 3, 0)


def test_reciprocal_rank():
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([True]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0
    assert math.isclose(reciprocal_rank([]), 0.0)
