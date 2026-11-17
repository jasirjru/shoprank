"""Ranking metrics: how good is one ranked list of products?

Spec = tests/test_metrics.py. Notes = docs/learn/week01-notes.md §2.

Conventions used everywhere in this project:
- A ranking is a list ordered best-first. Position 1 is the top result.
- Gains are the ESCI gains (E=1.0, S=0.1, C=0.01, I=0.0), already in ranked order.
- Discount for position i (1-based) is 1 / log2(i + 1). Linear gains (no 2**g - 1).
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")


def dcg_at_k(ranked_gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain of the first k results."""
    _check_k(k)
    # Position 1 divides by log2(2) = 1 (full credit); each lower position counts a little less.
    return sum(gain / math.log2(position + 1)
               for position, gain in enumerate(ranked_gains[:k], start=1))


def ndcg_at_k(ranked_gains: Sequence[float], k: int) -> float:
    """DCG@k divided by the best possible DCG@k (same gains sorted high-to-low).

    In Task 1 every candidate is judged, so the ideal ranking comes from this same list.
    Returns 0.0 when no relevant item exists (ideal DCG is 0).
    """
    ideal = dcg_at_k(sorted(ranked_gains, reverse=True), k)
    return dcg_at_k(ranked_gains, k) / ideal if ideal > 0 else 0.0


def recall_at_k(ranked_relevant: Sequence[bool], total_relevant: int, k: int) -> float:
    """Fraction of all relevant items that appear in the first k results."""
    _check_k(k)
    if total_relevant == 0:
        return 0.0
    return sum(ranked_relevant[:k]) / total_relevant  # True counts as 1, False as 0


def reciprocal_rank(ranked_relevant: Sequence[bool]) -> float:
    """1 / position of the first relevant result; 0.0 if none. (MRR = mean over queries.)"""
    for position, is_relevant in enumerate(ranked_relevant, start=1):
        if is_relevant:
            return 1.0 / position
    return 0.0
