"""Paired bootstrap confidence interval: is system B really better than system A?

Spec = tests/test_bootstrap.py. Notes = docs/learn/week01-notes.md §3.

Paired = both systems are scored on the SAME queries, so we resample per-query DIFFERENCES.
That cancels out "this query is just hard for everyone" and gives a much tighter interval.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def paired_bootstrap_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean_delta, low, high) for delta = candidate - baseline, per query."""
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(candidate, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"inputs differ in length: {a.size} vs {b.size}")
    if a.size == 0:
        raise ValueError("inputs are empty")

    deltas = b - a
    rng = np.random.default_rng(seed)  # fixed seed -> same interval every run
    # Each row is one resample: n query indices drawn WITH replacement. Shape (n_resamples, n).
    picks = rng.integers(0, deltas.size, size=(n_resamples, deltas.size))
    resample_means = deltas[picks].mean(axis=1)

    tail = (1.0 - confidence) / 2 * 100  # 2.5 for a 95% interval
    low, high = np.percentile(resample_means, [tail, 100 - tail])
    return float(deltas.mean()), float(low), float(high)
