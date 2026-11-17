"""Spec for your paired bootstrap."""
import numpy as np
import pytest

from shoprank.eval.bootstrap import paired_bootstrap_ci


def test_identical_systems_give_zero():
    a = [0.2, 0.5, 0.9, 0.4]
    assert paired_bootstrap_ci(a, a) == (0.0, 0.0, 0.0)


def test_constant_improvement_is_exact():
    a = [0.1, 0.2, 0.3]
    b = [x + 0.1 for x in a]
    mean, lo, hi = paired_bootstrap_ci(a, b)
    assert mean == pytest.approx(0.1)
    assert lo == pytest.approx(0.1)
    assert hi == pytest.approx(0.1)


def test_interval_contains_mean_and_is_deterministic():
    rng = np.random.default_rng(1)
    a = rng.uniform(size=200)
    b = a + rng.normal(0.02, 0.1, size=200)
    first = paired_bootstrap_ci(a, b, seed=7)
    assert first == paired_bootstrap_ci(a, b, seed=7)
    mean, lo, hi = first
    assert lo < mean < hi
    assert mean == pytest.approx(float(np.mean(b - a)))


def test_more_queries_give_narrower_interval():
    rng = np.random.default_rng(2)
    a = rng.uniform(size=2000)
    b = a + rng.normal(0.02, 0.1, size=2000)
    _, lo_small, hi_small = paired_bootstrap_ci(a[:100], b[:100])
    _, lo_big, hi_big = paired_bootstrap_ci(a, b)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_bad_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [])
