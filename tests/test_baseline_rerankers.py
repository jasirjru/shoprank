"""Logic tests for the zero-shot reranker baselines (no models, no GPU)."""
import numpy as np
import pandas as pd
import pytest

from shoprank.eval.baseline_rerankers import (
    QWEN_SUFFIX,
    compare,
    length_order,
    qwen_document,
    qwen_query,
    score_sorted,
)


def test_qwen_document_truncates_text_but_keeps_suffix():
    doc = qwen_document("x" * 5000, max_chars=100)
    assert doc == "<Document>: " + "x" * 100 + QWEN_SUFFIX


def test_qwen_query_carries_instruction_and_query():
    q = qwen_query("red shoes", "find shoes")
    assert "<Instruct>: find shoes\n<Query>: red shoes\n" in q


def test_length_order_sorts_short_first():
    pairs = [("aaaa", "bbbb"), ("a", "b"), ("aa", "bb")]
    assert length_order(pairs).tolist() == [1, 2, 0]


def test_score_sorted_restores_original_order():
    pairs = [("long query", "long text here"), ("a", "b"), ("mid", "mid2")]
    scores = score_sorted(pairs, lambda part, bs: np.array([len(q) + len(t) for q, t in part]), 2, chunk=2)
    assert scores.tolist() == [len("long query") + len("long text here"), 2, len("mid") + len("mid2")]


def test_score_sorted_halves_batch_on_oom():
    seen = []

    def predict(part, bs):
        seen.append(bs)
        if bs > 4:
            raise MemoryError
        return np.zeros(len(part))

    score_sorted([("a", "b")] * 5, predict, 16, oom=MemoryError)
    assert seen == [16, 8, 4]


def test_score_sorted_raises_at_batch_one():
    def predict(part, bs):
        raise MemoryError

    with pytest.raises(MemoryError):
        score_sorted([("a", "b")], predict, 1, oom=MemoryError)


def test_compare_aligns_on_query_id():
    ref = pd.DataFrame({"query_id": [2, 1, 3], "ndcg_full": [0.5, 0.5, 0.5]})
    cand = pd.DataFrame({"query_id": [1, 2, 3], "ndcg_full": [0.6, 0.6, 0.6]})
    res = compare(cand, ref)
    assert res["delta"] == pytest.approx(0.1)
    assert res["low"] <= res["delta"] <= res["high"]


def test_compare_rejects_mismatched_queries():
    ref = pd.DataFrame({"query_id": [1, 2], "ndcg_full": [0.5, 0.5]})
    cand = pd.DataFrame({"query_id": [1, 3], "ndcg_full": [0.6, 0.6]})
    with pytest.raises(ValueError):
        compare(cand, ref)
