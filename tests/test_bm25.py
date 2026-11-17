"""Spec for your BM25. Expected numbers computed by hand from the formula in bm25.py (k1=1.5, b=0.75)."""
import pytest

from shoprank.retrieval.bm25 import BM25, tokenize

DOCS = [["red", "shoe"], ["red", "red", "dress", "long"], ["blue", "shoe", "shoe"]]


@pytest.fixture
def bm25():
    return BM25(k1=1.5, b=0.75).fit(DOCS)


def test_tokenize_is_given():
    assert tokenize("iPhone 12-Pro, RED!") == ["iphone", "12", "pro", "red"]
    assert tokenize(None) == []


def test_idf(bm25):
    assert bm25.idf("red") == pytest.approx(0.47000362924573563)    # in 2 of 3 docs
    assert bm25.idf("dress") == pytest.approx(0.9808292530117263)   # in 1 of 3 docs: rarer -> higher
    assert bm25.idf("unknown") == pytest.approx(bm25.idf("unknown"))  # defined, positive
    assert bm25.idf("unknown") > bm25.idf("dress")


def test_scores_match_hand_calculation(bm25):
    q = ["red", "shoe"]
    assert bm25.score(q, 0) == pytest.approx(1.1058908923429074)
    assert bm25.score(q, 1) == pytest.approx(0.6064562958009492)
    assert bm25.score(q, 2) == pytest.approx(0.6714337560653366)


def test_duplicate_query_terms_count_once(bm25):
    assert bm25.score(["red", "red", "shoe"], 0) == pytest.approx(bm25.score(["red", "shoe"], 0))


def test_unknown_terms_score_zero(bm25):
    assert bm25.score(["laptop"], 0) == 0.0


def test_top_k_order_and_ties(bm25):
    assert [i for i, _ in bm25.top_k(["red", "shoe"], k=3)] == [0, 2, 1]
    assert len(bm25.top_k(["red", "shoe"], k=2)) == 2
    assert [i for i, _ in BM25().fit([["a"], ["a"]]).top_k(["a"], k=2)] == [0, 1]  # tie -> lower index first


def test_term_frequency_saturates():
    m = BM25(k1=1.5, b=0.0).fit([["x"], ["x", "x"], ["x"] * 10, ["y"]])
    s1, s2, s10 = (m.score(["x"], i) for i in range(3))
    assert s1 < s2 < s10
    assert (s10 - s2) < 8 * (s2 - s1)  # 8 more copies add less than 8x the first extra copy


def test_b_zero_ignores_document_length():
    m = BM25(k1=1.5, b=0.0).fit([["x"], ["x", "filler", "filler", "filler"]])
    assert m.score(["x"], 0) == pytest.approx(m.score(["x"], 1))


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        BM25().fit([])
