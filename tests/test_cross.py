"""Zero-shot cross-encoder plumbing, tested with a fake model (no download, no GPU)."""
import numpy as np
import pandas as pd
import pytest

from shoprank.cross.zero_shot import product_texts, score_pairs


@pytest.fixture
def pairs():
    return pd.DataFrame({"query_id": [1, 1, 2], "query": ["red shoe", "red shoe", "tv"],
                         "product_id": ["A", "B", "A"]})


@pytest.fixture
def products():
    return pd.DataFrame({"product_id": ["A", "B"], "title": ["Red Shoe", None], "brand": ["Nike", "Sony"]})


def test_product_texts_keep_row_order_and_fill_missing(pairs, products):
    assert product_texts(pairs, products, ["title"]) == ["Red Shoe", "", "Red Shoe"]
    assert product_texts(pairs, products, ["title", "brand"]) == ["Red Shoe Nike", "Sony", "Red Shoe Nike"]


def test_score_pairs_sends_query_with_its_own_product(pairs, products):
    seen = []

    def fake_predict(batch):
        seen.extend(batch)
        return np.arange(len(batch), dtype=float)

    scores = score_pairs(pairs, product_texts(pairs, products, ["title"]), fake_predict)
    assert seen == [("red shoe", "Red Shoe"), ("red shoe", ""), ("tv", "Red Shoe")]
    assert scores.tolist() == [0.0, 1.0, 2.0]


def test_score_pairs_rejects_wrong_length(pairs, products):
    with pytest.raises(ValueError):
        score_pairs(pairs, product_texts(pairs, products, ["title"]), lambda b: np.zeros(2))
