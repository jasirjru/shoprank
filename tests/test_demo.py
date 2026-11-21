"""Section 17 demo: input parsing and ranking logic, without gradio or a real model."""
from __future__ import annotations

import numpy as np
import pytest

from shoprank.serve.demo import parse_products, rank_rows


class FakeScorer:
    fields = ("title", "brand")
    name = "fake"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def score(self, queries, texts):
        self.texts = list(texts)
        return np.array([float("cable" in t) + 0.1 * len(t.split()) for t in texts])


def test_parse_products_splits_fields_and_numbers_lines():
    products = parse_products("USB cable | Anker | Black\n\n  HDMI cable  \nCharger | | White | 20W fast")
    assert [p["product_id"] for p in products] == ["p1", "p2", "p3"]
    assert products[0] == {"product_id": "p1", "title": "USB cable", "brand": "Anker", "color": "Black",
                           "bullets": None}
    assert products[1]["title"] == "HDMI cable" and products[1]["brand"] is None
    assert products[2]["brand"] is None and products[2]["color"] == "White" and products[2]["bullets"] == "20W fast"


def test_parse_products_keeps_extra_separators_in_bullets():
    products = parse_products("a | b | c | d | e")
    assert products[0]["bullets"] == "d | e"


def test_rank_rows_orders_best_first_with_training_text():
    scorer = FakeScorer()
    rows, status = rank_rows(scorer, " usb cable ", "Charger | Apple\nUSB cable | Anker\nHDMI cable")
    assert [r[2] for r in rows] == ["USB cable", "HDMI cable", "Charger"]
    assert [r[0] for r in rows] == [1, 2, 3]
    assert scorer.texts == ["Charger Apple", "USB cable Anker", "HDMI cable"]  # same join as training
    assert status.startswith("3 products scored")


@pytest.mark.parametrize(("query", "products", "message"), [
    ("   ", "USB cable", "query"),
    ("usb", "", "products"),
    ("usb", " | Anker", "title"),
    ("x" * 201, "USB cable", "query"),
])
def test_rank_rows_rejects_bad_input_with_the_api_rules(query, products, message):
    with pytest.raises(ValueError, match=message):
        rank_rows(FakeScorer(), query, products)
