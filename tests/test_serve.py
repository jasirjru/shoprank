"""Section 15 API: validation at the boundary, ranking order, training/serving text parity, error handling."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from shoprank.cross.zero_shot import product_texts
from shoprank.serve.app import MAX_PRODUCTS, create_app
from shoprank.serve.ranker import length_sorted_batches, serving_text

FIELDS = ["title", "brand", "color", "bullets"]


class FakeScorer:
    """Scores = number of query words found in the product text (deterministic, no model)."""
    fields, name = FIELDS, "fake"

    def __init__(self, fail: bool = False):
        self.fail, self.calls = fail, []

    def score(self, queries, texts):
        if self.fail:
            raise RuntimeError("model crashed")
        self.calls.append(texts)
        return np.array([sum(w in t.lower() for w in q.lower().split()) for q, t in zip(queries, texts)], float)


def client(scorer=None) -> TestClient:
    return TestClient(create_app(lambda: scorer or FakeScorer()))


def body(query="red usb cable", n=3):
    return {"query": query, "products": [
        {"product_id": "a", "title": "Blue HDMI cable"},
        {"product_id": "b", "title": "USB cable", "color": "red", "brand": "Anker"},
        {"product_id": "c", "title": "USB charger", "bullets": "fast"},
    ][:n]}


def test_rank_returns_every_product_best_first():
    with client() as c:
        r = c.post("/rank", json=body())
    assert r.status_code == 200
    assert [x["product_id"] for x in r.json()["results"]] == ["b", "a", "c"]  # a, c tie at 1: id order


def test_health_reports_the_model():
    with client() as c:
        assert c.get("/health").json() == {"status": "ok", "model": "fake"}


@pytest.mark.parametrize("bad", [
    {"query": "   ", "products": body()["products"]},                                   # blank query
    {"query": "x" * 201, "products": body()["products"]},                               # query too long
    {"query": "cable", "products": []},                                                 # no products
    {"query": "cable", "products": [{"product_id": "a"}]},                              # missing title
    {"query": "cable", "products": [{"product_id": "a", "title": "t"}] * 2},            # duplicate ids
    {"query": "cable", "products": [{"product_id": "a", "title": "t", "price": 3}]},    # unknown field
    {"query": "cable", "products": [{"product_id": str(i), "title": "t"} for i in range(MAX_PRODUCTS + 1)]},
])
def test_invalid_requests_get_422_and_never_reach_the_model(bad):
    scorer = FakeScorer()
    with client(scorer) as c:
        assert c.post("/rank", json=bad).status_code == 422
    assert scorer.calls == []


def test_model_failure_is_a_500_without_internal_details():
    with client(FakeScorer(fail=True)) as c:
        r = c.post("/rank", json=body())
    assert r.status_code == 500 and r.json() == {"detail": "scoring failed"}


def test_serving_text_equals_training_text():
    products = pd.DataFrame({"product_id": ["a", "b"], "title": ["USB cable", "Lamp"], "brand": ["Anker", None],
                             "color": [None, "red"], "bullets": ["6 ft", None]})
    pairs = pd.DataFrame({"product_id": ["a", "b"]})
    train_side = product_texts(pairs, products, FIELDS)
    serve_side = [serving_text(row, FIELDS) for row in products.astype(object).where(products.notna(), None)
                  .to_dict("records")]
    assert serve_side == train_side


def test_length_sorted_batches_cover_every_index_once():
    batches = length_sorted_batches([5, 1, 9, 3, 7], batch_size=2)
    assert sorted(np.concatenate(batches).tolist()) == [0, 1, 2, 3, 4]
    assert batches[0].tolist() == [1, 3]  # shortest first


def test_percentiles_of_known_latencies():
    from shoprank.serve.loadtest import percentiles

    out = percentiles(list(range(1, 101)))  # 1..100 ms
    assert out["p50_ms"] == 50.5 and out["p99_ms"] == pytest.approx(99.0, abs=0.1)


def test_model_calls_never_overlap_beyond_the_limit():
    import threading
    import time

    class SlowScorer(FakeScorer):
        active = peak = 0
        lock = threading.Lock()

        def score(self, queries, texts):
            with self.lock:
                SlowScorer.active += 1
                SlowScorer.peak = max(SlowScorer.peak, SlowScorer.active)
            time.sleep(0.05)
            with self.lock:
                SlowScorer.active -= 1
            return super().score(queries, texts)

    from concurrent.futures import ThreadPoolExecutor

    with TestClient(create_app(SlowScorer, max_concurrent=1)) as c, ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(lambda _: c.post("/rank", json=body()).status_code, range(8)))
    assert codes == [200] * 8 and SlowScorer.peak == 1


def test_oversized_body_is_refused_before_parsing():
    from shoprank.serve.app import MAX_BODY_BYTES

    scorer = FakeScorer()
    with client(scorer) as c:
        r = c.post("/rank", content=b"x" * (MAX_BODY_BYTES + 1), headers={"Content-Type": "application/json"})
    assert r.status_code == 413 and scorer.calls == []


def test_blank_title_is_rejected():
    with client() as c:
        r = c.post("/rank", json={"query": "cable", "products": [{"product_id": "a", "title": "   "}]})
    assert r.status_code == 422
