"""Project page: examples mined from the real API, rendered into one static page with escaped text."""
import json
from pathlib import Path

import pandas as pd

from shoprank.serve.showcase import api_products, bm25_order, check, ndcg_full
from shoprank.site import EXAMPLES_PATH, OUT_PATH, build, csp_headers, page, proof_line


def item(pid, label, title="t"):
    return {"product_id": pid, "title": title, "brand": None, "label": label}


def example(bm25_score=0.4, ours=0.9, title="t"):
    return {"query_id": 1, "query": "iphone xs pro", "query_used": "iphone xs pro", "note": "n",
            "ndcg": {"bm25": bm25_score, "shoprank": ours},
            "bm25": [item("a", "Irrelevant", title), item("b", "Exact")],
            "shoprank": [item("b", "Exact"), item("a", "Irrelevant", title)]}


def test_ndcg_full_is_one_for_the_ideal_order_and_lower_otherwise():
    gains = {"a": 1.0, "b": 0.0}
    assert ndcg_full(["a", "b"], gains) == 1.0
    assert ndcg_full(["b", "a"], gains) < 1.0


def test_bm25_order_breaks_ties_by_product_id_like_the_api():
    assert bm25_order(["c", "a", "b"], [1.0, 2.0, 1.0]) == ["a", "b", "c"]


def test_api_products_drops_missing_fields_and_respects_limits():
    rows = pd.DataFrame({"product_id": ["x"], "title": ["T" * 6000], "brand": [None], "color": [float("nan")],
                         "bullets": ["  "]})
    [product] = api_products(rows)
    assert product == {"product_id": "x", "title": "T" * 5000}


def test_check_reports_queries_whose_api_order_changed():
    pairs = pd.DataFrame({"query_id": [1, 1], "product_id": ["a", "b"], "title": ["A", "B"], "brand": [None, None],
                          "color": [None, None], "bullets": [None, None]})
    data = {"examples": [{"query_id": 1, "query": "q", "query_used": "q",
                          "shoprank": [{"product_id": "b"}, {"product_id": "a"}]}]}
    assert check(data, pairs, lambda q, p: ("q", ["b", "a"])) == []
    assert check(data, pairs, lambda q, p: ("q", ["a", "b"])) == ["1"]


def test_page_escapes_product_text_and_shows_every_example():
    html = page({"examples": [example(title="<script>alert(1)</script>"), example(0.9, 0.5)]}, "", "")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert html.count('role="tabpanel"') == 2
    assert html.count("card-flag") == 1  # only the example where ShopRank loses is flagged


def test_proof_line_is_built_from_the_example_data():
    line = proof_line(example())
    assert "iphone xs pro" in line and "number 2" in line and "puts 1 right product first" in line


def test_csp_allows_exactly_the_inline_script():
    csp = csp_headers("x()")["headers"][0]["headers"][0]["value"]
    assert "script-src 'sha256-" in csp and "unsafe-inline" not in csp.split("script-src")[1].split(";")[0]


def test_committed_page_matches_the_generator():
    """site/public is what Vercel serves: rebuild with `uv run python -m shoprank.site` after any change."""
    html, headers = build()
    assert Path(OUT_PATH).read_text(encoding="utf-8") == html
    assert json.loads((Path(OUT_PATH).parent / "vercel.json").read_text(encoding="utf-8")) == headers
    assert len(json.loads(Path(EXAMPLES_PATH).read_text(encoding="utf-8"))["examples"]) == 8
