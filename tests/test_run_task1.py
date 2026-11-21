"""Task-1 evaluation runner: the code behind every dev/test nDCG in the README."""
from __future__ import annotations

import json
import math
import sys

import pandas as pd
import pytest

from shoprank.eval import run_task1
from shoprank.eval.run_task1 import evaluate, load_task1, save


def write_tables(folder) -> None:
    pd.DataFrame({
        "query_id": [1, 1, 1, 2, 2, 3],
        "query": ["usb cable", "usb cable", "usb cable", "red mug", "red mug", "lamp"],
        "product_id": ["c", "a", "b", "d", "e", "f"],
        "gain": [0.1, 1.0, 0.0, 1.0, 0.01, 1.0],
        "esci_label": ["S", "E", "I", "E", "C", "E"],
        "small_version": [1, 1, 1, 1, 0, 1],
        "split": ["dev", "dev", "dev", "dev", "dev", "train"],
    }).to_parquet(folder / "judgments.parquet")
    pd.DataFrame({
        "product_id": list("abcdef"), "title": ["USB cable", "HDMI", "USB plug", "Red mug", "Mug", "Lamp"],
        "brand": [None] * 6, "bullets": [None] * 6, "color": [None] * 6,
    }).to_parquet(folder / "products.parquet")


def test_load_task1_keeps_only_small_version_of_the_split_sorted_by_product(tmp_path):
    write_tables(tmp_path)
    queries, products = load_task1("dev", tmp_path)
    assert set(queries) == {1, 2}  # query 3 is train
    assert queries[1] == ("usb cable", [("a", 1.0, "E"), ("b", 0.0, "I"), ("c", 0.1, "S")])
    assert queries[2][1] == [("d", 1.0, "E")]  # e is large-version only
    assert set(products) == {"a", "b", "c", "d"}  # corpus = this split's small-version candidates


def test_evaluate_matches_hand_computed_ndcg():
    queries = {7: ("q", [("x", 0.0, "I"), ("y", 0.1, "S"), ("z", 1.0, "E")])}
    worst = {"x": 3.0, "y": 2.0, "z": 1.0}  # ranks I, S, E
    [r] = evaluate(queries, lambda q, pids: [worst[p] for p in pids])
    dcg = 0.0 + 0.1 / math.log2(3) + 1.0 / math.log2(4)
    ideal = 1.0 + 0.1 / math.log2(3)
    assert r["ndcg_full"] == pytest.approx(dcg / ideal)
    assert r["rr_exact"] == pytest.approx(1 / 3)
    [best] = evaluate(queries, lambda q, pids: [-worst[p] for p in pids])
    assert best["ndcg_full"] == pytest.approx(1.0) and best["rr_exact"] == 1.0


def test_evaluate_breaks_ties_by_product_id():
    queries = {1: ("q", [("b", 1.0, "E"), ("a", 0.0, "I")])}
    [r] = evaluate(queries, lambda q, pids: [0.5, 0.5])
    assert r["rr_exact"] == 0.5  # tie -> "a" first, so the exact product is second


def test_save_writes_means_and_per_query_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(run_task1, "RUNS", tmp_path)
    rows = [{"query_id": 1, "ndcg10": 1.0, "ndcg_full": 0.8, "rr_exact": 1.0, "recall10_exact": 1.0},
            {"query_id": 2, "ndcg10": 0.0, "ndcg_full": 0.4, "rr_exact": 0.5, "recall10_exact": 0.0}]
    summary = save("demo", "dev", rows, {"ranker": "x"})
    assert summary["ndcg_full"] == pytest.approx(0.6) and summary["queries"] == 2
    assert json.loads((tmp_path / "demo.json").read_text())["mrr_exact"] == pytest.approx(0.75)
    assert len(pd.read_parquet(tmp_path / "demo.parquet")) == 2


def test_test_split_is_locked_without_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_task1", "--ranker", "random", "--name", "x", "--split", "test"])
    with pytest.raises(SystemExit):
        run_task1.main()


def test_test_results_are_write_once(tmp_path, monkeypatch):
    monkeypatch.setattr(run_task1, "RUNS", tmp_path)
    rows = [{"query_id": 1, "ndcg10": 1.0, "ndcg_full": 1.0, "rr_exact": 1.0, "recall10_exact": 1.0}]
    save("x_test", "test", rows, {})
    with pytest.raises(SystemExit, match="write-once"):
        save("x_test", "test", rows, {})
    save("x_dev", "dev", rows, {})
    save("x_dev", "dev", rows, {})  # dev may be overwritten
