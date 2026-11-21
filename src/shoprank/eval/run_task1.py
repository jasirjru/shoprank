"""ESCI Task-1 evaluation runner: rerank each query's judged candidates, score with nDCG@10.

Usage:
    uv run python -m shoprank.eval.run_task1 --ranker random --name random
    uv run python -m shoprank.eval.run_task1 --ranker bm25 --fields title --name bm25_title
    uv run python -m shoprank.eval.run_task1 --compare bm25_title bm25_title_brand_bullets

Rules:
    - Default split is dev. The test split needs --allow-test: tune on dev, report test once.
    - Every run writes per-query results, so any two runs can be compared with a paired bootstrap.
"""
from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path

import duckdb
import pandas as pd

from shoprank.eval.bootstrap import paired_bootstrap_ci
from shoprank.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from shoprank.retrieval.bm25 import BM25, tokenize

PROCESSED = Path("data/processed")
RUNS = PROCESSED / "runs"
K = 10
RANDOM_SEED = 0

Ranker = Callable[[str, list[str]], list[float]]  # (query, candidate product_ids) -> scores


def load_task1(split: str, processed: Path = PROCESSED) -> tuple[dict, dict]:
    """Return {query_id: (query, [(product_id, gain, label), ...])} and {product_id: product row}."""
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT query_id, query, product_id, gain, esci_label
        FROM '{(processed / "judgments.parquet").as_posix()}'
        WHERE small_version = 1 AND split = ?
        ORDER BY query_id, product_id
    """, [split]).fetchall()
    queries: dict = {}
    for qid, query, pid, gain, label in rows:
        queries.setdefault(qid, (query, []))[1].append((pid, gain, label))
    # Corpus = this split's candidate products only. BM25 stats (IDF, avgdl) use product text, never
    # labels, so this is not leakage; it also keeps a pure-Python BM25 within 16 GB RAM.
    products = {
        r[0]: {"title": r[1], "brand": r[2], "bullets": r[3], "color": r[4]}
        for r in con.execute(f"""
            SELECT p.product_id, p.title, p.brand, p.bullets, p.color
            FROM '{(processed / "products.parquet").as_posix()}' p
            SEMI JOIN (SELECT product_id FROM '{(processed / "judgments.parquet").as_posix()}'
                       WHERE small_version = 1 AND split = ?) j USING (product_id)
        """, [split]).fetchall()
    }
    return queries, products


def random_ranker(_: dict) -> Ranker:
    rng = random.Random(RANDOM_SEED)
    return lambda query, pids: [rng.random() for _ in pids]


def bm25_ranker(fields: list[str], k1: float, b: float) -> Callable[[dict], Ranker]:
    def build(products: dict) -> Ranker:
        ids = sorted(products)
        docs = [tokenize(" ".join(products[p][f] or "" for f in fields)) for p in ids]
        model = BM25(k1=k1, b=b).fit(docs)
        index = {p: i for i, p in enumerate(ids)}

        def rank(query: str, pids: list[str]) -> list[float]:
            q = tokenize(query)
            return [model.score(q, index[p]) for p in pids]
        return rank
    return build


def evaluate(queries: dict, ranker: Ranker) -> list[dict]:
    results = []
    for qid, (query, cands) in queries.items():
        pids = [p for p, _, _ in cands]
        scores = ranker(query, pids)
        order = sorted(range(len(cands)), key=lambda i: (-scores[i], pids[i]))  # deterministic ties
        gains = [cands[i][1] for i in order]
        exact = [cands[i][2] == "E" for i in order]
        results.append({
            "query_id": qid,
            "ndcg10": ndcg_at_k(gains, K),
            # Official ESCI Task-1 metric: trec_eval `-m ndcg.1=0,2=0.01,3=0.1,4=1`, no cutoff = whole list.
            "ndcg_full": ndcg_at_k(gains, len(gains)),
            "rr_exact": reciprocal_rank(exact),
            "recall10_exact": recall_at_k(exact, sum(exact), K),
        })
    return results


def save(name: str, split: str, results: list[dict], config: dict) -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    if split == "test" and ((RUNS / f"{name}.json").exists() or (RUNS / f"{name}.parquet").exists()):
        raise SystemExit(f"run {name} on test already exists; test results are write-once")
    con = duckdb.connect()
    con.register("r", pd.DataFrame(results))
    con.execute(f"COPY r TO '{(RUNS / f'{name}.parquet').as_posix()}' (FORMAT parquet)")
    n = len(results)
    summary = {
        "name": name, "split": split, "queries": n, **config,
        "ndcg10": sum(r["ndcg10"] for r in results) / n,
        "ndcg_full": sum(r["ndcg_full"] for r in results) / n,
        "mrr_exact": sum(r["rr_exact"] for r in results) / n,
        "recall10_exact": sum(r["recall10_exact"] for r in results) / n,
    }
    (RUNS / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def compare(a: str, b: str) -> None:
    q = "SELECT query_id, ndcg10, ndcg_full FROM '{}' ORDER BY query_id"
    ra = duckdb.sql(q.format((RUNS / f"{a}.parquet").as_posix())).fetchall()
    rb = duckdb.sql(q.format((RUNS / f"{b}.parquet").as_posix())).fetchall()
    if [x[0] for x in ra] != [x[0] for x in rb]:
        raise SystemExit("Runs cover different queries; compare runs from the same split.")
    for col, label in ((1, "nDCG@10"), (2, "nDCG (full list)")):
        mean, lo, hi = paired_bootstrap_ci([x[col] for x in ra], [x[col] for x in rb])
        verdict = "B better" if lo > 0 else "A better" if hi < 0 else "no clear difference"
        print(f"{label} delta ({b} - {a}): {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")


def main() -> None:
    p = argparse.ArgumentParser(description="ESCI Task-1 evaluation")
    p.add_argument("--ranker", choices=["random", "bm25"])
    p.add_argument("--fields", default="title", help="comma list from: title,brand,bullets,color")
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--split", default="dev", choices=["train", "dev", "dev2", "test"])
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--name")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = p.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if not args.ranker or not args.name:
        p.error("--ranker and --name are required unless using --compare")
    if args.split == "test" and not args.allow_test:
        p.error("test split is locked: tune on dev, then pass --allow-test for the final report only")

    queries, products = load_task1(args.split)
    fields = args.fields.split(",")
    is_bm25 = args.ranker == "bm25"
    build = bm25_ranker(fields, args.k1, args.b) if is_bm25 else random_ranker
    config = {"ranker": args.ranker, "fields": fields, "k1": args.k1, "b": args.b} if is_bm25 else {"ranker": "random"}
    summary = save(args.name, args.split, evaluate(queries, build(products)), config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
