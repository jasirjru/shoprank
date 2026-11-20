"""Turn each (query, product) pair into numeric clues ("features") a model can learn from.

Usage:
    uv run python -m shoprank.features.build          # builds train + dev, cached as Parquet

Rule: features may use the query text and the product text ONLY. Never the label, never anything
computed from labels (that would be leakage — the model would be "told the answer").
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from shoprank.retrieval.bm25 import BM25, tokenize

PROCESSED = Path("data/processed")
LABEL_TO_INT = {"I": 0, "C": 1, "S": 2, "E": 3}  # ordered relevance grade for ranking models
BM25_FIELDS = ("title", "brand", "bullets", "description")
NUMBER_RE = re.compile(r"\d+")

FEATURES = [
    "bm25_title", "bm25_brand", "bm25_bullets", "bm25_description",
    "title_coverage",      # share of query words found in the title
    "title_jaccard",       # overlap of word sets: |q ∩ t| / |q ∪ t|
    "phrase_in_title",     # whole query appears as-is in the title
    "brand_in_query",      # product's brand words all appear in the query
    "color_in_query",      # product's color words all appear in the query
    "numbers_matched",     # numbers in the query that also appear in the title (e.g. "12", "55")
    "numbers_missing",     # numbers in the query that the title does NOT contain
    "query_len", "title_len", "has_description",
]


def pair_features(query: str, product: dict) -> dict:
    """Text-match features for one (query, product). Pure function: same input -> same output."""
    q = tokenize(query)
    q_set = set(q)
    title = tokenize(product.get("title"))
    t_set = set(title)
    brand = set(tokenize(product.get("brand")))
    color = set(tokenize(product.get("color")))
    q_nums = set(NUMBER_RE.findall(query or ""))
    t_nums = set(NUMBER_RE.findall(product.get("title") or ""))
    return {
        "title_coverage": len(q_set & t_set) / len(q_set) if q_set else 0.0,
        "title_jaccard": len(q_set & t_set) / len(q_set | t_set) if q_set | t_set else 0.0,
        "phrase_in_title": float(bool(q) and " ".join(q) in " ".join(title)),
        "brand_in_query": float(bool(brand) and brand <= q_set),
        "color_in_query": float(bool(color) and color <= q_set),
        "numbers_matched": float(len(q_nums & t_nums)),
        "numbers_missing": float(len(q_nums - t_nums)),
        "query_len": float(len(q)),
        "title_len": float(len(title)),
        "has_description": float(bool(product.get("description"))),
    }


def load_split(split: str, processed: Path = PROCESSED) -> tuple[pd.DataFrame, dict]:
    """Task-1 judgments for one split, sorted by query (rankers need each query's rows together)."""
    con = duckdb.connect()
    pairs = con.execute(f"""
        SELECT query_id, query, product_id, esci_label, gain
        FROM '{(processed / "judgments.parquet").as_posix()}'
        WHERE small_version = 1 AND split = ?
        ORDER BY query_id, product_id
    """, [split]).df()
    products = con.execute(f"""
        SELECT p.product_id, p.title, p.brand, p.bullets, p.description, p.color
        FROM '{(processed / "products.parquet").as_posix()}' p
        SEMI JOIN (SELECT DISTINCT product_id FROM '{(processed / "judgments.parquet").as_posix()}'
                   WHERE small_version = 1 AND split = ?) j USING (product_id)
    """, [split]).df()
    # Parquet nulls arrive as NaN, and NaN is truthy in Python ("has a description" would be wrong).
    products = products.astype(object).where(products.notna(), None)
    return pairs, products.set_index("product_id").to_dict("index")


def build(split: str, processed: Path = PROCESSED) -> pd.DataFrame:
    pairs, products = load_split(split, processed)

    # One BM25 per product field. Corpus = this split's products (text only, no labels).
    ids = sorted(products)
    index = {p: i for i, p in enumerate(ids)}
    bm25 = {f: BM25().fit([tokenize(products[p][f]) for p in ids]) for f in BM25_FIELDS}

    rows = []
    for query, pid in zip(pairs["query"], pairs["product_id"], strict=True):
        q = tokenize(query)
        feats = {f"bm25_{f}": bm25[f].score(q, index[pid]) for f in BM25_FIELDS}
        feats.update(pair_features(query, products[pid]))
        rows.append(feats)

    out = pd.concat([pairs.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["label"] = out["esci_label"].map(LABEL_TO_INT)
    return out[["query_id", "query", "product_id", "esci_label", "gain", "label", *FEATURES]]


def cached(split: str, processed: Path = PROCESSED) -> pd.DataFrame:
    path = processed / f"features_{split}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = build(split, processed)
    df.to_parquet(path, index=False)
    return df


def main() -> None:
    for split in ("train", "dev", "dev2"):
        df = cached(split)
        print(f"{split}: {len(df):,} pairs, {df['query_id'].nunique():,} queries, {len(FEATURES)} features")


if __name__ == "__main__":
    main()
