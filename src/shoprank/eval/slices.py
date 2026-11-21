"""Error analysis by query slice: where does each model win or lose? (Section 13, plan step 2.)

Usage:
    uv run python -m shoprank.eval.slices bm25_title lambdamart minilm_ft_mse_fast deberta_v3_base_ft_mse

Every tag is a simple rule that can be explained in one sentence. Amazon's own query `source` labels
(data/raw/shopping_queries_dataset_sources.csv: negations, parse_pattern, behavioral, nlqec, other) are
added as a second, independent slicing.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from shoprank.eval.bootstrap import paired_bootstrap_ci
from shoprank.features.build import PROCESSED, load_split

SOURCES = Path("data/raw/shopping_queries_dataset_sources.csv")
NEGATION = re.compile(r"\b(not|no|non|without|except|excluding|w/o)\b")
SPANISH_WORDS = {"para", "con", "sin", "mujer", "hombre", "niños", "niñas", "niño", "niña", "de", "el", "la",
                 "los", "las", "y", "zapatos", "camisa", "juguete", "juguetes"}
ACCENTED = re.compile(r"[áéíóúñü¿¡]")
MIN_BRAND_LEN = 3  # "hp", "lg" would match too many queries by accident


def length_bucket(query: str) -> str:
    n = len(query.split())
    return "1-2 words" if n <= 2 else "3-5 words" if n <= 5 else "6+ words"


def query_tags(query: str, brands: set[str]) -> dict:
    """Rule-based tags for one query. `brands` = lowercase brands of this query's candidate products."""
    q = query.lower()
    words = set(q.split())
    brand_hit = any(len(b) >= MIN_BRAND_LEN and re.search(rf"\b{re.escape(b)}\b", q) for b in brands)
    return {
        "length": length_bucket(q),
        "has_number": bool(re.search(r"\d", q)),
        "negation": bool(NEGATION.search(q)),
        "spanish_like": bool(ACCENTED.search(q)) or len(words & SPANISH_WORDS) >= 2
                        or bool(words & {"para", "mujer", "hombre", "niños", "niñas"}),
        "brand": bool(brand_hit),
    }


def tag_queries(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per query_id with its tags. `pairs` needs query_id, query, brand."""
    rows = []
    for qid, g in pairs.groupby("query_id", sort=True):
        brands = {str(b).lower().strip() for b in g["brand"].dropna() if str(b).strip()}
        rows.append({"query_id": qid, **query_tags(g["query"].iloc[0], brands)})
    return pd.DataFrame(rows)


def slice_report(tags: pd.DataFrame, runs: dict[str, pd.DataFrame], a: str, b: str) -> pd.DataFrame:
    """Mean full-list nDCG per slice for every run, plus b - a with a paired bootstrap 95% CI."""
    table = tags.copy()
    for name, r in runs.items():
        table = table.merge(r[["query_id", "ndcg_full"]].rename(columns={"ndcg_full": name}), on="query_id")
    slices = [("all", pd.Series(True, index=table.index))]
    slices += [(f"length={v}", table["length"] == v) for v in sorted(table["length"].unique())]
    slices += [(c, table[c]) for c in ("has_number", "negation", "spanish_like", "brand")]
    if "source" in table:
        slices += [(f"source={v}", table["source"] == v) for v in sorted(table["source"].dropna().unique())]
    out = []
    for label, mask in slices:
        part = table[mask]
        if len(part) < 2:
            continue
        delta, lo, hi = paired_bootstrap_ci(part[a], part[b])
        out.append({"slice": label, "queries": len(part), **{n: round(part[n].mean(), 4) for n in runs},
                    f"{b} - {a}": f"{delta:+.4f} [{lo:+.4f}, {hi:+.4f}]"})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-slice nDCG on dev; the last two runs are compared with a CI")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    pairs, products = load_split("dev")
    pairs = pairs.assign(brand=pairs["product_id"].map(lambda p: products[p].get("brand")))
    tags = tag_queries(pairs)
    if SOURCES.exists():
        tags = tags.merge(pd.read_csv(SOURCES), on="query_id", how="left")
    runs = {n: pd.read_parquet(PROCESSED / "runs" / f"{n}.parquet") for n in args.runs}
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(slice_report(tags, runs, args.runs[-2], args.runs[-1]).to_string(index=False))


if __name__ == "__main__":
    main()
