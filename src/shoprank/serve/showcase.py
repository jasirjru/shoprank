"""Examples for the public project page: BM25 vs the served ShopRank API on real dev queries.

Mine every dev query (never test), rank its judged candidates two ways, and keep the numbers:
    uv run python -m shoprank.serve.showcase mine      # -> data/processed/showcase_candidates.parquet
    uv run python -m shoprank.serve.showcase export 123 456 ...   # chosen query ids -> site/data/examples.json
    uv run python -m shoprank.serve.showcase check     # parity: the API still gives the exported order

ShopRank's order comes from the real API code (create_app with the INT8 ONNX model and the spell corrector),
called in-process, so the page shows exactly what the service returns. BM25 is the title-only BM25 baseline
used everywhere in the project (feature `bm25_title`), ties broken by product id like the API.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from shoprank.eval.metrics import ndcg_at_k
from shoprank.serve.app import MAX_LONG_FIELD_CHARS, MAX_SHORT_FIELD_CHARS

PROCESSED = Path("data/processed")
CANDIDATES_PATH = PROCESSED / "showcase_candidates.parquet"
EXAMPLES_PATH = Path("site/data/examples.json")
PRODUCT_FIELDS = ("title", "brand", "color", "bullets")
LABEL_NAMES = {"E": "Exact", "S": "Substitute", "C": "Complement", "I": "Irrelevant"}

RankFn = Callable[[str, list[dict]], tuple[str, list[str]]]


@dataclass(frozen=True)
class Ranking:
    query_used: str
    order: list[str]


def ndcg_full(order: Sequence[str], gains: Mapping[str, float]) -> float:
    """Full-list nDCG (the project's main metric) of one ranked list of product ids."""
    ranked = [gains[pid] for pid in order]
    return ndcg_at_k(ranked, len(ranked))


def bm25_order(product_ids: Sequence[str], bm25: Sequence[float]) -> list[str]:
    """Best first; ties broken by product id, the same rule as the API."""
    return [pid for _, pid in sorted(zip(bm25, product_ids), key=lambda x: (-x[0], x[1]))]


def clip(value: object, limit: int) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    return text[:limit] if text.strip() else None


def api_products(rows: pd.DataFrame) -> list[dict]:
    """Request bodies for /rank, within the API's field limits (the tokenizer cuts at 128 tokens anyway)."""
    limits = {"title": MAX_LONG_FIELD_CHARS, "brand": MAX_SHORT_FIELD_CHARS, "color": MAX_SHORT_FIELD_CHARS,
              "bullets": MAX_LONG_FIELD_CHARS}
    out = []
    for row in rows.itertuples(index=False):
        item = {"product_id": row.product_id}
        item.update({f: clip(getattr(row, f), limits[f]) for f in PRODUCT_FIELDS})
        out.append({k: v for k, v in item.items() if v is not None})
    return out


def api_ranker() -> tuple[RankFn, Callable[[], None]]:
    """The real service, in-process: returns rank(query, products) and a close() function."""
    from fastapi.testclient import TestClient

    from shoprank.serve.app import create_app

    client = TestClient(create_app())
    client.__enter__()

    def rank(query: str, products: list[dict]) -> tuple[str, list[str]]:
        r = client.post("/rank", json={"query": query, "products": products})
        r.raise_for_status()
        data = r.json()
        return data["query_used"], [x["product_id"] for x in data["results"]]

    return rank, lambda: client.__exit__(None, None, None)


def load_dev(processed: Path = PROCESSED) -> pd.DataFrame:
    pairs = pd.read_parquet(processed / "features_dev.parquet",
                            columns=["query_id", "query", "product_id", "esci_label", "gain", "bm25_title"])
    products = pd.read_parquet(processed / "products.parquet", columns=["product_id", *PRODUCT_FIELDS],
                               filters=[("product_id", "in", pairs["product_id"].unique().tolist())])
    return pairs.merge(products, on="product_id", how="left", validate="many_to_one")


def query_summary(group: pd.DataFrame, rank: RankFn) -> dict | None:
    if group["title"].isna().any() or (group["title"].str.strip() == "").any():
        return None  # the API needs a title for every product
    gains = dict(zip(group["product_id"], group["gain"]))
    bm25 = bm25_order(group["product_id"].tolist(), group["bm25_title"].tolist())
    query_used, ours = rank(group["query"].iat[0], api_products(group))
    return {"query_id": int(group["query_id"].iat[0]), "query": group["query"].iat[0], "query_used": query_used,
            "n": len(group), "n_exact": int((group["esci_label"] == "E").sum()),
            "ndcg_bm25": ndcg_full(bm25, gains), "ndcg_shoprank": ndcg_full(ours, gains)}


def mine(pairs: pd.DataFrame, rank: RankFn) -> pd.DataFrame:
    rows = [query_summary(g, rank) for _, g in pairs.groupby("query_id", sort=True)]
    table = pd.DataFrame([r for r in rows if r is not None])
    table["delta"] = table["ndcg_shoprank"] - table["ndcg_bm25"]
    table["corrected"] = table["query"].str.strip() != table["query_used"]
    return table


def example_record(group: pd.DataFrame, rank: RankFn, note: str) -> dict:
    gains = dict(zip(group["product_id"], group["gain"]))
    by_id = group.set_index("product_id")
    bm25 = bm25_order(group["product_id"].tolist(), group["bm25_title"].tolist())
    query_used, ours = rank(group["query"].iat[0], api_products(group))

    def items(order: list[str]) -> list[dict]:
        return [{"product_id": pid, "title": str(by_id.at[pid, "title"]),
                 "brand": clip(by_id.at[pid, "brand"], 80), "label": LABEL_NAMES[by_id.at[pid, "esci_label"]]}
                for pid in order]

    return {"query_id": int(group["query_id"].iat[0]), "query": group["query"].iat[0], "query_used": query_used,
            "note": note, "ndcg": {"bm25": round(ndcg_full(bm25, gains), 4),
                                    "shoprank": round(ndcg_full(ours, gains), 4)},
            "bm25": items(bm25), "shoprank": items(ours)}


# Saved per-query dev results of other systems (same judged lists, full-list nDCG), shown beside each example.
OTHER_RUNS = {"bge": "zeroshot_bge_v2m3_dev", "qwen3": "zeroshot_qwen3_06b_dev", "teacher": "deberta_oof_avg5"}


def other_scores(qids: Sequence[int], runs_dir: Path = PROCESSED / "runs") -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {int(q): {} for q in qids}
    for key, name in OTHER_RUNS.items():
        run = pd.read_parquet(runs_dir / f"{name}.parquet", columns=["query_id", "ndcg_full"]).set_index("query_id")
        for q, scores in out.items():
            scores[key] = round(float(run.at[q, "ndcg_full"]), 4)
    return out


def export(pairs: pd.DataFrame, chosen: Sequence[tuple[int, str]], rank: RankFn,
           others: dict[int, dict[str, float]] | None = None) -> dict:
    groups = {qid: g for qid, g in pairs.groupby("query_id")}
    examples = [example_record(groups[qid], rank, note) for qid, note in chosen]
    for ex in examples:
        ex["others"] = (others or {}).get(ex["query_id"], {})
    return {"split": "dev", "metric": "nDCG over the full judged list (Exact 1, Substitute 0.1, Complement 0.01)",
            "baseline": "BM25 on product titles", "model": "ShopRank API: spell correction + INT8 MiniLM",
            "others": {"bge": "BGE-reranker-v2-m3 (568M), zero-shot", "qwen3": "Qwen3-Reranker-0.6B (596M), zero-shot",
                       "teacher": "5 DeBERTa-v3-base fold models, mean score"},
            "examples": examples}


def check(data: Mapping, pairs: pd.DataFrame, rank: RankFn) -> list[str]:
    """Query ids whose API order or corrected query no longer matches the exported file."""
    groups = {qid: g for qid, g in pairs.groupby("query_id")}
    bad = []
    for ex in data["examples"]:
        query_used, order = rank(ex["query"], api_products(groups[ex["query_id"]]))
        if query_used != ex["query_used"] or order != [x["product_id"] for x in ex["shoprank"]]:
            bad.append(str(ex["query_id"]))
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description="Examples for the ShopRank project page")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mine")
    e = sub.add_parser("export")
    e.add_argument("choices", nargs="+", help="query_id=note, e.g. 1234='typo fixed'")
    sub.add_parser("check")
    args = ap.parse_args()
    pairs = load_dev()
    rank, close = api_ranker()
    try:
        if args.cmd == "mine":
            table = mine(pairs, rank)
            table.to_parquet(CANDIDATES_PATH, index=False)
            print(f"{len(table)} dev queries -> {CANDIDATES_PATH}; mean nDCG BM25 {table.ndcg_bm25.mean():.4f}, "
                  f"ShopRank {table.ndcg_shoprank.mean():.4f}; corrected {int(table.corrected.sum())}")
        elif args.cmd == "export":
            chosen = [(int(c.split("=", 1)[0]), c.split("=", 1)[1] if "=" in c else "") for c in args.choices]
            EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = export(pairs, chosen, rank, other_scores([q for q, _ in chosen]))
            EXAMPLES_PATH.write_text(json.dumps(data, indent=1), encoding="utf-8", newline="\n")
            print(f"{len(chosen)} examples -> {EXAMPLES_PATH}")
        else:
            bad = check(json.loads(EXAMPLES_PATH.read_text(encoding="utf-8")), pairs, rank)
            print("parity OK" if not bad else f"parity FAILED for query ids {bad}")
            raise SystemExit(1 if bad else 0)
    finally:
        close()


if __name__ == "__main__":
    main()
