"""Week 4: score DEV with a pretrained cross-encoder, no training on ESCI (zero-shot).

Usage:
    uv run python -m shoprank.cross.zero_shot                       # title only, MiniLM-L-12
    uv run python -m shoprank.cross.zero_shot --fields title,brand --name minilm_zs_tb

The model reads "[CLS] query [SEP] product text [SEP]" and outputs one relevance logit.
It was trained on MS MARCO (web passages), never on ESCI, so this measures what pretrained
language knowledge alone gives before any fine-tuning (Week 5).
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from shoprank.eval.run_task1 import save
from shoprank.features.build import PROCESSED, cached
from shoprank.ltr.train import per_query_results

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Amazon's ESCI baseline starts from this model


def product_texts(pairs: pd.DataFrame, products: pd.DataFrame, fields: Sequence[str]) -> list[str]:
    """One text per (query, product) row, fields joined with spaces; missing fields become ''."""
    merged = pairs[["product_id"]].merge(products, on="product_id", how="left", validate="many_to_one")
    cols = [merged[f].fillna("").astype(str) for f in fields]
    text = cols[0]
    for c in cols[1:]:
        text = text + " " + c
    return text.str.strip().tolist()


def score_pairs(pairs: pd.DataFrame, texts: list[str],
                predict: Callable[[list[tuple[str, str]]], np.ndarray]) -> np.ndarray:
    scores = np.asarray(predict(list(zip(pairs["query"].tolist(), texts, strict=True))), dtype=float)
    if scores.shape != (len(pairs),):
        raise ValueError(f"expected {len(pairs)} scores, got shape {scores.shape}")
    return scores


def main() -> None:
    p = argparse.ArgumentParser(description="Zero-shot cross-encoder on DEV")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--fields", default="title", help="comma list from: title,brand,color,bullets,description")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--name", default="minilm_zs_title")
    args = p.parse_args()

    import torch
    from sentence_transformers import CrossEncoder

    dev = cached("dev")
    products = pd.read_parquet(PROCESSED / "products.parquet")
    texts = product_texts(dev, products, args.fields.split(","))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(args.model, device=device, max_length=args.max_length)

    start = time.perf_counter()
    scores = score_pairs(dev, texts, lambda pr: model.predict(pr, batch_size=args.batch_size))
    seconds = time.perf_counter() - start
    config = {"model": args.model, "fields": args.fields.split(","), "max_length": args.max_length,
              "device": device, "pairs": len(dev), "seconds": round(seconds, 1),
              "pairs_per_s": round(len(dev) / seconds)}
    summary = save(args.name, "dev", per_query_results(dev.assign(score=scores), "score"), config)
    print(f"{args.name}: nDCG full {summary['ndcg_full']:.4f}  nDCG@10 {summary['ndcg10']:.4f}  "
          f"({len(dev):,} pairs in {seconds:.1f}s, {len(dev) / seconds:.0f} pairs/s on {device})")


if __name__ == "__main__":
    main()
