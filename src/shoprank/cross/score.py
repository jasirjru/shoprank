"""Score every (query, product) pair of a split with a saved fine-tuned cross-encoder.

Usage:
    uv run python -m shoprank.cross.score deberta_v3_base_ft_mse --split dev
    uv run python -m shoprank.cross.score deberta_v3_base_ft_mse --split test --allow-test   # section 16, once

Writes data/processed/scores/<run>_<split>.parquet with one row per pair (query_id, product_id, esci_label,
gain, score). Used by error analysis (worst queries), blending/stacking (section 14) and the final test run.
The product fields are read from the run's saved config, so scoring uses the same input as training.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from shoprank.cross.finetune import MODELS, Prepared4DMask, predict
from shoprank.cross.zero_shot import product_texts
from shoprank.features.build import PROCESSED, load_split

SCORES = PROCESSED / "scores"
LOCKED = {"test"}
BATCH = 128


def check_split(split: str, allow_test: bool) -> None:
    """Same rule as run_task1: the official test set is scored only on purpose (section 16)."""
    if split in LOCKED and not allow_test:
        raise SystemExit(f"split '{split}' is locked; pass --allow-test only for the final, one-time run")


def score_frame(pairs: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    if len(scores) != len(pairs):
        raise ValueError(f"{len(scores)} scores for {len(pairs)} pairs")
    out = pairs[["query_id", "product_id", "esci_label", "gain"]].copy()
    out["score"] = np.asarray(scores, dtype=np.float32)
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-pair scores for a saved cross-encoder run")
    ap.add_argument("run")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    check_split(args.split, args.allow_test)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    config = json.loads((PROCESSED / "runs" / f"{args.run}.json").read_text(encoding="utf-8"))  # flat summary
    fields, max_length = config["fields"], config.get("max_length", 128)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs, _ = load_split(args.split)
    products = pd.read_parquet(PROCESSED / "products.parquet", columns=["product_id", *fields])
    texts = product_texts(pairs, products, fields)
    model_dir = MODELS / args.run
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32).to(device)
    net = Prepared4DMask(model) if model.config._attn_implementation == "sdpa" else model

    start = time.perf_counter()
    scores = predict(net, tokenizer, pairs["query"].tolist(), texts, BATCH, max_length, device, bf16=True)
    seconds = time.perf_counter() - start
    SCORES.mkdir(parents=True, exist_ok=True)
    path = SCORES / f"{args.run}_{args.split}.parquet"
    score_frame(pairs, scores).to_parquet(path, index=False)
    print(f"{args.run} on {args.split}: {len(pairs):,} pairs in {seconds:.1f} s "
          f"({len(pairs) / seconds:,.0f} pairs/s) -> {path}")


if __name__ == "__main__":
    main()
