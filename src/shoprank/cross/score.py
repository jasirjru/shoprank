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


def score_pairs(model_dir, pairs: pd.DataFrame, fields: list[str], max_length: int = 128,
                batch_size: int = BATCH) -> np.ndarray:
    """Score every pair with the saved model in `model_dir`, reading the same product fields as in training."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    products = pd.read_parquet(PROCESSED / "products.parquet", columns=["product_id", *fields])
    texts = product_texts(pairs, products, fields)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32).to(device)
    net = Prepared4DMask(model) if model.config._attn_implementation == "sdpa" else model
    scores = predict(net, tokenizer, pairs["query"].tolist(), texts, batch_size, max_length, device, bf16=True)
    del model, net
    if device == "cuda":
        torch.cuda.empty_cache()
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-pair scores for a saved cross-encoder run")
    ap.add_argument("run")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    check_split(args.split, args.allow_test)
    path = SCORES / f"{args.run}_{args.split}.parquet"
    if args.split in LOCKED and path.exists():  # the official test set is scored once: never overwrite
        raise SystemExit(f"{path} exists; test scores are write-once")

    config = json.loads((PROCESSED / "runs" / f"{args.run}.json").read_text(encoding="utf-8"))  # flat summary
    pairs, _ = load_split(args.split)
    start = time.perf_counter()
    scores = score_pairs(MODELS / args.run, pairs, config["fields"], config.get("max_length", 128))
    seconds = time.perf_counter() - start
    SCORES.mkdir(parents=True, exist_ok=True)
    score_frame(pairs, scores).to_parquet(path, index=False)
    print(f"{args.run} on {args.split}: {len(pairs):,} pairs in {seconds:.1f} s "
          f"({len(pairs) / seconds:,.0f} pairs/s) -> {path}")


if __name__ == "__main__":
    main()
