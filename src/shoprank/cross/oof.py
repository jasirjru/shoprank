"""K-fold out-of-fold (OOF) cross-encoder scores for stacking and distillation (sections 14-15).

Usage (one fold per process, so GPU and RAM are freed between folds):
    for k in 0 1 2 3 4; do
      uv run python -m shoprank.cross.oof --model microsoft/deberta-v3-base --fields title,brand,color \
          --lr 2e-5 --warmup-steps 1000 --fold $k --name deberta_oof
    done

For fold k, the model trains on every TRAIN query not in fold k and then scores:
  - the fold-k train pairs (out-of-fold: the model never saw these queries) -> stacking / distillation targets
  - dev and dev2 (the stacker's inputs at evaluation time are the average over the 5 fold models)
Folds are assigned by query (a query is never split) with a hash independent of the LambdaMART hold-out.
There is no checkpoint choice inside a fold (the last step is kept), so no extra hold-out is needed;
section 13 showed the best hold-out checkpoint is the last one for DeBERTa.
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch

from shoprank.cross.finetune import (
    DEFAULT_MODEL,
    MODELS,
    FitConfig,
    Prepared4DMask,
    fit,
    predict,
)
from shoprank.cross.score import SCORES, score_frame
from shoprank.cross.zero_shot import product_texts
from shoprank.features.build import PROCESSED, cached, load_split
from shoprank.hub import revision_for

FOLD_HASH_KEY = "shoprank-oof-k!!"  # 16 characters, different from the hold-out hash
NO_EVAL = 10**9  # eval_every larger than any run: only the final step is "evaluated", so last = kept
BATCH = 128


def query_fold(query_ids: pd.Series, n_folds: int) -> pd.Series:
    """Fold number (0..n_folds-1) per row, by query_id hash.

    The id is hashed as TEXT: pandas ignores `hash_key` for integers, so an integer hash would equal the
    hold-out hash, and hash % 10 == 0 (hold-out) would put every hold-out query in fold 0 (caught by a test).
    """
    h = pd.util.hash_pandas_object(query_ids.astype(str), index=False, hash_key=FOLD_HASH_KEY)
    return (h % n_folds).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one OOF fold and score its held-out queries + dev/dev2")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fields", default="title")
    ap.add_argument("--lr", type=float, default=FitConfig.lr)
    ap.add_argument("--warmup-steps", type=int, default=FitConfig.warmup_steps)
    ap.add_argument("--seed", type=int, default=FitConfig.seed)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    if not 0 <= args.fold < args.n_folds:
        raise SystemExit(f"--fold must be in 0..{args.n_folds - 1}")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    fields = args.fields.split(",")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    products = pd.read_parquet(PROCESSED / "products.parquet", columns=["product_id", *fields])
    train = cached("train")[["query_id", "query", "product_id", "esci_label", "gain"]]
    in_fold = query_fold(train["query_id"], args.n_folds) == args.fold
    fit_part, oof_part = train[~in_fold].reset_index(drop=True), train[in_fold].reset_index(drop=True)
    print(f"fold {args.fold}/{args.n_folds}: fit {fit_part['query_id'].nunique():,} q / {len(fit_part):,} pairs; "
          f"out-of-fold {oof_part['query_id'].nunique():,} q / {len(oof_part):,} pairs", flush=True)

    revision = revision_for(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=revision, num_labels=1, dtype=torch.float32).to(device)
    net = Prepared4DMask(model) if model.config._attn_implementation == "sdpa" else model
    cfg = FitConfig(lr=args.lr, warmup_steps=args.warmup_steps, eval_every=NO_EVAL, seed=args.seed)
    history = fit(net, tokenizer, fit_part["query"].tolist(), product_texts(fit_part, products, fields),
                  fit_part["gain"].to_numpy(), cfg, device, evaluate=lambda m: 0.0,
                  log=lambda s: print(s, flush=True))
    tag = f"{args.name}_fold{args.fold}"
    model.save_pretrained(MODELS / tag)
    tokenizer.save_pretrained(MODELS / tag)

    SCORES.mkdir(parents=True, exist_ok=True)
    parts = {"oof": oof_part}
    for split in ("dev", "dev2"):
        parts[split], _ = load_split(split)
    for label, pairs in parts.items():
        s = predict(net, tokenizer, pairs["query"].tolist(), product_texts(pairs, products, fields),
                    BATCH, cfg.max_length, device, cfg.bf16)
        score_frame(pairs, s).to_parquet(SCORES / f"{tag}_{label}.parquet", index=False)
    print(f"{tag}: trained {history['train_seconds']:.0f}s; scored "
          + ", ".join(f"{k} {len(v):,}" for k, v in parts.items()), flush=True)
    del model, net
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
