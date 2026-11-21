"""Section 14: combine rankers. A z-scored blend, then LightGBM stacking on out-of-fold (OOF) scores.

Usage:
    uv run python -m shoprank.ltr.stack            # fit on TRAIN (OOF scores), report on dev
    uv run python -m shoprank.ltr.stack --dev2     # ONE confirmation look at dev2, after every choice is frozen

Inputs come from shoprank.cross.oof: data/processed/scores/<model>_fold<k>_{oof,dev,dev2}.parquet.
- TRAIN rows get the score of the one fold model that never saw their query (out-of-fold), so the blend
  weights and the stacker learn from honest scores, not from scores of a model that memorised the labels.
- dev / dev2 rows get the mean of the 5 fold models (each fold model saw 80% of TRAIN).
Primary sources: KDD Cup 2022 top teams all blended several cross-encoders; 1st and 2nd place fed
cross-encoder scores plus per-query statistics into a LightGBM ranker. We copy that, on our 14 features.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from shoprank.cross.oof import query_fold
from shoprank.cross.score import SCORES
from shoprank.features.build import FEATURES, cached
from shoprank.ltr.train import RUNS, report, train_lambdamart

OOF_MODELS = ("deberta_oof", "minilm_oof")
N_FOLDS = 5
KEY = ["query_id", "product_id"]
BLEND_STEP = 0.05                     # weight grid for the z-score blend
BLEND_EXTRA = ("bm25_title",)         # an untrained score, honest on TRAIN as it is
QUERY_STATS = ("z", "pct", "gap", "qmax", "qmed", "qmin")
RELATIVE_STATS = ("z", "pct")         # scale-free: immune to the train (1 model) vs dev (mean of 5) shift
MIN_STD = 1e-6


def load_oof_scores(name: str, split: str, n_folds: int = N_FOLDS, scores_dir=SCORES) -> pd.DataFrame:
    """KEY + one column `name`. TRAIN: concatenated fold-k OOF files. dev/dev2: mean over the fold models."""
    if split == "train":
        parts = [pd.read_parquet(scores_dir / f"{name}_fold{k}_oof.parquet", columns=[*KEY, "score"])
                 for k in range(n_folds)]
        for k, part in enumerate(parts):  # fold k's file must hold exactly the fold-k queries
            if (query_fold(part["query_id"], n_folds) != k).any():
                raise ValueError(f"{name}: fold {k} OOF file holds queries of another fold")
        out = pd.concat(parts, ignore_index=True)
        if out.duplicated(KEY).any():
            raise ValueError(f"{name}: a train pair has scores from two folds")
        return out.rename(columns={"score": name})
    frames = [pd.read_parquet(scores_dir / f"{name}_fold{k}_{split}.parquet", columns=[*KEY, "score"])
              .set_index(KEY)["score"] for k in range(n_folds)]
    mean = pd.concat(frames, axis=1, join="outer").mean(axis=1, skipna=False)
    if mean.isna().any():
        raise ValueError(f"{name}: fold models scored different {split} pairs")
    return mean.rename(name).reset_index()


def attach_scores(feats: pd.DataFrame, scores: list[pd.DataFrame]) -> pd.DataFrame:
    """Left-join each score table on (query_id, product_id); every pair must get every score."""
    runs = (feats["query_id"] != feats["query_id"].shift()).sum()
    if runs != feats["query_id"].nunique():  # LightGBM groups need each query's rows together
        raise ValueError("rows of a query are not contiguous")
    out = feats
    for s in scores:
        out = out.merge(s, on=KEY, how="left", validate="one_to_one")
    missing = out[[c for s in scores for c in s.columns if c not in KEY]].isna().any()
    if missing.any():
        raise ValueError(f"pairs without a score: {missing[missing].index.tolist()}")
    return out


def per_query_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    """(x - query mean) / query std. A query whose scores are all equal gets 0 (no information)."""
    x = df[col].astype("float64")
    g = x.groupby(df["query_id"], sort=False)
    std = g.transform("std", ddof=0)
    return ((x - g.transform("mean")) / std.where(std > MIN_STD)).fillna(0.0)


def add_query_features(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Per-query context for each score: a raw 0.7 means little until you know the query's best is 0.9."""
    new = {}
    for c in cols:
        g = df.groupby("query_id", sort=False)[c]
        new[f"{c}_z"] = per_query_zscore(df, c)
        new[f"{c}_pct"] = g.rank(pct=True)
        new[f"{c}_gap"] = df[c] - g.transform("max")
        new[f"{c}_qmax"], new[f"{c}_qmed"], new[f"{c}_qmin"] = (
            g.transform("max"), g.transform("median"), g.transform("min"))
    return df.assign(**new), list(new)


def mean_ndcg(df: pd.DataFrame, score_col: str) -> float:
    """Mean full-list nDCG over queries, vectorised. Same tie rule as ltr.train.per_query_results."""
    def dcg(sorted_df: pd.DataFrame) -> pd.Series:
        pos = sorted_df.groupby("query_id", sort=False).cumcount().to_numpy() + 1
        return (sorted_df["gain"] / np.log2(pos + 1)).groupby(sorted_df["query_id"]).sum()

    ranked = df.sort_values(["query_id", score_col, "product_id"], ascending=[True, False, True])
    ideal = df.sort_values(["query_id", "gain"], ascending=[True, False])
    d, i = dcg(ranked), dcg(ideal)
    return float((d / i.where(i > 0)).fillna(0.0).mean())


def weight_grid(n: int, step: float = BLEND_STEP) -> list[tuple[float, ...]]:
    """All weight vectors of length n on a `step` grid that sum to 1."""
    ticks = round(1 / step)
    return [tuple(c / ticks for c in combo) + ((ticks - sum(combo)) / ticks,)
            for combo in itertools.product(range(ticks + 1), repeat=n - 1) if sum(combo) <= ticks]


def blend(df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    return np.sum([w * per_query_zscore(df, c).to_numpy() for c, w in weights.items()], axis=0)


def fit_blend(train: pd.DataFrame, cols: list[str], step: float = BLEND_STEP) -> tuple[dict[str, float], float]:
    """Pick blend weights by mean nDCG on TRAIN (OOF scores). dev is never used to choose them."""
    z = {c: per_query_zscore(train, c).to_numpy() for c in cols}
    base = train[["query_id", "product_id", "gain"]]
    best = max(((mean_ndcg(base.assign(s=sum(w * z[c] for c, w in zip(cols, ws, strict=True))), "s"), ws)
                for ws in weight_grid(len(cols), step)), key=lambda t: t[0])
    return dict(zip(cols, best[1], strict=True)), best[0]


def frame(split: str) -> pd.DataFrame:
    return attach_scores(cached(split), [load_oof_scores(m, split) for m in OOF_MODELS])


def main() -> None:
    ap = argparse.ArgumentParser(description="z-score blend and LightGBM stacking on OOF scores")
    ap.add_argument("--dev2", action="store_true", help="also score dev2 ONCE (final confirmation)")
    ap.add_argument("--force", action="store_true", help="re-score dev2 although it was scored before")
    args = ap.parse_args()

    train, dev = frame("train"), frame("dev")
    print(f"train {len(train):,} pairs / {train['query_id'].nunique():,} q (OOF scores) | dev {len(dev):,} pairs")
    for m in OOF_MODELS:
        print(f"  {m}: train OOF nDCG {mean_ndcg(train, m):.4f}")
        report(f"{m}_avg5", dev, dev[m].to_numpy(), {"model": m, "folds": N_FOLDS, "combine": "mean"})

    blend_cols = [*OOF_MODELS, *BLEND_EXTRA]
    weights, train_ndcg = fit_blend(train, blend_cols)
    print(f"  blend weights (chosen on train OOF, nDCG {train_ndcg:.4f}): {weights}")
    report("zblend", dev, blend(dev, weights), {"model": "zblend", "weights": weights})

    train, qcols = add_query_features(train, list(OOF_MODELS))
    dev, _ = add_query_features(dev, list(OOF_MODELS))
    # stack_lgbm: raw scores + all query stats (KDD-style). stack_lgbm_rel: only scale-free z / percentile,
    # added after review: raw scores shift between TRAIN (one fold model) and dev (mean of 5 models).
    variants = {
        "stack_lgbm": [*FEATURES, *OOF_MODELS, *qcols],
        "stack_lgbm_rel": [*FEATURES, *(c for c in qcols if c.rsplit("_", 1)[1] in RELATIVE_STATS)],
    }
    stackers = {}
    for name, feats in variants.items():
        stackers[name] = model = train_lambdamart(train, feats)
        report(name, dev, model.predict(dev[feats]),
               {"model": "lambdamart_stack", "features": feats, "trees": model.best_iteration_})
        importance = pd.Series(model.booster_.feature_importance("gain"), index=feats)
        print(f"  {name} top features (gain share):",
              (importance / importance.sum()).sort_values(ascending=False).head(6).round(3).to_dict())

    if args.dev2:
        if (RUNS / "zblend_dev2.json").exists() and not args.force:
            raise SystemExit("dev2 was already scored once; pass --force only with a written reason")
        dev2 = frame("dev2")
        report("deberta_oof_avg5_dev2", dev2, dev2["deberta_oof"].to_numpy(), {"model": "deberta_oof"}, "dev2")
        report("zblend_dev2", dev2, blend(dev2, weights), {"model": "zblend", "weights": weights}, "dev2")
        dev2, _ = add_query_features(dev2, list(OOF_MODELS))
        for name, feats in variants.items():
            report(f"{name}_dev2", dev2, stackers[name].predict(dev2[feats]),
                   {"model": "lambdamart_stack", "features": feats}, "dev2")


if __name__ == "__main__":
    main()
