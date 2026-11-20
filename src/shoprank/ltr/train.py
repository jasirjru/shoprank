"""Week 2: learn to rank from features. Trains models on TRAIN, reports on DEV.

Usage:
    uv run python -m shoprank.ltr.train

Models:
    logreg_scratch   our NumPy logistic regression, pointwise: "is this product Exact?"
    logreg_sklearn   scikit-learn's version, as a check on ours
    logreg_graded    multinomial LR on all 4 labels, ranked by expected gain sum_k p(k) * gain(k):
                     same label information as LambdaMART, so the LR-vs-LambdaMART gap is about model type
    lambdamart       LightGBM LambdaMART, listwise: optimizes nDCG of each query's list directly
    lambdamart_no_bm25 / lambdamart_bm25_only   ablations: which features carry the signal?

Each run is saved like Week 1 runs, so they compare with:
    uv run python -m shoprank.eval.run_task1 --compare bm25_title lambdamart
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SkLogReg

from shoprank.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from shoprank.eval.run_task1 import RUNS, K, save
from shoprank.features.build import FEATURES, cached
from shoprank.ltr.logreg import LogisticRegression, Standardizer

SEED = 0
# LightGBM's default gain is 2**label - 1. We pass the ESCI gains so it optimizes the metric we report.
ESCI_LABEL_GAIN = [0.0, 0.01, 0.1, 1.0]  # index = label: I=0, C=1, S=2, E=3
VALID_BUCKETS = 10                        # 1 in 10 TRAIN queries held out for early stopping


def per_query_results(df: pd.DataFrame, score_col: str) -> list[dict]:
    results = []
    for qid, g in df.groupby("query_id", sort=True):
        g = g.sort_values([score_col, "product_id"], ascending=[False, True])  # deterministic ties
        gains = g["gain"].tolist()
        exact = (g["esci_label"] == "E").tolist()
        results.append({
            "query_id": qid,
            "ndcg10": ndcg_at_k(gains, K),
            "ndcg_full": ndcg_at_k(gains, len(gains)),
            "rr_exact": reciprocal_rank(exact),
            "recall10_exact": recall_at_k(exact, sum(exact), K),
        })
    return results


def groups(df: pd.DataFrame) -> np.ndarray:
    """LightGBM needs the size of each query's block of rows, in row order."""
    return df.groupby("query_id", sort=False).size().to_numpy()


def is_holdout(query_ids: pd.Series) -> pd.Series:
    """True for the 1-in-10 TRAIN queries held out for early stopping / checkpoint choice."""
    return pd.util.hash_pandas_object(query_ids, index=False) % VALID_BUCKETS == 0


def train_lambdamart(train: pd.DataFrame, feats: list[str]) -> lgb.LGBMRanker:
    # Early stopping on held-out TRAIN queries, so dev stays a clean measurement.
    is_valid = is_holdout(train["query_id"])
    fit_part, valid_part = train[~is_valid], train[is_valid]
    model = lgb.LGBMRanker(
        objective="lambdarank", label_gain=ESCI_LABEL_GAIN, n_estimators=2000, learning_rate=0.05,
        num_leaves=31, min_child_samples=50, random_state=SEED, verbose=-1,
    )
    # Stop on FULL-list nDCG (the headline metric): eval_at = longest list, so no list is cut at K.
    valid_groups = groups(valid_part)
    model.fit(
        fit_part[feats], fit_part["label"], group=groups(fit_part),
        eval_X=(valid_part[feats],), eval_y=(valid_part["label"],), eval_group=[valid_groups],
        eval_at=[int(valid_groups.max())], callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def report(name: str, df: pd.DataFrame, scores: np.ndarray, config: dict, split: str = "dev") -> dict:
    scored = df.assign(score=scores)
    summary = save(name, split, per_query_results(scored, "score"), config)
    print(f"{name:<22} [{split}] nDCG full {summary['ndcg_full']:.4f}   nDCG@10 {summary['ndcg10']:.4f}")
    return summary


def main() -> None:
    train, dev = cached("train"), cached("dev")
    print(f"train {len(train):,} pairs / {train['query_id'].nunique():,} queries | "
          f"dev {len(dev):,} pairs / {dev['query_id'].nunique():,} queries")

    # Pointwise logistic regression: target = 1 if Exact, else 0.
    scaler = Standardizer.fit(train[FEATURES].to_numpy())
    X_train, X_dev = scaler.transform(train[FEATURES].to_numpy()), scaler.transform(dev[FEATURES].to_numpy())
    y_train = (train["label"] == 3).to_numpy(dtype=float)
    ours = LogisticRegression.fit(X_train, y_train, lr=0.5, epochs=500, l2=1e-4)
    report("logreg_scratch", dev, ours.predict_proba(X_dev), {"model": "logreg_scratch", "epochs": 500})
    sk = SkLogReg(C=1 / (1e-4 * len(y_train)), max_iter=2000).fit(X_train, y_train)
    report("logreg_sklearn", dev, sk.predict_proba(X_dev)[:, 1], {"model": "logreg_sklearn"})

    # Graded LR: all 4 labels, ranked by expected gain.
    graded = SkLogReg(C=1 / (1e-4 * len(y_train)), max_iter=2000).fit(X_train, train["label"].to_numpy())
    label_gain = np.array([ESCI_LABEL_GAIN[c] for c in graded.classes_])
    report("logreg_graded", dev, graded.predict_proba(X_dev) @ label_gain, {"model": "logreg_graded"})

    # Listwise LambdaMART + two ablations.
    bm25_feats = [f for f in FEATURES if f.startswith("bm25_")]
    variants = {
        "lambdamart": FEATURES,
        "lambdamart_no_bm25": [f for f in FEATURES if f not in bm25_feats],
        "lambdamart_bm25_only": bm25_feats,
    }
    importances, models = {}, {}
    for name, feats in variants.items():
        model = models[name] = train_lambdamart(train, feats)
        report(name, dev, model.predict(dev[feats]),
               {"model": "lambdamart", "features": feats, "trees": model.best_iteration_})
        importances[name] = dict(sorted(
            zip(feats, model.booster_.feature_importance("gain").round(1).tolist(), strict=True),
            key=lambda kv: -kv[1]))
    (RUNS / "week2_feature_importance.json").write_text(json.dumps(importances, indent=2), encoding="utf-8")
    print("feature importance (gain), full model:", importances["lambdamart"])

    # dev2: a second held-out bucket, scored ONCE to confirm the choice made on dev (selection bias check).
    dev2 = cached("dev2")
    report("logreg_graded_dev2", dev2, graded.predict_proba(scaler.transform(dev2[FEATURES].to_numpy())) @ label_gain,
           {"model": "logreg_graded"}, split="dev2")
    report("lambdamart_dev2", dev2, models["lambdamart"].predict(dev2[FEATURES]),
           {"model": "lambdamart", "trees": models["lambdamart"].best_iteration_}, split="dev2")


if __name__ == "__main__":
    main()
