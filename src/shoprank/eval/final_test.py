"""Section 16: the ONE official test run, exactly as pre-registered in docs/lab_notebook.md (21 Nov 2026).

Usage:
    uv run python -m shoprank.eval.run_task1 --ranker bm25 --fields title --split test --allow-test \
        --name bm25_title_test
    uv run python -m shoprank.eval.final_test --allow-test

Systems (declared before any test score was seen; no choice is made from these numbers):
    primary    deberta_oof_avg5   mean of the 5 DeBERTa-v3-base fold models (title, brand, colour, bullets)
    second     stack_lgbm         LightGBM stack on 14 features + DeBERTa/MiniLM OOF scores
    reported   lambdamart, minilm_ft_mse_fast, minilm_distil_w10_bullets (served model), bm25_title (above)
Every model and the stacker are already frozen: the stacker is re-fitted with the same seed and data, and its
dev nDCG is printed to prove it is the same model as on dev.
Per-pair model scores are cached in data/processed/scores/*_test.parquet, so a crash resumes without
re-scoring. The per-query run files (<system>_test) are written once; a second run is refused.
Every nDCG is re-checked with pytrec_eval (the trec_eval C code), an implementation independent of ours.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from shoprank.cross.finetune import MODELS
from shoprank.cross.score import SCORES, score_frame, score_pairs
from shoprank.eval.run_task1 import RUNS, compare
from shoprank.features.build import FEATURES, cached, load_split
from shoprank.ltr import stack
from shoprank.ltr.train import per_query_results, report, train_lambdamart

FOLD_SETS = {"deberta_oof": ["title", "brand", "color", "bullets"], "minilm_oof": ["title"]}
SINGLE_RUNS = {"minilm_ft_mse_fast": ["title"], "minilm_distil_w10_bullets": ["title", "brand", "color", "bullets"]}
GUARD_RUN = "deberta_oof_avg5_test"
TEST_BATCH = 64  # 128 ran out of GPU memory on DeBERTa with bullets (21 Nov)
TREC_SCALE = 100  # trec_eval needs integer grades: E/S/C/I = 100/10/1/0; nDCG is unchanged by scaling gains


def check_once(force: bool, runs_dir=RUNS) -> None:
    if (runs_dir / f"{GUARD_RUN}.json").exists() and not force:
        raise SystemExit("the official test run was already made; it is scored once (pass --force only to "
                         "re-create lost files, with a written reason)")


def cached_scores(tag: str, model_dir, pairs: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    path = SCORES / f"{tag}_test.parquet"
    if path.exists():
        return pd.read_parquet(path)
    out = score_frame(pairs, score_pairs(model_dir, pairs, fields, batch_size=TEST_BATCH))
    out.to_parquet(path, index=False)
    print(f"  scored {tag} on test ({len(out):,} pairs)", flush=True)
    return out


def trec_ndcg(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Per-query full-list nDCG and nDCG@10 from pytrec_eval. Note: trec_eval breaks score ties by document id
    in reverse order, ours by product_id ascending, so tied scores (BM25 zeros) can differ slightly."""
    import pytrec_eval

    qrels, run = {}, {}
    for qid, pid, gain, score in zip(df["query_id"], df["product_id"], df["gain"], df[score_col], strict=True):
        qrels.setdefault(str(qid), {})[str(pid)] = round(float(gain) * TREC_SCALE)
        run.setdefault(str(qid), {})[str(pid)] = float(score)
    res = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg", "ndcg_cut_10"}).evaluate(run)
    return pd.DataFrame([{"query_id": int(q), "ndcg_full": m["ndcg"], "ndcg10": m["ndcg_cut_10"]}
                         for q, m in res.items()])


def save_system(name: str, pairs: pd.DataFrame, scores: np.ndarray, config: dict) -> dict:
    """Save `<name>_test` and re-check its nDCG with trec_eval."""
    scored = pairs[["query_id", "product_id", "esci_label", "gain"]].assign(score=np.asarray(scores))
    summary = report(f"{name}_test", scored, scored["score"].to_numpy(), config, "test")
    ours = pd.DataFrame(per_query_results(scored, "score")).set_index("query_id")
    trec = trec_ndcg(scored, "score").set_index("query_id").loc[ours.index]
    diff = (ours["ndcg_full"] - trec["ndcg_full"]).abs()
    print(f"    trec_eval nDCG full {trec['ndcg_full'].mean():.4f} (ours {ours['ndcg_full'].mean():.4f}; "
          f"max per-query diff {diff.max():.2e}, queries differing > 1e-6: {(diff > 1e-6).sum()})", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="One-time official test run (section 16)")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not args.allow_test:
        raise SystemExit("test split is locked: pass --allow-test for the one pre-registered run")
    check_once(args.force)

    pairs, _ = load_split("test")
    print(f"official test: {pairs['query_id'].nunique():,} queries / {len(pairs):,} pairs", flush=True)
    SCORES.mkdir(parents=True, exist_ok=True)
    for name, fields in FOLD_SETS.items():
        for k in range(stack.N_FOLDS):
            cached_scores(f"{name}_fold{k}", MODELS / f"{name}_fold{k}", pairs, fields)
    single = {run: cached_scores(run, MODELS / run, pairs, fields) for run, fields in SINGLE_RUNS.items()}

    train, test = cached("train"), cached("test")
    lm = train_lambdamart(train, FEATURES)
    save_system("lambdamart", test, lm.predict(test[FEATURES]), {"model": "lambdamart", "trees": lm.best_iteration_})
    for run, frame in single.items():
        save_system(run, frame, frame["score"].to_numpy(), {"model": run})

    test_frame = stack.frame("test")
    save_system("deberta_oof_avg5", test_frame, test_frame["deberta_oof"].to_numpy(),
                {"model": "deberta_oof", "folds": stack.N_FOLDS, "combine": "mean"})
    train_frame, qcols = stack.add_query_features(stack.frame("train"), list(stack.OOF_MODELS))
    feats = [*FEATURES, *stack.OOF_MODELS, *qcols]
    stacker = train_lambdamart(train_frame, feats)
    dev_frame, _ = stack.add_query_features(stack.frame("dev"), list(stack.OOF_MODELS))
    print(f"  stacker re-fit check: dev nDCG {stack.mean_ndcg(dev_frame.assign(s=stacker.predict(dev_frame[feats])), 's'):.4f}"
          " (dev run: 0.8879)", flush=True)
    test_frame, _ = stack.add_query_features(test_frame, list(stack.OOF_MODELS))
    save_system("stack_lgbm", test_frame, stacker.predict(test_frame[feats]),
                {"model": "lambdamart_stack", "features": feats, "trees": stacker.best_iteration_})

    print("\npaired bootstrap 95% CIs on test:")
    for a, b in [("bm25_title_test", "deberta_oof_avg5_test"), ("lambdamart_test", "deberta_oof_avg5_test"),
                 ("deberta_oof_avg5_test", "stack_lgbm_test"), ("minilm_ft_mse_fast_test", "minilm_distil_w10_bullets_test"),
                 ("bm25_title_test", "minilm_distil_w10_bullets_test"),
                 ("minilm_distil_w10_bullets_test", "deberta_oof_avg5_test")]:
        if (RUNS / f"{a}.parquet").exists():
            compare(a, b)


if __name__ == "__main__":
    main()
