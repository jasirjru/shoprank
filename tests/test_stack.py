"""Section 14 stacking helpers: honest OOF score loading, per-query features, fast nDCG, blend weights."""
import numpy as np
import pandas as pd
import pytest

from shoprank.cross.oof import query_fold
from shoprank.ltr.stack import (
    add_query_features,
    attach_scores,
    fit_blend,
    load_oof_scores,
    mean_ndcg,
    per_query_zscore,
    weight_grid,
)
from shoprank.ltr.train import per_query_results


def _pairs(n_queries: int = 30, per_query: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice(list("ESCI"), size=n_queries * per_query)
    return pd.DataFrame({
        "query_id": np.repeat(np.arange(n_queries), per_query),
        "product_id": [f"p{i}" for i in range(n_queries * per_query)],
        "esci_label": labels,
        "gain": pd.Series(labels).map({"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}).to_numpy(),
        "score": rng.normal(size=n_queries * per_query).round(1),  # rounding makes ties
    })


def test_fast_ndcg_matches_the_reference_per_query_loop_including_ties():
    df = _pairs()
    reference = np.mean([r["ndcg_full"] for r in per_query_results(df, "score")])
    assert mean_ndcg(df, "score") == pytest.approx(reference, abs=1e-12)


def test_zscore_is_per_query_and_a_flat_query_gets_zero():
    df = pd.DataFrame({"query_id": [1, 1, 2, 2], "s": [1.0, 3.0, 5.0, 5.0]})
    assert per_query_zscore(df, "s").tolist() == pytest.approx([-1.0, 1.0, 0.0, 0.0])


def test_query_features_describe_each_score_relative_to_its_query():
    df = pd.DataFrame({"query_id": [1, 1, 1], "s": [0.2, 0.5, 0.9]})
    out, cols = add_query_features(df, ["s"])
    assert set(cols) == {"s_z", "s_pct", "s_gap", "s_qmax", "s_qmed", "s_qmin"}
    assert out["s_gap"].tolist() == pytest.approx([-0.7, -0.4, 0.0])
    assert out["s_pct"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert "s_z" not in df.columns  # input not mutated


def test_weight_grid_sums_to_one_and_covers_the_corners():
    grid = weight_grid(3, step=0.25)
    assert all(sum(w) == pytest.approx(1.0) for w in grid)
    assert (1.0, 0.0, 0.0) in grid and (0.0, 0.0, 1.0) in grid
    assert len(grid) == 15  # C(4 + 2, 2)


def test_fit_blend_puts_the_weight_on_the_informative_score():
    df = _pairs(n_queries=60)
    df["good"] = df["gain"] + np.random.default_rng(1).normal(scale=0.01, size=len(df))
    df["noise"] = np.random.default_rng(2).normal(size=len(df))
    weights, best = fit_blend(df, ["good", "noise"], step=0.25)
    assert weights["good"] >= 0.75
    assert best == pytest.approx(mean_ndcg(df, "good"), abs=0.02)


def _write_folds(tmp_path, split_rows: dict[str, list[tuple[int, str, float]]]) -> None:
    for fname, rows in split_rows.items():
        pd.DataFrame(rows, columns=["query_id", "product_id", "score"]).to_parquet(tmp_path / fname)


def _query_in_fold(fold: int, n_folds: int = 2) -> int:
    qids = pd.Series(np.arange(100))
    return int(qids[query_fold(qids, n_folds) == fold].iloc[0])


def test_train_scores_come_from_the_out_of_fold_files(tmp_path):
    q0, q1 = _query_in_fold(0), _query_in_fold(1)
    _write_folds(tmp_path, {"m_fold0_oof.parquet": [(q0, "a", 0.1)], "m_fold1_oof.parquet": [(q1, "b", 0.2)]})
    out = load_oof_scores("m", "train", n_folds=2, scores_dir=tmp_path).set_index("query_id")["m"]
    assert out.to_dict() == {q0: 0.1, q1: 0.2}


def test_a_fold_file_holding_another_folds_query_is_refused(tmp_path):
    q0 = _query_in_fold(0)
    _write_folds(tmp_path, {"m_fold0_oof.parquet": [(q0, "a", 0.1)], "m_fold1_oof.parquet": [(q0, "a", 0.3)]})
    with pytest.raises(ValueError, match="another fold"):
        load_oof_scores("m", "train", n_folds=2, scores_dir=tmp_path)


def test_dev_scores_are_the_mean_of_the_fold_models(tmp_path):
    _write_folds(tmp_path, {"m_fold0_dev.parquet": [(1, "a", 0.0), (1, "b", 1.0)],
                            "m_fold1_dev.parquet": [(1, "b", 3.0), (1, "a", 2.0)]})
    out = load_oof_scores("m", "dev", n_folds=2, scores_dir=tmp_path).set_index("product_id")["m"]
    assert out.to_dict() == {"a": 1.0, "b": 2.0}
    _write_folds(tmp_path, {"m_fold1_dev.parquet": [(1, "a", 2.0)]})
    with pytest.raises(ValueError):
        load_oof_scores("m", "dev", n_folds=2, scores_dir=tmp_path)


def test_attach_scores_refuses_pairs_without_a_score():
    feats = pd.DataFrame({"query_id": [1, 1], "product_id": ["a", "b"], "x": [0, 1]})
    scores = pd.DataFrame({"query_id": [1], "product_id": ["a"], "m": [0.5]})
    with pytest.raises(ValueError):
        attach_scores(feats, [scores])


def test_attach_scores_refuses_a_query_split_into_two_blocks():
    feats = pd.DataFrame({"query_id": [1, 2, 1], "product_id": ["a", "b", "c"]})
    scores = pd.DataFrame({"query_id": [1, 2, 1], "product_id": ["a", "b", "c"], "m": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="contiguous"):
        attach_scores(feats, [scores])


def test_float32_rounding_noise_in_a_flat_query_is_not_blown_up():
    df = pd.DataFrame({"query_id": [1, 1], "s": np.array([0.5, 0.5 + 1e-8], dtype=np.float32)})
    assert per_query_zscore(df, "s").tolist() == [0.0, 0.0]
