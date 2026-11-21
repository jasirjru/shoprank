"""Section 16 one-time test run: the once-guard and the independent trec_eval re-check."""
import numpy as np
import pandas as pd
import pytest

from shoprank.eval.final_test import GUARD_RUN, check_once, trec_ndcg
from shoprank.ltr.train import per_query_results


def test_second_test_run_is_refused_unless_forced(tmp_path):
    check_once(force=False, runs_dir=tmp_path)  # first run allowed
    (tmp_path / f"{GUARD_RUN}.json").write_text("{}")
    with pytest.raises(SystemExit):
        check_once(force=False, runs_dir=tmp_path)
    check_once(force=True, runs_dir=tmp_path)


def test_trec_eval_agrees_with_our_ndcg_when_scores_have_no_ties():
    rng = np.random.default_rng(0)
    n_q, per_q = 40, 8
    labels = rng.choice(list("ESCI"), size=n_q * per_q)
    df = pd.DataFrame({
        "query_id": np.repeat(np.arange(n_q), per_q),
        "product_id": [f"p{i:04d}" for i in range(n_q * per_q)],
        "esci_label": labels,
        "gain": pd.Series(labels).map({"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}).to_numpy(),
        "score": rng.permutation(n_q * per_q).astype(float),  # all distinct
    })
    ours = pd.DataFrame(per_query_results(df, "score")).set_index("query_id").sort_index()
    trec = trec_ndcg(df, "score").set_index("query_id").sort_index()
    assert np.allclose(ours["ndcg_full"], trec["ndcg_full"], atol=1e-6)
    assert np.allclose(ours["ndcg10"], trec["ndcg10"], atol=1e-6)
