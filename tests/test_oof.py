"""Out-of-fold folds: every train query in exactly one fold, folds by query (never split a query)."""
import numpy as np
import pandas as pd

from shoprank.cross.oof import query_fold
from shoprank.ltr.train import is_holdout


def test_each_query_gets_one_fold_and_its_rows_stay_together():
    qids = pd.Series(np.repeat(np.arange(5000), 3))
    folds = query_fold(qids, n_folds=5)
    assert set(folds.unique()) == {0, 1, 2, 3, 4}
    assert folds.groupby(qids).nunique().max() == 1


def test_folds_are_balanced_and_repeatable():
    qids = pd.Series(np.arange(10000))
    folds = query_fold(qids, n_folds=5)
    share = folds.value_counts(normalize=True)
    assert share.min() > 0.18 and share.max() < 0.22
    assert (folds == query_fold(qids, n_folds=5)).all()


def test_folds_do_not_line_up_with_the_lambdamart_holdout():
    qids = pd.Series(np.arange(10000))
    folds, hold = query_fold(qids, n_folds=5), is_holdout(qids)
    for k in range(5):  # every fold holds some hold-out queries and some not
        assert 0.05 < hold[folds == k].mean() < 0.15
