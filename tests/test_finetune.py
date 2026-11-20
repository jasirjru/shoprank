"""Fine-tune loop, tested with a tiny fake model and tokenizer (no download, no GPU)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from shoprank.cross.finetune import (
    FitConfig,
    batch_indices,
    fit,
    predict,
    warmup_linear,
)
from shoprank.ltr.train import is_holdout


class FakeTokenizer:
    """Encodes a (query, text) pair as one number: the text length."""

    def __call__(self, queries, texts, **kwargs):
        return {"x": torch.tensor([[float(len(t))] for t in texts])}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return SimpleNamespace(logits=self.linear(x / 10.0))


def test_warmup_linear_rises_then_falls_to_zero():
    assert warmup_linear(0, warmup=10, total=110) == 0.0
    assert warmup_linear(5, warmup=10, total=110) == 0.5
    assert warmup_linear(10, warmup=10, total=110) == 1.0
    assert warmup_linear(60, warmup=10, total=110) == 0.5
    assert warmup_linear(110, warmup=10, total=110) == 0.0


def test_batch_indices_cover_every_row_once_and_repeat_with_seed():
    a = batch_indices(10, batch_size=4, seed=42)
    assert [len(b) for b in a] == [4, 4, 2]
    assert sorted(np.concatenate(a).tolist()) == list(range(10))
    assert all((x == y).all() for x, y in zip(a, batch_indices(10, 4, seed=42), strict=True))


def test_is_holdout_is_one_in_ten_queries_and_keeps_query_rows_together():
    qids = pd.Series(np.repeat(np.arange(2000), 3))
    mask = is_holdout(qids)
    assert 0.07 < mask.mean() < 0.13
    assert mask.groupby(qids).nunique().max() == 1  # a query is never split


def test_predict_returns_one_score_per_pair_in_row_order():
    model = FakeModel()
    scores = predict(model, FakeTokenizer(), ["q"] * 5, ["a", "bbb", "bb", "bbbb", "bbbbb"],
                     batch_size=2, max_length=8, device="cpu")
    assert scores.shape == (5,)
    order = np.argsort(scores)  # linear in text length, so order is by length (either direction)
    assert order.tolist() in ([0, 2, 1, 3, 4], [4, 3, 1, 2, 0])


def test_fit_lowers_mse_and_keeps_the_best_checkpoint():
    torch.manual_seed(0)
    texts = ["x" * n for n in range(1, 41)]
    targets = np.array([n / 40 for n in range(1, 41)])  # learnable: gain grows with length
    model = FakeModel()
    before = np.mean((predict(model, FakeTokenizer(), [""] * 40, texts, 8, 8, "cpu") - targets) ** 2)

    seen_steps = []

    def evaluate(m):  # higher is better; reward low MSE on the training pairs
        seen_steps.append(True)
        return -float(np.mean((predict(m, FakeTokenizer(), [""] * 40, texts, 8, 8, "cpu") - targets) ** 2))

    cfg = FitConfig(lr=0.05, warmup_steps=2, batch_size=8, epochs=20, eval_every=10, max_length=8,
                    seed=42, bf16=False)
    history = fit(model, FakeTokenizer(), [""] * 40, texts, targets, cfg, "cpu", evaluate)

    after = np.mean((predict(model, FakeTokenizer(), [""] * 40, texts, 8, 8, "cpu") - targets) ** 2)
    assert after < before / 2
    assert len(seen_steps) == len(history["evals"]) == 10  # 100 steps / eval_every 10
    best = max(history["evals"], key=lambda e: e["score"])
    assert history["best_step"] == best["step"]
    assert -after == pytest.approx(best["score"])  # model ends holding the best weights
