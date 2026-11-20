"""Fine-tune loop, tested with a tiny fake model and tokenizer (no download, no GPU)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from shoprank.cross.finetune import (
    FitConfig,
    Prepared4DMask,
    class_ids,
    collate,
    compute_loss,
    fit,
    length_bucketed_batches,
    predict,
    score_from_logits,
    warmup_linear,
)
from shoprank.ltr.train import is_holdout


class FakeTokenizer:
    """One token per text character, unpadded, like a real tokenizer called without padding."""

    pad_token_id = 0

    def __call__(self, queries, texts, truncation=True, max_length=None, **kwargs):
        ids = [[7] * min(len(t), max_length or len(t)) for t in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}


class FakeModel(torch.nn.Module):
    """Scores a pair from its real (unpadded) length, so padding must not change the score."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, input_ids, attention_mask):
        length = attention_mask.sum(dim=1, keepdim=True).float()
        return SimpleNamespace(logits=self.linear(length / 10.0))


def padded_tokens(batches, lengths):
    return sum(lengths[b].max() * len(b) for b in batches)


def test_warmup_linear_rises_then_falls_to_zero():
    assert warmup_linear(0, warmup=10, total=110) == 0.0
    assert warmup_linear(5, warmup=10, total=110) == 0.5
    assert warmup_linear(10, warmup=10, total=110) == 1.0
    assert warmup_linear(60, warmup=10, total=110) == 0.5
    assert warmup_linear(110, warmup=10, total=110) == 0.0


def test_length_bucketed_batches_cover_every_row_once_and_repeat_with_seed():
    lengths = np.random.default_rng(0).integers(1, 100, size=1000)
    a = length_bucketed_batches(lengths, batch_size=32, seed=42, chunk_batches=4)
    assert sorted(len(b) for b in a)[1:] == [32] * 31  # one short batch (1000 = 31 x 32 + 8)
    assert sorted(np.concatenate(a).tolist()) == list(range(1000))
    b = length_bucketed_batches(lengths, batch_size=32, seed=42, chunk_batches=4)
    assert all((x == y).all() for x, y in zip(a, b, strict=True))


def test_length_bucketed_batches_cut_padding_but_keep_batch_order_random():
    lengths = np.random.default_rng(0).integers(1, 100, size=3200)
    rng = np.random.default_rng(42).permutation(3200)
    random_batches = [rng[i:i + 32] for i in range(0, 3200, 32)]
    bucketed = length_bucketed_batches(lengths, batch_size=32, seed=42, chunk_batches=10)
    assert padded_tokens(bucketed, lengths) < 0.7 * padded_tokens(random_batches, lengths)
    means = [lengths[b].mean() for b in bucketed]
    assert means != sorted(means)  # not short-first: the model still sees lengths in random order


def test_collate_pads_to_the_longest_row_in_the_batch_not_to_max_length():
    feats = {"input_ids": [[5, 6], [5, 6, 7, 8, 9], [5, 6, 7]], "attention_mask": [[1] * 2, [1] * 5, [1] * 3]}
    out = collate(feats, np.array([0, 2]), pad_id=0, device="cpu")
    assert out["input_ids"].tolist() == [[5, 6, 0], [5, 6, 7]]
    assert out["attention_mask"].tolist() == [[1, 1, 0], [1, 1, 1]]


def test_prepared_4d_mask_passes_a_boolean_key_mask_and_keeps_other_inputs():
    seen = {}

    class Inner(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            seen["ids"], seen["mask"] = input_ids, attention_mask
            return SimpleNamespace(logits=torch.zeros(len(input_ids), 1))

    ids, mask = torch.tensor([[5, 6, 0], [5, 6, 7]]), torch.tensor([[1, 1, 0], [1, 1, 1]])
    Prepared4DMask(Inner())(input_ids=ids, attention_mask=mask)
    assert seen["ids"] is ids
    assert seen["mask"].shape == (2, 1, 1, 3) and seen["mask"].dtype == torch.bool
    assert seen["mask"][:, 0, 0, :].tolist() == [[True, True, False], [True, True, True]]


def _tiny_bert(attn: str):
    from transformers import BertConfig, BertForSequenceClassification

    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=32, num_labels=1)
    cfg._attn_implementation = attn
    return BertForSequenceClassification(cfg).eval()


def test_prepared_4d_mask_gives_the_same_logits_as_a_2d_mask_on_a_real_bert():
    model = _tiny_bert("sdpa")
    ids = torch.tensor([[2, 5, 6, 0, 0], [2, 5, 6, 7, 8]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    with torch.no_grad():
        plain = model(input_ids=ids, attention_mask=mask).logits
        wrapped = Prepared4DMask(model)(input_ids=ids, attention_mask=mask).logits
        unpadded = model(input_ids=ids[:1, :3], attention_mask=mask[:1, :3]).logits
    assert torch.allclose(plain, wrapped, atol=1e-6)
    assert torch.allclose(wrapped[0], unpadded[0], atol=1e-5)  # padding really is ignored


def test_prepared_4d_mask_refuses_eager_attention_where_a_bool_mask_would_be_added():
    with pytest.raises(ValueError, match="sdpa"):
        Prepared4DMask(_tiny_bert("eager"))


def test_score_from_one_logit_is_the_logit_itself():
    assert score_from_logits(torch.tensor([[0.3], [-2.0]])).tolist() == pytest.approx([0.3, -2.0])


def test_score_from_four_logits_is_the_expected_gain():
    sure_exact = torch.tensor([[50.0, 0.0, 0.0, 0.0]])
    uniform = torch.zeros(1, 4)
    assert score_from_logits(sure_exact).item() == pytest.approx(1.0)
    assert score_from_logits(uniform).item() == pytest.approx((1.0 + 0.1 + 0.01 + 0.0) / 4)


def test_compute_loss_is_mse_on_gain_or_cross_entropy_on_class():
    logits1, gains = torch.tensor([[0.5], [0.0]]), torch.tensor([1.0, 0.0])
    assert compute_loss(logits1, gains, "mse").item() == pytest.approx(0.125)
    logits4, classes = torch.tensor([[2.0, 0.0, 0.0, 0.0]]), torch.tensor([1])
    expected = torch.nn.functional.cross_entropy(logits4, classes).item()
    assert compute_loss(logits4, classes, "ce").item() == pytest.approx(expected)
    with pytest.raises(ValueError):
        compute_loss(logits1, gains, "hinge")


def test_class_ids_follow_e_s_c_i_order():
    assert class_ids(pd.Series(["E", "S", "C", "I", "E"])).tolist() == [0, 1, 2, 3, 0]
    with pytest.raises(ValueError):
        class_ids(pd.Series(["E", "X"]))


def test_fit_with_cross_entropy_learns_the_classes():
    class FourWay(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 4)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.linear(attention_mask.sum(1, keepdim=True).float() / 10.0))

    torch.manual_seed(0)
    texts = ["x" * n for n in range(1, 41)]
    classes = np.array([0 if n > 30 else 3 for n in range(1, 41)])  # long texts are Exact
    model = FourWay()
    cfg = FitConfig(lr=0.05, warmup_steps=2, batch_size=8, epochs=30, eval_every=50, max_length=64,
                    seed=42, bf16=False, loss="ce")
    fit(model, FakeTokenizer(), [""] * 40, texts, classes, cfg, "cpu", lambda m: 0.0)
    scores = predict(model, FakeTokenizer(), [""] * 40, texts, 8, 64, "cpu")
    assert scores[classes == 0].min() > scores[classes == 3].max()  # every Exact ranked above every Irrelevant


def test_is_holdout_is_one_in_ten_queries_and_keeps_query_rows_together():
    qids = pd.Series(np.repeat(np.arange(2000), 3))
    mask = is_holdout(qids)
    assert 0.07 < mask.mean() < 0.13
    assert mask.groupby(qids).nunique().max() == 1  # a query is never split


def test_predict_returns_one_score_per_pair_in_row_order():
    model = FakeModel()
    scores = predict(model, FakeTokenizer(), ["q"] * 5, ["a", "bbb", "bb", "bbbb", "bbbbb"],
                     batch_size=2, max_length=64, device="cpu")
    assert scores.shape == (5,)
    order = np.argsort(scores)  # linear in text length, so order is by length (either direction)
    assert order.tolist() in ([0, 2, 1, 3, 4], [4, 3, 1, 2, 0])


def test_fit_lowers_mse_and_keeps_the_best_checkpoint():
    torch.manual_seed(0)
    texts = ["x" * n for n in range(1, 41)]
    targets = np.array([n / 40 for n in range(1, 41)])  # learnable: gain grows with length
    model = FakeModel()
    before = np.mean((predict(model, FakeTokenizer(), [""] * 40, texts, 8, 64, "cpu") - targets) ** 2)

    seen_steps = []

    def evaluate(m):  # higher is better; reward low MSE on the training pairs
        seen_steps.append(True)
        return -float(np.mean((predict(m, FakeTokenizer(), [""] * 40, texts, 8, 64, "cpu") - targets) ** 2))

    cfg = FitConfig(lr=0.05, warmup_steps=2, batch_size=8, epochs=20, eval_every=10, max_length=64,
                    seed=42, bf16=False)
    history = fit(model, FakeTokenizer(), [""] * 40, texts, targets, cfg, "cpu", evaluate)

    after = np.mean((predict(model, FakeTokenizer(), [""] * 40, texts, 8, 64, "cpu") - targets) ** 2)
    assert after < before / 2
    assert len(seen_steps) == len(history["evals"]) == 10  # 100 steps / eval_every 10
    best = max(history["evals"], key=lambda e: e["score"])
    assert history["best_step"] == best["step"]
    assert -after == pytest.approx(best["score"])  # model ends holding the best weights
