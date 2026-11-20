import random

import torch
import torch.nn.functional as F

from shoprank.dense.hashing import EMPTY_ID, N_BUCKETS, pack, text_ids
from shoprank.dense.model import TwoTower, info_nce
from shoprank.dense.train import sample_hard_negatives


def test_misspelling_shares_pieces():
    anna, ana, laptop = set(text_ids("anna")), set(text_ids("ana")), set(text_ids("laptop"))
    assert len(anna & ana) >= 2       # "#an" and "na#" are shared
    assert not (anna & laptop)


def test_ids_are_stable_and_in_range():
    assert text_ids("Red Shoe") == text_ids("red shoe")
    assert all(0 < i < N_BUCKETS for i in text_ids("iphone 12 pro"))
    assert text_ids(None) == [EMPTY_ID]


def test_pack_offsets():
    flat, offsets = pack([[5, 6], [7], [8, 9, 10]])
    assert flat.tolist() == [5, 6, 7, 8, 9, 10]
    assert offsets.tolist() == [0, 2, 3]


def test_info_nce_equals_cross_entropy():
    torch.manual_seed(0)
    q = F.normalize(torch.randn(8, 16), dim=-1)
    p = F.normalize(torch.randn(8, 16), dim=-1)
    expected = F.cross_entropy(q @ p.T / 0.1, torch.arange(8))
    assert torch.allclose(info_nce(q, p, 0.1), expected, atol=1e-6)


def test_matching_pairs_give_lower_loss():
    torch.manual_seed(0)
    q = F.normalize(torch.randn(8, 16), dim=-1)
    assert info_nce(q, q, 0.1) < info_nce(q, q.roll(1, dims=0), 0.1)


def test_same_query_products_are_not_punished():
    q = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), dim=-1)  # same query twice
    p = q.clone()                                                    # two Exact products, identical
    same = torch.ones(2, 2, dtype=torch.bool)
    assert info_nce(q, p, 0.1, same_query=same).item() == 0.0       # nothing to push apart
    assert info_nce(q, p, 0.1).item() > 0.5                          # unmasked: wrongly penalized


def test_tiny_model_can_overfit_one_batch():
    """Classic sanity check: if a model cannot memorize 8 examples, the training code is broken."""
    torch.manual_seed(0)
    model = TwoTower(dim=16)
    texts = [f"product {w}" for w in ["red", "blue", "green", "black", "white", "gold", "pink", "gray"]]
    qs, ps = pack([text_ids(t.split()[1]) for t in texts]), pack([text_ids(t) for t in texts])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(200):
        loss = info_nce(model.query(*qs), model.product(*ps), temperature=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_extra_negative_columns_are_plain_cross_entropy_over_all_products():
    torch.manual_seed(0)
    q = F.normalize(torch.randn(4, 16), dim=-1)
    p = F.normalize(torch.randn(6, 16), dim=-1)  # 4 positives + 2 extra hard negatives
    expected = F.cross_entropy(q @ p.T / 0.1, torch.arange(4))
    no_mask = torch.zeros(4, 6, dtype=torch.bool)
    assert torch.allclose(info_nce(q, p, 0.1, same_query=no_mask), expected, atol=1e-6)


def test_a_hard_negative_close_to_the_query_raises_the_loss():
    q = torch.eye(2)
    easy = torch.cat([q, torch.tensor([[0.0, -1.0]])])          # extra product far from query 0
    hard = torch.cat([q, F.normalize(torch.tensor([[1.0, 0.1]]), dim=-1)])  # extra product near query 0
    assert info_nce(q, hard, 0.1) > info_nce(q, easy, 0.1)


def test_sample_hard_negatives_uses_only_the_same_querys_non_exact_products():
    hard_by_qid = {7: [[1], [2]], 8: [[3]]}  # query 9 has no non-Exact products
    rows, ids = sample_hard_negatives([7, 9, 8, 7], hard_by_qid, random.Random(0))
    assert rows == [0, 2, 3]  # row 1 (query 9) is skipped
    assert ids[0] in ([1], [2]) and ids[1] == [3] and ids[2] in ([1], [2])
    again = sample_hard_negatives([7, 9, 8, 7], hard_by_qid, random.Random(0))
    assert (rows, ids) == again  # same seed, same choice
