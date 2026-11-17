"""Two-tower model + InfoNCE loss. Week 3, Part D.

    query text   ──► query tower   ──► 128 numbers (a point in space)  ┐
                                                                       ├─ cosine similarity = score
    product text ──► product tower ──► 128 numbers                     ┘

Training pulls each query close to its Exact product and pushes it away from the OTHER products
in the same batch ("in-batch negatives") — free negatives, no extra labelling.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from shoprank.dense.hashing import N_BUCKETS


class Tower(nn.Module):
    def __init__(self, dim: int = 128, n_buckets: int = N_BUCKETS) -> None:
        super().__init__()
        # One learnable vector per hash bucket; a text = the average of its pieces' vectors.
        self.embed = nn.EmbeddingBag(n_buckets, dim, mode="mean")
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, flat_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # L2-normalize: every vector has length 1, so dot product = cosine similarity.
        return F.normalize(self.mlp(self.embed(flat_ids, offsets)), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.query = Tower(dim)    # queries are short and messy
        self.product = Tower(dim)  # titles are longer and structured -> separate weights


def info_nce(q: torch.Tensor, p: torch.Tensor, temperature: float,
             same_query: torch.Tensor | None = None) -> torch.Tensor:
    """Contrastive loss: row i of q should match column i of p, not the other columns.

    logits[i, j] = similarity(query i, product j) / temperature
    loss = mean over i of  -log( exp(logits[i,i]) / sum_j exp(logits[i,j]) )   (softmax cross-entropy)

    same_query[i, j] = True when rows i and j come from the same query. Then product j is ALSO a
    correct answer for query i, so it must not be punished as a negative ("false negative") -> masked.
    """
    logits = q @ p.T / temperature
    if same_query is not None:
        off_diagonal = ~torch.eye(len(q), dtype=torch.bool, device=q.device)
        logits = logits.masked_fill(same_query & off_diagonal, float("-inf"))
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # log softmax, numerically safe
    return -log_probs.diagonal().mean()
