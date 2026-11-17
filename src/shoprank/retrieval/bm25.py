"""BM25 keyword ranking, written from the formula.

Spec = tests/test_bm25.py. Notes = docs/learn/week01-notes.md §4.
Source: Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and Beyond", §3.

Formula (Lucene-style IDF, never negative):
    idf(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
    score(q, d) = sum over DISTINCT terms t in q of
                  idf(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
where N = number of docs, n_t = docs containing t, f(t,d) = count of t in d,
|d| = length of d in tokens, avgdl = average doc length.
"""
from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str | None) -> list[str]:
    """Lowercase and split into runs of letters/digits. 'iPhone 12-Pro' -> ['iphone', '12', 'pro']."""
    return TOKEN_RE.findall(text.lower()) if text else []


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1  # how fast repeated words stop helping (saturation)
        self.b = b    # how much long documents are penalized (0 = never, 1 = fully)

    def fit(self, docs: list[list[str]]) -> BM25:
        """Precompute everything scoring needs, once, so each query is fast."""
        if not docs:
            raise ValueError("cannot fit BM25 on an empty corpus")
        self.term_counts = [Counter(doc) for doc in docs]          # f(t, d) for every doc
        self.doc_lengths = [len(doc) for doc in docs]              # |d|
        self.n_docs = len(docs)                                    # N
        self.avgdl = sum(self.doc_lengths) / self.n_docs           # average doc length
        self.doc_freq = Counter(t for doc in docs for t in set(doc))  # n_t: docs containing t
        return self

    def idf(self, term: str) -> float:
        n_t = self.doc_freq.get(term, 0)
        # Rare term -> big value. The "1 +" keeps it positive even for terms in most docs.
        return math.log(1 + (self.n_docs - n_t + 0.5) / (n_t + 0.5))

    def score(self, query: list[str], doc_index: int) -> float:
        counts = self.term_counts[doc_index]
        # Length normalization: >1 for longer-than-average docs, which shrinks their score.
        length_norm = 1 - self.b + self.b * self.doc_lengths[doc_index] / self.avgdl
        total = 0.0
        for term in set(query):  # each distinct query word counts once
            f = counts.get(term, 0)
            if f:
                # f*(k1+1)/(f+k1*norm) rises with f but levels off: repeats have diminishing returns.
                total += self.idf(term) * f * (self.k1 + 1) / (f + self.k1 * length_norm)
        return total

    def top_k(self, query: list[str], k: int) -> list[tuple[int, float]]:
        """Best k (doc_index, score) pairs, highest score first; ties broken by lower doc_index."""
        scored = ((i, self.score(query, i)) for i in range(self.n_docs))
        return sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:k]
