# ShopRank

A learned product search and ranking engine on the Amazon Shopping Queries (ESCI) dataset:
BM25 + fine-tuned bi-encoder retrieval, LightGBM LambdaMART, a cross-encoder reranker, and a small
QLoRA query parser — every stage measured against human relevance labels with confidence intervals.

Status: Week 1 (data, metrics, BM25). Plan and rationale:
`C:\Studio\AI-ML\Projects\_research\2026-11-flagship-ai-ml-project.md`.

## Setup

```bash
uv sync
uv run python -m shoprank.data.download   # ~1.16 GB, resumable, writes data/raw/manifest.json
uv run python -m shoprank.data.ingest     # US-English tables + train/dev/test split
uv run pytest -q
```

## Results

| System | dev nDCG (full list) | dev nDCG@10 | full-list Δ vs previous (95% CI) | test nDCG (full list) |
|---|---|---|---|---|
| Random | 0.7500 | 0.5786 | — | — |
| BM25 (title, k1 1.5, b 0.75) | 0.8193 | 0.6907 | +0.0693 [+0.0622, +0.0763] | — |
| Logistic regression, 14 features (from scratch) | 0.8296 | 0.7069 | +0.0104 [+0.0067, +0.0141] | — |
| LambdaMART, 14 features | 0.8359 | 0.7167 | +0.0062 [+0.0032, +0.0093] | — |
| Two-tower from scratch (char-trigram hashing, InfoNCE) | 0.7849 | 0.6405 | −0.0343 [−0.0403, −0.0283] vs BM25 | — |

Dev = 2,099 ESCI Task-1 US queries. Field and parameter ablations: `docs/learn/week01-notes.md` §5;
feature ablations: `docs/learn/week02-notes.md` page 6.

Public reference (Task 1 test): official baseline 0.850, KDD Cup 2022 winner 0.9043. Those use the
official metric — trec_eval `ndcg.1=0,2=0.01,3=0.1,4=1` over the **whole candidate list** — averaged
over US, ES and JP. Our "nDCG (full list)" uses the same gains and no cutoff, but US only, so the
comparison is indicative, not exact.

## Data

Shopping Queries Dataset (Reddy et al., 2022), Apache-2.0: https://github.com/amazon-science/esci-data
