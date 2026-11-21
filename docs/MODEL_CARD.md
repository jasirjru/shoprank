# Model card: ShopRank served re-ranker (`minilm_distil_w10_bullets`, INT8 ONNX)

## What it does
Given a shopping query and a list of candidate products (title, optional brand, colour, bullet points), it
scores each product's relevance so the list can be sorted best first. It re-ranks; it does not retrieve.

## Model
- Architecture: cross-encoder, MiniLM-L12 (`cross-encoder/ms-marco-MiniLM-L-12-v2` start point), 33M
  parameters, one output score. Input: `query [SEP] title brand colour bullets`, cut at 128 tokens.
- Training: ESCI Task 1 train, US English, 80% of the official training queries (285k-358k pairs). Target:
  the out-of-fold score of a 5-model DeBERTa-v3-base ensemble (knowledge distillation), MSE loss, 1 epoch,
  lr 7e-6, batch 32, seed 42.
- Serving: ONNX Runtime, transformer kernel fusion, dynamic INT8 weights (per channel), 32 MB file.

## Evaluation
| Data | Metric | Value |
|---|---|---|
| Official ESCI Task 1 test, US (8,956 queries), PyTorch model | nDCG, full list | 0.8674 [0.8643, 0.8703] |
| same | nDCG@10 | 0.7743 [0.7702, 0.7783] |
| dev (2,001 held-out training queries), INT8 ONNX | nDCG, full list | 0.8654 (fp32 ONNX 0.8662) |
Gains: Exact 1, Substitute 0.1, Complement 0.01, Irrelevant 0. 95% CIs: query bootstrap.
For comparison on the same test: BM25 0.8202, Amazon's MiniLM recipe 0.8544, the DeBERTa teacher 0.8877.

## Speed (laptop CPU, 16 threads, one model call at a time)
p50 69 ms, p95 232 ms for one request of ~16 candidates; ~230 pairs/s (~11 requests/s) per process.

## Intended use
Re-ranking a short candidate list (up to 200 products) for English product-search queries, e.g. after a
keyword or vector retrieval step. A teaching and portfolio project, not a production system.

## Not intended / known limits
- Other languages (ES, JP data exist in ESCI but were not used), other domains than Amazon-style products.
- Retrieval from a full catalogue: it must be given candidates.
- Weakest on negation queries ("without", "not"): dev nDCG 0.8378 on 158 such queries vs 0.8667 overall.
- Labels are noisy: in a 100-pair audit of confident disagreements, 75 human labels looked wrong under the
  ESCI guidelines; scores above ~0.90 are not realistic on this data.
- Scores are relevance scores for ranking within one query, not calibrated probabilities; do not compare
  them across queries or use them as thresholds without calibration.
- Bias and safety: trained on Amazon US queries and human labels, it inherits their brand and popularity
  patterns; no fairness analysis was done.

## Data and licence
Shopping Queries Dataset (Reddy et al., 2022), Apache-2.0. Base model `cross-encoder/ms-marco-MiniLM-L-12-v2`
(Apache-2.0). Teacher `microsoft/deberta-v3-base` (MIT).
