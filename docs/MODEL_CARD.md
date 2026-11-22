# Model card: ShopRank served re-ranker (`minilm_distil_w10_bullets`, INT8 ONNX)

## What it does
Given a shopping query and a list of candidate products (title, optional brand, colour, bullet points), it
scores each product's relevance so the list can be sorted best first. It re-ranks; it does not retrieve.

## Model
- Architecture: cross-encoder, MiniLM-L12 (`cross-encoder/ms-marco-MiniLM-L-12-v2` start point), 33M
  parameters, one output score. Input: `query [SEP] title brand colour bullets`, cut at 128 tokens.
- Training: ESCI Task 1 train, US English, 323,302 pairs (~16,000 queries, ~77% of the official training
  queries; the rest is dev, dev2 and a 10% early-stopping hold-out). Target (knowledge distillation): for each
  pair, the score of the DeBERTa-v3-base fold model that never saw that query (out-of-fold), MSE loss,
  1 epoch, lr 7e-6 with 5,000 linear warm-up steps, batch 32, bf16, seed 42.
- Serving: ONNX Runtime, transformer kernel fusion, dynamic INT8 weights (per channel), 32 MB file.

## Evaluation
| Data | Metric | Value |
|---|---|---|
| Official ESCI Task 1 test, US (8,956 queries), PyTorch model | nDCG, full list | 0.8674 [0.8643, 0.8703] |
| same | nDCG@10 | 0.7743 [0.7702, 0.7783] |
| dev (2,001 held-out training queries), INT8 ONNX | nDCG, full list | 0.8654 (fp32 ONNX 0.8662) |
Gains: Exact 1, Substitute 0.1, Complement 0.01, Irrelevant 0. 95% CIs: query bootstrap.
For comparison on the same test: BM25 0.8202, Amazon's MiniLM recipe 0.8544, the DeBERTa teacher 0.8877.
Ready-made rerankers, zero-shot on dev (same product text): BGE-reranker-v2-m3 (568M) 0.8585 and
Qwen3-Reranker-0.6B (596M) 0.8540, both below this model's 0.8667 (paired CIs exclude 0), at 17x its size.

## Speed (laptop CPU, 16 threads, one model call at a time)
p50 69 ms, p95 232 ms for one request of ~16 candidates; ~230 pairs/s (~11 requests/s) per process.

## Download and use
Published files (identical in both places, SHA-256 in `SHA256SUMS`): `model_int8.onnx`, `tokenizer.json`,
`tokenizer_config.json`, `serving.json` (input fields, max length), `quality.json` (dev results), and since
v1.1 `spell_vocab.json` (catalogue word counts for query spell correction).
- Hugging Face: https://huggingface.co/jasirjru/shoprank-reranker-minilm-int8
- GitHub release v1.1: https://github.com/jasirjru/shoprank/releases/tag/v1.1
- In this repo: `uv run python -m shoprank.serve.fetch` downloads them into the folder the API reads and
  checks every SHA-256.

Without this repo (onnxruntime + transformers only):
```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

path = snapshot_download("jasirjru/shoprank-reranker-minilm-int8")
tokenizer = AutoTokenizer.from_pretrained(path)
session = ort.InferenceSession(f"{path}/model_int8.onnx", providers=["CPUExecutionProvider"])
query = "usb c cable 6ft"
products = ["Anker USB C to USB C cable, 6 ft Anker Black", "HDMI cable 10 ft"]  # title brand colour bullets
enc = tokenizer([query] * len(products), products, truncation=True, max_length=128, padding=True,
                return_tensors="np")
scores = session.run(["logits"], {k: v.astype(np.int64) for k, v in enc.items()})[0].ravel()
print(sorted(zip(scores, products), reverse=True))  # higher score = more relevant
```
Build each product text exactly as in training: title, brand, colour, bullets joined by single spaces,
missing fields left out.

## Intended use
Re-ranking a short candidate list (up to 200 products) for English product-search queries, e.g. after a
keyword or vector retrieval step. A teaching and portfolio project, not a production system.

## Not intended / known limits
- Other languages (ES, JP data exist in ESCI but were not used), other domains than Amazon-style products.
- Retrieval from a full catalogue: it must be given candidates.
- The ranker alone is sensitive to typos: one swapped letter pair in the longest query word lowers dev nDCG
  from 0.8654 to 0.8392 (INT8). Since v1.1 the API corrects the query first against the catalogue vocabulary
  (`spell_vocab.json`, one-edit corrections), which wins back 87-90% of the loss on dev and dev2 with no harm
  to clean queries; misspellings common in titles are not fixed, and rare non-English words can be wrongly
  "corrected"; words with attached punctuation or digits are left alone. The vocabulary is built from the
  titles, brands and colours of all 1.2M catalogue products (product text only, no labels or queries). Missing
  brand, colour and bullets cost ~0.01; upper/lower case makes no difference.
- Weaker on negation queries ("without", "not"): dev nDCG 0.8378 on 158 such queries vs 0.8667 overall. The
  model does read the negation word (deleting it changes the ranking a lot), so the cause is something else.
- Labels are noisy (ESCI paper: 91% annotator agreement). In a 100-pair sample chosen to find errors (the
  teacher's most extreme disagreements, judged by an LLM reviewer and spot-checked), 75 labels looked wrong.
  Label noise limits how high nDCG can go; how high exactly was not measured.
- Scores are relevance scores for ranking within one query, not calibrated probabilities; do not compare
  them across queries or use them as thresholds without calibration.
- Bias and safety: trained on Amazon US queries and human labels, it inherits their brand and popularity
  patterns; no fairness analysis was done.

## Data and licence
Shopping Queries Dataset (Reddy et al., 2022), Apache-2.0. Base model `cross-encoder/ms-marco-MiniLM-L-12-v2`
(Apache-2.0). Teacher `microsoft/deberta-v3-base` (MIT).
