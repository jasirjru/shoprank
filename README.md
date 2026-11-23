# ShopRank

**Live page: [shoprank-jasirjru.vercel.app](https://shoprank-jasirjru.vercel.app)**: real searches ranked by keyword
search and by ShopRank, side by side. Model: [Hugging Face](https://huggingface.co/jasirjru/shoprank-reranker-minilm-int8).

ShopRank re-orders product search results so the right product comes first. On Amazon's official Shopping Queries
test (8,956 searches, scored once) the served model scores nDCG 0.867 against 0.820 for keyword search (BM25), and
puts an exact match at #1 for 74.7% of searches against 62.7%. Compressed to a 32 MB INT8 file (0.001 lower on the
dev set), it ranks a 16-product search in about 70 ms (median) on a laptop CPU. Built with Python, PyTorch, ONNX
Runtime, FastAPI and Docker.

Product search re-ranking on the Amazon Shopping Queries dataset (ESCI, Task 1, US English): from BM25 to a
DeBERTa-v3 cross-encoder ensemble, distilled into a small model served on a CPU. Every test number has a 95%
confidence interval, every run has a prediction written before it (`docs/lab_notebook.md`), and the official
test set was scored once, with the systems declared in advance.

## Results on the official test set (scored once)

ESCI Task 1 test, US, small version: 8,956 queries / 181,701 judged (query, product) pairs. Metric: nDCG over
the full candidate list with the official gains (Exact 1, Substitute 0.1, Complement 0.01, Irrelevant 0).
95% CIs from 10,000 query bootstrap resamples.

| System | Test nDCG [95% CI] | Test nDCG@10 |
|---|---|---|
| BM25 (title) | 0.8202 [0.8170, 0.8232] | 0.6942 |
| LambdaMART, 14 text features | 0.8337 [0.8305, 0.8368] | 0.7172 |
| MiniLM-L12 cross-encoder, Amazon's recipe (our run) | 0.8544 [0.8513, 0.8573] | 0.7527 |
| **MiniLM distilled from DeBERTa, + bullets (served model)** | **0.8674 [0.8643, 0.8703]** | 0.7743 |
| **DeBERTa-v3-base x5 (5-fold models, mean score): primary** | **0.8877 [0.8849, 0.8905]** | 0.8092 |
| LightGBM stack on DeBERTa + MiniLM scores (declared second system) | 0.8883 [0.8855, 0.8910] | 0.8099 |

Paired differences on test: primary vs BM25 +0.0676 [+0.0646, +0.0705]; vs LambdaMART +0.0541
[+0.0513, +0.0568]; stack vs primary +0.0006 [-0.0003, +0.0014] (no clear difference); served model vs
Amazon-recipe MiniLM +0.0130 [+0.0112, +0.0148].
Every test nDCG was re-computed with pytrec_eval (trec_eval): the means agree within 0.0003, and exactly once
tied scores are ordered the same way (ties come from bf16 scores; their order moves the mean by ~0.00005).

**Published references:** Amazon's fine-tuned MiniLM cross-encoder scores 0.857 on the English part of the
KDD Cup 2022 *public* test (ESCI paper, Table 4; trained on the full US train, title only, max length 512); a
public same-setup project ([Anankin77/esci-search-reranking](https://github.com/Anankin77/esci-search-reranking),
LightGBM with MiniLM-L6 features) scores 0.854 on the US test used here. Both lie below the primary system's interval,
but Amazon's number is on a different sample of US test queries, so it is a reference, not a paired comparison.
Our own run of Amazon's recipe (adapted: bf16, length-bucketed batches, max length 128, 90% of our train
split) gives 0.8544, consistent with their 0.857. The KDD Cup 2022 winners (0.904) report on all three
locales (US, ES, JP) with large ensembles, so that number is not comparable.

**Gain convention:** we use the paper's gains (Substitute 0.1, Complement 0.01). Amazon's released evaluation
code swaps them (S 0.01, C 0.1); under that convention our numbers are 0.8724 (top) / 0.8499 (served) /
0.8354 (Amazon's recipe, vs 0.83 in Amazon's README). An independent paper (SQID, 2024) re-ran Amazon's recipe
on this same US test with the paper's gains: 0.8562 (ours: 0.8544). Full comparison with the KDD Cup 2022
field: `docs/analysis/benchmark_position_2026-11-21.md`.

## Serving

`POST /rank` (FastAPI) first fixes typos in the query against the catalogue vocabulary (v1.1, see below), then
re-ranks the candidates with the distilled MiniLM, exported to ONNX and quantised
to INT8 (32 MB; dev nDCG 0.8654 vs 0.8662 fp32). Load test with real dev queries (median 16 candidates, title
+ brand + colour + bullets), laptop CPU (16 cores), client and server on the same machine:

| Model | Parallel requests | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|---|
| INT8 | 1 | 69 ms | 232 ms | 255 ms | 217 pairs/s |
| INT8 | 16 | 1,436 ms | 1,784 ms | 1,950 ms | 230 pairs/s |
| fp32 | 1 | 115 ms | 313 ms | 335 ms | 143 pairs/s |

How to read this: 69 ms is one search at a time. The server deliberately runs one model call at a time (`SHOPRANK_MAX_CONCURRENT=1`), so with 16 searches arriving together they queue, and each waits for the ones ahead of it (about 1.4 s at p50). Throughput stays about 230 pairs/s. Serving many users at once needs more processes or machines, a smaller model, or batching; that has not been built here.

At high load, latency is queueing: one process handles about 11 requests/s. The 5-model DeBERTa ensemble is
0.020 nDCG better but scores ~89 pairs/s on an RTX 4060 GPU vs 230 pairs/s for the INT8 MiniLM on CPU, so it
is used offline as the teacher.

**Run it without training or a GPU.** The served model (INT8 ONNX, 32 MB) and the spell vocabulary are published with
v1.1 on the [GitHub release](https://github.com/jasirjru/shoprank/releases/tag/v1.1) and on
[Hugging Face](https://huggingface.co/jasirjru/shoprank-reranker-minilm-int8); every file is checked by SHA-256.

```bash
# Option 1: Docker (CPU image, ~45 packages, no torch; the build downloads and verifies the model)
docker build -t shoprank . && docker run --rm -p 8000:8000 shoprank

# Option 2: uv, serving packages only (~45 packages, ~200 MB, no torch; Linux/macOS shell)
uv venv .venv-serve && sh scripts/serve_requirements.sh
VIRTUAL_ENV=.venv-serve uv pip install --require-hashes -r requirements-serve.txt
PYTHONPATH=src .venv-serve/bin/python -m shoprank.serve.fetch      # or --source huggingface
PYTHONPATH=src .venv-serve/bin/uvicorn shoprank.serve.app:app --port 8000

# Option 3: the full environment (also trains; installs CUDA torch, several GB)
uv sync --locked && uv run python -m shoprank.serve.fetch && uv run uvicorn shoprank.serve.app:app --port 8000

curl -X POST localhost:8000/rank -H "Content-Type: application/json" \
  -d '{"query": "usb c cable 6ft", "products": [{"product_id": "a", "title": "Anker USB C cable, 6 ft"},
       {"product_id": "b", "title": "HDMI cable 10 ft"}]}'
```
The response's `query_used` is the query after spell correction. `/health` reports `"spell_correction": true`
when the vocabulary was found (in the model folder, else `data/processed/spell_vocab.json`); without it the API
ranks the query as typed. `SHOPRANK_SPELL=0` turns correction off.

### Demo

A Gradio page over the same INT8 model and input rules as `/rank` (local only, no public link):

```bash
uv sync --group demo
uv run --group demo python -m shoprank.serve.demo   # http://127.0.0.1:7860
```
Type a query and one product per line (`title | brand | colour | bullets`); the page shows the list best
first and the scoring time (about 10 ms for 4 products, measured once).

## What mattered (dev, 2,001 queries; paired 95% CIs)

| Change | Effect on dev nDCG |
|---|---|
| Fine-tuned MiniLM cross-encoder vs LambdaMART | +0.019 |
| DeBERTa-v3-base vs MiniLM (same fields) | +0.0244 [+0.0197, +0.0292] |
| + bullet points as input (DeBERTa) | +0.0072 [+0.0039, +0.0106] |
| + brand and colour fields (DeBERTa) | +0.0002 [-0.0029, +0.0032] (nothing) |
| mean of 5 fold models vs 1 model | +0.0028 [+0.0004, +0.0051] |
| z-score blend / LightGBM stack vs the 5-model mean | -0.0003 / +0.0000 (nothing) |
| MiniLM: + bullets | +0.0068 [+0.0032, +0.0104] |
| MiniLM: + learn DeBERTa's out-of-fold scores (distillation) | +0.0064 [+0.0038, +0.0090] |
| 4-class cross-entropy instead of MSE on the gain | -0.0056 [-0.0094, -0.0018] |
| MiniLM: 3.3x more labelled pairs (large ESCI version, title only) | +0.0052 [+0.0023, +0.0082] |
| Typo queries: spell correction before ranking (v1.1) | +0.0252 [+0.0211, +0.0292], 89% of the typo loss; dev2 +0.0287 |
| Clean queries: spell correction before ranking | +0.0003 [-0.0008, +0.0015] (no harm); dev2 +0.0020 |
| Typo queries: train the ranker on misspelled queries instead | +0.0041 [+0.0019, +0.0062], 14% (rejected) |
| Ready-made BGE-reranker-v2-m3 (568M, zero-shot) vs our served 33M MiniLM | -0.0081 [-0.0130, -0.0032] (0.8585 vs 0.8667) |
| Ready-made Qwen3-Reranker-0.6B (596M, zero-shot) vs our served MiniLM | -0.0127 [-0.0178, -0.0076] (0.8540) |

Seed noise (3 seeds): DeBERTa spread 0.0016, MiniLM 0.0003 (title only) and 0.0017 (distilled + bullets); no
change smaller than that is claimed.
Hardest queries: negation ("without", "not"); BM25 0.748, a single DeBERTa 0.839 on dev (mostly ESCI's
built-hard "negations" query source; the model does read the negation word).

## Honest limits

- US English only, small version of Task 1 (re-ranking of judged candidates, no retrieval stage). The trained
  two-tower retriever stayed below BM25 (0.790 vs 0.816 on dev).
- Less training data than the official setup: dev (2,001) and dev2 (1,053 queries), 14.6% of the official
  training queries, are held out for model choices. Each DeBERTa fold model trains on 80% of the remaining
  train split (~68% of the official training queries); the served MiniLM on 323,302 pairs (~77%; a further
  10% of train is its early-stopping hold-out).
- Label noise: in a 100-pair sample of the DeBERTa model's most extreme disagreements with the labels (judged
  against the ESCI guidelines by an LLM reviewer, spot-checked by hand; 50 rows come from 10 queries), 75
  labels looked wrong (`docs/analysis/`). The sample is chosen to find errors, so it does not estimate the
  overall error rate (the ESCI paper reports 91% annotator agreement).
- Some choices rest on one seed (the bullets ablation), checked against the 3-seed spread.
- Scores pass through bf16, so 14% of a query's products can tie; the tie order moves nDCG by ~0.00005.
- The INT8 model was not scored on test (the test set is spent); its dev loss is 0.0008.
- Latency is from one laptop, client on the same machine; no network, no autoscaling.
- Typos: one mistake in a query costs the ranker ~0.028 dev nDCG. Catalogue spell correction (v1.1) wins back
  87-90% of that on dev and dev2 (synthetic one-edit typos), but misses misspellings that are common in product
  titles and can turn rare non-English words into English ones ("llaves" -> "leaves"). Real typo logs were not
  available, so the typos are synthetic. Words with attached punctuation ("cabel,") or digits are never
  corrected. The vocabulary comes from all 1.2M catalogue products' titles, brands and colours, including
  products that appear in test pairs (product text only, no labels or queries).

## Reproduce

```bash
sudo apt-get install -y libgomp1             # Linux only: OpenMP runtime LightGBM needs (not on fresh Ubuntu/WSL)
uv sync --locked
uv run python -m shoprank.data.download      # ~1.16 GB, resumable
uv run python -m shoprank.data.ingest        # US tables + train/dev/dev2/test split (test = official test)
uv run python -m shoprank.features.build     # 14 text features
uv run pytest -q && uv run ruff check src tests
# DeBERTa 5-fold models (~30-35 min each on an RTX 4060), MiniLM folds (~5-6 min each):
for k in 0 1 2 3 4; do
  uv run python -m shoprank.cross.oof --model microsoft/deberta-v3-base --fields title,brand,color,bullets \
      --lr 2e-5 --warmup-steps 1000 --fold $k --name deberta_oof
  uv run python -m shoprank.cross.oof --fold $k --name minilm_oof
done
uv run python -m shoprank.ltr.stack                                    # blend + stack on dev
uv run python -m shoprank.cross.finetune --teacher deberta_oof --teacher-weight 1.0 \
    --fields title,brand,color,bullets --name minilm_distil_w10_bullets
uv run python -m shoprank.serve.spell build                            # spell vocabulary (needed before export)
uv run python -m shoprank.serve.onnx_export minilm_distil_w10_bullets  # ONNX + INT8 + dev check; copies the vocabulary
```
The test split is locked in code; `--allow-test` exists for the one final run (`shoprank.eval.final_test`).

## Documentation

- `docs/lab_notebook.md`: every run, with the prediction written before it.
- `docs/learn/`: lessons per section (beginner level, with check questions and answers).
- `docs/analysis/project_status_2026-11-21.md`: final status, known limits, what would come next.
- `docs/analysis/`: error analysis, label audit and the 21 Nov 2026 audit (leakage, claims, robustness).

## Data

Shopping Queries Dataset (Reddy et al., 2022), Apache-2.0: https://github.com/amazon-science/esci-data
