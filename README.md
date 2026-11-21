# ShopRank

Product search re-ranking on the Amazon Shopping Queries dataset (ESCI, Task 1, US English): from BM25 to a
DeBERTa-v3 cross-encoder ensemble, distilled into a small model served on a CPU. Every number has a 95%
confidence interval, a prediction written before the run (`docs/lab_notebook.md`), and the official test set
was scored once, with the systems declared in advance.

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
tied scores are ordered the same way.

**Published references:** Amazon's fine-tuned MiniLM baseline 0.857 (English); a public same-setup project
0.854 (US test). Both lie below the primary system's interval. Our own run of Amazon's recipe gives 0.8544,
close to their 0.857, which supports that the metric and data match. The KDD Cup 2022 winners (0.904) report
on all three locales (US, ES, JP) with large ensembles, so that number is not comparable.

## Serving

`POST /rank` (FastAPI) re-ranks a query's candidates with the distilled MiniLM, exported to ONNX and quantised
to INT8 (32 MB; dev nDCG 0.8654 vs 0.8662 fp32). Load test with real dev queries (median 16 candidates, title
+ brand + colour + bullets), laptop CPU (16 cores), client and server on the same machine:

| Model | Parallel requests | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|---|
| INT8 | 1 | 69 ms | 232 ms | 255 ms | 217 pairs/s |
| INT8 | 16 | 1,436 ms | 1,784 ms | 1,950 ms | 230 pairs/s |
| fp32 | 1 | 115 ms | 313 ms | 335 ms | 143 pairs/s |

At high load, latency is queueing: one process handles about 11 requests/s. The DeBERTa ensemble is 0.020
nDCG better but needs a GPU and is ~2.6x slower there, so it is used offline as the teacher.

```bash
uv run uvicorn shoprank.serve.app:app --port 8000
curl -X POST localhost:8000/rank -H "Content-Type: application/json" \
  -d '{"query": "usb c cable 6ft", "products": [{"product_id": "a", "title": "Anker USB C cable, 6 ft"},
       {"product_id": "b", "title": "HDMI cable 10 ft"}]}'
```

### Demo

A Gradio page over the same INT8 model and input rules as `/rank` (local only, no public link):

```bash
uv sync --group demo
uv run --group demo python -m shoprank.serve.demo   # http://127.0.0.1:7860
```
Type a query and one product per line (`title | brand | colour | bullets`); the page shows the list best
first and the scoring time (about 10 ms for 4 products).

## What mattered (dev, 2,001 queries; paired 95% CIs)

| Change | Effect on dev nDCG |
|---|---|
| Fine-tuned MiniLM cross-encoder vs LambdaMART | +0.019 |
| DeBERTa-v3-base vs MiniLM (same fields) | +0.0244 [+0.0197, +0.0292] |
| + bullet points as input (DeBERTa) | +0.0072 [+0.0039, +0.0106] |
| + brand and colour fields | -0.0002 [-0.0032, +0.0029] (nothing) |
| mean of 5 fold models vs 1 model | +0.0028 [+0.0004, +0.0051] |
| z-score blend / LightGBM stack vs the 5-model mean | -0.0003 / +0.0000 (nothing) |
| MiniLM: + bullets | +0.0068 [+0.0032, +0.0104] |
| MiniLM: + learn DeBERTa's out-of-fold scores (distillation) | +0.0064 [+0.0038, +0.0090] |
| 4-class cross-entropy instead of MSE on the gain | -0.0056 [-0.0094, -0.0018] |

Seed noise (3 seeds): DeBERTa spread 0.0016, MiniLM 0.0003; no change smaller than that is claimed.
Hardest queries: negation ("without", "not"); BM25 0.748, DeBERTa 0.839 on dev.

## Honest limits

- US English only, small version of Task 1 (re-ranking of judged candidates, no retrieval stage). The trained
  two-tower retriever stayed below BM25 (0.790 vs 0.816 on dev).
- Models train on 80% of the official training queries; 20% is held out as dev and dev2 for model choices.
- Label quality caps the score: in a 100-pair audit of the model's most confident disagreements, 75 labels
  looked wrong under the ESCI guidelines (`docs/analysis/`).
- Some choices rest on one seed (the bullets ablation), checked against the 3-seed spread.
- Scores pass through bf16, so 14% of a query's products can tie; the tie order moves nDCG by ~0.00005.
- The INT8 model was not scored on test (the test set is spent); its dev loss is 0.0008.
- Latency is from one laptop, client on the same machine; no network, no autoscaling.

## Reproduce

```bash
uv sync --locked
uv run python -m shoprank.data.download      # ~1.16 GB, resumable
uv run python -m shoprank.data.ingest        # US tables + train/dev/dev2/test split (test = official test)
uv run python -m shoprank.features.build     # 14 text features
uv run pytest -q && uv run ruff check src tests
# DeBERTa 5-fold models (~36 min each on an RTX 4060), MiniLM folds (~5 min each):
for k in 0 1 2 3 4; do
  uv run python -m shoprank.cross.oof --model microsoft/deberta-v3-base --fields title,brand,color,bullets \
      --lr 2e-5 --warmup-steps 1000 --fold $k --name deberta_oof
  uv run python -m shoprank.cross.oof --fold $k --name minilm_oof
done
uv run python -m shoprank.ltr.stack                                    # blend + stack on dev
uv run python -m shoprank.cross.finetune --teacher deberta_oof --teacher-weight 1.0 \
    --fields title,brand,color,bullets --name minilm_distil_w10_bullets
uv run python -m shoprank.serve.onnx_export minilm_distil_w10_bullets  # ONNX + INT8 + dev check
```
The test split is locked in code; `--allow-test` exists for the one final run (`shoprank.eval.final_test`).

## Documentation

- `docs/lab_notebook.md`: every run, with the prediction written before it.
- `docs/learn/`: lessons per section (beginner level, with check questions and answers).
- `docs/HANDOFF.md`: project state and decisions.

## Data

Shopping Queries Dataset (Reddy et al., 2022), Apache-2.0: https://github.com/amazon-science/esci-data
