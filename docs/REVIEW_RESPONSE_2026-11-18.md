# Response to the external review (ChatGPT, 18 Nov 2026)

Each item was checked against the code and the processed data on 18 Nov 2026. **Verified** means a command was run or the code was read; **hypothesis** means it has not been tested yet.

## Verified facts

| Check | Result | Evidence |
|---|---|---|
| Task 1 data version | `small_version = 1`, US locale | `run_task1.py:41`, `ingest.py` |
| Split sizes, Task 1 (rows / queries) | train 376,968 / 18,789; dev 42,685 / 2,099; test 181,701 / 8,956 | DuckDB count on `judgments.parquet` |
| Official test matches | 8,956 US test queries, the same as the ESCI README Table 1 | – |
| Dev is our own split | 1 in 10 *official-train* queries chosen by MD5 of `query_id`; ESCI has no official dev split | `ingest.py:49` |
| Reviewer's training-time "discrepancy" | **Reviewer error.** 1,393,063 is the large-version train used for Tasks 2/3. Task 1 train has 376,968 pairs; 376,968 / 131 ≈ 48 min. The 131 pairs/s came from a synthetic-token benchmark, so real-data throughput is still to be measured. | Counts above; `docs/STACK.md` |
| nDCG | Full list, gains 1 / 0.1 / 0.01 / 0, ideal order from the same candidates; equal to `pytrec_eval` (maximum difference 5.55e-16 on 500 queries) | `docs/STACK.md` |
| Bootstrap | Paired, query level: resamples per-query differences of the same queries; 10,000 resamples; fixed seed | `eval/bootstrap.py` |
| Standardisation | Fit on train only, reused on dev/test | `ltr/logreg.py:24`, `ltr/train.py:85` |
| LambdaMART early stopping | 10% of *train* queries held out (pandas hash of `query_id`); dev not used | `ltr/train.py:57-59` |
| Label gains in LightGBM | `label_gain = [0, 0.01, 0.1, 1.0]` | `ltr/train.py:31,61` |
| Two-tower false negatives | Same-query pairs masked in InfoNCE | `dense/train.py:75` |
| Catalogue reuse | 22.7% of dev candidate products also appear in train candidates. Not leakage: no feature is a learned per-product statistic. | DuckDB |

## Issues found (several were not in the review)

1. **Near-duplicate queries (verified, small).** 103 normalised query texts map to more than one `query_id`. In Task 1, **6** normalised texts span more than one split, so a few near-identical queries cross the train/dev/test boundary. Fix: assign splits by normalised query text. The expected effect on dev is tiny (6 texts against 2,099 dev queries), but the fix is still needed for rigour.
2. **Logistic regression trained on Exact vs rest only** (`ltr/train.py:87`). It discards the Substitute and Complement grades that LambdaMART uses, so the 0.830 vs 0.836 comparison mixes model type with label information. Hypothesis: a graded or ordinal logistic regression would narrow the gap.
3. **Early-stopping metric mismatch.** LightGBM stops on nDCG@10 (`K = 10`), but the headline metric is full-list nDCG. Probably a small effect; to verify.
4. **BM25 IDF is fit per split on that split's candidate products** (`features/build.py:87`). This uses no labels, so it is not leakage, but the statistics come from about 20 candidates per query, not from a catalogue. A production system would use a catalogue-wide IDF, so serving needs a fixed, saved index.
5. **Selection on dev.** About 15 configurations were compared on the same 2,099 dev queries, so the "best" dev number carries mild selection bias. Mitigation: freeze the final model choice on a second held-out bucket (dev-2) and run the test set once.
6. **Spanish queries inside the US set** (found in week-1 error analysis). Their effect on scores has not been measured.

## Is 0.836 trustworthy?

Yes, **as a dev-set number**:
- the split is query-level;
- there is no label leakage in the features;
- scaling uses train statistics only;
- early stopping is held out from dev;
- the metric is verified against `pytrec_eval`;
- the improvement over BM25 has a 95% CI of [+0.0127, +0.0205].

Caveats: mild dev-selection bias and the 6 near-duplicate texts. It is **not** comparable to the published 0.857, which is the official *test* set, and not to the 0.9043, which is test-set and averaged over three locales.

## Why the two-tower model loses (hypotheses to test, in order of cost)

1. **Training task differs from the evaluation task** (cheapest test). Negatives are random in-batch products (easy), while Task 1 ranks about 20 same-query candidates (hard). Test: add the same query's non-Exact candidates as hard negatives.
2. **No pretrained language knowledge.** Test: a zero-shot pretrained bi-encoder (MiniLM) with the same evaluation.
3. **Bag-of-trigrams ignores word order.** Test: a pretrained encoder.

Each test is a single controlled run with a CI. Insufficient data is *not* claimed as the cause until 1 and 2 are tested.

## Agree / disagree with the review

- **Agree:**
  - Production serving is the biggest portfolio gap.
  - Run a MiniLM cross-encoder before DeBERTa. Stronger reason: Amazon's 0.857 baseline **is** a fine-tuned `cross-encoder/ms-marco-MiniLM-L-12-v2`, so MiniLM first replicates the published baseline.
  - Use out-of-fold scores for stacking, and try a simple blend first.
  - Run feature ablations and error analysis.
- **Disagree:**
  - The training-time discrepancy is the reviewer's counting error (see above).
  - Dataset version: already verified as Task 1 small, US.
- **Missing from the review:** items 2-6 above.

## Prioritised roadmap (about 4 weeks)

| Week | Work | Gate |
|---|---|---|
| 1. Rigour fixes (CPU) | Split by normalised text; rebuild features; re-run BM25 / LR / LambdaMART with CIs; graded LR; full-list early stopping; dev-2 bucket; ablations; error report by query type | Numbers change only within documented reasons |
| 2. Pretrained baselines (cheap GPU) | Zero-shot MiniLM cross-encoder; two-tower plus same-query hard negatives; measure real throughput | The two-tower failure mechanism is identified |
| 3. Fine-tuning | Fine-tune MiniLM-L-12 (aim: reproduce about 0.857 on dev); then DeBERTa-v3-base only if MiniLM leaves a clear gap; expected-gain scoring; blend with LambdaMART; out-of-fold stacking | Decision recorded with CIs |
| 4. Serving | FastAPI rerank API (query + candidates in, ranked list out); saved catalogue BM25 index; model versioning; CPU/GPU p50/p95 latency and throughput; quality vs latency table; small demo; README; **one** test-set run | Reproducible from a clean clone |

**Avoid for now:** DeBERTa-large and ensembles chasing 0.90; the QLoRA query parser; MLOps platforms; multilingual locales; re-implementing more models from scratch.
