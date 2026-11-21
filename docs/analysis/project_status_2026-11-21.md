# ShopRank: final status (21 Nov 2026, v1.1.2)

The build is complete. This page says what exists, how good it is, how to run it, what is known to be weak, and
what would come next. Details: `README.md`, `docs/MODEL_CARD.md`, `docs/lab_notebook.md`, `CHANGELOG.md`.

## What exists
| Piece | Where | Status |
|---|---|---|
| Data pipeline (ESCI US, small version) | `src/shoprank/data`, `features` | reproduced bit for bit from a clean clone |
| Baselines: BM25, logistic regression, LambdaMART | `retrieval`, `ltr` | done |
| Cross-encoders: MiniLM, DeBERTa-v3-base, 5-fold out-of-fold, stacking | `cross`, `ltr/stack.py` | done |
| Distilled MiniLM, INT8 ONNX | `cross/finetune.py`, `serve/onnx_export.py` | served model |
| Query spell correction | `serve/spell.py` | in the API since v1.1 |
| API, demo, Docker image | `serve/app.py`, `serve/demo.py`, `Dockerfile` | CI builds and tests the image |
| Model download with checksums | `serve/fetch.py`; GitHub release v1.1; Hugging Face | public |
| Evaluation tools | `eval/` (bootstrap CIs, trec_eval check, slices, typo and spell tests) | done |
| CI | `.github/workflows/ci.yml`: tests, Docker smoke test, vulnerability + secret scans | green |
| Lessons | `docs/learn/sections09-11.md`, `section12.md` ... `section18.md` | written, to be studied |

## How good it is
| System | Official test nDCG [95% CI] (scored once) |
|---|---|
| BM25 | 0.8202 [0.8170, 0.8232] |
| MiniLM, Amazon's recipe (our run) | 0.8544 [0.8513, 0.8573] |
| Served: distilled MiniLM, INT8, CPU | 0.8674 [0.8643, 0.8703] (PyTorch weights; INT8 -0.0008 on dev) |
| Top: DeBERTa-v3-base, mean of 5 fold models | 0.8877 [0.8849, 0.8905] |

- Serving: p50 69 ms, p95 232 ms per request (~16 candidates), ~230 pairs/s per CPU process.
- Typos: one typo costs the ranker ~0.028 (dev); spell correction wins back 87-90% (dev and dev2), no loss on
  clean queries.
- Ready-made rerankers, zero-shot on dev: BGE-reranker-v2-m3 (568M) 0.8585 and Qwen3-Reranker-0.6B (596M) 0.8540,
  both below our served 33M model (0.8667; paired CIs exclude 0). In-domain distillation beats size here.

- More labelled data (large ESCI version, 3.3x pairs) gives the small MiniLM +0.0052 [+0.0023, +0.0082] on dev,
  about what distillation gave; distilling on the large version (needs ~7 GPU-hours of teacher folds) is untested.

## How to run it (no GPU, no training)
```bash
docker build -t shoprank . && docker run --rm -p 8000:8000 shoprank
# or, without Docker and without torch: see README "Serving", Option 2 (scripts/serve_requirements.sh)
```

## Known limits (also in README and model card)
- English (US) only; re-ranks given candidates, no retrieval over the full catalogue.
- Labels are noisy; the label error rate was not measured on a random sample.
- Typos in the evaluation are synthetic; the corrector skips words with digits or attached punctuation, can turn
  rare non-English words into English, and cannot fix misspellings that are common in product titles.
- Latency is from one laptop; no load balancer, autoscaling, authentication or rate limiting.
- The official test set is spent; v1.1 changes are judged on dev and dev2 only.
- Found by the final full-code audit and documented rather than changed (changing them would move published
  numbers after the fact): scores pass through bf16, so ~16% of a single model's pairs tie within a query (4% for
  the 5-model mean; tie order moves nDCG by ~0.00005); BM25 statistics (IDF, average length) are computed per
  split's candidate products, so LambdaMART/stacker BM25 features shift slightly in scale between splits; the
  `phrase_in_title` feature is a substring test ("pen" matches "pencil"); the BM25 tokenizer drops non-ASCII.
- Labels: on 200 random dev pairs, two AI judges agree with each other 87.5% of the time but contest the ESCI
  label on 28.5% [22.7, 35.1] (mostly Exact vs Substitute); an upper bound, since judges can share a bias.

## Where the evidence is
- Every run and its prediction: `docs/lab_notebook.md` (predictions written before each run, wrong ones kept).
- Audits: `docs/analysis/audit_2026-11-21.md` (leakage, claims, robustness), independent code reviews (sections
  15-16 and v1.0..v1.1, recorded in the notebook), clean-clone reproductions in WSL.
- Trained artefacts that took GPU hours (5 DeBERTa teachers, served PyTorch model, all scores and run records)
  are backed up in a private Hugging Face repository.

## What would come next (not started)
1. Human pass over the 57 contested rows of the random label audit (`label_audit_random200_dev.csv`,
   column `your_label`): turns the AI-judge upper bound (28.5%) into a real label error rate.
2. Retrieval stage on the large ESCI version: BM25 + embeddings over the 1.2M-product catalogue, then re-rank.
3. Real typo logs (or mined misspellings) to replace synthetic typos; correct words with attached punctuation.
4. Serving depth: /metrics, latency under a fixed arrival rate, authentication.

## Maintenance notes
- `uv.lock` pins every package; CI fails on a stale lock. Run osv-scanner (CI does) before any release.
- Base models are pinned by commit in `src/shoprank/hub.py`; the published model by SHA-256 in `serve/fetch.py`.
- When a release asset changes: create the GitHub release BEFORE pushing a commit that points `fetch` at it.
