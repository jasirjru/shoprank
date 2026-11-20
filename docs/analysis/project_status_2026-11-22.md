# ShopRank: final status (22 Nov 2026, v1.1.1)

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
| Lessons | `docs/learn/section09-18` | written, to be studied |

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
- Modern ready-made rerankers (zero-shot, dev): see the "Modern rerankers" entry in the lab notebook.

## How to run it (no GPU, no training)
```bash
docker build -t shoprank . && docker run --rm -p 8000:8000 shoprank
# or: uv sync --locked && uv run python -m shoprank.serve.fetch && uv run uvicorn shoprank.serve.app:app
```

## Known limits (also in README and model card)
- English (US) only; re-ranks given candidates, no retrieval over the full catalogue.
- Labels are noisy; the label error rate was not measured on a random sample.
- Typos in the evaluation are synthetic; the corrector skips words with digits or attached punctuation, can turn
  rare non-English words into English, and cannot fix misspellings that are common in product titles.
- Latency is from one laptop; no load balancer, autoscaling, authentication or rate limiting.
- The official test set is spent; v1.1 changes are judged on dev and dev2 only.

## Where the evidence is
- Every run and its prediction: `docs/lab_notebook.md` (predictions written before each run, wrong ones kept).
- Audits: `docs/analysis/audit_2026-11-21.md` (leakage, claims, robustness), independent code reviews (sections
  15-16 and v1.0..v1.1, recorded in the notebook), clean-clone reproductions in WSL.
- Trained artefacts that took GPU hours (5 DeBERTa teachers, served PyTorch model, all scores and run records)
  are backed up in a private Hugging Face repository.

## What would come next (not started)
1. Random-sample label audit (200 pairs, two judges, kappa): the realistic ceiling.
2. Retrieval stage on the large ESCI version: BM25 + embeddings over the 1.2M-product catalogue, then re-rank.
3. Real typo logs (or mined misspellings) to replace synthetic typos; correct words with attached punctuation.
4. Serving depth: /metrics, latency under a fixed arrival rate, authentication.

## Maintenance notes
- `uv.lock` pins every package; CI fails on a stale lock. Run osv-scanner (CI does) before any release.
- Base models are pinned by commit in `src/shoprank/hub.py`; the published model by SHA-256 in `serve/fetch.py`.
- When a release asset changes: create the GitHub release BEFORE pushing a commit that points `fetch` at it.
