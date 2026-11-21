# ShopRank: is it complete, and what is left? (21 Nov 2026)

Builds on `audit_2026-11-21.md` (leakage, claims, robustness) and the WSL clean-clone reproduction.
New checks today: dependency vulnerabilities (osv-scanner on `uv.lock`), untested modules, model loading,
and what a fresh clone can actually run.

## Verdict

**Complete as an offline re-ranking study.** Sections 13-17 are delivered: test numbers with CIs (scored once),
independent metric check, ablations, distillation, INT8 serving with a load test, demo, model card, public
repo, CI green, bit-for-bit reproduction of the CPU pipeline from a clean clone.

**Not complete as a system someone else can use or trust in production.** Six gaps block that:

| # | Gap | Evidence | Why it matters |
|---|---|---|---|
| G1 | A fresh clone cannot serve or demo | served model (128 MB PyTorch, 32 MB INT8) exists only in gitignored `data/`; rebuilding it needs ~3.5 GPU-hours (5 DeBERTa folds + distillation) | "clone and run" fails at the most visible step |
| G2 | Upstream models not pinned | `from_pretrained("microsoft/deberta-v3-base")` etc. without `revision=` | a changed upstream checkpoint silently changes results |
| G3 | Known vulnerabilities in the lock | torch 2.11.0 (GHSA-rrmf-rvhw-rf47, CVSS 5.3, fixed 2.13.0); setuptools 81.0.0 (PYSEC-2026-3447 6.1, GHSA-xmj3-9gf7-h52w 1.0; fixed 84.1.0) | public repo with known CVEs |
| G4 | Typo fragility | one swapped letter pair: dev nDCG 0.8654 -> 0.8392 | real queries have typos; largest measured weakness |
| G5 | No model-quality gate in CI | CI runs unit tests only; behavioural tests ran once by hand | a regression would pass CI |
| G6 | Critical code without tests | `eval/run_task1.py` (computes every dev number), `serve/onnx_export.py` (export + INT8), `data/download.py` | the metric runner and the serving artefact path are unchecked by CI |

Important but not blocking:

| # | Gap | Note |
|---|---|---|
| G7 | Label error rate unknown | the 100-pair audit hunted errors; a random sample is needed for a rate and a ceiling |
| G8 | No modern off-the-shelf reranker baseline | "why fine-tune instead of downloading a 2025-26 reranker?" is the first interview question; zero-shot bge-reranker-v2 / Qwen3-Reranker on dev answers it |
| G9 | Serving is a demo, not a service | no Dockerfile, no /metrics, no request ids, load test is closed-loop only (no latency-vs-arrival-rate curve) |
| G10 | No retrieval stage | ESCI Task 1 hands over the candidates; a real system finds them in 1.2M products |

Hygiene: `shoprank` console script prints "Hello from shoprank!"; learning code `scratch/` sits inside the
package; no LICENSE (owner's decision; Apache-2.0 would match the data and base models); GitHub Actions warns
that Node 20 actions are deprecated; no version tag or changelog.

Not worth doing (decided): DeBERTa-v3-large (8 GB GPU, teacher-only gain ~0.003-0.006), more stacking
(measured: nothing), other locales (a different project), LLM fine-tuning.

## Plan (in order; each phase ends with tests + CI green + notebook entry)

Rules for every phase: prediction and pass threshold written before each run; dev for choices, dev2 once to
confirm; the official test set is not scored again (a new served model is reported on dev/dev2 only, or in a
second, clearly labelled and pre-registered test look of one final model); ask before downloads, commits,
pushes and publications; warn before any job over 2 GPU-hours.

**Phase 0: freeze a baseline (15 min).** Tag the current state `v1.0` with a CHANGELOG entry. Every upgrade
below is compared against v1.0.

**Phase A: supply chain and code health (half a day, no GPU).**
1. Upgrade torch to 2.13 and setuptools to >= 84.1; `uv lock`. Re-verify: unit tests, ONNX parity, served
   model dev nDCG within 0.0005 of 0.8654 (GPU numerics may move), osv-scanner clean. (G3)
2. Pin every Hugging Face model by commit `revision` in one constants module. (G2)
3. Tests for `run_task1` (tiny synthetic split, known nDCG), `onnx_export` (export a tiny random BERT, check
   ONNX vs PyTorch parity and that INT8 runs), `download` (resume logic with a fake server). (G6)
4. Add osv-scanner and gitleaks to CI. Replace the "Hello" console script with a small CLI or remove it;
   move `scratch/` out of the package; bump the deprecated actions.

**Phase B: anyone can run it (half a day).** (G1, G9 part)
1. Publish the served model (INT8 ONNX + tokenizer + model card) as a Hugging Face model repo or a GitHub
   release asset (ask first). `shoprank.serve.fetch` downloads it by pinned revision and checks SHA-256.
2. Dockerfile for the CPU API (INT8), non-root user, health check; README "serve in two commands".
3. CI smoke job: fetch the 32 MB model, rank the README example, assert the order.

**Phase C: measure the measuring stick (1 day, no GPU except G8).**
1. Random 200-pair dev label audit, stratified by label and query source; two judges (owner + LLM with the
   ESCI guidelines); Cohen's kappa, error rate with a Wilson CI, model nDCG on cleaned pairs. (G7)
2. Modern reranker baselines zero-shot on dev (e.g. bge-reranker-v2-m3, Qwen3-Reranker-0.6B), with latency
   on the same CPU/GPU; paired CIs vs the served model and the teacher. (G8; ~1-2 GB download, ask)

**Phase D: the model upgrade (1 day, ~1 GPU-hour).** (G4, G5)
1. Typo-consistency distillation: a share of training queries gets a realistic typo (adjacent-key, drop,
   swap; plus misspellings mined from ESCI queries), the target stays the teacher score of the clean query.
   Pass: clean dev nDCG drops by no more than the seed spread; typo dev nDCG recovers at least half of 0.026.
   3 seeds; dev2 once.
2. Re-export ONNX INT8, re-run the load test, update the model card (v1.1).
3. Behavioural tests (typo, case, word order, missing fields) on a fixed 200-query dev subset, thresholds in
   CI against the published model.

**Phase E: serving depth (1 day).** (G9)
Prometheus /metrics (latency histogram, requests, errors, queue wait), request ids in logs, readiness vs
liveness, open-loop load test at fixed arrival rates to find the knee of the latency curve, p99 target stated
(e.g. p99 < 300 ms at N requests/s on 16 cores) and the max sustainable rate per process.

**Phase F (optional, the next big piece, ~1 week): retrieval stage.** (G10)
Large version of ESCI, full US catalogue; BM25 (bm25s) + a small off-the-shelf embedding model (FAISS) fused
with reciprocal rank fusion; recall@100/500 on dev; then the served re-ranker on top; end-to-end nDCG and
latency. Only after A-E.

**Then freeze (v1.1 or v2.0 tag) and start the study block.**

## Definition of done (acceptance checklist)
- [ ] `git clone` -> `docker run` (or two `uv` commands) serves the model with no GPU and no training.
- [ ] CI: lint, unit tests, vulnerability scan, secret scan, smoke test, behavioural thresholds, all green.
- [ ] osv-scanner: 0 known vulnerabilities; all upstream models pinned by revision.
- [ ] Every module that produces a reported number or a served artefact has a test.
- [ ] Label error rate measured on a random sample, with a CI; modern-reranker baseline reported.
- [ ] Served model typo drop at most half of today's 0.026, clean dev nDCG unchanged within the seed spread.
- [ ] Latency-vs-arrival-rate curve and a stated p99 target in the README.
- [ ] README, model card, notebook and lessons match the final numbers; tag + changelog.
