# Changelog

Versions are git tags. Every quality number below is on the split named next to it; the official test set
was scored once, for v1.0.

## Unreleased

## v1.1.1 — 22 Nov 2026: release hardening (no change to any reported number)
Independent code review (separate reviewer agent) of v1.0..v1.1: no defect in the ranking path; fixed:
- Spell correction could be silently off after the README "Reproduce" steps (vocabulary not in the model
  folder): `onnx_export` now copies `spell_vocab.json`, the API also looks at `data/processed/spell_vocab.json`,
  and the README lists `serve.spell build` and how `/health` shows the state.
- CI Docker smoke test now asserts `spell_correction: true` and that "usb c cbale 6ft" is corrected.
- Tests for `speller_from_env` (vocabulary found / missing / switched off).
- `query_used` keeps the input exactly (spacing too) when nothing was corrected; the demo no longer reports a
  "correction" for a double space. Model input is unchanged (the tokenizer ignores spacing).
- `SHOPRANK_SPELL` accepts 0/false/off/no.
- `fetch` checks every existing file before downloading anything (a refusal never leaves the folder
  half-updated) and reports network errors clearly.
- Documented, not changed (the evaluated corrector stays exactly as measured): words with attached punctuation
  or digits are not corrected; the vocabulary uses all catalogue products' text, incl. products in test pairs.
- Apache-2.0 LICENSE; package version 1.1.1; repo topics and Hugging Face link; lesson 18; stale v1.0 wording.
- Serving-only install without torch (`scripts/serve_requirements.sh`, shared by Dockerfile and README), after
  the WSL fresh-clone test showed the old README path pulled the full CUDA training stack.
- `eval/baseline_rerankers.py`: zero-shot BGE-reranker-v2-m3 (dev 0.8585) and Qwen3-Reranker-0.6B (0.8540)
  vs our served 33M MiniLM (0.8667): -0.0081 [-0.0130, -0.0032] and -0.0127 [-0.0178, -0.0076].

## v1.1 — 22 Nov 2026: query spell correction
- `serve/spell.py`: catalogue vocabulary (titles, brands, colours of 1.2M US products, 184,594 words) and
  one-edit (Damerau-Levenshtein, SymSpell-style) correction of rare query words; 0.01 ms per query.
- Dev (3 models): typo queries +0.0252 [+0.0211, +0.0292] and +0.0226 [+0.0185, +0.0267], 88-89% of the typo
  loss; clean +0.0003 [-0.0008, +0.0015]. Confirmed once on dev2: +0.0287 / +0.0263 (87% / 90%), clean +0.0020.
- API and demo correct the query before ranking (`query_used` in the response, `spell_correction` in /health,
  `SHOPRANK_SPELL=0` turns it off); `spell_vocab.json` published next to the model; fetch points at v1.1.
- Docs: products.parquet is US-only (1.2M products, small + large versions); an earlier note said three locales.

### Before v1.1 (since v1.0)
Phase A, supply chain and code health (plan: `docs/analysis/completion_plan_2026-11-21.md`).
- Security: torch 2.11.0+cu128 -> 2.13.0+cu126 (GHSA-rrmf-rvhw-rf47), setuptools 81.0.0 -> 84.2.1
  (PYSEC-2026-3447, GHSA-xmj3-9gf7-h52w); osv-scanner now reports none. Results unchanged: served MiniLM dev
  nDCG identical (0.86668), DeBERTa fold 0 +0.00017.
- Reproducibility: Hugging Face base models pinned by commit (`src/shoprank/hub.py`).
- Tests: metric runner (`eval/run_task1`), ONNX export + INT8 path (tiny random BERT), resumable download,
  model pins; 124 -> 138 tests.
- CI: security job (osv-scanner, gitleaks; binaries verified by SHA-256); actions pinned to commit SHAs.
- Removed the placeholder `shoprank` console script.
- Docs: completion plan (`docs/analysis/completion_plan_2026-11-21.md`).

Phase B, anyone can run it.
- Served model published: GitHub release v1.0 assets and Hugging Face `jasirjru/shoprank-reranker-minilm-int8`.
- `shoprank.serve.fetch`: download + SHA-256 check; tests.
- Dockerfile: CPU serving image without torch (45 packages); CI job builds it and checks `/rank`.
- Model card: download and usage without this repo.

Phase D, typo robustness (negative result, recorded).
- `cross/typos.py` (synthetic typos), `--typo-rate` in `cross.finetune` (consistency distillation),
  `eval/typo_robustness.py` (clean vs typo dev, seed-averaged paired CIs); `scripts/phase_d.sh`.
- Result on dev (3 seeds each): typo queries +0.0041 [+0.0019, +0.0062], 14% of the 0.028 typo loss, at a clean
  cost of -0.0015 [-0.0028, -0.0003]. Fails the pre-registered 50% rule: served model unchanged, dev2 unspent.
- Training reads only the products its pairs use (`read_products`): 1.2M -> 336k rows, identical texts.

## v1.0 — 21 Nov 2026 (commit 3390899)
Baseline for every later change.
- Top model: DeBERTa-v3-base, mean of 5 fold models. Official test nDCG 0.8877 [0.8849, 0.8905].
- Served model: MiniLM-L12 distilled from DeBERTa, with bullets, INT8 ONNX on CPU. Test nDCG 0.8674
  [0.8643, 0.8703] (PyTorch); dev 0.8654 (INT8). p50 69 ms / p95 232 ms per request, 230 pairs/s.
- Baselines on test: BM25 0.8202, LambdaMART 0.8337, MiniLM with Amazon's recipe 0.8544.
- FastAPI `/rank`, Gradio demo, model card, lessons 13-17, CI (ruff + 124 tests).
- Reproduced from a clean clone in WSL: data hashes, BM25 and LambdaMART dev nDCG bit for bit.
- Known limits: typo sensitivity (one typo: dev 0.8654 -> 0.8392), 3 known dependency vulnerabilities
  (torch 2.11, setuptools 81), served model not downloadable (lives only on the build machine).
