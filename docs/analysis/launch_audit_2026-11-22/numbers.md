# Launch audit: numbers and claims (22 Nov 2026)

Scope: every number on the live page (`site/public/index.html`, built by `src/shoprank/site.py` from
`site/data/examples.json`), `README.md`, `docs/MODEL_CARD.md` and the Hugging Face card, checked against
`data/processed/runs/*`, `docs/lab_notebook.md` and `CHANGELOG.md`. Nothing was edited, and no model was run.
The test set was not re-scored: every test figure below comes from reading the saved per-query
`*_test.parquet` files.

Checks on the files themselves:
- The live page (https://shoprank-jasirjru.vercel.app, fetched 22 Nov) is byte-identical to the committed
  `site/public/index.html`, and that file is identical to what `shoprank.site.build()` generates now.
- The HF card (`https://huggingface.co/jasirjru/shoprank-reranker-minilm-int8/raw/main/README.md`, 105 lines)
  is `docs/MODEL_CARD.md` with YAML metadata and one link line added above it. Every number is identical, so
  the MODEL_CARD verdicts below also cover the HF card.

## Summary of findings

There are no BLOCKERs. Every reported number matches its evidence file. All the issues below are about
framing.

| # | Rank | Finding | Where |
|---|---|---|---|
| F1 | SHOULD | The headline "Puts the right product first" and the page title "...in Amazon product search" overstate the result. The top product is Exact on 74.7% of test searches (keyword search: 62.7%). The title also reads as if Amazon uses the model. | site.py:162, 180 |
| F2 | SHOULD | The 5 winning examples are hand-picked from the top 0.2-10% of dev wins, and the page does not say they were chosen. ShopRank beats keyword search on 65.5% of test searches and does worse on 32.8%. | site.py:196 |
| F3 | SHOULD | The label "Classic ranking with hand-made rules" is wrong for LambdaMART, which is a learned model (gradient-boosted trees over 14 hand-made features). | site.py:30 |
| F4 | SHOULD | "It runs on a normal laptop" and "16 products in 0.07 seconds on a laptop": the 69 ms is the **median**, measured on a 16-physical-core laptop CPU. That is not a typical laptop, and p95 is 232 ms. | site.py:183, 213 |
| F5 | SHOULD | "Wins back almost 90% of the loss" omits that the typos were **synthetic** one-edit typos. The measured range is 87-90%. The page shows a real typo example right beside it, which invites misreading. | site.py:214 |
| F6 | SHOULD | The headline 0.867 is the PyTorch model. The 32 MB INT8 file that "runs on a laptop" was never scored on test (dev: -0.0008). The engineers section says so, but the first screen does not. | site.py:181-183, 32 |
| F7 | NICE | The row "Amazon's published method (my re-run)" can be read as Amazon's published score. Amazon's own number is 0.857, on a different query sample. | site.py:31 |
| F8 | NICE | "both need a GPU" (BGE-reranker-v2-m3, Qwen3-Reranker-0.6B) is not measured. Both run on CPU, only slower. | MODEL_CARD.md:25, HF card:45 |
| F9 | NICE | The single-system test CIs for two systems cannot be reproduced exactly with the repo's bootstrap at seed 0: served [0.8645, 0.8703] vs reported [0.8643, 0.8703]; Amazon recipe [0.8515, 0.8574] vs [0.8513, 0.8573]. No code in the repo produces these single-system intervals. All paired CIs reproduce exactly. | README.md:21-22, MODEL_CARD.md:19 |
| F10 | NICE | Small wording points: "Five large models" (they are DeBERTa-v3-**base**); "per 16-candidate request" (16 is the median); "~230 pairs/s ... one model call at a time" (230 is at 16 parallel clients, 217 at 1); the "lg refridgerators" example gives BM25 the typo as typed. | site.py:211, 234; MODEL_CARD.md:28; examples.json:835 |
| F11 | NICE | On the HF card, "In this repo: `uv run ...`" refers to the GitHub repo, not the HF repo. | HF card:56 |

## 1. Claims table

Evidence paths are relative to the repo root. "runs/" means `data/processed/runs/`.

### Live page (`src/shoprank/site.py` → `site/public/index.html`)

| Claim | Where | Evidence file:line | Verdict | Fix |
|---|---|---|---|---|
| 0.867 ShopRank, 0.820 keyword search | site.py:181 (index.html:235), meta site.py:163 | runs/minilm_distil_w10_bullets_test.json `ndcg_full` 0.86736 → 0.867 (4 dp 0.8674); runs/bm25_title_test.json 0.82017 → 0.820 | ok (rounding consistent: 3 dp on the first screen, 4 dp in the engineers section) | see F6 for the label |
| Results table 0.8202 / 0.8337 / 0.8544 / 0.8674 / 0.8877, shown as 0.820 / 0.834 / 0.854 / 0.867 / 0.888 | site.py:28-34 | runs/{bm25_title,lambdamart,minilm_ft_mse_fast,minilm_distil_w10_bullets,deberta_oof_avg5}_test.json; lab_notebook.md:532-538 | ok | F3, F7 for labels |
| "8,956 searches, checked once at the end" | site.py:204 | every *_test.json `"queries": 8956`; judgments.parquet: 8,956 small-version US test queries / 181,701 pairs; lab_notebook.md:526-531 (BM25 scored once before an OOM crash, not re-scored) | ok | none |
| "Real searches the model never saw while learning" | site.py:196 | Split by normalised query text (src/shoprank/data/ingest.py:30, 45-66). Training reads only `split='train'` (cross/finetune.py:101-114, 337). The OOF teacher folds train only on train (cross/oof.py:9-12, 71). Early stopping uses a train hold-out (cross/finetune.py:341). I counted the overlap of normalised query text: dev∩train 0, dev2∩train 0, test∩train 0, and dev query_ids in train 0. All 6 example query_ids are `split='dev'` in judgments.parquet. | ok (weights never trained on these queries). Caveat: dev was used to choose the model, and the examples were hand-picked from dev with known outcomes (F2). | F2 |
| Example grades (6 examples, bm25 / ShopRank): 0.38/1.00, 0.49/0.99, 0.47/0.85, 0.59/0.97, 0.78/1.00, 1.00/0.57 | examples.json `ndcg` | I recomputed all 6 (not just 2) independently from the listed labels, full list of 16 candidates (8 shown + "8 more"), linear gains E1 S0.1 C0.01 I0, discount 1/log2(i+1). All 12 values match to 4 dp: iphone xs pro 0.3770/1.0000; holey jeans 0.4922/0.9927; lenovo ram 0.4702/0.8494; usb 3.0 0.5903/0.9720; lg refridgerators 0.7815/0.9996; m23 1.0000/0.5706. Both lists contain the same 16 products with the same labels, and those labels match judgments.parquet for all 6 queries. Top-8-only nDCG would differ a lot (e.g. iphone xs pro BM25 0.018), so full-list scoring was correctly used. | ok | none |
| Proof line: "the first right product is number 12. ShopRank puts 3 right products first." | site.py:139-150 (generated) | examples.json, iphone xs pro: BM25 labels CCCCCCCCCCCECCEE (first E at 12); ShopRank EEEC... | ok | none |
| "ranks 16 products in 0.07 seconds on a laptop" | site.py:213 | data/processed/serving/loadtest_int8_t16_q1.json: median 16 candidates, concurrency 1, p50 69.1 ms, p95 232.4 ms, mean 93.2 ms; lab_notebook.md:577-581 (16 threads; machine has 16 physical cores, lab_notebook.md:570-571) | unclear → SHOULD (F4): it is the median, on a high-core-count laptop | see F4 |
| "Shrunk to 32 MB" | site.py:213 | data/processed/models/onnx/minilm_distil_w10_bullets/model_int8.onnx = 33,848,219 bytes = 32.3 MiB | ok | none |
| "It wins back almost 90% of the loss" | site.py:214 | runs/spell_eval_dev.json: recovered 0.89 (transpose) and 0.88 (neighbour) on dev; dev2 87% / 90% (lab_notebook.md:791-792). Typos are synthetic. | ok numerically, SHOULD for framing (F5) | see F5 |
| "keeps most of their skill" (student vs teacher) | site.py:212 | Student gain over BM25 is 0.0472 vs the teacher's 0.0676, so the student keeps 70% of the teacher's gain. In absolute terms 0.8674 vs 0.8877. | ok ("most" = 70%) | NICE: "keeps about 70% of their gain over keyword search" |
| "Five large models" | site.py:211 | 5 × DeBERTa-v3-base (README.md:23) | unclear (they are the *base* size) | NICE (F10) |
| "Big teacher model (needs a GPU)" | site.py:33 | README.md:60-62: ~89 pairs/s on an RTX 4060; CPU not measured | ok as practical guidance | none |
| "Amazon's published method (my re-run)" 0.8544 | site.py:31 | runs/minilm_ft_mse_fast_test.json 0.85438; README.md:37-38 (adapted recipe: bf16, max length 128, 90% of our train split) | ok number; label ambiguous | F7 |
| Engineers section: 0.8877 [0.8849, 0.8905] | site.py:232 | My recompute with `paired_bootstrap_ci(zeros, x)`, seed 0: [0.88491, 0.8905] | ok | none |
| Engineers section: 0.8674 [0.8643, 0.8703] | site.py:233 | Same recompute: [0.86448, 0.87026] → [0.8645, 0.8703] | unclear (lower bound off by 0.0002) | F9 |
| +0.013 [+0.011, +0.015] over the Amazon-recipe re-run | site.py:233 | My recompute (paired, seed 0): +0.0130 [+0.0112, +0.0148]; lab_notebook.md:540-541 | ok | none |
| INT8 dev -0.0008; p50 69 ms / p95 232 ms "per 16-candidate request" on a 16-thread laptop CPU | site.py:234 | models/onnx/.../quality.json 0.8654 vs 0.8662; loadtest_int8_t16_q1.json | ok (16 is the median) | NICE: "per request (median 16 candidates)" |
| One synthetic typo costs 0.028; recovers 87-90% | site.py:235 | spell_eval_dev.json gap 0.02831; lab_notebook.md:791-792; typo-augmented training 14% (CHANGELOG.md:158-159) | ok | none |
| BGE 0.8585, Qwen3 0.8540 on dev, both below the served model (paired CIs exclude 0) | site.py:236 | runs/zeroshot_bge_v2m3_dev.json 0.85855; zeroshot_qwen3_06b_dev.json 0.85398. My recompute: -0.0081 [-0.0130, -0.0032] and -0.0127 [-0.0178, -0.0076] vs minilm_distil_w10_bullets (dev 0.86668) | ok | none |
| "m23 for cars ... showed stickers, not McLaren M23 model cars" | site.py:221; examples.json:1041 | examples.json: ShopRank #1 "Navy Logo (M23) USN Vinyl Decal Sticker", #2 "MSA OFFROAD WHEELS M23", Exact model cars at #3-4 | ok | none |
| "nivda" → "nivea" | site.py:222 | lab_notebook.md:909-910 | ok | none |
| "A 32 MB re-ranking model that beats keyword search ... nDCG 0.867 vs 0.820" | site.py:163 | as above | ok | none |

### README.md

| Claim | Where | Evidence | Verdict | Fix |
|---|---|---|---|---|
| 8,956 queries / 181,701 judged pairs | README.md:13 | judgments.parquet (small, US, test): 8,956 q / 181,701 rows | ok | none |
| Test table nDCG and nDCG@10 (6 systems) | README.md:19-24 | runs/*_test.json (ndcg_full, ndcg10): 0.8202/0.6942, 0.8337/0.7172, 0.8544/0.7527, 0.8674/0.7743, 0.8877/0.8092, 0.8883/0.8099 | ok | none |
| Single-system CIs | README.md:19-24 | Recomputed with the repo bootstrap, seed 0: BM25 [0.8170, 0.8232(5)], LambdaMART [0.8305, 0.8368], recipe [0.8515, 0.8574], served [0.8645, 0.8703], DeBERTa [0.8849, 0.8905] | ok for 3 rows; recipe and served differ by ≤0.0002 | F9 |
| Paired: +0.0676 [+0.0646, +0.0705]; +0.0541 [+0.0513, +0.0568]; +0.0006 [-0.0003, +0.0014]; +0.0130 [+0.0112, +0.0148] | README.md:26-28 | My recompute matches all four exactly; lab_notebook.md:540-542 | ok | none |
| pytrec_eval agreement within 0.0003 | README.md:29-30 | lab_notebook.md:546-550 (log only; not re-run) | ok (as recorded) | none |
| Published refs 0.857, 0.854, SQID 0.8562, KDD 0.904 | README.md:32-45 | benchmark_position_2026-11-21.md, sources listed there | ok (see section 3) | none |
| Swapped-gain numbers 0.8724 / 0.8499 / 0.8354 | README.md:43 | benchmark_position_2026-11-21.md:12-17 | unclear: not re-checked (needs pair-level test scores, which this audit must not re-score) | none |
| INT8 32 MB, dev 0.8654 vs 0.8662 | README.md:51 | quality.json | ok | none |
| Load-test table | README.md:56-58 | loadtest_int8_t16_q1.json (69/232/255 ms, 217 pairs/s at 1; 1,436/1,784/1,950 ms, 230 at 16); lab_notebook.md:581-583 for fp32 | ok | none |
| "about 11 requests/s"; ensemble 0.020 better, ~89 pairs/s | README.md:60-61 | loadtest json 10.7-11.3 req/s; 0.8877-0.8674 = 0.0204; lab_notebook.md:601 | ok | none |
| Dev table: spell +0.0252 [+0.0211, +0.0292], 89%, dev2 +0.0287; clean +0.0003 [-0.0008, +0.0015]; BGE/Qwen deltas | README.md:114-118 | spell_eval_dev.json; lab_notebook.md:791-792; recompute above | ok | none |
| Other dev deltas (rows 104-113), seed spreads | README.md:104-121 | lab_notebook.md entries | not re-verified one by one (dev, not part of launch claims) | none |
| 2,001 / 1,053 held out = 14.6%; ~68%; 323,302 pairs (~77%) | README.md:129-132 | 3,054 / 20,888 = 14.6%; 0.8 × 17,823 / 20,888 = 68%; 16,091 / 20,888 = 77% (lab_notebook.md:111-112, 146-148) | ok | none |
| INT8 not scored on test; dev loss 0.0008 | README.md:139 | quality.json | ok | none |

### docs/MODEL_CARD.md (identical text on the HF card, lines +20)

| Claim | Where | Evidence | Verdict | Fix |
|---|---|---|---|---|
| 323,302 pairs, ~16,000 queries, ~77% | MODEL_CARD.md:10 | lab_notebook.md:146-148 (16,091 q / 323,302 pairs) | ok | none |
| Test 0.8674 [0.8643, 0.8703]; nDCG@10 0.7743 [0.7702, 0.7783] | MODEL_CARD.md:19-20 | json; my recompute of nDCG@10 CI [0.7702, 0.7783] matches; nDCG CI see F9 | ok / F9 | F9 |
| dev INT8 0.8654 (fp32 0.8662) | MODEL_CARD.md:21 | quality.json | ok | none |
| BM25 0.8202, recipe 0.8544, teacher 0.8877 | MODEL_CARD.md:23 | json | ok | none |
| BGE 0.8585, Qwen 0.8540 < 0.8667 "and both need a GPU" | MODEL_CARD.md:24-25 | run files ok; "need a GPU" not measured | F8 | F8 |
| p50 69 ms, p95 232 ms, ~230 pairs/s (~11 req/s) "one model call at a time" | MODEL_CARD.md:27-28 | loadtest json: 230 pairs/s is at 16 clients; 217 at 1 | ok (230 is the sustained throughput with one call at a time and a queue) | NICE (F10) |
| typo: 0.8654 → 0.8392 (INT8) | MODEL_CARD.md:66-67 | serving/robustness_int8_dev.json `typo_longest_word` 0.8392 | ok | none |
| negation: 0.8378 on 158 queries vs 0.8667 | MODEL_CARD.md:73 | I recomputed it from runs/minilm_distil_w10_bullets.parquet with the `slices.NEGATION` regex: 158 queries, 0.8378 vs 0.8667 | ok | none |
| 75 of 100 labels looked wrong; 91% agreement | MODEL_CARD.md:75-76 | README.md:133-136; CHANGELOG.md:103-105 | ok (as recorded) | none |

## 2. Over-claiming check (judgement, with exact wording)

Evidence for the numbers in this section is in section 4.

**F1 (SHOULD). Headline and page title.**
- What the data shows: on the official test, the top product is Exact for 74.7% of searches with ShopRank,
  vs 62.7% with keyword search (+12.0 points [+10.9, +13.1]). "Puts the right product first" reads as an
  absolute claim, and one search in four is a miss. The `<title>` "...in Amazon product search" can be read
  as Amazon using the model.
- `src/shoprank/site.py:180`
  - old: `<h1 id="headline">Puts the right product first in shop search</h1>`
  - new: `<h1 id="headline">Puts the right product first more often in shop search</h1>`
- Optional first-screen fact (the number is stronger than the slogan), in the lede at site.py:183, add:
  `On Amazon's official test, the top result is the right product for 75% of searches, against 63% for keyword search.`
- `src/shoprank/site.py:162`
  - old: `<title>ShopRank: puts the right product first in Amazon product search</title>`
  - new: `<title>ShopRank: re-ranking for product search, tested on Amazon's Shopping Queries data</title>`

**F2 (SHOULD). Hand-picked examples are not labelled as such.** The 5 wins sit at the 99.9th, 99.6th, 97.5th,
97.6th and 89.7th percentiles of the per-query dev gain (`data/processed/showcase_candidates.parquet`). ShopRank
is better on 65.3% of dev queries and worse on 33.7% (lab_notebook.md:905-907); on test, better on 65.5% and
worse on 32.8%. One failure is shown, which helps, but a skeptic will ask how the examples were chosen.
- `src/shoprank/site.py:196`
  - old: `Real searches the model never saw while learning. Same products, two orders. The marks are Amazon's human labels.`
  - new: `Real searches the model never saw while learning, hand-picked to show typical wins and one failure. Across all 8,956 test searches ShopRank ranks better than keyword search on 65% and worse on 33%. Same products, two orders. The marks are Amazon's human labels.`

**F3 (SHOULD). LambdaMART label.** LambdaMART is a learned ranker over 14 hand-made features, not hand-made rules.
- `src/shoprank/site.py:30`
  - old: `("Classic ranking with hand-made rules", 0.8337, ""),`
  - new: `("Classic machine-learned ranking (LambdaMART)", 0.8337, ""),`

**F4 (SHOULD). "Normal laptop" and "0.07 seconds".** The latency was measured with 16 ONNX threads on a
laptop with 16 physical cores (lab_notebook.md:570-571). 69 ms is the median, and 1 request in 20 takes
more than 232 ms. "Runs on a laptop CPU" is true for the INT8 model. "Normal laptop" is not shown for
4-8 core machines.
- `src/shoprank/site.py:183`
  - old: `It runs on a normal laptop.`
  - new: `It runs on a laptop CPU, no GPU needed.`
- `src/shoprank/site.py:213`
  - old: `Shrunk to 32&nbsp;MB, it ranks 16 products in 0.07 seconds on a laptop.`
  - new: `Shrunk to 32&nbsp;MB, it ranks a typical list of 16 products in about 0.07 seconds (median) on a 16-core laptop CPU.`

**F5 (SHOULD). Synthetic typos.**
- `src/shoprank/site.py:214`
  - old: `Typos hurt the ranking, so a spell checker fixes the search first. It wins back almost 90% of the loss.`
  - new: `Typos hurt the ranking, so a spell checker fixes the search first. On made-up one-letter typos it wins back 87-90% of the loss.`

**F6 (SHOULD). The headline number is not the shipped file's number.** 0.8674 is the PyTorch model. The
INT8 file was only scored on dev (0.0008 lower), and the API with spell correction was not scored on test.
This is disclosed in site.py:234 and README.md:139, but not next to the number.
- `src/shoprank/site.py:32`
  - old: `("ShopRank, this model", 0.8674, "this page"),`
  - new: `("ShopRank, this model (full precision)", 0.8674, "this page"),`
- Or add to the lede (site.py:183) after "where 1.0 is a perfect order.":
  ` The 32 MB laptop version scores 0.001 lower on the development set.`

**F7 (NICE). Amazon re-run label.**
- `src/shoprank/site.py:31`
  - old: `("Amazon's published method (my re-run)", 0.8544, "published baseline"),`
  - new: `("Amazon's published recipe, re-trained by me", 0.8544, "published baseline"),`

**F8 (NICE). "both need a GPU".**
- `docs/MODEL_CARD.md:25` (and HF card line 45)
  - old: `both below this model's 0.8667 (paired CIs exclude 0), and both need a GPU.`
  - new: `both below this model's 0.8667 (paired CIs exclude 0), and both about 17x larger.`

**F9 (NICE). Single-system CIs.** With `paired_bootstrap_ci(zeros, x)` (seed 0, 10,000 resamples, queries
sorted by id, which is also the file order), I get: served [0.8645, 0.8703], recipe [0.8515, 0.8574],
BM25 [0.8170, 0.8232(5)]. The reported values are [0.8643, 0.8703], [0.8513, 0.8573] and [0.8170, 0.8232].
The differences are at most 0.0002. No script in the repo writes these intervals. They come from a run not
in the repo, probably a different seed or resampler. Either note "single-system CIs from an ad hoc run" or
replace them with the seed-0 values above. Paired CIs reproduce exactly.

**F10 (NICE). Wording details.**
- site.py:211: `Five large models` → `Five DeBERTa models`.
- site.py:234: `per 16-candidate request` → `per request (median 16 candidates)`.
- MODEL_CARD.md:28 / HF:48: `~230 pairs/s (~11 requests/s) per process` → `217-230 pairs/s (~11 requests/s) per process`.
- examples.json:835 note: add `Keyword search gets the query as typed.` Keyword search is scored on the
  typo, so part of this win is the spell checker's, not the ranker's.

**F11 (NICE). HF card.** HF card:56: `- In this repo:` → `- In the GitHub repo:`.

Checked and not over-claiming: the meta description ("beats keyword search ... 0.867 vs 0.820") is correct.
"Checked once at the end" matches the record. "Amazon's human labels" is correct. The failure section shows a
real failure and a real spell-checker error. The footer states that the page is not affiliated with Amazon.

## 3. Public ESCI Task 1 US-test results newer than benchmark_position_2026-11-21.md

The method was general web searches (no personal data) plus reading the candidate pages. I found no public
result on the official ESCI Task 1 US test (full-list nDCG, judged candidates) above ShopRank's served 0.867
or top 0.888. I also found no comparable result newer than the ones already listed in
`docs/analysis/benchmark_position_2026-11-21.md`. Candidates found and why they are not comparable:

| Source | What it reports | Comparable? |
|---|---|---|
| https://huggingface.co/albertobarnabo/ecommerce-product-search-reranker | full-list nDCG 0.8602 on the US test, paper gains | yes. Already listed (benchmark doc §2), and below 0.867 |
| https://huggingface.co/lv12/esci-ms-marco-MiniLM-L-12-v2 | self-reported ndcg@10 85.46, mrr@10 91.81; split, locale and gains not stated | no (nDCG@10, unknown split) |
| https://huggingface.co/ramitha2002/genieai-product-reranker | NDCG@10 0.8796 on 424 *validation* queries in 10 gift categories, own gains (S 0.70, C 0.35) | no |
| https://arxiv.org/abs/2601.05588 (Autoregressive Ranking, 2026) | ESCI-derived set of 310 examples with Gecko-similarity targets, not ESCI labels (Table 2) | no |
| https://arxiv.org/abs/2503.24289 (Rec-R1, 2025) | ESCI retrieval NDCG@100 and re-ranking NDCG@10 on 4 category subsets with its own retrieved candidates | no |
| https://arxiv.org/abs/2508.14180 (RewardRank, 2025) | uses Amazon KDD Cup data with LLM-simulated users; no Task 1 nDCG | no |
| https://ceur-ws.org/Vol-3843/paper_16.pdf (Walmart KD, eCom'24) | in-house Walmart data | no |
| https://arxiv.org/abs/2504.03947 (2025) | BRIGHT benchmark, not ESCI | no |

Verdict: "above every public baseline and replication on the same test" (benchmark doc §3) still holds as far
as I could source. This is an absence of evidence from general searches, not a full literature survey. If the
page or post says "top", phrase it as "above every public result I found on the same US test (Nov 2026)".
The page currently makes no "top" claim, which is the safer position. The KDD Cup 2022 field (0.893-0.904,
three locales, private test) stays non-comparable, and the README already says so.

## 4. Success@1 and MRR from the saved test files (read only)

Definitions: success@1 = share of queries with `rr_exact == 1`. MRR = mean `rr_exact`. `rr_exact` is
1/position of the first Exact product over the full judged list, and 0 if there is none
(`eval/metrics.py:47-52`). Ties are broken by product_id ascending (`run_task1.py:83`).

| System (`*_test.parquet`, 8,956 queries) | nDCG (full) | success@1 | MRR | MRR in json |
|---|---|---|---|---|
| bm25_title_test | 0.8202 | 0.6268 | 0.7554 | 0.7554 |
| lambdamart_test | 0.8337 | 0.6580 | 0.7766 | 0.7766 |
| minilm_ft_mse_fast_test (Amazon recipe) | 0.8544 | 0.7161 | 0.8166 | 0.8166 |
| minilm_distil_w10_bullets_test (served) | 0.8674 | 0.7468 | 0.8369 | 0.8369 |
| deberta_oof_avg5_test (teacher) | 0.8877 | 0.7903 | 0.8661 | 0.8661 |

No-Exact queries: 1 of 8,956 (0.011%). It is the same query in every file, and I confirmed it against the
test labels in judgments.parquet (the query has products, none labelled Exact). It counts as `rr_exact = 0`,
i.e. a miss in success@1 and 0 in MRR. Excluding it moves success@1 and MRR by +0.0001 at most. No query has
nDCG 0. All five files cover the same query ids.

Paired bootstrap, served minus BM25 (10,000 query resamples, `np.random.default_rng(0)`, the same method as
`eval/bootstrap.py`):
- **success@1: +0.1199 [+0.1091, +0.1309]**. The served model gets the top product right where BM25 does not on
  1,859 queries, and the reverse happens on 785.
- MRR: +0.0815 [+0.0748, +0.0884]
- nDCG: +0.0472 [+0.0444, +0.0500]

Single-system success@1 95% CIs: BM25 [0.6169, 0.6368], served [0.7378, 0.7556], teacher [0.7817, 0.7987].
Per-query nDCG, served vs BM25 on test: better on 5,870 (65.5%), worse on 2,937 (32.8%), tied on 149.
