# Lab notebook

One entry per experiment or decision, newest at the bottom. Failed experiments stay in.

```
## YYYY-MM-DD — title
Prediction: what I expect and why (written BEFORE running)
Change: what changed (command / commit)
Result: metric + 95% CI, split
Conclusion: keep / revert / next
```

## 2026-11-06 — Week 1 setup
Change: repo, uv env, ESCI download + ingest + query-group split, Task-1 runner, Week 1 specs.
Result: ingest tests 5/5 pass. Specs for metrics/BM25/bootstrap verified against a private reference.
Found: DuckDB stores literal 0.1 as DECIMAL; gains cast to DOUBLE. DuckDB hash() replaced with MD5
for a version-stable split.
Conclusion: next, implement Week 1 tasks.

## 2026-11-06 — Data in; official metric checked
Change: downloaded ESCI (manifest hashes in data/raw/manifest.json); ingest in 36 s.
Result: US = 97,345 queries, 1.82M judgments (E 69%, S 20%, I 9%, C 2%), 1.22M products;
Task-1 dev 2,099 queries, Task-1 test 8,956 queries.
Found: public 0.850 / 0.9043 use nDCG over the FULL candidate list (trec_eval, no cutoff), averaged over
3 locales. Runner now reports nDCG@10 and full-list nDCG.
Conclusion: compare our full-list nDCG with public numbers as a guide only (US-only).

## 2026-11-06 — Week 1 BM25 experiments (dev, 2,099 queries)
Prediction: BM25 title > random; adding bullets helps (more words to match).
Change: metrics, bootstrap, BM25 implemented (30/30 tests). Runs: random, title, title+brand,
title+brand+bullets, b in {0, 0.75, 1}, k1 in {0.9, 1.5, 3}.
Result (full-list nDCG): random 0.7500; bm25_title 0.8193 (+0.0693 [+0.0622, +0.0763]).
Fields: no clear gain (CIs cross 0). b=0: −0.0105 [−0.0136, −0.0076]. k1=3: −0.0018 [−0.0029, −0.0007].
Conclusion: bullets prediction was wrong. BM25 plateaus ~0.82; tuning won't close the gap to 0.85+.
Keep bm25_title (k1 1.5, b 0.75) as the baseline. Error analysis: long NL queries, ignored constraints
(gender), accessory-vs-product, spelling, Spanish queries, and label noise (see week01-notes §6).

## 2026-11-07 — Week 2 learning to rank (dev, 2,099 queries)
Prediction: learned models beat BM25; LambdaMART beats logistic regression (listwise > pointwise).
Change: 14 text-match features; logistic regression from scratch (NumPy GD + L2) and scikit-learn;
LightGBM LambdaMART (label_gain = ESCI gains, early stopping on 10% held-out TRAIN queries).
Result (full-list nDCG): logreg ours 0.8296 = sklearn 0.8296; LambdaMART 0.8359 (+0.0166 [+0.0127,
+0.0205] vs BM25; +0.0062 [+0.0032, +0.0093] vs logreg). Ablations: no BM25 feats 0.8322 (−0.0037),
BM25 only 0.8223 (−0.0135). 274 trees. Top features: title_coverage, bm25_title, title_jaccard,
numbers_missing.
Bugs: NaN treated as truthy (fixed at load), wrong Jaccard expectation in my test, DECIMAL in fixture.
Conclusion: both predictions right. Word-match features plateau ~0.836; next: learned meaning (Weeks 3–5).

## 2026-11-08 — Week 3 two-tower from scratch (dev, 2,099 queries)
Prediction: beats random, probably below BM25 (no pretraining, easy negatives).
Change: scratch autograd + MLP (match PyTorch/finite differences); PyTorch two-tower, char-trigram hashing
(262,144 buckets, dim 128), InfoNCE with in-batch negatives + same-query mask, AdamW 1e-3, 3 epochs,
163,125 train Exact pairs.
Result (full-list nDCG): 0.7849 (batch 256, temp 0.05): +0.0350 vs random, −0.0343 [−0.0403, −0.0283] vs BM25.
Batch 32/1024: no clear difference. Temp 0.2: −0.0096; temp 1.0: −0.0243 (loss stuck near ln 256).
Peak GPU 1.3 GB. Overfit-one-batch plateau 0.37 explained by 41/256 duplicate product texts.
Conclusion: prediction right. Causes: no pretrained knowledge, easy in-batch vs hard Task-1 negatives,
no word order. Next: pretrained transformer (Week 4), hard negatives (Week 5).
Note: temp 1.0 run took 5,554 s because a game shared the GPU (55 s normally).

## 2026-11-20 — Rigour fixes (split by query text, graded LR, full-list early stopping, dev-2)
Prediction: numbers move only slightly; graded LR narrows the LR-vs-LambdaMART gap (it now sees S/C labels).
Change:
- Split by normalised query text (lowercase, non-[a-z0-9] runs to one space, trim), MD5 into 20 buckets:
  0-1 dev (~10%), 2 dev2 (~5%), rest train. Official-train rows whose text also occurs in test: dropped.
- Recount (Task 1, lowercase BEFORE the regex): 6 texts spanned two splits; 12 texts had >1 query_id
  (the earlier "103" was a counting bug). 15 Task-1 queries (367 rows) dropped. Cross-split texts now 0 (test added).
- logreg_graded: multinomial LR on 4 labels, ranked by expected gain sum_k p(k) * gain(k).
- LambdaMART early stopping on full-list nDCG (eval_at = longest list), not nDCG@10.
- Sizes (Task 1): train 17,819 q / 357,702 pairs; dev 2,000 / 40,294; dev2 1,054 / 21,290; test 8,956 unchanged.
Result (full-list nDCG, NEW dev, 2,000 queries; only 178 overlap the old dev, so not paired with old runs):
random 0.7479; BM25 0.8165; logreg 0.8258 (+0.0093 [+0.0053, +0.0133] vs BM25); logreg_graded 0.8258
(+0.0000 [−0.0011, +0.0011] vs logreg); LambdaMART 0.8342 (+0.0177 [+0.0137, +0.0216] vs BM25;
+0.0084 [+0.0052, +0.0117] vs graded LR), 231 trees. Ablations: no BM25 0.8268, BM25 only 0.8216.
dev2 (1,054 queries, scored once): BM25 0.8190; graded LR 0.8258; LambdaMART 0.8355
(+0.0165 [+0.0104, +0.0226] vs BM25; +0.0097 [+0.0050, +0.0146] vs graded LR).
Conclusion: graded-LR prediction WRONG: S/C labels add nothing to a linear model, so the gap to LambdaMART
is model type (non-linear feature interactions + listwise loss), not label information. Ranking of models
holds on dev2, so dev selection bias did not change the decision. Old runs: data/archive_pre_textsplit/.
Two-tower runs not re-run yet (old dev); re-run in the transformer section.
Correction (same day): the [a-z0-9] normaliser turned every non-Latin query (Korean, Chinese, Arabic)
into '' so unrelated queries merged into one "text"; one was in test, so 4 unrelated Task-1 train queries
were dropped. Normaliser now keeps letters/digits of any script ([^\p{L}\p{N}]+); test added. Re-run:
Sizes: train 17,823 q; dev 2,001; dev2 1,053; dropped 11. Full-list nDCG on dev: random 0.7500;
BM25 0.8163; logreg 0.8256 (+0.0093 [+0.0054, +0.0133]); logreg_graded 0.8254 (−0.0002 [−0.0012, +0.0009]
vs logreg); LambdaMART 0.8342 (+0.0180 [+0.0140, +0.0219] vs BM25; +0.0088 [+0.0055, +0.0121] vs graded
LR), 259 trees; no BM25 0.8266; BM25 only 0.8213. dev2: BM25 0.8191; graded LR 0.8260; LambdaMART 0.8364
(+0.0173 [+0.0113, +0.0234] vs BM25; +0.0104 [+0.0056, +0.0152] vs graded LR). Conclusions unchanged.
Early-stopping check (same data, same seed): eval_at=10 and eval_at=136 (full list) both stop at
259 trees, so dev full nDCG is identical (0.8342). The metric fix is correct but has no measured effect
here; the earlier 274 -> 231/259 tree change came from the new split, not from the stopping metric.

## 2026-11-20 — Week 4 zero-shot cross-encoder (dev, 2,001 queries)
Prediction: beats BM25 (pretrained meaning), below LambdaMART (never saw ESCI labels).
Change: cross-encoder/ms-marco-MiniLM-L-12-v2 (MS MARCO web passages, no ESCI training), input
"[CLS] query [SEP] product title [SEP]", max_length 128, RTX 4060, batch 64.
Result (full-list nDCG): title 0.8332 (+0.0169 [+0.0124, +0.0214] vs BM25; LambdaMART − it = +0.0011
[−0.0037, +0.0058], no clear difference). Title+brand 0.8334 (+0.0003 [−0.0014, +0.0019], no gain).
Speed: 40,304 pairs in 15.9 s = 2,534 pairs/s. Per query: MiniLM better on 897, LambdaMART on 1,035,
tie 69; per-query nDCG correlation 0.734.
Conclusion: prediction half wrong. With zero ESCI labels it TIES LambdaMART trained on 357,759 labelled
pairs. Low per-query correlation = different mistakes, so a blend (Week 6) should help. Next: fine-tune on
ESCI train (Week 5), target ~0.857 (Amazon's fine-tuned MiniLM, test set, not directly comparable).
