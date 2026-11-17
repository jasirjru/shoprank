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
