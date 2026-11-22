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

## 2026-11-20 — Section 12a: fine-tune MiniLM, Amazon's recipe (MSE on gain)
Prediction (written before the run): dev full nDCG 0.845-0.855, above zero-shot 0.8332 and LambdaMART
0.8342 with CIs above zero. Amazon's 0.857 is on the official test set, so not directly comparable.
Change: cross-encoder/ms-marco-MiniLM-L-12-v2, one output, MSE on the gain (E 1, S 0.1, C 0.01, I 0),
1 epoch, lr 7e-6, 5,000 warmup steps then linear decay, batch 32, AdamW wd 0.01, grad clip 1.0, seed 42,
title only, max_length 128 (<0.01% of train pairs longer; p99 69 tokens), bf16 autocast. Fit on 16,091
train queries / 323,302 pairs; checkpoint chosen every 1,000 steps on the 1,732-query train hold-out
(same hash as LambdaMART), never on dev. Code: src/shoprank/cross/finetune.py.
Measured speed (300-step smoke run): 14 s per 300 steps = ~690 pairs/s, so ~8 min per epoch.
Start loss 38.9: zero-shot logits are on a ±10 scale, gains on 0-1, so MSE first has to shrink the scale.
Result (full-list nDCG, dev 2,001 queries): 0.8549 (nDCG@10 0.7534). vs zero-shot +0.0218
[+0.0176, +0.0260]; vs LambdaMART +0.0207 [+0.0154, +0.0260]; vs BM25 +0.0386 [+0.0331, +0.0443].
Training: 444 s for 10,104 steps (measured, ~730 pairs/s). Hold-out nDCG: 0.8357 at step 300, dipped to
0.7999 at step 1,000 (output scale shrinking, loss 18.3), 0.8501 at 5,000, best 0.8581 at step 10,000.
Hold-out still rising slowly at the end, so 1 epoch may be under-trained (not tested).
Conclusion: prediction right (inside 0.845-0.855, near the top). Fine-tuned MiniLM is the new best single
model, clearly above LambdaMART. Model saved in data/processed/models/minilm_ft_mse/. Next: 12b, same
setup with 4-class cross-entropy + expected gain.

## 2026-11-20 — Section 12a speed-up: profile, pre-tokenise, length-bucketed batches
Prediction (written before profiling): the main waste is padding. Random batches pad every pair to the
longest pair in the batch, so about 40-50% of processed tokens are padding. Tokenising on the CPU inside
the loop costs about 10-15% of each step (explains the 78-87% GPU busy). After pre-tokenising once and
length-bucketed batches: epoch 444 s -> about 220-280 s. Dev nDCG within about ±0.003 of 0.8549 (batches
differ, so the result cannot be bit-identical).
Profile 1 (old loop; py-spy 40 s at 200 Hz on the training thread, nvidia-smi every 1 s): 46 ms/step.
Of the loop: 51% `loss.item()` (CPU waiting for the GPU), 27% forward (Python launching kernels), 7%
AdamW step, 6% tokenising, 4% grad clip. GPU 83% busy (76-90), 2.1 GB VRAM, 65 W. Padding measured on all
323,302 fit pairs: random batches 47.1% padding tokens, length-bucketed (100-batch chunks) 1.7%: 1.86x
fewer tokens. Tokenising everything once takes 7.8 s. (Tokenising guess 10-15% was too high: 6%.)
Run 2 (pre-tokenise + length-bucketed + loss kept on GPU): train 396 s (was 444 s, only 1.12x).
Dev 0.8535; vs old 12a -0.0014 [-0.0034, +0.0005], no clear difference. Dev scoring 12.9 s -> 7.0 s
(1.84x, matches the token cut). GPU busy fell to 72% (55-83).
Profile 2 (new loop, 30 s): `collate` 48% of the loop, and 98% of that is one line, `.to(device)`.
A copy from pageable RAM waits for all queued GPU work, so it became the new wait point; after it the
GPU queue is empty and the CPU launches small forward kernels one by one (GPU idles between them).
Prediction for run 3 (pinned memory + non_blocking copies for inputs and labels): GPU busy above 90%,
train 230-280 s, dev nDCG again within the noise of 0.8549.
Run 3 (pinned + non_blocking only): train 404 s, dev 0.8530, GPU 69% busy. No gain: prediction wrong.
Profile 3 (20 s): collate fell to 1%, but forward rose to 47.5%; the deepest frame is
transformers 5.23 masking_utils.py:366 `padding_mask.all()` inside a Python `if` (decides whether the
mask can be skipped). Reading that answer is a third wait point. Fix: `Prepared4DMask` passes a ready 4D
boolean mask, which transformers uses as-is. Checked on 5,000 real dev pairs with the 12a model: max score
difference 0.00e+00 (fp32 and bf16).
Prediction for run 4 (all three fixes): GPU busy above 90%, train 230-280 s, dev within noise of 0.8549.
Result run 4 (pre-tokenise + length-bucketed + loss on GPU + pinned copies + Prepared4DMask): train
316 s per epoch (was 444 s: 1.41x faster). GPU 91% SM busy (nvitop), 68 C, 66 W, 2.2 GB VRAM. Training
process ~5 GB RAM (pre-tokenised lists add ~0.8 GB). Dev 0.8534 (nDCG@10 0.7512); vs old 12a
-0.0015 [-0.0035, +0.0003], no clear difference. Dev scoring 12.9 s -> 7.0 s.
The three new-loop runs gave 0.8535 / 0.8530 / 0.8534 (same seed; GPU kernels are not bit-exact), all a
little under 0.8549. A small real cost from length-bucketed batches cannot be ruled out (CI upper end
+0.0003); separating it from batch-order luck needs several seeds (not done).
Conclusion: prediction partly wrong. Padding was the biggest waste (47%), but removing it alone gave only
1.12x, because the loop was also waiting for the GPU at three hidden sync points (`loss.item()`, pageable
`.to(device)`, transformers' `padding_mask.all()`). Each fix moved the wait to the next point; only
removing all three let the CPU run ahead. Lesson: re-profile after every fix. The GPU is now the limit
(91% busy). 12b uses this same loop, and its baseline is `minilm_ft_mse_fast`, so only the loss changes.

## 2026-11-20 — Section 12b: 4-class cross-entropy + expected gain (vs 12a MSE)
Prediction (written before coding and running): dev 0.850-0.858; difference vs `minilm_ft_mse_fast`
(same fast loop) inside [-0.004, +0.004], probably with a CI that covers zero. Reason: MSE on gains
1 / 0.1 / 0.01 / 0 treats S, C and I as almost the same target (0.1 vs 0), while CE learns all four
classes, so it could rank S above C/I better. Risk: the 4-class head starts from random weights (the
pretrained MS MARCO head has one output), and lr 7e-6 is small for a new head, so early hold-out nDCG
may start lower than 12a.
Change: only the loss and the head. Head: 4 outputs (E, S, C, I), new random weights. Loss: cross-entropy
on the class. Score: expected gain P(E) + 0.1 P(S) + 0.01 P(C). Everything else as 12a fast run:
lr 7e-6, 5,000 warmup steps, batch 32, 1 epoch, seed 42, title only, max_length 128, bf16.
Result (dev, 2,001 queries): full nDCG 0.8478 (nDCG@10 0.7430). Train 322 s, best hold-out step 9,000.
vs 12a fast (same loop, MSE): -0.0056 [-0.0094, -0.0018], MSE better (nDCG@10 -0.0083 [-0.0137, -0.0028]).
vs 12a original: -0.0071 [-0.0110, -0.0033]. vs LambdaMART +0.0136 [+0.0083, +0.0188]. vs zero-shot
+0.0147 [+0.0104, +0.0190].
Hold-out nDCG CE vs MSE: step 1,000 0.7268 vs 0.7975; 5,000 0.8260 vs 0.8524; 9,000 0.8449 vs 0.8568.
CE was behind at every checkpoint, and the gap shrank from 0.071 to 0.012 but never closed.
Conclusion: prediction wrong. Under Amazon's recipe (lr 7e-6, 5,000 warmup, 1 epoch), 4-class CE loses
to MSE with a CI below zero. This is not a clean "MSE beats CE" result: CE also starts from a new random
head, while MSE keeps the pretrained MS MARCO relevance head, and lr 7e-6 is small for a new head.
The hold-out curve says the CE model was still catching up. Fair re-test (not run): give the new head a
higher LR (for example 1e-3) or train longer, then compare again. Winner for section 13 for now: MSE.
Code review (python-reviewer agent, Sonnet, then checked by hand): Prepared4DMask is only correct with
sdpa attention (eager attention adds the mask, modeling_bert.py:128). Added a guard and a tiny-BERT test:
wrapper logits = 2D-mask logits, padding ignored. 71 tests pass.

## 2026-11-20 — Section 12c: two-tower re-run on the new dev, then same-query hard negatives
Prediction, re-run (written before running): same settings as Week 3 (batch 256, temp 0.05, 3 epochs, AdamW
1e-3), now on the text split (154,614 train Exact pairs, was 163,125). Dev full nDCG 0.775-0.795, still
below BM25 0.8163 with a CI below zero. The split fix removes leaked near-duplicate queries, so a small
drop vs the old 0.7849 is possible.
Result, re-run: 0.7865 (nDCG@10 0.6445), 56 s; vs BM25 -0.0298 [-0.0362, -0.0237]. Prediction right
(old split 0.7849: the split fix barely changed it). Still well below BM25.
Prediction, hard negatives (written before running): change only one thing, `--hard-negatives`: each
positive also brings one random non-Exact (S/C/I) product of the SAME query as an extra negative column
(never masked). Dev training now matches the test task (re-rank one query's candidates). Expect +0.015 to
+0.035 vs 0.7865 with a CI above zero, dev 0.800-0.820, probably still below BM25 0.8163 (no pretraining,
no word order). Caveat: the shared RNG also changes the batch order, not only the negatives.
Result, hard negatives: 0.7904 (nDCG@10 0.6494), 56 s, 203,145 non-Exact train products available.
vs in-batch only (0.7865): +0.0039 [+0.0009, +0.0069], a small real gain (nDCG@10 +0.0049 [+0.0007, +0.0091]).
vs BM25: -0.0259 [-0.0324, -0.0197], still clearly worse.
Conclusion: prediction wrong (expected +0.015 to +0.035). Same-query hard negatives help, but explain only
about 0.004 of the 0.030 gap to BM25 (~13%). Most of the gap comes from the model itself: no pretrained
knowledge, a bag of character trigrams averaged into one vector (no word order), and a small MLP. Note each
positive gets only 1 hard negative next to 255 easy in-batch columns, so the hard signal is diluted; more
hard negatives per positive was not tested. Review hypothesis 1 ("easy negatives explain the loss"):
partly true, not the main cause. Section 12 gate met: MSE beats CE (-0.0056 [-0.0094, -0.0018]).

## 2026-11-20 — Section 13: DeBERTa-v3-base cross-encoder (MSE on gain)
Setup bug found first (smoke run): loss NaN from step 2, even at lr ~0, also in fp32 autocast. Probes: grads
finite, weights finite, but after ONE AdamW step at lr 0, 201 of 202 weights were NaN. Root cause: the
DeBERTa-v3 checkpoint is stored in fp16 and transformers 5 loads it as fp16; in fp16 AdamW's eps (1e-8)
rounds to 0 and 99.7% of word-embedding grads are 0, so the update is 0/0 = NaN. Fused AdamW and SGD
were fine (fp32 inside). Fix: load with dtype=float32 (fp32 master weights, bf16 autocast for the maths)
plus a guard in fit() that refuses non-fp32 trainable weights (test added). MiniLM's checkpoint is fp32.
Smoke run after the fix (500 steps, measurement only): hold-out 0.8652, 75 s per 500 steps, peak VRAM
6.95 GB of 8. Also: tokens stored as int32 arrays and only needed product columns read; training process
3.7 GB RAM (MiniLM 12a used ~5 GB).
Prediction (written before the full run): dev full nDCG 0.865-0.880, above fine-tuned MiniLM 0.8534
(same loop) with a CI above zero. Epoch ~25 min train time.
Change: microsoft/deberta-v3-base, new 1-output head, MSE on gain (12a/12b winner), query + title + brand
+ color, max_length 128, batch 32, bf16 autocast, lr 2e-5, 1,000 warmup steps then linear decay, 1 epoch,
seed 42, checkpoint on the train hold-out every 2,500 steps.
Result (dev, 2,001 queries): full nDCG 0.8778 (nDCG@10 0.7952). Train 1,461 s (24 min), peak VRAM 6.95 GB
(nvidia-smi showed 7.9 GB reserved, no slowdown: ~145 s per 1,000 steps throughout). Hold-out: 0.8772 at
step 2,500, 0.8791 at 5,000, 0.8796 at 7,500, 0.8804 at 10,104 (best = last; rising slowly, no overfit).
vs MiniLM fine-tuned (fast loop) +0.0244 [+0.0197, +0.0292]; vs LambdaMART +0.0436 [+0.0377, +0.0495];
vs BM25 +0.0616 [+0.0554, +0.0678].
Conclusion: prediction right (0.865-0.880, near the top). New best single model. Not yet attributed: this
run changes model, fields (+brand+colour) and LR (2e-5, 1,000 warmup) together; the section 13 ablations
separate them. Seed spread still unknown (next step), so no claim below that spread.

## 2026-11-20 — Noise: seed spread for fine-tuned MiniLM and DeBERTa-v3-base
Why: three MiniLM runs with the same seed already gave 0.8530-0.8549, so differences of ~0.002 may be noise.
A claimed win must be bigger than the seed spread. The seed changes the batch order, the new head's start
weights and dropout.
Prediction (before running): seeds 42 / 43 / 44, same recipes as `minilm_ft_mse_fast` (0.8534) and
`deberta_v3_base_ft_mse` (0.8778). Spread (max - min) about 0.002-0.004 for each model; DeBERTa's
+0.024 over MiniLM stays far outside it.

## 2026-11-20 — Error analysis by query slice (dev, 2,001 queries)
Prediction (before running): DeBERTa's gain over MiniLM is largest on negation and 6+ word queries (needs
meaning and word order) and smallest on 1-2 word queries. Spanish-like queries score lowest for every model
(English pretraining). Brand queries score high for all models, since the brand word matches the title.
Code: src/shoprank/eval/slices.py (rule-based tags, tests/test_slices.py) plus Amazon's query `source` labels.
Result (full-list nDCG; BM25 / LambdaMART / MiniLM-ft / DeBERTa; DeBERTa - MiniLM with 95% CI):
- all (2,001 q): 0.8163 / 0.8342 / 0.8534 / 0.8778; +0.0244 [+0.0197, +0.0292]
- 1-2 words (379): 0.8084 / 0.8313 / 0.8552 / 0.8739; +0.0187 [+0.0083, +0.0293]
- 3-5 words (1,334): 0.8191 / 0.8358 / 0.8509 / 0.8789; +0.0281 [+0.0223, +0.0339]
- 6+ words (288): 0.8138 / 0.8311 / 0.8629 / 0.8780; +0.0152 [+0.0024, +0.0274]
- has number (413): 0.8033 / 0.8340 / 0.8584 / 0.8770; +0.0186 [+0.0089, +0.0286]
- negation rule (158): 0.7481 / 0.7683 / 0.8273 / 0.8391; +0.0118 [-0.0012, +0.0244]
- brand in query (338): 0.8292 / 0.8526 / 0.8591 / 0.8881; +0.0290 [+0.0170, +0.0414]
- Amazon source: negations (141) 0.7466 / 0.7689 / 0.8295 / 0.8395, +0.0100 [-0.0032, +0.0238];
  parse_pattern (181) 0.8427 / 0.8475 / 0.8690 / 0.8931, +0.0240 [+0.0133, +0.0349];
  behavioral (62) 0.8478 / 0.8602 / 0.8691 / 0.8741, +0.0050 [-0.0153, +0.0259];
  other (1,609) 0.8191 / 0.8380 / 0.8534 / 0.8797, +0.0263 [+0.0207, +0.0319]
- spanish_like rule: only 4 queries, too few to read; nlqec source: 8 queries, too few.
Conclusion: prediction mostly wrong.
- Negation is the hardest slice for every model. The big jump there is lexical -> transformer (MiniLM
  +0.079 over BM25 vs +0.037 overall), not MiniLM -> DeBERTa (+0.012, CI touches 0).
- DeBERTa's gain is largest on 3-5 word and brand queries; the brand gain is confounded with the added brand
  field (ablation next).
- Our negation rule and Amazon's `negations` source agree (158 vs 141 queries, same scores), which
  cross-checks the rule.
Next: per-pair scores for the 50 worst DeBERTa queries and a 100-pair label audit (needs the GPU free).
Result, MiniLM (fast loop): seed 42 0.8534, seed 43 0.8536, seed 44 0.8537: spread 0.0003 (prediction of
0.002-0.004 was too pessimistic). Consequence: the old loop's 0.8549 (random batches) is 0.0013 above all three
seeds, about 4x the seed spread. So length-bucketed batches most likely cost a small real ~0.001 of nDCG
(the earlier paired CI, -0.0015 [-0.0035, +0.0003], could not separate it). Trade-off accepted for now:
1.41x faster training; noted as a known cost. DeBERTa seeds 43/44 running.
Result, DeBERTa-v3-base: seed 42 0.8778, seed 43 0.8785, seed 44 0.8794: mean 0.8786, spread 0.0016
(seed 43 vs 42: +0.0006 [-0.0021, +0.0033]; seed 44 vs 42: +0.0015 [-0.0014, +0.0044], both "no clear
difference"). MiniLM mean 0.8536, spread 0.0003.
Conclusion: prediction right for DeBERTa (0.002-0.004 expected, 0.0016 seen); MiniLM is even more stable.
DeBERTa - MiniLM = +0.025 on the means, ~15x DeBERTa's spread: real. Rule for every later DeBERTa change:
a claimed gain must exceed ~0.002 AND have a paired CI above zero. Training time per seed: 24-26 min.

## 2026-11-20 — Section 13 ablations: which input fields matter? (DeBERTa-v3-base, MSE, seed 42)
Base: title + brand + colour = 0.8778 (seed 42); 3-seed mean 0.8786, spread 0.0016. Threshold for a real
effect: > ~0.002 and a paired CI that excludes zero. One change per run, everything else fixed.
Primary sources: 1st place used `color: brand: description: <title + bullets + description>` at length 128;
2nd place title + brand + colour at length 400. Loss ablation (listwise) skipped: 2nd place reported MSE
beat pairwise and converged faster, and 12b already favoured MSE over 4-class CE on this data.
Prediction (before running):
- A1 title only: 0.000 to -0.005 vs base (brand helps brand queries; zero-shot MiniLM showed brand adds
  ~0, but a fine-tuned model can use it).
- A2 + bullets (title, brand, colour, bullets; cut at 128 tokens): +0.002 to +0.008 vs base (more product
  facts, as 1st place), at ~1.5-2x training time.

Result (dev, 2,001 queries; lr 2e-5, warmup 1000, max_length 128, seed 42, one seed each):

| Run | Fields | nDCG full | nDCG@10 | vs base (full list, paired 95% CI) | Train time |
|---|---|---|---|---|---|
| base `deberta_v3_base_ft_mse` | title, brand, colour | 0.8778 | 0.7952 | | 1,461 s |
| A1 `deberta_abl_title` | title | 0.8777 | 0.7941 | -0.0002 [-0.0032, +0.0029] no difference | 1,621 s |
| A2 `deberta_abl_bullets` | title, brand, colour, bullets | **0.8851** | **0.8045** | **+0.0072 [+0.0039, +0.0106]** | 2,387 s |

A2 vs A1: +0.0074 [+0.0039, +0.0109]. nDCG@10, A2 vs base: +0.0093 [+0.0045, +0.0143].
Peak VRAM 6.94 / 6.99 GB. A2's best checkpoint was the last step (10,104), as for the base run.

By slice (A2 - base): negation +0.0140 [+0.0049, +0.0234], has a number +0.0096 [+0.0030, +0.0161],
6+ words +0.0087 [+0.0008, +0.0167], 3-5 words +0.0081; brand queries +0.0019 [-0.0068, +0.0099] (no
difference). Title-only vs base on brand queries: 0.8886 vs 0.8881, so the brand field adds nothing even there.

Conclusions:
- A1 prediction (0 to -0.005) right at the edge: brand + colour add nothing measurable. The brand is
  usually already in the title (the 0.029 DeBERTa-over-MiniLM gain on brand queries comes from the model,
  not the brand field).
- A2 prediction (+0.002 to +0.008) right: +0.0072, about 4.5x the DeBERTa 3-seed spread (0.0016) and 0.0065
  above the 3-seed mean (0.8786), so a real win even though A2 has one seed. Cost: 1.63x training time.
- Bullets help most where the title cannot answer: negation ("without X" needs the product's facts) and
  numbers (sizes, counts, wattage live in the bullets).
- Not done: bullets at max_length 256 (plan item). At 128 tokens the bullets are cut after the first ~60-80
  tokens; 256 would cost ~2x again (~80 min per run). Deferred: the stack and the distillation come first.
Decision: the overnight OOF run uses title, brand, colour, bullets (night.sh picked it by the > base + 0.002
rule, written before the result was known).

## 2026-11-20 — Worst-50 queries and 100-pair label-noise audit (DeBERTa seed 42, dev)
Files: docs/analysis/worst50_deberta_dev.csv, docs/analysis/label_audit100_deberta_dev.csv (built from per-pair
scores, src/shoprank/cross/score.py). Audit set = the 50 label-I pairs DeBERTa scores highest (0.95-1.05) and the
50 label-E pairs it scores lowest (0.02-0.08). Judged against the official ESCI definitions by a reviewer agent
(Sonnet), then spot-checked by hand (4 rows and all counts re-computed; one count corrected: 50, not 55, of the
100 rows come from the top-10 queries; 60 distinct queries; 31172 "cybling womens shoes" alone gives 16 rows).
Prediction: none written before this audit (it is descriptive, not an experiment); recorded as a gap.
Result (row level): label likely wrong 75, model wrong 14, ambiguous 11 (label-I group 37/7/6, label-E group
38/7/5). Verified examples: "cybling womens shoes" -> CYBLING shoes labelled I (label noise); "23mm socket" ->
a 23mm socket labelled I (noise); "white out" -> the film Whiteout (2009) scored 1.04 (model wrong: keyword
collision); "hat for toilet" -> specimen collector pan labelled E, scored 0.045 (label right: model lacks slang).
Model-error types (14 rows): slang/jargon 7 (one query), wrong product type from keyword overlap 5, title collision 2.
Worst 50 (nDCG 0.281-0.506): ambiguous/garbled query 18, label noise 7, numbers/model-number spec 7, few Exact 6,
other 5, brand confusion 3, negation 2, accessory vs main 2. 25 of the 50 have exactly one Exact product, so their
nDCG is unstable (one Exact ranked 3rd instead of 1st costs a lot).
Conclusion: among the strongest disagreements most errors are label errors, but they cluster in few queries, so
75% over-states the query-level rate. It still supports the paper's 91% agreement ceiling: the remaining headroom
over 0.879 is small and partly unreachable. The most fixable real error is "wrong product type from keyword
overlap" (needs meaning, not word match). Section 16 should also report nDCG on queries with >= 2 Exact products.

## 2026-11-21 (overnight) — 5-fold out-of-fold (OOF) scores: DeBERTa-v3-base and MiniLM
Purpose: honest train-set scores for stacking (section 14) and teacher targets for distillation (section 15).
Fields for DeBERTa chosen automatically by the chain script: bullets only if `deberta_abl_bullets` beats the
base 0.8778 by > 0.002, else title+brand+colour. Folds by query (text hash, independent of the hold-out).
Prediction (before running): each fold model's dev nDCG within ~0.003 of its full model (DeBERTa ~0.875-0.880,
MiniLM ~0.851-0.854; 80% of the training data, no checkpoint choice); the 5-fold average on dev slightly above a
single fold model (+0.001 to +0.004, an ensemble effect). DeBERTa fold ~25 min, MiniLM fold ~6 min.
Addendum (01:05, before any fold result): the rule picked title, brand, colour, bullets (A2 0.8851). The
prediction above shifts with it: DeBERTa fold models ~0.881-0.886 on dev, 5-fold mean ~0.885-0.889. Fold time
~35 min (bullets make inputs longer), so DeBERTa folds end ~04:00 and MiniLM folds ~04:30.

## 2026-11-21 (overnight) — Section 14: z-scored blend and LightGBM stacking (code: src/shoprank/ltr/stack.py)
What the primary sources did: KDD Cup 2022 Task 1 top teams blended several cross-encoders; 1st and 2nd place
fed cross-encoder scores plus per-query statistics into a LightGBM ranker. Copied: LightGBM LambdaMART on
[14 text features + cross-encoder scores + per-query z, percentile, gap-to-best, query max/median/min].
Changed: two cross-encoders (not 5-10, GPU budget); every learned weight is fitted on TRAIN out-of-fold scores
(blend weights by grid search, stacker with the same early-stopping hold-out as LambdaMART), never on dev.
dev2 is scored once, after the dev numbers are written down, for all three systems at once (no choice on dev2).
Known risk: TRAIN rows carry one fold model's score, dev rows the mean of 5 fold models (smoother scores).
The per-query rank / z features should make the stacker robust to that shift.
Prediction (before running), dev full-list nDCG:
- `deberta_oof_avg5` 0.885-0.889; `minilm_oof_avg5` 0.853-0.856.
- `zblend` (DeBERTa, MiniLM, BM25-title z-scores): DeBERTa weight >= 0.7; +0.000 to +0.003 over
  `deberta_oof_avg5` (MiniLM is weaker and highly correlated, BM25 adds exact-match signal).
- `stack_lgbm`: +0.002 to +0.006 over `deberta_oof_avg5`; top features: the DeBERTa score and its query-z /
  gap. If below the blend, the train/dev score shift is the first suspect.
- dev2: same order of the three systems, each within +-0.005 of its dev value.

Result (dev, 2,001 queries; all weights and stackers fitted on TRAIN OOF scores):

| Run | nDCG full | nDCG@10 | Paired 95% CI (full list) |
|---|---|---|---|
| `deberta_abl_bullets` (one model, 100% of TRAIN) | 0.8851 | 0.8045 | |
| `deberta_oof_avg5` (mean of the 5 fold models) | **0.8879** | **0.8094** | vs one model +0.0028 [+0.0004, +0.0051] |
| `zblend` (z: 0.8 DeBERTa, 0.1 MiniLM, 0.1 BM25-title) | 0.8876 | 0.8079 | vs avg5 -0.0003 [-0.0019, +0.0013] |
| `stack_lgbm` (14 features + scores + query stats) | 0.8879 | 0.8083 | vs avg5 +0.0000 [-0.0018, +0.0019] |
| `stack_lgbm_rel` (14 features + z / percentile only) | 0.8875 | 0.8075 | vs avg5 -0.0003 [-0.0022, +0.0015] |
| `minilm_oof_avg5` | 0.8523 | 0.7503 | vs `minilm_ft_mse_fast` -0.0012 [-0.0027, +0.0004] |

Per fold model on dev: DeBERTa 0.8858 / 0.8835 / 0.8849 / 0.8844 / 0.8850 (mean 0.8847, spread 0.0022);
MiniLM 0.8517 / 0.8516 / 0.8516 / 0.8538 / 0.8528. TRAIN OOF nDCG: DeBERTa 0.8828, MiniLM 0.8538.
Stacker feature importance (gain share): DeBERTa query-z 0.81, raw DeBERTa 0.07, DeBERTa percentile 0.03,
MiniLM query-z 0.02; the 14 text features together < 0.02.
`stack_lgbm_rel` was added after the reviewer agent flagged the train (one model) vs dev (mean of 5) score
shift. It is not better, so the shift is not what limits the stack.

Prediction check:
- `deberta_oof_avg5` 0.885-0.889: right (0.8879). Fold models were as good as the 100% model (0.8847 vs
  0.8851): 20% less data cost almost nothing; averaging 5 of them gained +0.0028 (CI excludes 0), about the
  size predicted for the ensemble effect.
- `minilm_oof_avg5` 0.853-0.856: wrong, a little low (0.8523). MiniLM fold models lose ~0.001-0.002 from 20%
  less data, and averaging similar MiniLMs gains little (their seed spread is only 0.0003).
- `zblend` +0.000 to +0.003: wrong (-0.0003, no difference). `stack_lgbm` +0.002 to +0.006: wrong (+0.0000).
Why stacking failed here (unlike the KDD Cup teams): they stacked 5-10 strong, *different* cross-encoders
(DeBERTa, BERT, COCO-LM, multilingual models; different inputs). We have one strong model and one much weaker
model of the same kind, plus text features that a cross-encoder already "reads". There is little the
stacker can learn that the DeBERTa score does not already contain.
Decision: the best dev system is `deberta_oof_avg5` (5 DeBERTa models averaged). The stack is not kept: same
nDCG at more complexity. Per the plan, dev2 is now scored once for all systems (`stack.py --dev2`).

Result, dev2 (1,053 queries, scored once at ~05:50, after everything above was written):

| Run | dev | dev2 | dev2 paired 95% CI vs `deberta_oof_avg5` |
|---|---|---|---|
| `lambdamart` | 0.8342 | 0.8364 | avg5 - LambdaMART +0.0448 [+0.0372, +0.0525] |
| `deberta_oof_avg5` | 0.8879 | 0.8812 | |
| `zblend` | 0.8876 | 0.8826 | +0.0014 [-0.0006, +0.0034] |
| `stack_lgbm` | 0.8879 | 0.8845 | **+0.0033 [+0.0008, +0.0058]** |
| `stack_lgbm_rel` | 0.8875 | 0.8831 | +0.0019 [-0.0005, +0.0042] |

dev + dev2 pooled (3,054 queries), `stack_lgbm` - `deberta_oof_avg5`: +0.0012 [-0.0003, +0.0027].
Prediction check: "same order, each within +-0.005 of dev" was wrong. The order changed (the stack is best on
dev2, tied on dev), and `deberta_oof_avg5` fell 0.0067.
Reading (honest):
- dev and dev2 disagree about the stack: 0.0000 on dev, +0.0033 on dev2. Pooled, the stack is at most a small
  gain (+0.0012, CI touches 0). Neither split alone is the truth; the pooled number is the best estimate.
- Selection bias check: LambdaMART goes UP 0.0022 from dev to dev2, DeBERTa goes DOWN 0.0067. Choices made on
  dev (DeBERTa learning rate, bullets, "avg5 is best") fit dev a little. So DeBERTa's dev numbers are probably
  optimistic by a few thousandths; the official test set (section 16) is the unbiased number.
- Nothing is re-tuned after this look. dev2 is now spent.
Open decision for the learner (before section 16): which system goes to the one-time test run. Option 1:
`deberta_oof_avg5` (decided on dev, simplest to serve). Option 2: also report `stack_lgbm` as a declared second
system (both scored once on test, no choice made from test).

## 2026-11-21 (overnight) — Distillation: MiniLM learns from DeBERTa's out-of-fold scores
Why: DeBERTa + bullets is the most accurate, but 184M parameters is slow on CPU for serving (section 15).
Distillation trains the small model to copy the big model's scores, which carry more information than the
4 human grades (e.g. "this Substitute is almost Exact"). Source: Hofstaetter et al. 2020 (cross-architecture
knowledge distillation for rankers) used Margin-MSE on teacher score differences; we start with the simplest
pointwise version: MSE on the teacher's score, because our recipe is already pointwise MSE on the gain.
Teacher = `deberta_oof` TRAIN scores (out-of-fold: the teacher never trained on the query it scores, so it
cannot simply hand back the memorised labels). Student = MiniLM, title only, Amazon recipe, seed 42, so the
only change vs `minilm_ft_mse_fast` (0.8534; 3-seed spread 0.0003) is the target.
Runs: `minilm_distil_w10` (target = teacher score), `minilm_distil_w05` (0.5 gain + 0.5 teacher).
Prediction (before running): w10 0.856-0.862 (+0.002 to +0.008 over 0.8534); w05 between 0.8534 and w10.
The student cannot reach the teacher (~0.886): it reads the title only, the teacher also read the bullets.
Result (dev, 2,001 queries; MiniLM, title only, seed 42; ~5 min training each, peak VRAM 1.37 GB):

| Run | Target | nDCG full | nDCG@10 | vs `minilm_ft_mse_fast` 0.8534 (paired 95% CI) |
|---|---|---|---|---|
| `minilm_distil_w10` | DeBERTa OOF score | **0.8575** | 0.7571 | **+0.0041 [+0.0017, +0.0065]** |
| `minilm_distil_w05` | 0.5 gain + 0.5 teacher | 0.8564 | 0.7561 | +0.0029 [+0.0011, +0.0048] |

w10 vs w05: +0.0011 [-0.0005, +0.0028] (no clear difference). Teacher (`deberta_oof_avg5`) is still
+0.0304 [+0.0257, +0.0349] above the w10 student.
Prediction check: w10 0.856-0.862: right (low end, 0.8575); w05 between 0.8534 and w10: right.
+0.0041 is ~14x MiniLM's seed spread (0.0003): a real gain from changing only the training target.
The student keeps only ~13% of the teacher's lead over plain MiniLM (0.0041 of 0.0345): reading the title
only, it cannot see what the teacher read in the bullets.

## 2026-11-21 (early morning) — Distillation with bullets: MiniLM student reads what the teacher read
One change vs `minilm_distil_w10`: fields title, brand, colour, bullets (the teacher's input), max_length 128.
Prediction (before running): 0.862-0.870 (+0.004 to +0.012 over `minilm_distil_w10`), training ~1.5x longer.
Still well below the teacher (0.8879): 33M vs 184M parameters.
Result: `minilm_distil_w10_bullets` 0.8667 (nDCG@10 0.7717): +0.0092 [+0.0056, +0.0128] over the title-only
student, +0.0133 [+0.0095, +0.0171] over plain MiniLM 0.8534. Teacher still +0.0212 [+0.0170, +0.0254] above.
Training 467 s (1.5x), dev scoring 26 s for 40,304 pairs on the GPU. Prediction 0.862-0.870: right.
Open confound: this run changed the input AND kept the teacher, so part of the +0.0133 may be bullets alone.

## 2026-11-21 (early morning) — Control: MiniLM with bullets, human labels only (no teacher)
One change vs `minilm_distil_w10_bullets`: target = gain (no teacher). Separates "bullets" from "distillation".
Prediction (before running): 0.858-0.862, i.e. distillation still worth +0.005 to +0.009 with bullets.
Result: `minilm_ft_bullets` 0.8602 (nDCG@10 0.7613), +0.0068 [+0.0032, +0.0104] over title-only MiniLM.
Distillation on top of bullets: `minilm_distil_w10_bullets` - `minilm_ft_bullets` = +0.0064 [+0.0038, +0.0090].
Prediction (0.858-0.862; distillation +0.005 to +0.009): right on both.
The +0.0133 of the best student splits cleanly into two real, roughly equal parts:

| Step (MiniLM, seed 42) | dev nDCG | Gain (paired 95% CI) |
|---|---|---|
| title, human labels (`minilm_ft_mse_fast`) | 0.8534 | |
| + bullets (`minilm_ft_bullets`) | 0.8602 | +0.0068 [+0.0032, +0.0104] |
| + DeBERTa teacher (`minilm_distil_w10_bullets`) | **0.8667** | +0.0064 [+0.0038, +0.0090] |
| teacher: mean of 5 DeBERTa | 0.8879 | +0.0212 above the student |

Caveat from the review: the teacher's OOF scores on the student's hold-out queries came from fold models that
trained on other hold-out queries; this only affects the student's checkpoint choice (best step), not dev.
Serving candidate for section 15: `minilm_distil_w10_bullets` (33M parameters, 26 s for 40k dev pairs on GPU).

## 2026-11-21 — Pre-registration for the one-time official test run (section 16), decided by the learner
Test split = official ESCI Task 1 test, US, small version: 8,956 queries / 181,701 pairs, never scored so far.
Declared before any test score is seen (no choice will be made from test):
- Primary system: `deberta_oof_avg5` (mean of the 5 DeBERTa-v3-base fold models; title, brand, colour, bullets).
- Declared second system: `stack_lgbm` (LightGBM stack on the 14 features + DeBERTa/MiniLM OOF scores).
- Also reported, not candidates: `bm25_title`, `lambdamart`, `minilm_ft_mse_fast`, and the serving model
  `minilm_distil_w10_bullets` (its test number goes in the section 15 quality-vs-latency table).
- External references: Amazon's fine-tuned MiniLM 0.857 (English, official test), public project 0.854.
- Each system scored exactly once; nDCG with bootstrap 95% CIs; re-checked with pytrec_eval (trec_eval,
  gains x100, which leaves nDCG unchanged).
Prediction (before running): `deberta_oof_avg5` 0.878-0.886 (dev 0.8879 is a little optimistic, dev2 0.8812);
`stack_lgbm` - `deberta_oof_avg5` between -0.001 and +0.003 (pooled dev+dev2 +0.0012); both above Amazon's
0.857 by 0.02-0.03; `minilm_distil_w10_bullets` 0.860-0.868; BM25 and LambdaMART within 0.005 of dev.

## 2026-11-21 — Section 15: ONNX export + INT8 of the served model (`minilm_distil_w10_bullets`)
What the sources do: ONNX Runtime's own guide for BERT-family models is export -> transformer optimizer
(fused attention / LayerNorm / GELU) -> dynamic INT8 quantisation; Amazon's esci-data baseline has no serving
code. Copied as is; per-channel weight quantisation (usually loses less accuracy than per-tensor).
Prediction (before running): ONNX fp32 matches PyTorch fp32 to < 1e-4 per score; INT8 model ~4x smaller
(~127 MB -> ~32 MB); INT8 dev nDCG within 0.002 of fp32 (0.8667), and 2-3x more pairs/s than fp32 on CPU.
(Timings in this step run while the GPU test scoring also uses the CPU; the load test later is the clean one.)
Result (dev, 2,001 queries / 40,304 pairs; files in data/processed/models/onnx/minilm_distil_w10_bullets/):

| Model | Size | dev nDCG full | Max abs score diff vs PyTorch fp32 (512 pairs) |
|---|---|---|---|
| PyTorch, bf16 autocast (training eval) | 127 MB | 0.8667 | |
| ONNX fp32, fused (`model_opt.onnx`) | 127 MB | 0.8662 | 9.5e-07 |
| ONNX INT8 dynamic, per-channel (`model_int8.onnx`) | **32 MB** | **0.8654** | 8.7e-02 |

Prediction check: parity < 1e-4 right (9.5e-07: the ONNX graph computes the same function); 4x smaller
right (127 -> 32 MB); INT8 within 0.002 right (-0.0008 vs ONNX fp32). The bf16 vs fp32 gap (0.0005) is
numeric noise of the same size. Speed: 156 vs 207 pairs/s (1.3x), but measured while the GPU test run was
also using the CPU, so not a fair number; the load test measures it cleanly.
Fix needed on the way: quantize_dynamic failed on the fused graph ("Unable to find data type ... shape
inference failed") because the fused ops are ONNX Runtime-only; extra_options DefaultTensorType=FLOAT fixed it,
as the error message itself suggests.

## 2026-11-21 — Section 15: ranking API load test (FastAPI + ONNX Runtime on CPU)
Setup: `uvicorn shoprank.serve.app:app`, 1 process, ONNX Runtime default threads (32 logical CPUs, laptop on
mains power, Turbo plan). Requests = real dev queries with their full candidate lists (median ~20 products,
title + brand + colour + bullets). Client and server on the same laptop (no network cost). 300 timed requests
per level after 20 warm-up requests. Run after the GPU test scoring finished, nothing else heavy running.
Prediction (before running):
- INT8 at concurrency 1: p50 30-80 ms, p95 < 150 ms per request (about 300-700 pairs/s).
- INT8 vs fp32 (`model_opt.onnx`): 1.5-2.5x more pairs/s at the same concurrency.
- Concurrency 16: throughput only +0-50% over concurrency 1 (one request already uses all cores), and
  p95 grows to several hundred ms (requests wait in line).

## 2026-11-21 — Section 16: official test results (scored once, 07:31-08:18, as pre-registered)
Official ESCI Task 1 test, US, small version: 8,956 queries / 181,701 pairs. 95% CIs: query bootstrap, 10,000
resamples. One incident: the first `final_test` attempt hit CUDA out-of-memory on the first DeBERTa fold at
batch 128 before any score was written; re-run at batch 64. BM25 had been scored once before the crash and was
not re-scored.

| System | nDCG full [95% CI] | nDCG@10 [95% CI] | dev | dev2 |
|---|---|---|---|---|
| BM25 (title) | 0.8202 [0.8170, 0.8232] | 0.6942 [0.6899, 0.6986] | 0.8163 | |
| LambdaMART (14 features) | 0.8337 [0.8305, 0.8368] | 0.7172 [0.7128, 0.7215] | 0.8342 | 0.8364 |
| MiniLM, Amazon recipe (ours) | 0.8544 [0.8513, 0.8573] | 0.7527 [0.7487, 0.7568] | 0.8534 | |
| MiniLM distilled + bullets (served) | 0.8674 [0.8643, 0.8703] | 0.7743 [0.7702, 0.7783] | 0.8667 | |
| **DeBERTa-v3-base x5 mean (primary)** | **0.8877 [0.8849, 0.8905]** | **0.8092 [0.8053, 0.8131]** | 0.8879 | 0.8812 |
| LightGBM stack (declared second) | 0.8883 [0.8855, 0.8910] | 0.8099 [0.8060, 0.8137] | 0.8879 | 0.8845 |

Paired (test): primary - BM25 +0.0676 [+0.0646, +0.0705]; primary - LambdaMART +0.0541 [+0.0513, +0.0568];
stack - primary +0.0006 [-0.0003, +0.0014] (no clear difference); served MiniLM - Amazon-recipe MiniLM
+0.0130 [+0.0112, +0.0148]; primary - served MiniLM +0.0204 [+0.0184, +0.0223].
External references: Amazon's fine-tuned MiniLM 0.857 (English) and the public project 0.854 are both below
the primary system's CI (+0.031 and +0.034). Our own run of Amazon's recipe gives 0.8544, close to their 0.857
(we train on 80% of their training data: dev and dev2 are held out), which supports that the metric matches.
Independent check: pytrec_eval (trec_eval) means equal ours within 0.0003 for every system. Per-query
differences came only from tied scores: trec_eval orders ties by document id descending, we ascending. With
trec_eval's tie rule our nDCG matches it exactly (max per-query diff 7.8e-16, served MiniLM). Ties are common
(14.4% of the served model's pairs) because scores pass through bf16; the tie order moves the mean by 0.00005.
Prediction check:
- primary 0.878-0.886: wrong, slightly better (0.8877). dev was not optimistic after all; dev2 (1,053 queries)
  was the low one. The selection-bias worry from dev2 did not show up on 8,956 test queries.
- stack - primary in [-0.001, +0.003]: right (+0.0006, CI holds 0). Stacking adds nothing measurable.
- above Amazon 0.857 by 0.02-0.03: right at the edge (+0.031).
- served MiniLM 0.860-0.868: right (0.8674). BM25 and LambdaMART within 0.005 of dev: right (+0.0039, -0.0005).
Test is now spent: no model, setting or choice changes because of these numbers.
Follow-up idea (not done): score heads in fp32 instead of bf16 to remove ties.
Result (dev requests: 2,001 queries, median 16 candidates, max 84; latency seen by the client):

| Model | Concurrency | p50 ms | p95 ms | p99 ms | req/s | pairs/s |
|---|---|---|---|---|---|---|
| INT8 | 1 | 67.8 | 232.7 | 261.9 | 11.0 | 222 |
| INT8 | 4 | 377.4 | 871.4 | 929.3 | 9.2 | 187 |
| INT8 | 16 | 1,880 | 4,018 | 4,855 | 7.5 | 151 |
| fp32 | 1 | 109.3 | 300.9 | 306.4 | 7.5 | 152 |
| fp32 | 4 | 408.2 | 927.6 | 982.5 | 8.5 | 172 |
| fp32 | 16 | 1,829 | 3,896 | 5,102 | 7.6 | 155 |

No errors. Prediction check: INT8 c=1 p50 30-80 ms right (68); p95 < 150 ms wrong (233: long tail from
requests with many candidates and long bullets); INT8 vs fp32 1.5-2.5x: just below (1.46x pairs/s at c=1);
c=16 throughput +0-50%: wrong, it FELL 32% (222 -> 151 pairs/s) and INT8's lead vanished.
Diagnosis: every request asks ONNX Runtime for all 32 logical CPUs, so 16 parallel requests run 16 x 32
threads on 16 physical cores: the threads fight (context switches, cache thrashing).
Next, one change: at most 1 model call at a time per process (a semaphore, others wait in line) and 16 threads
(physical cores; hyper-threads add little to matrix maths).
Prediction (before running): INT8 c=1 >= 222 pairs/s; c=4 and c=16 keep the c=1 throughput (within 10%),
with p50 growing about linearly with concurrency (queueing, not fighting).
Result (SHOPRANK_THREADS=16, SHOPRANK_MAX_CONCURRENT=1; same requests):

| Model | Concurrency | p50 ms | p95 ms | p99 ms | req/s | pairs/s |
|---|---|---|---|---|---|---|
| INT8 | 1 | 69.1 | 232.4 | 254.7 | 10.7 | 217 |
| INT8 | 4 | 327.7 | 572.2 | 704.7 | 11.3 | 229 |
| INT8 | 16 | 1,436 | **1,784** (was 4,018) | **1,950** (was 4,855) | 11.3 | **230** (was 151) |
| fp32 | 1 | 115.2 | 313.2 | 335.3 | 7.1 | 143 |
| fp32 | 16 | 2,166 | 2,607 | 2,762 | 7.4 | 151 |

Prediction check: right. Throughput no longer falls with load (INT8 217 -> 230 pairs/s, +6%), p95 at
concurrency 16 is 2.3x lower, and INT8 keeps a 1.5x lead over fp32 under load. Latency at high load is now
pure queueing (p50 grows ~linearly with concurrency), which is the honest capacity limit of one laptop
process: ~11 requests/s of ~16 candidates with bullets. Kept as the default (MAX_CONCURRENT=1).

Quality vs latency (served-model options; nDCG on test where it exists, else dev):

| Option | Hardware | Test nDCG | Dev nDCG | Throughput | Latency (1 request, ~16 candidates) |
|---|---|---|---|---|---|
| BM25 title | CPU | 0.8202 | 0.8163 | not measured in the API (no neural model) | |
| MiniLM distilled + bullets, INT8 ONNX (**served**) | CPU, 16 threads | (0.8674 PyTorch) | 0.8654 | 230 pairs/s | p50 69 ms, p95 232 ms |
| same, fp32 ONNX | CPU, 16 threads | (0.8674 PyTorch) | 0.8662 | 151 pairs/s | p50 115 ms, p95 313 ms |
| DeBERTa-v3-base x5 mean | GPU (RTX 4060), bf16 | 0.8877 | 0.8879 | ~89 pairs/s (443 per model, offline batches) | not served |

The INT8 model was not scored on test (test is spent); its dev loss vs fp32 is 0.0008. The DeBERTa ensemble
is 0.020 nDCG better on test but needs a GPU and is ~2.6x slower even there: the distilled MiniLM is the
practical serving choice; DeBERTa stays the offline teacher.

## 2026-11-21 — Reviewer agent (Sonnet) on sections 15-16 code
No HIGH issues; serving input matches training input (text join, tokenizer call, pair order, truncation).
Fixed: max_length taken from the run config in the ONNX export; ranker refuses a non-1-logit model; tokenizer
calls under a lock (HF fast tokenizers are not thread-safe); 1 MB body limit (413) and Content-Length required
for POST; 503 after 10 s waiting for the model instead of an unbounded queue; blank titles rejected; load test
counts latency/throughput for successful requests only and survives client exceptions (the measured runs had 0
errors, so their numbers are unchanged); CI limited to main + PRs, read-only token, cancel superseded runs.
Not changed on purpose: `final_test.py` stays exactly as it ran (it is the record of the one-time test).
Its known weaknesses, for an honest record: the once-guard file is written mid-run, so a crash before it would
allow a silent re-write of earlier run files; the score cache is keyed by name only (no model hash); fields
for the single MiniLM runs are hard-coded rather than read from their run files (they match:
`minilm_ft_mse_fast` = title, `minilm_distil_w10_bullets` = title, brand, colour, bullets); the trec_eval check
prints rather than asserts. None of these happened in this run (one crash, before any model score or guard).
Load-test limits stated in the code: closed-loop clients, client and server on one CPU, p99 from 300 requests.

## 2026-11-21 — Audit: robustness of the served model (INT8 ONNX, dev), predictions written before the run
Behavioural tests in the CheckList style: change the input in one controlled way, re-score all 2,001 dev
queries with the served INT8 model, compare dev nDCG (labels unchanged) with the unperturbed run.
Predictions:
- unperturbed: 0.8654 (same as quality.json; checks the harness).
- query in UPPER CASE: |change| < 0.0001 (the MiniLM tokenizer lower-cases).
- one typo in the longest query word (two neighbouring letters swapped): -0.010 to -0.025.
- query words in reverse order: -0.005 to -0.015.
- product as title only (brand, colour, bullets missing at serving time): -0.008 to -0.015.
- negation queries (slices.NEGATION, about 158 queries): delete the negation word. If the model reads the
  negation, scores should move a lot; prediction: they barely move (Spearman of scores > 0.9) and nDCG on
  that slice changes by less than 0.01, i.e. the model mostly ignores negation.

Result (dev, 2,001 queries, served INT8 model, labels unchanged; one run each, no CI computed: the 3-seed
spread of MiniLM is 0.0003, so changes of 0.01 and more are far outside noise):

| Perturbation | dev nDCG | change | negation slice (158 q) | Prediction |
|---|---|---|---|---|
| none | 0.8654 | | 0.8347 | 0.8654: right |
| query in UPPER CASE | 0.8654 | 0.0000 | 0.8347 | < 0.0001: right |
| one typo, longest query word | 0.8392 | **-0.0262** | 0.8043 | -0.010 to -0.025: wrong, worse |
| query words reversed | 0.8537 | -0.0117 | 0.7687 | -0.005 to -0.015: right |
| title only (no brand/colour/bullets) | 0.8557 | -0.0097 | 0.8188 | -0.008 to -0.015: right |
| negation word deleted | 0.8581 | -0.0073 | **0.7585** (-0.0762) | slice change < 0.01, Spearman > 0.9: **wrong** |

Negation: within-query Spearman between scores with and without the negation word, median 0.50 (prediction
> 0.9). The model does read "without / not / no": deleting the word re-orders the list and costs 0.076 on that
slice. So the slice's low score (0.835 vs 0.865) is not "the model ignores negation"; more likely those
queries are harder or noisier in other ways (not yet checked).
Typos: one swapped letter pair costs 0.026, more than the whole distillation gain (+0.0064) and bullets
(+0.0068) together. This is the served model's largest measured weakness.
Files: data/processed/serving/robustness_int8_dev.json; script kept in the session scratchpad (audit only).

## 2026-11-21 — Reproduction from a clean clone (WSL Ubuntu 24.04) and first CI run
CI (GitHub Actions, ubuntu-latest, push of 5128234): `uv sync --locked` (95 packages incl. torch 2.11.0+cu128),
ruff clean, 124 tests passed in 26 s. Green on the first run.
Clean clone: `git clone https://github.com/jasirjru/shoprank` in WSL (no files copied from Windows), then the
README steps: uv sync --locked, download, ingest, features, pytest + ruff, BM25 title and LambdaMART on dev.
Predictions:
- raw file SHA-256 identical to the Windows manifest (unless Amazon changed the files upstream).
- ingest_stats.json identical (split is a deterministic hash of query_id).
- BM25 title dev nDCG identical to 4 decimals (0.8163); LambdaMART dev within 0.001 of 0.8342 (LightGBM is
  deterministic for the same data, but float summation order may differ across OS/builds).
- Risk: ingest or features may run out of memory in WSL's 8 GB.

Result (WSL clean clone of github.com/jasirjru/shoprank at 5128234; data downloaded fresh, ~12 min):
| Check | Windows record | WSL clean clone | Prediction |
|---|---|---|---|
| raw file SHA-256 (manifest.json) | | identical | right |
| ingest_stats.json (split sizes) | | identical | right |
| features_dev (40,304 x 20) | | max abs difference 1.4e-14 | (not predicted) |
| tests / ruff | 124 passed / clean | 124 passed / clean | |
| BM25 title, dev nDCG | 0.8162856959701303 | 0.8162856959701303 | identical: right |
| LambdaMART, dev nDCG (259 trees) | 0.834246248084562 | 0.834246248084562 | within 0.001: right (identical) |
Stage times in WSL: download 701 s, ingest 38 s, features 55 s, tests 23 s, BM25 2 s. Peak RAM was fine.
One reproduction bug found: LightGBM needs the OpenMP runtime `libgomp1`, a system library missing on a fresh
Ubuntu 24.04 WSL (`OSError: libgomp.so.1: cannot open shared object file`); GitHub's runner has it, so CI
did not catch it. Fixed in the README reproduce steps (`sudo apt-get install -y libgomp1`).
Not re-run in the clone (GPU hours): the cross-encoders, OOF folds and distillation; their code paths are
covered by the unit tests and by the run records.

## 2026-11-21 — Phase A: dependency upgrade (torch 2.11 cu128 -> 2.13 cu126, setuptools 81 -> >= 84.1)
Why: osv-scanner on uv.lock found GHSA-rrmf-rvhw-rf47 (torch, memory corruption in torch.jit.script; our code
never calls torch.jit, so not directly exposed) and two setuptools issues (macOS NFD file names; legacy egg
install on Windows, a path uv does not use). torch 2.11 caps setuptools < 82, so one torch upgrade fixes all
three. The cu128 index stops at torch 2.11; 2.13.0 exists for cu126 (same CUDA 12 family, Ada sm_89 GPUs
supported) and cu130/cu132. cu126 chosen as the smaller change; driver 610.62 supports it.
Re-verification and predictions (v1.0 numbers in brackets):
- unit tests: 124 pass; ruff clean; osv-scanner: 0 known vulnerabilities.
- served MiniLM (PyTorch, bf16) re-scored on dev: within 0.0005 of 0.8667 (new GPU kernels change bf16
  rounding and tie order slightly).
- DeBERTa fold-0 model re-scored on dev: within 0.0005 of its stored v1.0 dev scores' nDCG.
- ONNX INT8 path does not use torch at inference: unchanged by definition (onnxruntime not upgraded).

Result (Phase A, 21 Nov 2026):
| Check | v1.0 (torch 2.11 cu128) | torch 2.13.0 cu126 | Prediction |
|---|---|---|---|
| served MiniLM, dev nDCG (PyTorch, bf16) | 0.86668 | 0.86668 | within 0.0005: right (identical) |
| DeBERTa fold 0, dev nDCG | 0.88578 | 0.88595 (+0.00017) | within 0.0005: right |
| DeBERTa fold 0, largest per-pair score change | | 0.031 (a few bf16 steps) | (not predicted) |
| unit tests | 124 | 138 (14 new) pass | right |
| osv-scanner on uv.lock | 3 known vulnerabilities | none | right |
setuptools 81.0.0 -> 84.2.1. All v1.0 numbers stand; the INT8 ONNX model was not re-exported (onnxruntime
unchanged, and the export path is now covered by a test on a tiny random BERT).
Also in Phase A: Hugging Face base models pinned by commit (`src/shoprank/hub.py`; DeBERTa main 8ccc9b6,
MiniLM 7b02352, the commits the v1.0 runs used); new tests for the metric runner (`run_task1`), the ONNX
export path and the resumable download; CI gains a security job (osv-scanner 2.7.0 and gitleaks 8.30.1,
binaries checked against their published SHA-256) and actions pinned to commit SHAs (checkout v7.0.1,
setup-uv v10.3.3, both over two weeks old); the placeholder `shoprank` console script is removed.

## 2026-11-21 — Phase B: the served model can be downloaded and run by anyone
- GitHub release v1.0 (tag on 3390899): model_int8.onnx (33,848,219 bytes), tokenizer.json,
  tokenizer_config.json, serving.json, quality.json, SHA256SUMS. Published files scanned first: no local paths
  or personal data (one grep hit was the vocabulary word "studio", one random bytes inside the weights).
- `shoprank.serve.fetch`: downloads from the release (or Hugging Face), writes a file only after its SHA-256
  matches the value pinned in code, never overwrites a different local model without --force. Live test from
  the release: 5/5 files verified.
- Serving without torch: `uv export --prune` of the training stack leaves 45 packages (199 MB venv instead of
  ~7 GB). Fresh venv + fetched model: /health ok, /rank orders the USB-C cable first, 41 ms.
- Dockerfile (python 3.12-slim and uv 0.12.17 pinned by digest, hashed requirements, model verified at build
  time, non-root user, health check). No Docker on this laptop: CI builds the image and runs a /rank smoke test.

## 2026-11-21 — Phase D: typo-consistency distillation (pre-registered before any run)
Problem (audit): one typo in a query costs the served model 0.026 dev nDCG (0.8654 -> 0.8392, INT8).
Change (one change only): `--typo-rate 0.3` in `cross.finetune`. 30% of training QUERIES get one synthetic
typo (`cross/typos.py`: transpose, delete, double, QWERTY-neighbour; one word of 4+ letters, never the first
letter); every pair keeps the target of the clean query (the DeBERTa out-of-fold score), so the student learns
that a misspelled query should score like the clean one. Everything else as `minilm_distil_w10_bullets`
(teacher deberta_oof, weight 1.0, fields title/brand/colour/bullets, lr 7e-6, warmup 5000, 1 epoch, batch 32,
checkpoint chosen on the CLEAN train hold-out). Rate 0.3 is set once, not tuned.
Runs: baseline seeds 43, 44 (seed 42 exists) and typo seeds 42, 43, 44: 5 runs, ~10-12 min each.
Measurement (dev, 2,001 queries, PyTorch bf16 scoring), per model: clean; `typo_transpose` and
`typo_neighbour` (`eval_typo`: one fixed mistake in the longest word, position from the query id). Arms are
compared on per-query nDCG averaged over the 3 seeds, paired bootstrap 95% CI over queries.
Pass rules (all must hold):
1. clean dev: mean change >= -max(0.0010, baseline 3-seed spread).
2. typo_transpose: recovers >= 50% of the baseline gap (baseline clean - baseline typo_transpose).
3. dev2 (1,053 queries, scored once, same models): the clean change is not below -0.002 and the typo gain
   keeps the same sign.
Predictions: baseline typo_transpose gap ~0.025 (audit: 0.026 on INT8); typo_neighbour gap similar or larger;
typo arm recovers 50-80% of both gaps; clean change between -0.002 and +0.001; baseline seed spread <= 0.001.
Caveat stated in advance: train and test typos come from the same generator family; this measures robustness to
these synthetic mistakes, not to all real misspellings.

Result (Phase D, dev; `data/processed/runs/typo_robustness_dev.json`):
| Dev, 2,001 queries, mean of 3 seeds | baseline | typo 0.3 | typo - baseline [95% CI] |
|---|---|---|---|
| clean | 0.8657 | 0.8642 | -0.0015 [-0.0028, -0.0003] |
| typo_transpose | 0.8374 | 0.8415 | +0.0041 [+0.0019, +0.0062] (14% of the 0.0283 gap) |
| typo_neighbour | 0.8400 | 0.8433 | +0.0032 [+0.0011, +0.0052] (13% of the 0.0257 gap) |
Per model (clean / transpose / neighbour): baseline s42 0.8667/0.8382/0.8404, s43 0.8650/0.8372/0.8402,
s44 0.8655/0.8368/0.8394; typo s42 0.8639/0.8427/0.8438, s43 0.8641/0.8404/0.8425, s44 0.8645/0.8412/0.8434.
Baseline seed spread (clean) 0.0017, larger than the 0.0003 measured for title-only MiniLM.
Prediction check: gap ~0.025: right (0.028 / 0.026); recovery 50-80%: WRONG (13-14%); clean change between
-0.002 and +0.001: right (-0.0015); baseline seed spread <= 0.001: wrong (0.0017).
Verdict by the pre-registered rules: rule 1 passes narrowly (-0.0015 >= -0.0017), rule 2 FAILS (14% < 50%).
dev2 is therefore not used (it stays unspent for a later candidate). The served model stays v1.0.
What this means: consistency training buys a real but small robustness gain (+0.004 on typo queries) at a real
small clean cost (-0.0015). A 33M-parameter student that sees each misspelled query once cannot learn to undo
the subword split a typo causes ("cbale" -> different word pieces). Likely better fix, not yet tested: correct
the query BEFORE ranking (spell correction against the catalogue vocabulary), which is how production search
handles typos; its ceiling is the clean score (0.8657), i.e. up to the full 0.028.
Incident: first attempt thrashed (process 10.3 GB, 1.9 GB resident: all 1.2M US products (small + large ESCI versions; not three locales, corrected 21 Nov) were
loaded); fixed by reading only the 336k products the pairs use (`cross.zero_shot.read_products`, identical
texts, tested). The run paused once mid-way (the system clock jumped, likely sleep; this machine's clock proved unreliable,
see git commit times); runs resumed, results unaffected.

## 2026-11-21 — Phase D2: query spell correction before ranking (pre-registered before any dev look)
Idea: fix the typo before the ranker sees it, the way production search does. `serve/spell.py`: vocabulary =
word counts from titles, brands and colours of the 1.2M US products (product text only, no labels, no
queries). A query word (letters only, 4+ chars) seen < 5 times in the catalogue is replaced by the most
frequent catalogue word exactly one Damerau-Levenshtein edit away, if that word is at least 10x more frequent.
Settings fixed now (5, 10x, one edit), not tuned on dev. The corrector does not know the evaluation typo
generator (it may also change first letters).
Measurement (dev, the 3 baseline v1.0-recipe models, seed-averaged, PyTorch bf16): clean, clean+spell,
typo_transpose, typo_transpose+spell, typo_neighbour, typo_neighbour+spell. Also: share of clean dev queries
the corrector changes (false corrections), share of typo queries restored exactly, time per query.
Pass rules:
1. clean+spell - clean >= -0.0010 (it must not hurt correct queries; paired CI reported).
2. typo+spell recovers >= 50% of the typo gap for BOTH typo kinds.
3. dev2 once (same rules' signs): clean change >= -0.002, typo gain > 0 for both kinds.
Predictions: restores 60-80% of typo queries exactly; changes 1-4% of clean queries (brand names, model
words, Spanish/garbled queries); clean change -0.0010 to +0.0005; recovers 50-75% of both gaps (above the 14%
of typo training); under 1 ms per query on CPU.

Result (Phase D2; `data/processed/runs/spell_eval_dev.json`, `spell_eval_dev2.json`; 3 baseline models averaged):
| | dev as typed | dev +spell | dev change [95% CI] | recovered | dev2 change [95% CI] | dev2 recovered |
|---|---|---|---|---|---|---|
| clean | 0.8657 | 0.8661 | +0.0003 [-0.0008, +0.0015] | | +0.0020 [-0.0002, +0.0042] | |
| typo_transpose | 0.8374 | 0.8626 | +0.0252 [+0.0211, +0.0292] | 89% of 0.0283 | +0.0287 [+0.0229, +0.0349] | 87% of 0.0330 |
| typo_neighbour | 0.8400 | 0.8627 | +0.0226 [+0.0185, +0.0267] | 88% of 0.0257 | +0.0263 [+0.0203, +0.0323] | 90% of 0.0293 |
Corrector statistics (dev / dev2): changes 5.9% / 4.9% of clean queries; restores 79.6% / 81.1% (transpose) and
80.3% / 84.6% (neighbour) of typo queries exactly; 0.003-0.01 ms per query.
Prediction check: restores 60-80%: right (80-85%, top of the range); changes 1-4% of clean queries: wrong
(5-6%), but those changes did not hurt (clean +0.0003 dev, +0.0020 dev2); clean change -0.0010..+0.0005: right
on dev; recovers 50-75%: wrong, better (87-90%); under 1 ms: right (0.01 ms).
Verdict: all three pre-registered rules pass, on dev and once on dev2. Spell correction before ranking goes
into serving (v1.1). It beats typo-consistency training (14%) by a wide margin at zero training cost.
Known misses seen while building (not tuned): misspellings frequent in titles stay ("wirless" occurs 5+ times),
and rare non-English words can be "corrected" into English ("llaves" -> "leaves").

## 2026-11-21 — Release hardening (v1.1.1): independent review, fixes, licence
Independent reviewer agent (Sonnet) on everything since v1.0: no defect in the ranking path; 8 findings, all
fixed or documented (CHANGELOG v1.1.1). The most important: following the README "Reproduce" steps gave an API
with spell correction silently OFF (vocabulary not in the model folder), and the CI Docker smoke test used a
query without a typo, so it could not catch that. Both fixed; tests added for the vocabulary lookup.
Deliberately NOT changed: the corrector's behaviour on words with attached punctuation ("cabel,") is left as
evaluated, so the published 87-90% describes exactly what is served; the limit is documented instead.
One behaviour change with no effect on scores: `query_used` returns the input unchanged (spacing included) when
no word was corrected; the tokenizer ignores spacing, so the model input is identical.
Also: Apache-2.0 licence, package version 1.1.1, repo topics + Hugging Face link, lesson 18 written.

## 2026-11-21 — Modern rerankers, zero-shot, on dev (does training our own pay off?)
Prediction (written in the plan as the interview question, no numbers fixed beforehand; recorded as a gap):
a strong 2025-26 general reranker should be close to, but below, an in-domain fine-tuned model.
Setup: same product text as the served model (title, brand, colour, bullets), dev only (2,001 queries),
models pinned by commit; paired bootstrap vs our per-query results. Script: `eval/baseline_rerankers.py`.
| Model (zero-shot) | params | max len | dev nDCG | nDCG@10 | vs served MiniLM [95% CI] | vs DeBERTa teacher | GPU pairs/s |
|---|---|---|---|---|---|---|---|
| BAAI/bge-reranker-v2-m3 @953dc6f | 568M | 256 | 0.8585 | 0.7584 | -0.0081 [-0.0130, -0.0032] | -0.0293 [-0.0342, -0.0245] | 212 |
| Qwen3-Reranker-0.6B (tomaarsen seq-cls) @6a5829f | 596M | 512 | 0.8540 | 0.7510 | -0.0127 [-0.0178, -0.0076] | -0.0339 [-0.0391, -0.0288] | 47 |
| ours: served MiniLM (distilled) | 33M | 128 | 0.8667 | 0.7717 | | | (CPU INT8: ~230) |
| ours: DeBERTa-v3-base x5 | 5 x 184M | 128 | 0.8879 | | | | |
Qwen3 used the generic instruction from its model card ("Given a web search query, retrieve relevant passages
that answer the query"); product text cut to 1,200 characters.
Reading: both 17x-larger general rerankers lose to our 33M in-domain distilled model, and by ~0.03 to the
in-domain teacher. In-domain fine-tuning on ESCI labels is worth more than model size here. Caveats: zero-shot
only (fine-tuning them on ESCI would likely close or reverse the gap, at 17x the serving cost); dev only, the
test set stays spent; the Qwen3 prompt was not tuned.

## 2026-11-21 — Fresh-clone acceptance test of v1.1 in WSL (separate agent)
`git clone` from GitHub, checkout v1.1 (40bace7), README steps only: `uv sync --locked` (torch 2.13.0+cu126),
173 tests pass, ruff clean, `serve.fetch` verifies 6/6 files incl. spell_vocab.json, API /health
spell_correction true, "usb c cbale 6ft" -> query_used "usb c cable 6ft", right product first (13 ms), empty
query -> 422. 8/8 steps pass. Its README findings (heavy torch install under "no GPU", stale v1.0 link, no
/health hint) were fixed in v1.1.1; `scripts/serve_requirements.sh` also checked on the v1.1.1 checkout.
Backup of GPU-hour artefacts (5 DeBERTa folds, served PyTorch model, run records, scores): private Hugging
Face repository, 181 files.

## 2026-11-21 — More labelled data: MiniLM on the LARGE ESCI version (pre-registered before the run)
Why: the benchmark research (`docs/analysis/benchmark_position_2026-11-21.md`) names more labelled data as the
biggest lever (KDD Cup 9th place: Task 2 data "improved the model substantially"). Cheapest honest test: the
served-size MiniLM, Amazon's recipe (title only, MSE on the gain, lr 7e-6, warmup 5000, 1 epoch, batch 32,
seed 42), trained on all LARGE-version train judgments: 63,553 queries / 1,181,741 pairs (small: 17,823 /
357,759; small is a subset; 0 queries shared with dev or dev2). Checkpoint on a 10% train hold-out as before
(eval every 5000 steps instead of 1000, to save time). Baseline: `minilm_ft_mse_fast` (same recipe, small
version), dev 0.8534. Judged on the usual dev (2,001 small-version queries), paired bootstrap. dev2 and test
untouched.
Prediction: +0.008 to +0.020 dev nDCG (3.6x more queries; title-only MiniLM gained +0.0068 from bullets and
+0.0064 from distillation, so data at this scale should matter at least as much). A gain below 0.002 (6x the
MiniLM seed spread 0.0003) would mean data is not the lever for this model size.

## 2026-11-21 — Label-noise estimate on a RANDOM sample (pre-registered before judging)
The 21 Nov audit hunted errors (most extreme disagreements), so it could not give an error rate. Now: 200 dev
pairs drawn uniformly at random (seed 2026; label mix E 84 / S 73 / I 34 / C 9, close to dev's 44/34/18/4%).
Two independent AI judges (a Sonnet agent and an Opus agent), each blind to the labels and to the other judge,
label every pair with the ESCI definitions only. Files: data/processed/label_audit/ (blind sheet, key, judgments).
Measures: each judge's agreement with the ESCI label (4 classes, and E vs not-E); Cohen's kappa between the two
judges; "likely label error" = both judges agree with each other AND disagree with the ESCI label, as a share of
200 with a Wilson 95% interval. A human (the owner) can later judge the same sheet for a human-vs-AI check.
Predictions: judge vs label 4-class agreement 65-80% (Substitute vs Exact is the hard boundary), E vs not-E
80-90%; judge-judge kappa 0.5-0.7; likely label errors 5-12% of pairs.

Result (random label audit; `docs/analysis/label_audit_random200_dev.csv` and `..._summary.json`):
| Measure | Result [Wilson 95%] | Prediction |
|---|---|---|
| judges agree with each other | 175/200 = 87.5% [82.2, 91.4]; Cohen's kappa 0.81 | kappa 0.5-0.7: wrong (higher) |
| Sonnet judge = ESCI label (4 classes) | 63.5% [56.6, 69.9]; Exact vs not 72.5% | 65-80% / 80-90%: wrong (lower) |
| Opus judge = ESCI label (4 classes) | 63.5% [56.6, 69.9]; Exact vs not 74.0% | same |
| both judges agree AND differ from the label | 57/200 = 28.5% [22.7, 35.1] | 5-12%: wrong (much higher) |
| same, Exact vs not-Exact only | 47/200 = 23.5% [18.2, 29.8] | |
Where they differ (judge consensus vs label): label E -> judges S 17; label S -> judges E 15; label E -> I 5;
label S -> I 6; the rest C/I. 32 of the 57 are the Exact/Substitute boundary.
Reading (careful): two AI judges that agree with each other are not ground truth: they can share a bias. Some
disagreements look like label errors ("boots for men work construction" -> WHITIN construction work boots,
labelled S), others like judge errors ("wig without bangs" -> a wig WITH bangs, labelled I, judges said S:
the label looks right). So 28.5% is an upper bound on how often a careful reader would contest the label, not
the error rate. The ESCI paper's own 91% annotator agreement (9% disagreement) is a lower reference. The true
noise level is between, and the Exact/Substitute boundary is genuinely subjective. Consequence for ShopRank:
part of the remaining gap to 1.0 cannot be closed by any model; at most it argues for a "graded" view of
Substitute vs Exact. A human pass over the 57 DISAGREE rows (column `your_label` in the CSV) would turn this
into a real error-rate estimate.

Result (large-version data, dev; `runs/minilm_ft_mse_large.json`): 0.8586 (nDCG@10 0.7605) vs small 0.8534:
+0.0052 [+0.0023, +0.0082] full list, +0.0093 [+0.0051, +0.0136] nDCG@10. 1,064,406 fit pairs, 1,009 s training
(title only; ~17 min), best step 30,000 of 33,263. Prediction +0.008 to +0.020: wrong (smaller), but the gain is
real (17x the MiniLM seed spread). Versus distillation on the small data (`minilm_distil_w10`, 0.8575): +0.0011
[-0.0017, +0.0040], no clear difference: 3.3x more labelled pairs buy about what the DeBERTa teacher's soft
scores buy. Combining both (distilling on the large version) would need teacher scores for 1.18M pairs, about
5 x 1.5 GPU-hours of DeBERTa fold training: not run (over the 2-GPU-hour limit without the owner's OK).
The hold-out nDCG during training (~0.956) is on large-version queries, whose candidate lists differ; not
comparable to dev.

## 21 Nov 2026: public project page (site/, for Vercel)

Goal: one static page that proves ShopRank works to a recruiter (30 s) and an ML hiring manager (3 min).
Research (sources in `docs/website/research.md`): people decide in about 10 s and spend 57% of viewing time above
the fold, so the first screen states the result against a baseline and shows one real example. New Gradio/Docker
Spaces now need Hugging Face PRO, so the page needs no server; a live demo can later run the INT8 model in the
browser (ONNX Runtime Web).

No prediction was written before this run (a lapse; it is a data export, not an experiment, and the expected
means were already known: INT8 + spell about 0.866, BM25 0.816).
Result (`uv run python -m shoprank.serve.showcase mine`, all 2,001 dev queries through the real API code):
ShopRank 0.8662, BM25 0.8163; ShopRank better on 65.3% of queries, worse on 33.7%, tied 1.0%.
The API figure is 0.0005 above the recorded INT8 + spell number (0.8657): same model and corrector, possibly
batch-padding differences in INT8 scoring; not investigated further (the page reports test numbers, not this).
Spell correction changed 116 of 2,001 dev queries; it fixed real typos ("lg refridgerators") and also changed
real names ("nivda" -> "nivea", "squire" -> "square", "tasers" -> "teasers"). The page states this limit.

Examples chosen by hand from `data/processed/showcase_candidates.parquet` (rules: 8-16 candidates, readable
query, no adult products): four clear wins (iphone xs pro 0.38 -> 1.00; holey jeans for babys 0.49 -> 0.99;
extendable ram for lenovo yoga 0.47 -> 0.85; usb 3.0 type a to micro a 0.59 -> 0.97), one real customer typo
fixed by spell correction (lg refridgerators 0.78 -> 1.00) and one honest failure (m23 for cars: BM25 1.00,
ShopRank 0.57). `showcase check` re-ranks the six through the API and confirms the page shows the same order
(parity OK). The page is generated by `python -m shoprank.site` (static HTML, no framework, strict CSP);
`tests/test_site.py` keeps the committed page in sync with the generator.
Not yet done: Lighthouse and axe runs (need a download), cold start (no server, so none), the 5-person test.

## 22 Nov 2026: launch audit (4 parallel reviewers) and page fixes

Reports in `docs/analysis/launch_audit_2026-11-22/` (code_repo, site, numbers, linkedin). No blocker in code,
numbers or links; 200 tests, CI, gitleaks and osv-scanner clean. Live page before fixes: Lighthouse mobile 86 /
100 / 100 / 100, desktop 99 / 100 / 100 / 100 (performance / accessibility / best practices / SEO); axe-core 0
violations. Blocker: no social preview tags or image.
New metric, read from the saved official-test runs (no new scoring): exact product at #1 (success@1) BM25 0.6268,
LambdaMART 0.6580, Amazon recipe 0.7161, served (PyTorch) 0.7468, DeBERTa teacher 0.7903; served - BM25 +0.1199
[+0.1091, +0.1309]. Dev through the real API (INT8 + spell): 0.7496 vs 0.6212 [+0.104, +0.152]; known-item
searches (model number) 0.782 vs 0.593 (413 queries), brand 0.766 vs 0.672 (338), general browsing 0.737 vs 0.618
(1,318). No prediction written for these read-only measurements.
Fixes: headline now "right product first in 3 of 4 searches" (75% vs 63%), honest labels (LambdaMART is learned,
"before compression", examples are hand-picked, 65% better / 33% worse overall, synthetic typos, laptop CPU),
known-item section, Open Graph tags + 1200x627 preview image, in-page example links fixed, tap targets 44 px,
fonts without the optical-size axis (about half the bytes), private path removed from docs/PLAN.md.
