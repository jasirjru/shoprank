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
