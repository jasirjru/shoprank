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
