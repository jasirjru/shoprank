# ShopRank lessons: section 12 (taught 20 Nov 2026)

Every number here comes from real runs on this project's data (dev = 2,001 queries).
The check-question answers are at the bottom. Answer first, then look.

---

# Section 12: Fine-tune MiniLM ("the England coach learns Indian pitches")

| Step | What changed | Dev nDCG (full list) |
|---|---|---|
| Zero-shot MiniLM (section 11) | nothing, pretrained on MS MARCO only | 0.8332 |
| LambdaMART (section 6) | 14 hand-made features | 0.8342 |
| 12a: fine-tune, MSE on gain | all 33M weights move | 0.8549 |
| 12a, fast loop | same maths, epoch 444 s -> 316 s | 0.8534 (same within noise) |
| 12b: 4-class cross-entropy | only the loss and the head | 0.8478 (worse, CI below 0) |

## Part 1: What fine-tuning changes

- Fine-tuning = keep training a pretrained model on your own labels. Every weight moves a little.
- One training step: 32 (query, title) pairs -> tokenise -> forward -> 32 scores -> loss -> backward ->
  AdamW step (weight = weight - LR x direction). 323,302 pairs / 32 = 10,104 steps = 1 epoch.
- LR 7e-6 (Amazon's recipe) with 5,000 warmup steps: the LR rises from 0 so new data does not shake the
  pretrained weights too hard, then falls to 0 so the weights settle.
- Real curve: hold-out nDCG 0.8357 at step 300, dipped to 0.7999 at step 1,000 (the model first had to
  shrink its scores from the MS MARCO +-10 scale to the 0-1 gain scale), best 0.8581 at step 10,000.

## Part 2.1: MSE on the gain (12a)

**Picture:** an exam where one question is worth 100 marks and the others 10, 1 and 0. The student
only studies the big one.

**MSE** = mean of (prediction - target)^2. Squaring makes every error positive and punishes big
errors much more: error 0.5 -> 0.25, error 0.1 -> 0.01.

**Real example, dev query `ipad bay bag`, 12a model:**

| Label | Target | 12a score | Title |
|---|---|---|---|
| E | 1.0 | +0.778 | Broad Bay Dolphins Tablet Bag |
| S | 0.1 | +0.316 | iPad Green Bay Packers Smart Case |
| I | 0.0 | +0.269 | Pacsafe Cross-Body Bag |
| E | 1.0 | +0.249 | Grayman Kato Tablet Mini-Messenger |
| C | 0.01 | +0.118 | Tablet Sleeve for iPad Pro |

An Irrelevant bag ranks above an Exact one. S, C and I targets are all between 0 and 0.1, so mixing
them up costs almost nothing (0.1^2 = 0.01), while missing an E by 0.5 costs 0.25.
Train labels: E 43.2%, S 35.3%, I 16.9%, C 4.6%.

**Interview answer:** "In 12a I trained with MSE on the gain, as in Amazon's baseline. Because the error
is squared, the loss is dominated by Exact versus not-Exact; S, C and I targets are all close to zero,
so the model gets little signal to separate them. It still reached 0.855 nDCG on dev."

## Part 2.2: Cross-entropy + expected gain (12b)

**Picture:** a weather forecast with a chance for every outcome (70% sun, 20% cloud ...) instead of one
number.

- The head has 4 outputs; softmax turns them into P(E), P(S), P(C), P(I), which add up to 1.
- CE loss = -log(chance given to the correct class): P = 0.73 -> 0.31; P = 0.18 -> 1.71; P = 0.14 -> 1.97.
- Ranking score = expected gain = P(E) x 1 + P(S) x 0.1 + P(C) x 0.01 + P(I) x 0.

**Real example, same query, 12b model:**

| Label | P(E) | P(S) | P(C) | P(I) | Expected gain | Title |
|---|---|---|---|---|---|---|
| E | 0.73 | 0.20 | 0.04 | 0.04 | 0.746 | Broad Bay Dolphins Tablet Bag |
| E | 0.36 | 0.56 | 0.02 | 0.06 | 0.415 | Kato iPad Mini-Messenger Bag |
| S | 0.31 | 0.23 | 0.08 | 0.38 | 0.334 | Green Bay Packers Smart Case |
| I | 0.21 | 0.63 | 0.02 | 0.14 | 0.273 | Pacsafe Cross-Body Bag |
| E | 0.18 | 0.48 | 0.03 | 0.32 | 0.229 | Grayman Kato Tablet Messenger |
| C | 0.15 | 0.20 | 0.06 | 0.59 | 0.171 | Tablet Sleeve for iPad Pro |

The Pacsafe bag costs 0.07 under MSE but 1.97 under CE: CE punishes S/C/I confusion much harder.
Still, 12b lost: 0.8478 vs 0.8534, -0.0056 [-0.0094, -0.0018]. Reasons: the 4-output head started from
random weights (MSE kept the pretrained head), LR 7e-6 is small for a new layer (the hold-out gap
shrank from 0.071 to 0.012 but never closed), and C is rare (4.6%). So: "CE lost under Amazon's
recipe", not "CE is always worse".

**Interview answer:** "In 12b I changed only the loss: four class probabilities with a softmax,
cross-entropy, ranked by expected gain. It scored 0.848 against 0.853 for MSE, a significant gap, mainly
because the new four-class head started from random weights at a learning rate of 7e-6."

## Part 3: Hard negatives

**Picture:** practising to spot fake notes with toy money (easy) vs near-perfect fakes (hard).

**Real example, `ipad bay bag`:** same-query non-Exact candidates share words with the query (Pacsafe
Cross-Body *Bag*, *iPad* 9.7 Case), so they are hard negatives. Random products from other queries in the
batch (Lido Coffee Grinder, RBX Camo Leggings) share no words: easy negatives.
56.8% of ESCI train candidates are not Exact, and a search engine retrieved them, so the cross-encoder
learned from hard negatives for free. The two-tower (section 7) used in-batch negatives (mostly easy)
and lost to BM25 (0.7849, old split); 12c tests adding same-query hard negatives.

**Interview answer:** "Hard negatives are wrong answers that look like right answers. ESCI candidates
already are hard negatives, so my cross-encoder learned from them by default; my two-tower used easy
in-batch negatives, which I think is a main reason it lost to BM25. Section 12c tests that."

## Part 4: Overfitting and checkpoint choice

**Picture:** a student who memorises past papers: practice marks rise, new mock-exam marks fall.

- **Epoch** = one pass over all training pairs: 323,302 / 32 = 10,104 steps, 316 s.
- Every 1,000 steps the 1,732-query train hold-out is scored and the best checkpoint is kept.
- 12a: hold-out 0.7975 (step 1,000) -> 0.8567 (7,000) -> 0.8574 (10,104): rising then flat, no overfitting.
- 12b: best at step 9,000 (0.8449); the last checkpoint is 0.8436. The dip at step 4,000 (0.8108 -> 0.8045)
  then recovery is noise; overfitting is a lasting fall.
- Choose on the hold-out, never dev: the best of 11 checkpoints chosen on dev would make dev optimistic
  (selection bias, section 9).
- The last logged train loss (0.0193) divides 104 steps by 1,000: a known logging bug, ignore it.

**Interview answer:** "I scored a 1,732-query train hold-out every 1,000 steps. The MSE model's hold-out
kept rising and then flattened, so one epoch did not overfit. I kept the best hold-out checkpoint and never
chose on dev, so the dev score stays unbiased."

## Bonus: speeding up the training loop (profile, fix, profile again)

- py-spy + nvidia-smi showed the CPU waiting for the GPU 51% of the loop, and 47% of tokens were padding.
- Fixes: tokenise once, length-bucketed batches (padding 47% -> 1.7%), and removing three hidden
  CPU-GPU waits: `loss.item()` every step, `.to(device)` from normal RAM, transformers' `padding_mask.all()`.
- Each fix moved the wait to the next point; only all three together worked: 444 s -> 316 s (1.41x),
  same accuracy. GPU now 91% busy, so it is the limit. Lesson: profile again after every fix.

---

## Check questions

1. (MSE) Target 1.0, model scores 0.6. Squared error? A) 0.4 B) 0.04 C) 0.6 D) 0.16
2. (Part 2.1) The model gives an Irrelevant product (target 0) a score of 0.1. Squared error, and what does
   it show? A) 0.9, pushed hard B) 0.1, big error C) 0.01, MSE barely pushes S, C, I apart D) 0
3. (Part 2.2) P(E) 0.5, P(S) 0.5, P(C) 0, P(I) 0. Expected gain? A) 1.0 B) 0.5 C) 0.6 D) 0.55
4. (Part 3) Why did the cross-encoder not need extra hard-negative mining? A) cross-encoders cannot learn
   from negatives B) ESCI candidates already are hard negatives (same query, 56.8% not Exact) C) random
   products are harder D) MSE ignores negatives
5. (Part 4) Why choose the checkpoint on the train hold-out, not dev? A) dev has no labels B) hold-out is
   faster C) to keep dev clean: best-of-11 on dev would make dev look better than reality D) hold-out has
   more queries

## Answer key

1. D (0.4^2 = 0.16). 2. C (0.1^2 = 0.01, 16x smaller than 0.16). 3. D (0.5 + 0.05 = 0.55). 4. B. 5. C.
