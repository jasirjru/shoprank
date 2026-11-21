# ShopRank lessons: section 13 (built 20 Nov 2026, study after the build)

Every number here comes from real runs on this project's data (dev = 2,001 queries).
The check-question answers are at the bottom. Answer first, then look.

| Model | Dev nDCG (full list) | vs fine-tuned MiniLM |
|---|---|---|
| BM25 | 0.8163 | |
| LambdaMART | 0.8342 | |
| MiniLM fine-tuned (section 12, fast loop) | 0.8534 | |
| DeBERTa-v3-base fine-tuned (title + brand + colour) | 0.8778 | +0.0244 [+0.0197, +0.0292] |
| **DeBERTa-v3-base + bullets** | **0.8851** | |

---

## Part 1: What DeBERTa is, and why it should beat MiniLM

**Picture:** MiniLM is a small, fast scooter; DeBERTa is a bigger car with a better engine. The engine
upgrade is "disentangled attention": like a coach who keeps *who* a fielder is and *where he stands
relative to the batsman* as two separate facts, instead of one blended description.

**Real numbers (measured on this machine):**

| | MiniLM-L12 | DeBERTa-v3-base |
|---|---|---|
| Layers | 12 | 12 |
| Width | 384 | 768 |
| Vocabulary | 30,522 | 128,100 |
| Parameters | 33M | 184M |

Tokens for `ipad bay bag` + `Turtle Beach headset Call of Duty`:
```
MiniLM:  [CLS] ipad bay bag [SEP] turtle beach heads ##et call of duty [SEP]
DeBERTa: [CLS] ▁ipad ▁bay ▁bag [SEP] ▁Turtle ▁Beach ▁headset ▁Call ▁of ▁Duty [SEP]
```
MiniLM lowercases and splits `headset`; DeBERTa keeps capitals (brands) and knows `headset` as one word.

**Disentangled attention:**
```
BERT / MiniLM:  token vector = word + position (added, mixed)
DeBERTa:        score(i, j) = word_i·word_j + word_i·position(i-j) + position(j-i)·word_j
```

**Interview answer:** "DeBERTa-v3-base has 184 million parameters against MiniLM's 33 million, keeps
capital letters and has a four times bigger vocabulary. Its key idea is disentangled attention: separate
vectors for a token's meaning and its relative position. It is the model family the KDD Cup 2022 winners
used on this dataset."

## Part 2: Mixed precision, and the real NaN bug

**One idea: tiny numbers can disappear.** A scale that only shows grams shows a grain of sugar as 0.

| Format | Bytes | Biggest number | 1e-8 stored as | (5.96e-8)^2 |
|---|---|---|---|---|
| fp32 | 4 | 3.4e38 | 1e-8 | 3.55e-15 |
| bf16 | 2 | 3.39e38 | 1e-8 | 3.55e-15 |
| fp16 | 2 | 65,504 | **0** | **0** |

**The bug (found today):** the DeBERTa-v3 checkpoint is stored in fp16, so its weights loaded as fp16.
AdamW divides by (sqrt of squared gradient + eps). In fp16, eps (1e-8) rounds to 0 and 99.7% of the
embedding gradients were 0, so the update was 0 / 0 = NaN, even with learning rate 0. After one step,
201 of 202 weight tensors were NaN.

**How it was found (method, not luck):** NaN at step 2 even at lr 2e-8 -> also in fp32 autocast, so not
bf16 -> gradients finite, weights finite -> one AdamW step at lr 0 made them NaN -> the gradient dtype was
float16 -> fused AdamW and SGD were fine. Root cause, then the fix.

**Fix:** load the weights as fp32 ("master weights"); bf16 autocast still does the matrix maths fast.
`fit()` now refuses half-precision weights, with a test.

**Interview answer:** "DeBERTa's weights loaded in fp16, where tiny numbers round to zero, so AdamW divided
zero by zero and produced NaN. Keeping fp32 master weights with bf16 autocast fixed it, and I added a guard
so training refuses half-precision weights."

## Part 3: Error analysis by query slice

**Picture:** a teacher who looks at *which* questions a class gets wrong, not only the average mark.

| Slice (queries) | BM25 | LambdaMART | MiniLM | DeBERTa | DeBERTa - MiniLM |
|---|---|---|---|---|---|
| all (2,001) | 0.8163 | 0.8342 | 0.8534 | 0.8778 | +0.0244 [+0.0197, +0.0292] |
| 1-2 words (379) | 0.8084 | 0.8313 | 0.8552 | 0.8739 | +0.0187 [+0.0083, +0.0293] |
| 3-5 words (1,334) | 0.8191 | 0.8358 | 0.8509 | 0.8789 | +0.0281 [+0.0223, +0.0339] |
| 6+ words (288) | 0.8138 | 0.8311 | 0.8629 | 0.8780 | +0.0152 [+0.0024, +0.0274] |
| has a number (413) | 0.8033 | 0.8340 | 0.8584 | 0.8770 | +0.0186 [+0.0089, +0.0286] |
| **negation (158)** | **0.7481** | 0.7683 | 0.8273 | 0.8391 | +0.0118 [-0.0012, +0.0244] |
| brand in query (338) | 0.8292 | 0.8526 | 0.8591 | 0.8881 | +0.0290 [+0.0170, +0.0414] |

- Negation ("without", "not") is the hardest slice for every model. BM25 rewards the very word the user
  said "not" to. The big jump on negation is word-matching -> transformer (MiniLM +0.079 over BM25), not
  MiniLM -> DeBERTa (+0.012, CI touches 0).
- Amazon's own query labels (`sources.csv`) agree: their `negations` queries (141) score like our rule (158).
- The brand gain is partly the added brand field: the ablation separates it.

**Interview answer:** "I sliced dev by query type. Negation queries are the hardest for every model; lexical
models score 0.75 there because they match the negated word, and cross-encoders fix most of that. DeBERTa's
extra gain over MiniLM is largest on 3-5 word and brand queries and not significant on negation."

## Part 4: Noise first: is a win bigger than luck?

**Picture:** weigh yourself three times in a row and the scale says 70.1, 70.3, 70.2. If a diet "makes you
0.1 kg lighter", you cannot tell that from the scale's wobble. Training has a wobble too: the random
seed changes the batch order, the new head's start weights and dropout.

**Real numbers (dev nDCG, full list, same recipe, only the seed changes):**

| Model | seed 42 | seed 43 | seed 44 | Mean | Spread (max - min) |
|---|---|---|---|---|---|
| MiniLM fine-tuned | 0.8534 | 0.8536 | 0.8537 | 0.8536 | 0.0003 |
| DeBERTa-v3-base | 0.8778 | 0.8785 | 0.8794 | 0.8786 | 0.0016 |

```
DeBERTa seeds        |--0.0016--|
                  0.8778     0.8794
Bullets run                              * 0.8851   (0.0065 above the mean: 4x the wobble)
```

**Rule written before the ablations:** a change counts only if it beats the base by more than ~0.002
(above the spread) AND its paired confidence interval excludes zero.

**Interview answer:** "Before any ablation I trained each model with three seeds. DeBERTa's dev nDCG moved
by 0.0016 between seeds, MiniLM's by 0.0003, so I only accepted changes larger than about 0.002 with a
paired bootstrap interval that excluded zero."

## Part 5: Ablations: which product fields matter?

**Picture:** a cook removes one ingredient at a time to find which one makes the dish. Change one thing,
keep everything else fixed, measure.

**Real numbers (DeBERTa-v3-base, seed 42, input cut at 128 tokens):**

| Input fields | Dev nDCG | vs title + brand + colour (paired 95% CI) | Train time |
|---|---|---|---|
| title only | 0.8777 | -0.0002 [-0.0032, +0.0029] (no difference) | 27 min |
| title + brand + colour (base) | 0.8778 | | 24 min |
| title + brand + colour + **bullets** | **0.8851** | **+0.0072 [+0.0039, +0.0106]** | 40 min |

Where the bullets help most (bullets - base):

| Slice | Gain |
|---|---|
| negation ("without", "not") | +0.0140 [+0.0049, +0.0234] |
| has a number | +0.0096 [+0.0030, +0.0161] |
| brand in query | +0.0019 [-0.0068, +0.0099] (no difference) |

```
query:  "phone case without magnet"
title:  "Spigen Tough Armor case for iPhone 13"          <- says nothing about magnets
bullets:"... built-in MagSafe magnet ring ..."           <- the answer is here
```

- Brand and colour add nothing: the brand is usually already written in the title.
- Bullets carry the facts a title leaves out (sizes, materials, "includes", "not compatible with"), so
  negation and number queries gain most. This matches the KDD Cup 1st-place input (title + bullets +
  description).
- Cost: 1.63x training time. Not tried yet: bullets at 256 tokens (another ~2x).

**Interview answer:** "I ablated the product fields one at a time. Brand and colour gave nothing because
the brand is usually in the title. Adding the bullet points gave +0.007 nDCG with a confidence interval
well above zero and about four times the seed spread, mostly on negation and number queries, where the
title cannot answer the question."

---

## Check questions

1. (Part 1) Which is NOT a reason DeBERTa-v3 should beat MiniLM? A) disentangled attention B) it was
   pretrained on ESCI labels C) 184M vs 33M parameters D) cased, 128k vocabulary
2. (Part 2) On fp16, what does 0.00000001 become? A) 1 B) 0 C) 100 D) unchanged
3. (Part 2) Why did fp32 weights fix the NaN? A) the GPU got faster B) eps and tiny squared gradients stay
   above 0, so no 0/0 C) fp32 removes zero gradients D) fp32 turns off dropout
4. (Part 3) On which slice do ALL models score lowest? A) brand queries B) 6+ word queries C) negation
   queries D) queries with numbers

5. (Part 4) DeBERTa's seed spread is 0.0016. A new idea scores +0.0010 with a CI of [-0.0020, +0.0040].
   What do you conclude? A) it is a win B) no evidence it helps: smaller than the wobble and the CI holds 0
   C) it is worse D) train one more epoch to be sure
6. (Part 5) Why did adding the brand field change nothing? A) DeBERTa cannot read brands B) the brand is
   usually already in the title C) brand queries are rare D) the brand field was empty
7. (Part 5) Which slice gained most from the bullets? A) brand queries B) 1-2 word queries C) negation
   queries D) Spanish queries

## Answer key

1. B (it saw ESCI only in our fine-tuning). 2. B. 3. B. 4. C (BM25 0.748, DeBERTa 0.839).
5. B. 6. B (title-only 0.8777 vs 0.8778; brand field filled for 94% of products, so D is wrong).
7. C (+0.0140).
