# Week 3 notes — neural networks from scratch, then PyTorch

★ = memorize. Read top to bottom.

---

## Page 1: The big idea

Weeks 1–2 matched **words**. A neural network can learn its own **representation** of text — a list of
numbers (a "vector") where similar meanings end up close together.

```
"running shoes"  ──► [0.12, -0.40, 0.88, ...]  ┐
                                                ├─ close together = similar
"sneakers"       ──► [0.10, -0.38, 0.91, ...]  ┘
"phone case"     ──► [-0.70, 0.22, 0.05, ...]     far away = different
```

This week: (A) build the engine that trains neural networks, by hand; (B) a tiny network on it;
(C) PyTorch on your GPU; (D) a real search model trained on ESCI.

---

## Page 2: Autograd — how a network learns (Part A) ★★

**The problem:** a network has thousands to millions of weights. Training needs, for EVERY weight,
"if I nudge this weight up a little, does the loss go up or down, and by how much?" = its **gradient**.

**The trick (autograd):** every number remembers **how it was made**. Then we walk backwards and apply
the **chain rule**.

```
a = 2, b = -3
c = a * b        → c remembers: "I am a × b"
L = c + 10       → L remembers: "I am c + 10"

backward:  dL/dL = 1
           dL/dc = 1            (+ passes the gradient through unchanged)
           dL/da = b × 1 = -3   (× sends each input the OTHER input times the gradient)
           dL/db = a × 1 =  2
```

Each operation only needs its **local rule**:

| Operation | Local rule (how gradient flows back) |
|---|---|
| a + b | both get the gradient unchanged |
| a × b | a gets b × grad, b gets a × grad |
| relu(a) | passes gradient if a > 0, blocks it (0) if a ≤ 0 |
| exp(a) | gets exp(a) × grad |
| log(a) | gets (1/a) × grad |

> ★ **Chain rule:** gradient of the result w.r.t. an input = product of local slopes along the path.
> ★ **If a value is used twice, its gradients ADD** (`x*x + x` → dx = 2x + 1). Our code uses `+=` for this.
> ★ Our engine (`scratch/autograd.py`) matches **PyTorch to 9 decimal places** and matches
>   **finite differences** (nudge the input by 0.000001 and measure) — the definition of a derivative.

**PyTorch does exactly this**, on whole tables of numbers (tensors), on the GPU. `loss.backward()` in
PyTorch = our `Value.backward()`.

---

## Page 3: A neural network (Part B)

- **Neuron:** weighted sum of inputs + bias, then **ReLU** (`max(0, x)`: "fire only if positive").
- **Layer:** many neurons reading the same inputs.
- **MLP (multi-layer perceptron):** layers stacked. Input → hidden → output.
- Why ReLU (a **non-linearity**)? Without it, stacking layers is still just one weighted sum — the network
  could only learn straight-line rules.
- **He initialization:** start weights small and random, scaled by √(2/inputs), so signals don't explode
  or vanish as they pass through layers.

The training loop is the **same 4 steps as Week 2**: predict → loss → backward → update.
New detail: **set gradients to zero before each backward**, because gradients add up (`+=`).

> ★ Forgetting `zero_grad()` is a classic bug: gradients from old steps pile up and training goes wrong.

---

## Page 4: The two-tower model (Part D) ★★

```
query text   ──► query tower   ──► 128 numbers  ┐
                                                ├──► cosine similarity = relevance score
product text ──► product tower ──► 128 numbers  ┘
```

**Step 1 — text to ids (`dense/hashing.py`):** each word → one id for the word + one id per 3-letter piece.
```
"anna" → word:anna, #an, ann, nna, na#
"ana"  → word:ana,  #an, ana, na#        ← shares 2 pieces: robust to misspellings
```
Pieces are **hashed** into 262,144 slots ("hashing trick"): no vocabulary file needed; new words still work.

**Step 2 — tower (`dense/model.py`):**
- `EmbeddingBag`: one learnable 128-number vector per slot; a text = **average** of its pieces' vectors.
- 2 small layers (Linear → ReLU → Linear).
- **Normalize** to length 1, so dot product = cosine similarity (−1 to 1).

**Why two towers?** Products can be encoded **once, ahead of time**. At search time only the query is
encoded, then compared to millions of stored product vectors very fast. This is how large-scale search works.

---

## Page 5: Contrastive learning — InfoNCE loss ★★

**Training data:** 163,125 (query, Exact product) pairs from train.

**Idea:** in a batch of 256 pairs, each query has 1 correct product and 255 **other products in the batch
that act as wrong answers** ("in-batch negatives") — free, no extra labels.

```
              product1  product2  product3 ...
query1     [   HIGH      low       low   ]    ← want the diagonal high
query2     [   low       HIGH      low   ]
query3     [   low       low       HIGH  ]
```

**Loss:** for each row, softmax over the 256 similarities, then −log(probability of the correct one).
This is just **cross-entropy** where the "correct class" is the matching product. (Our test checks it equals
PyTorch's `cross_entropy`.)

**Temperature** divides similarities before softmax:
- small (0.05) → differences get magnified → sharp, confident softmax → strong learning signal
- large (1.0) → cosine is only between −1 and 1, so all logits are close → softmax nearly flat → weak signal

**False negatives:** if the same query appears twice in a batch with two different Exact products, each is a
CORRECT answer for the other row. Punishing it would teach the wrong thing → we **mask** those cells.

> ★ InfoNCE = "pick the right partner out of the batch". ★ In-batch negatives = other rows' positives.
> ★ Temperature controls how sharp the softmax is. ★ Mask false negatives.

---

## Page 6: Results (dev, 2,099 queries, full-list nDCG)

| Model | nDCG full | nDCG@10 | Δ (95% CI) |
|---|---:|---:|---|
| Random | 0.7500 | 0.5786 | — |
| **Two-tower (from scratch), batch 256, temp 0.05** | **0.7849** | 0.6405 | +0.0350 [+0.0283, +0.0413] vs random |
| BM25 title | 0.8193 | 0.6907 | two-tower is **−0.0343 [−0.0403, −0.0283]** vs BM25 |
| LambdaMART (Week 2) | 0.8359 | 0.7167 | best so far |

**Experiments (vs batch 256, temp 0.05):**

| Change | nDCG full | Δ (95% CI) | Verdict |
|---|---:|---|---|
| batch 32 | 0.7847 | −0.0002 [−0.0045, +0.0041] | no difference |
| batch 1024 | 0.7817 | −0.0032 [−0.0065, +0.0000] | no clear difference |
| temperature 0.2 | 0.7754 | −0.0096 [−0.0139, −0.0052] | worse |
| temperature 1.0 | 0.7607 | −0.0243 [−0.0302, −0.0184] | much worse; loss barely fell (5.14 → 4.82, near ln 256 = 5.55 = random guessing) |

Peak GPU memory: **1.3 GB** of 8 GB. Training: ~55 s for 3 epochs (when the GPU isn't shared).

---

## Page 7: Why the neural model LOST to BM25 (the most important page) ★★

This is a real, common result — and explaining it is exactly what interviewers test.

1. **No language knowledge.** It started from random numbers and saw only 163k pairs. It can't know that
   "sneakers" ≈ "trainers" unless it saw that in training. → **Week 4: start from a pretrained transformer**
   that already learned language from billions of words.
2. **Training ≠ the test task.** Training negatives are random other products in the batch (a shoe vs a
   phone — easy). Task 1 asks to order ~20 products that are ALL about the same thing (iphone vs iphone
   case — hard). The model learned the easy job. → **Week 5: hard negatives** (similar-but-wrong products).
3. **Bag of pieces ignores word order.** "case for iphone" and "iphone for case" look identical.
   → **Week 4: attention** reads word order and context.

> ★ **A new model that loses to the baseline is still a result.** Report it with its CI, explain why,
> and use the reasons to plan next steps. Never hide it.

---

## Page 8: Engineering lessons

| Lesson | What happened |
|---|---|
| **Overfit-one-batch sanity check** | a correct training loop must be able to memorize one batch. Ours dropped from 5.60 to 0.37 — but not to 0. |
| **Investigate surprises** | 41 of 256 rows in that batch share product text with another row, so some "wrong" columns are identical to the right one → loss can't reach 0. Data, not a bug. |
| **Batch size didn't matter here** | the real limit was the model and data, not the number of negatives. |
| **GPU contention** | the same run took 55 s alone and 5,554 s while a game used the GPU. Benchmark on an idle machine. |

---

## Page 9: Stack added this week

| Tool | Job | Underneath |
|---|---|---|
| **PyTorch 2.11 (CUDA 12.8)** | tensors, autograd, GPU training | records operations on tensors, then runs the chain rule backward (like our engine) using GPU kernels |
| **`nn.EmbeddingBag`** | text pieces → averaged vector | a big lookup table (262,144 × 128) + averaging, in one fast operation |
| **AdamW** | the optimizer (update rule) | per-weight adaptive step sizes + weight decay (shrinks weights slightly, like L2) |
| **CUDA** | runs math on the NVIDIA GPU | thousands of small cores doing matrix multiplications in parallel |

---

## Page 10: Self-test

1. What does autograd do, in one sentence?
2. `y = x*x + x`, x = 3. What is dy/dx, and why must gradients be **added** in code?
3. Why do neural networks need a non-linearity like ReLU?
4. What are in-batch negatives? What is a false negative, and how did we handle it?
5. What does temperature do? Why did temperature 1.0 train badly?
6. Why did the two-tower model lose to BM25? Give two reasons and the fix for each.
7. What is the overfit-one-batch test, and what did ours reveal?

### Model answers
1. It records how every value was computed, then applies the chain rule backwards to get the gradient of
   the loss with respect to every weight.
2. dy/dx = 2x + 1 = 7. x is used in three places; each use contributes a gradient, and the total effect is
   their sum — so the code accumulates with `+=`.
3. Without it, stacked layers collapse into one linear function, so the network can only learn straight-line
   (linear) relationships.
4. Other pairs' products in the same batch, used as wrong answers for free. A false negative is a product in
   the batch that is actually correct for the query (same query appears twice); we masked it out of the loss.
5. It scales similarities before softmax: low temperature = sharper distribution, stronger gradients. With
   temperature 1.0, cosine values in [−1, 1] give nearly equal logits, softmax is almost flat, and the loss
   stayed near ln(256) — the model barely learned.
6. (a) No pretrained language knowledge → use a pretrained transformer (Week 4). (b) Easy random negatives
   vs hard Task-1 candidates → train with hard negatives (Week 5). Also: bag-of-pieces ignores word order →
   attention (Week 4).
7. Train on one small batch many times; a correct pipeline should memorize it (loss → ~0). Ours fell to 0.37
   because 41 rows had duplicate product text — a data property, not a code bug.
