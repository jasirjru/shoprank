# ShopRank: weeks 1-3 in one page

Read this first. Then go into `week01-notes.md`, `week02-notes.md` and `week03-notes.md` for detail.

## The story in 5 lines

1. **Week 1:** got the Amazon data, built the scoreboard (nDCG), and wrote a **hand-made rule** (BM25). Score: **0.819**.
2. BM25 tuning stopped helping, so **Week 2** gave the computer **14 clues** and let it **learn** the rule (logistic regression, then LambdaMART). Score: **0.836**.
3. Clues still only match **words**, so **Week 3** built a **neural network** that learns **meaning** (two-tower model). Score: **0.785**, **worse** than BM25.
4. It lost for 3 known reasons, and each reason is a later week: no language knowledge (week 4), easy training examples (week 5), ignores word order (week 4).
5. Amazon's own model scores **0.857**. Weeks 4-6 aim to reach it.

```
random 0.750 ─► BM25 0.819 ─► LogReg 0.830 ─► LambdaMART 0.836 ─► (two-tower 0.785, a dead end) ─► goal 0.857
   week 1          week 1         week 2          week 2                   week 3                     weeks 4-6
```

## One running example: query "samsung tv 55 black"

| Week | What the system does with it |
|---|---|
| 1 (BM25) | Counts matching words. "samsung" is rare, so it scores high; "tv" is common, so it scores lower. It can't tell a TV from a TV remote. |
| 2 (features + learning) | Measures 14 clues: are all 4 words in the title? Is "55" in the title (`numbers_missing`)? Is the brand Samsung? The model **learned** that a missing number is a bad sign. |
| 3 (two-tower) | Turns the query and each product into 128 numbers and checks how close they are. It knows meaning only from 163k training pairs, so it is weaker than word matching. |

## Week 1: data, scoreboard, first rule

**Built:** download, ingest, split, metrics, bootstrap, BM25, and 30 tests.

**Key points:**
- **Data:** 97,345 US queries, 1.82M human labels. The labels are 69% Exact, so random order already scores 0.75.
- **Gains:** Exact 1, Substitute 0.1, Complement 0.01, Irrelevant 0.
- **Split by query, never by row.** The same query must not appear in both train and dev. The test set is locked.
- **nDCG:** relevant items near the top earn more (divide by log2(position + 1)). Normalise by the best possible order, so the score is always between 0 and 1.
- **Paired bootstrap:** is the gain real or luck? If the 95% interval stays above 0, it is real; if it crosses 0, it is not proven.
- **BM25 has three ideas:**
  - rare words matter more (IDF);
  - repeating a word helps less and less (k1);
  - long text is penalised (b).
- **Result:** BM25 0.819. Tuning gave nothing, because BM25 has hit its ceiling.
- **It fails on:**
  - long sentences;
  - ignored constraints ("girls");
  - accessory versus product (iPhone versus case);
  - spelling;
  - wrong labels in the data itself.

## Week 2: learning to rank

**Built:** 14 features, logistic regression from scratch (it matches scikit-learn exactly), and LightGBM LambdaMART.

**Key points:**
- **Machine learning means learning the rule from examples** instead of writing it by hand.
- **Leakage rule:** features may use the query and product text, never the label.
- **The training loop** (used by every later model): predict, measure the loss, compute the gradient, update. Repeat.
- **Pointwise vs listwise:**
  - pointwise (logistic regression) judges each product alone;
  - listwise (LambdaMART) judges the order of the whole list.
- **LambdaMART** = boosted trees (each tree fixes the last one's mistakes; 274 trees) + lambdas (push hardest on swaps that change nDCG most).
- **Early stopping used held-out *train* queries, not dev**, so dev stays clean.
- **Result:** LambdaMART 0.836, +0.0166 over BM25, and the gain is proven.
- **Top features:** title_coverage, bm25_title, title_jaccard, numbers_missing.

## Week 3: neural networks

**Built:** an autograd engine from scratch (it matches PyTorch to 9 decimals), a small MLP, and a two-tower model in PyTorch on the GPU.

**Key points:**
- **Autograd:** every number remembers how it was made; walk backwards with the chain rule. If a value is used twice, its gradients add.
- **ReLU** is needed; without it, stacked layers collapse into one straight line.
- **Two towers:**
  - the query and the product each become 128 numbers, compared by cosine similarity;
  - products can be encoded ahead of time, which is how large-scale search works.
- **Text to numbers:** words plus 3-letter pieces, hashed into 262,144 slots. "anna" and "ana" share pieces, so it handles misspellings.
- **InfoNCE loss:** pick the right product out of the batch; the other rows' products act as wrong answers. Temperature 0.05 works; 1.0 barely learns. False negatives are masked.
- **Result:** 0.785, worse than BM25. **Explaining why** is the most important interview answer:
  1. it has no language knowledge;
  2. its training negatives were easy;
  3. it ignores word order.

## Five questions to test yourself (answers are in the week notes)

1. Why does random order already score 0.75?
2. What three ideas make up BM25?
3. Why didn't we use dev for early stopping?
4. Pointwise vs listwise: which won, and by how much?
5. Give three reasons the two-tower model lost to BM25, and the fix for each.
