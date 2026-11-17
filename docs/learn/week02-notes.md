# Week 2 notes — teaching a model to rank

★ = memorize. Read top to bottom.

---

## Page 1: The big idea

Week 1: **we** wrote the scoring rule (BM25 formula).
Week 2: we give the computer many **clues** and the right answers, and it **learns** the scoring rule.

```
Week 1:  query + product ──► BM25 formula (hand-written) ──► score
Week 2:  query + product ──► 14 clues ──► model (learned from 18,789 labelled queries) ──► score
```

> ★ **Machine learning = learning a rule from examples instead of writing it by hand.**

---

## Page 2: Features (the clues)

A **feature** is one number describing a (query, product) pair. We built 14:

| Feature | Plain meaning | Example: query "samsung tv 55 black" |
|---|---|---|
| bm25_title / brand / bullets / description | Week 1 BM25 score on each text field | high if words match |
| title_coverage | share of query words found in the title | 4 of 4 → 1.0 |
| title_jaccard | shared words ÷ all distinct words | 4 ÷ 6 = 0.67 |
| phrase_in_title | whole query appears as-is | 0 or 1 |
| brand_in_query | product's brand is in the query | "samsung" → 1 |
| color_in_query | product's color is in the query | "black" → 1 |
| numbers_matched / numbers_missing | numbers like "55", "12" found / not found in title | "55" found → 1 / 0 |
| query_len, title_len, has_description | sizes and presence | — |

> ★ **Feature engineering** = turning domain knowledge ("shoppers care about brand and size") into numbers.
>
> ★ **Leakage rule:** features may use **query text and product text only — never the label.**
> Test `test_features_do_not_depend_on_labels` flips every label and checks no feature changes.
> Example of a forbidden feature: "how often this product was labelled Exact" — that is the answer itself.

---

## Page 3: Logistic regression (the simplest learner) ★★

**Question it answers:** "What is the probability this product is Exact?"

**How it works:**
1. Give each feature a **weight** (how important it is). Start all at 0.
2. Score = w1·feature1 + w2·feature2 + … + bias.
3. **Sigmoid** squashes any score into 0..1 → a probability.

$$p = \sigma(w \cdot x + b), \qquad \sigma(z) = \frac{1}{1+e^{-z}}$$

**How it learns — the 4-step loop (used in EVERY model later, including neural networks):**

```
repeat:
  1. predict      p = sigmoid(X·w + b)
  2. measure loss  how wrong? (log loss)
  3. gradient     which way to change each weight to reduce the loss
  4. update       w = w − learning_rate × gradient     ("step downhill")
```

**Log loss** (how wrong): for a true Exact product, predicting 0.9 costs little; predicting 0.1 costs a lot.

**Two helpers:**
- **Standardize** features (mean 0, std 1) so a feature measured in big numbers (title_len ≈ 20) doesn't
  dominate one in small numbers (coverage ≈ 0.5). ★ Use **train** mean/std on dev and test too.
- **L2 penalty** = small cost for big weights → prevents over-trusting any single feature (less overfitting).

> ★ We wrote it from scratch in NumPy (`ltr/logreg.py`). It matches scikit-learn: weights equal within
> 0.0001 on a test problem, and on real data both score **0.8296**. That proves we understand it.

---

## Page 4: Pointwise vs listwise ★

| | Pointwise (logistic regression) | Listwise (LambdaMART) |
|---|---|---|
| Looks at | one product at a time | the whole list for a query |
| Learns | "is this product Exact?" | "is this ORDER good?" |
| Optimizes | log loss | nDCG directly |
| Problem | doesn't care about ranking between products | — |

**Analogy:** pointwise = grading each student alone. Listwise = deciding the class ranking.
Search needs a ranking, so listwise fits the task better.

> ★ Learning to rank has 3 families: **pointwise** (score each item), **pairwise** (is A better than B?),
> **listwise** (optimize the whole list's metric). LambdaMART is pairwise/listwise.

---

## Page 5: LambdaMART (our best model this week) ★★

**Two parts:**
1. **MART = gradient-boosted decision trees.**
   - A **decision tree** asks yes/no questions: "coverage > 0.8? → brand_in_query = 1? → score +0.3".
   - **Boosting** = build tree 1, see its mistakes, build tree 2 to fix them, tree 3 fixes what's left…
     Final score = sum of all trees. Our model used **274 trees**.
2. **Lambda = the ranking trick.** For every pair of products in a query, it asks "if I swapped these two,
   how much would nDCG change?" and pushes hardest on the swaps that matter most (near the top, big label gap).

**Settings that matter:**
- `label_gain = [0, 0.01, 0.1, 1.0]` → tells LightGBM our ESCI gains. Its default (2^label − 1) would
  optimize a *different* nDCG than the one we report.
- **Early stopping**: keep adding trees until a held-out slice of **train** stops improving (100 trees with
  no gain), then stop. ★ We did NOT use dev for this — dev must stay a clean measurement.
- `learning_rate 0.05`: each tree only corrects a little → slower but more reliable.

> ★ Why trees for this data: our features are a small table of numbers. Gradient-boosted trees are usually
> the strongest model for tabular data, are fast, and handle features on different scales without standardizing.

---

## Page 6: Results (dev, 2,099 queries, full-list nDCG)

| Model | nDCG full | nDCG@10 | Δ vs BM25 title (95% CI) |
|---|---:|---:|---|
| Random | 0.7500 | 0.5786 | −0.0693 |
| BM25 title (Week 1) | 0.8193 | 0.6907 | — |
| LambdaMART, BM25 features only | 0.8223 | 0.6961 | +0.0031 [+0.0004, +0.0058] |
| Logistic regression (ours) | 0.8296 | 0.7069 | +0.0104 [+0.0067, +0.0141] |
| Logistic regression (scikit-learn) | 0.8296 | 0.7070 | identical to ours (Δ 0.0000) |
| LambdaMART, no BM25 features | 0.8322 | 0.7111 | — |
| **LambdaMART, all 14 features** | **0.8359** | **0.7167** | **+0.0166 [+0.0127, +0.0205]** |
| Official baseline (trained neural model, test, 3 locales) | 0.850 | — | — |

Head-to-head: LambdaMART vs our logistic regression +0.0062 [+0.0032, +0.0093] → listwise wins.

**What it tells us:**
1. **Learning beats hand-tuning.** Week 1 tuning of BM25 gave nothing; learning from features gave +0.0166.
2. **The new clues matter more than BM25 combinations:** BM25-only model +0.003; all features +0.0166.
3. **Every feature group helps:** removing BM25 features costs −0.0037; removing the others costs −0.0135.
4. **Listwise > pointwise**, as theory predicts.
5. **Still a gap to 0.850.** These features still only match *words*. Next: models that learn *meaning*.

**Most useful features (LightGBM "gain" importance):** title_coverage, bm25_title, title_jaccard,
numbers_missing, bm25_bullets. ★ `numbers_missing` ranking 4th shows sizes/models ("55 inch",
"iphone 12") matter a lot to shoppers — domain knowledge turned into a number paid off.

---

## Page 7: Bugs found this week (learn from them)

| Bug | Cause | Lesson |
|---|---|---|
| Product with no description counted as "has description"; tokenizer crashed | Parquet empty values become `NaN`, and `NaN` is "true" in Python | clean missing values where data is **loaded** |
| My own test expected Jaccard 4/7 | I counted wrong; the code (4/6) was right | tests can be wrong too — check by hand |
| Test data again stored 0.1 as DECIMAL | same DuckDB trap as Week 1 | write down traps once you meet them |

---

## Page 8: Stack added this week

| Tool | Job | Underneath |
|---|---|---|
| **LightGBM** | LambdaMART ranker | builds trees by histogram-binning feature values (fast), adds them one by one (boosting) |
| **scikit-learn** | reference logistic regression | solver L-BFGS finds the same minimum our gradient descent walks to |
| **pandas** | the feature table | DataFrame = table in memory; `groupby("query_id")` handles per-query work |

---

## Page 9: Glossary

- **Feature**: one number describing a (query, product) pair.
- **Weight**: how much the model trusts a feature.
- **Loss**: a number for "how wrong" the model is. Training = making it small.
- **Gradient**: direction to change weights to reduce loss.
- **Learning rate**: step size of each change.
- **Overfitting**: memorizing training data; good on train, bad on new data.
- **Regularization (L2)**: penalty that keeps weights small to reduce overfitting.
- **Boosting**: each new tree fixes the previous trees' mistakes.
- **Early stopping**: stop training when held-out performance stops improving.
- **Feature importance**: how much each feature helped the trees.

---

## Page 10: Self-test

1. What is a feature? Give two from our project and what they mean.
2. Name the 4 steps of the training loop.
3. Why must dev/test use the **train** mean and std when standardizing?
4. Pointwise vs listwise — what's the difference, and which won here?
5. In LambdaMART, what does "boosting" mean?
6. Why didn't we use the dev set for early stopping?
7. Give an example of a feature that would be leakage.

### Model answers
1. A number describing a query-product pair. `title_coverage` = share of query words in the title;
   `numbers_missing` = query numbers (like "55") absent from the title.
2. Predict → measure loss → compute gradient → update weights (step downhill). Repeat.
3. The model learned using train's scale. Dev/test must be measured with the same ruler; computing their
   own mean/std would also use information from the evaluation data (a form of leakage).
4. Pointwise scores each product alone ("is it Exact?"); listwise optimizes the whole ranked list (nDCG).
   Listwise (LambdaMART) won: +0.0062 [+0.0032, +0.0093] over logistic regression.
5. Trees are built one after another; each new tree learns to correct the mistakes of all previous trees.
6. Dev is our measuring stick. If it chooses when to stop, dev scores become optimistic — we tuned on it.
   So we held out 10% of **train** queries for early stopping.
7. "Fraction of times this product was labelled Exact in the dataset" — it is computed from the labels,
   i.e. the answer we're trying to predict.
