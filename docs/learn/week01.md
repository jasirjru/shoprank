# Week 1 — Data, statistics, IR metrics, BM25

Goal: by Friday you can (1) explain how a search result list is scored, (2) say whether one system
is really better than another, and (3) build keyword search from the formula. Everything later in
ShopRank is measured with what you build this week.

## How we work

Claude builds and runs the code; you learn how it works. Start with **`week01-notes.md`**: the stack,
the important lines of each file, the results and what to remember. This file is the background theory.

| Built | File | Checked by | Primary source |
|---|---|---|---|
| Ranking metrics | `src/shoprank/eval/metrics.py` | `tests/test_metrics.py` (incl. scikit-learn cross-check) | Manning et al., *Introduction to IR*, ch. 8 |
| Paired bootstrap | `src/shoprank/eval/bootstrap.py` | `tests/test_bootstrap.py` | ISLR §5.2 |
| BM25 | `src/shoprank/retrieval/bm25.py` | `tests/test_bm25.py` (hand-computed values) | Robertson & Zaragoza 2009, §3 |
| Debugging exercise | `exercises/week01_buggy_dcg.py` | answer in notes §7 | — |

## 1. The data (ESCI)

- Amazon took real customer queries, pulled about 20 candidate products per query, and had humans
  label each pair: **E**xact, **S**ubstitute, **C**omplement, **I**rrelevant.
- `shoprank.data.ingest` keeps US English and writes three tables:
  `judgments` (query, product, label, gain, split), `queries`, `products`.
- **Task 1** (`small_version = 1`): rerank each query's ~20 judged candidates. This is what the
  public 0.850 baseline and 0.9043 winner were scored on.

## 2. Splits and leakage

- **Leakage** = information from the evaluation data sneaks into training or tuning, so scores look
  better than they will be on new data.
- We split **by query**, never by row. If "red shoes" had some products in train and others in dev,
  a model could memorize that query and look great on dev.
- `test` = ESCI's official test split, locked. The runner refuses `--split test` without `--allow-test`.
  Rule: tune on dev, report test once at the end.
- `dev` = 1 in 10 train queries, chosen by MD5 of `query_id`. MD5 never changes, so the split is
  identical on every machine and every run (tested in `tests/test_ingest.py`).
- BM25's corpus statistics use product text only, no labels, so computing them is not leakage.

## 3. Ranking metrics

- A result list is good if the best products are **near the top**. Position 1 matters most.
- **Gain** per label: E=1.0, S=0.1, C=0.01, I=0.0. A substitute is worth a little; irrelevant is worth nothing.
- **DCG@k** = Σ (gain at position i) / log2(i + 1), for i = 1..k. Lower positions are discounted.
- **nDCG@k** = DCG@k / ideal DCG@k (same gains sorted best-first). Always between 0 and 1, so queries
  with many or few relevant items are comparable.
- **Recall@k** = share of all relevant items found in the top k. **Reciprocal rank** = 1 / position of
  the first relevant item; its mean over queries is **MRR**.
- **Two versions in this project.** nDCG@10 = what a shopper sees on page 1. nDCG over the **full
  candidate list** (k = list length) = the official ESCI Task-1 metric (trec_eval with the same gains,
  no cutoff). The runner reports both. Public numbers (0.850, 0.9043) are full-list and average three
  locales; ours is US only, so treat the comparison as a guide.
- Worked example (from the tests): ranked gains [1.0, 0.1, 0.0, 0.01]
  DCG@4 = 1/1 + 0.1/1.585 + 0/2 + 0.01/2.322 = 1.0674. Ideal [1.0, 0.1, 0.01, 0.0] = 1.0681.
  nDCG@4 = 0.99935. Only the last two are swapped, and they barely matter.

## 4. Is a difference real? Paired bootstrap

- You test system A and B on the same 2,000 queries. B averages 0.012 higher. Could a different set
  of 2,000 queries have shown the opposite? The bootstrap answers that without formulas.
- Steps: per-query difference d = B − A. Resample the d values with replacement, same count, 10,000
  times; take the mean each time. The middle 95% of those means is the **95% confidence interval**.
- If the interval does not include 0, the improvement is unlikely to be luck.
- **Paired** matters: some queries are just hard for every system. Using per-query differences
  removes that shared difficulty, so the interval is much narrower than comparing two averages.

## 5. BM25

Three ideas, one formula (in `bm25.py`):
1. **Rare words matter more** (IDF). "kindle" in 20 of 40,000 products tells you far more than "for".
2. **Repeats help, with diminishing returns** (saturation, controlled by k1). A title that says
   "case case case" is not 3x more about cases.
3. **Long documents get a penalty** (length normalization, controlled by b). A 300-word listing
   matches more words by chance than a 6-word title.

k1 ≈ 1.2–2.0 and b ≈ 0.75 are common starting points; you will measure them on dev.

## 6. Experiments (after tests pass)

Run each on dev, record results in `docs/lab_notebook.md`:

```
uv run python -m shoprank.eval.run_task1 --ranker random --name random
uv run python -m shoprank.eval.run_task1 --ranker bm25 --fields title --name bm25_title
uv run python -m shoprank.eval.run_task1 --ranker bm25 --fields title,brand --name bm25_title_brand
uv run python -m shoprank.eval.run_task1 --ranker bm25 --fields title,brand,bullets --name bm25_tbb
uv run python -m shoprank.eval.run_task1 --compare random bm25_title
uv run python -m shoprank.eval.run_task1 --compare bm25_title bm25_tbb
```

Before running, **write your predictions** in the notebook:
- Random nDCG@10 will be roughly ____ (hint: most ESCI candidates are labelled E, so random is NOT low).
- Adding bullets will help / hurt because ____.

Then: pick the 20 dev queries where BM25 does worst. For each, write one line: why did keyword
matching fail? (synonym, brand, number, negation, spelling, other). That list is next week's motivation.

## 7. Checkpoint (answer without notes; Claude grades)

1. Why is nDCG a better fit than accuracy for search results?
2. Why must we split by query? Describe a concrete leakage that row-splitting would cause in ESCI.
3. What does IDF capture, and why does BM25 use `ln(1 + …)` instead of plain `ln(N / n)`?
4. What do k1 and b each control? Predict what b = 0 does to a 300-word product listing.
5. Your new system gains +0.008 nDCG with 95% CI [−0.002, +0.018]. What do you conclude, and what
   would you do next?

**Debugging task (no AI):** `exercises/week01_buggy_dcg.py` — find both planted bugs.
