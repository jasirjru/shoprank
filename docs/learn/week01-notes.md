# Week 1 notes — what was built, how it works, what to remember

Read top to bottom. Each section: **what it is → the important lines → remember**.
"Remember" items are what interviewers ask. Model answers to the checkpoint are at the end.

---

## 0. The stack this week (and why each piece)

| Tool | Job here | Why this one | What happens underneath |
|---|---|---|---|
| **Python 3.12** | everything | ML ecosystem standard | — |
| **uv** | creates the project's own environment and installs exact versions (`uv.lock`) | fast; same versions on every machine | resolves dependencies once, writes a lock file, installs into `.venv/` |
| **DuckDB** | reads the 1.1 GB Parquet, filters, joins, splits | SQL on files, no server, uses all CPU cores, fits 16 GB RAM | columnar engine: reads only the columns a query needs, processes them in batches |
| **Parquet** | storage format for raw and processed tables | ~10x smaller than CSV, typed, fast column reads | data stored column by column, compressed (zstd) |
| **NumPy** | bootstrap resampling | vectorized math: 10,000 resamples in one call | operations run in C over whole arrays, not Python loops |
| **pytest** | 30 tests define correct behaviour | standard Python testing | finds `test_*.py`, runs each `test_*` function, reports failures |
| **scikit-learn** | only in tests, as an independent check of our nDCG | trusted reference implementation | — |
| **Ruff** | lint (style + bug patterns) | very fast | static analysis of the code |

**Remember:** we use plain Python for BM25 and metrics *on purpose* — to understand them. Later weeks swap
in fast libraries (`bm25s`, FAISS) and we check they give the same numbers.

---

## 1. Data pipeline (`src/shoprank/data/`)

**Flow:**
```
download.py  ──► data/raw/*.parquet + manifest.json (bytes + SHA-256 per file)
ingest.py    ──► data/processed/judgments.parquet, queries.parquet, products.parquet
```

**Important lines in `ingest.py`:**
```sql
CAST(CASE esci_label WHEN 'E' THEN 1.0 ... END AS DOUBLE) AS gain
```
→ turns a human label into a number. Without the CAST, DuckDB stores 0.1 as an exact DECIMAL, which does
not equal Python's float 0.1. A test caught this.

```sql
CASE WHEN split = 'test' THEN 'test'
     WHEN md5_number(CAST(query_id AS VARCHAR)) % 10 = 0 THEN 'dev'
     ELSE 'train' END AS split
```
→ the split. Official test stays test. 1 in 10 train **queries** become dev, chosen by an MD5 hash of
the query id, so it never changes between runs, machines or library versions.

**Numbers (US English):** 97,345 queries · 1.82M labels · 1.22M products.
Labels: Exact 69% · Substitute 20% · Irrelevant 9% · Complement 2%.

**Remember:**
1. **Split by query, never by row.** Row-splitting puts the same query in train and dev, so a model can
   memorize it and dev scores lie. `tests/test_ingest.py::test_no_query_in_two_splits` enforces this.
2. **Test set is locked.** The runner refuses `--split test` without `--allow-test`. Tune on dev only.
3. **Hash the raw files** (manifest) so every result is tied to exact inputs.
4. **69% Exact** means "random order" already scores well. Always report a random baseline.

---

## 2. Ranking metrics (`src/shoprank/eval/metrics.py`)

**DCG — the core line:**
```python
sum(gain / math.log2(position + 1) for position, gain in enumerate(ranked_gains[:k], start=1))
```
- `ranked_gains[:k]` → only the top k count.
- `enumerate(..., start=1)` → positions 1, 2, 3, …
- `/ math.log2(position + 1)` → position 1 divides by 1 (full credit), position 3 by 2, position 10 by 3.46.

**nDCG — normalize to 0..1:**
```python
ideal = dcg_at_k(sorted(ranked_gains, reverse=True), k)
return dcg_at_k(ranked_gains, k) / ideal if ideal > 0 else 0.0
```
- ideal = the same products in the best possible order.
- `if ideal > 0` → a query with no relevant products would divide by zero.

**Recall@k** = relevant found in top k ÷ all relevant. **Reciprocal rank** = 1 ÷ position of the first
relevant item (MRR = its average).

**Two nDCG versions in this project:**
- `ndcg10` = what the shopper sees on page 1.
- `ndcg_full` = **official ESCI Task-1 metric** (whole candidate list). Verified from Amazon's script:
  `trec_eval -m 'ndcg.1=0,2=0.01,3=0.1,4=1'`. Public 0.850 / 0.9043 are this metric, averaged over US+ES+JP.

**Remember:**
1. DCG formula and *why log*: attention drops with position, smoothly.
2. *Why normalize*: so a query with 3 relevant items and one with 20 are on the same 0–1 scale.
3. nDCG cares about **order of graded items**; recall only about **whether** items appear in top k.

---

## 3. Paired bootstrap (`src/shoprank/eval/bootstrap.py`)

**Question:** B scored higher than A on 2,099 queries. Real, or luck of which queries we used?

**The important lines:**
```python
deltas = b - a                                           # per-query improvement
picks = rng.integers(0, deltas.size, size=(10_000, n))   # 10,000 resamples, WITH replacement
resample_means = deltas[picks].mean(axis=1)              # mean improvement in each resample
low, high = np.percentile(resample_means, [2.5, 97.5])   # middle 95% = confidence interval
```

**Reading the result:** interval entirely above 0 → B better. Entirely below → A better. Crosses 0 → no
clear difference (don't claim an improvement).

**Remember:**
1. **Paired** = compare per-query differences. Hard queries are hard for both systems; differencing cancels
   that, so the interval is much narrower than comparing two averages.
2. **With replacement** = some queries picked twice, some not at all; that mimics "a different sample of queries".
3. **Fixed seed** = same interval every run (reproducible).

---

## 4. BM25 (`src/shoprank/retrieval/bm25.py`)

**`fit` precomputes, once:** word counts per product, product lengths, average length, and for each word
how many products contain it (`doc_freq`). Then scoring a query is cheap.

**IDF — rare words matter more:**
```python
math.log(1 + (N - n_t + 0.5) / (n_t + 0.5))
```
"kindle" in 20 of 40,000 products → large. "for" in 30,000 → near 0. The `1 +` keeps it positive.

**Score — the heart:**
```python
length_norm = 1 - b + b * doc_length / avgdl
total += idf * f * (k1 + 1) / (f + k1 * length_norm)
```
- `f` = how often the word appears in this product. More is better, but **levels off** (k1 controls how fast).
- `length_norm` > 1 for long products → smaller score. `b` controls how strong (0 = off, 1 = full).
- Loop over `set(query)` → each query word counts once.

**Remember:**
1. Three ideas: **rare words matter more (IDF), repeats saturate (k1), long docs are penalized (b)**.
2. BM25 has **no idea of meaning**: "sneakers" ≠ "trainers", "phone" vs "phone case" look alike.
3. It is the baseline every learned model must beat.

---

## 5. Results (dev, 2,099 Task-1 queries)

| Run | nDCG full | nDCG@10 | MRR (Exact) | vs bm25_title (full-list Δ, 95% CI) |
|---|---:|---:|---:|---|
| random | 0.7500 | 0.5786 | 0.6205 | −0.0693 [−0.0763, −0.0622] |
| **bm25_title** (k1 1.5, b 0.75) | **0.8193** | 0.6907 | 0.7494 | — |
| bm25 title+brand | 0.8207 | 0.6927 | 0.7510 | +0.0015 [−0.0002, +0.0032] no clear diff |
| bm25 title+brand+bullets | 0.8199 | 0.6927 | 0.7507 | +0.0006 [−0.0034, +0.0046] no clear diff |
| bm25_title, b = 0 | 0.8087 | 0.6747 | 0.7269 | **−0.0105 [−0.0136, −0.0076] worse** |
| bm25_title, b = 1 | 0.8185 | 0.6894 | 0.7463 | −0.0008 [−0.0022, +0.0006] no clear diff |
| bm25_title, k1 = 0.9 | 0.8200 | 0.6912 | 0.7523 | +0.0007 [−0.0002, +0.0016] no clear diff |
| bm25_title, k1 = 3 | 0.8175 | 0.6880 | 0.7450 | −0.0018 [−0.0029, −0.0007] slightly worse |

Public reference (test, 3 locales): official baseline 0.850 · winner 0.9043.

**What the numbers say:**
1. BM25 beats random clearly (+0.069).
2. **Tuning BM25 barely moves it.** Extra fields and k1 give nothing measurable. BM25 has hit its ceiling
   (~0.82). → To go further we need models that understand meaning. That is Weeks 2–6.
3. **Length normalization matters** (b = 0 is clearly worse): long titles match words by chance.
4. Without confidence intervals, "title+brand +0.0015" would look like a win. It isn't proven.

---

## 6. Where BM25 fails (worst dev queries)

| Failure type | Example query | What BM25 put first | Fix later in ShopRank |
|---|---|---|---|
| **Long natural-language request** | "One that's fast, reliable and has a good processor… light-weight" | funny T-shirts (matched "good", "like") | fine-tuned bi-encoder (meaning) — Week 5 |
| **Constraint ignored** | "air force 1 shoes **girls**" | men's Nike Air Force 1 | query parser (gender/size) — Week 7; features — Week 2 |
| **Accessory vs product** | "iphone xs pro" | screen protectors (Complement) | cross-encoder reads query+product together — Week 6 |
| **Spelling** | "ana outfit for 12 months" (Anna) | Cinderella/Elsa costumes | subword tokenization + dense models — Week 4–5 |
| **Non-English in US set** | "sillon una pieza para sexo" | children's chair | multilingual model or language detection — stretch |
| **Label noise** | "iqos heat not burn" → Exact = Duracell batteries; "anziner" → Exact = tea lights | — | **can't be fixed**; report it, don't overfit to it |

**Remember:** an error taxonomy turns "the model is bad" into a **plan**. And some errors are the data's
fault — a good engineer finds and says that.

---

## 7. Checkpoint — questions and model answers

Try to answer out loud first, then read the answer.

1. **Why nDCG instead of accuracy for search?**
   Search is about **order**, and users mostly see the top. Accuracy treats every item equally and ignores
   position. nDCG rewards relevant items at the top and supports graded relevance (E > S > C > I).

2. **Why split by query? Concrete ESCI leakage example.**
   If "iphone xs pro" had 8 judged products in train and 8 in dev, a model could learn "this query →
   phones, not protectors" from train and look great on the same query in dev. Real users send new
   queries, so dev must contain unseen queries.

3. **What does IDF capture? Why `ln(1 + …)`?**
   How rare (informative) a word is across products. Plain `ln((N − n + 0.5)/(n + 0.5))` goes **negative**
   for words in more than half the products, which would make matching a common word *lower* the score.
   Adding 1 inside the log keeps it ≥ 0.

4. **What do k1 and b control? b = 0 on a 300-word listing?**
   k1: how quickly repeated words stop adding score. b: how much long documents are penalized.
   b = 0 turns length normalization off, so a 300-word listing that mentions many words by chance competes
   equally with a precise 6-word title — our dev run confirms b = 0 is clearly worse (−0.0105).

5. **+0.008 nDCG with 95% CI [−0.002, +0.018]. Conclusion?**
   Not proven: the interval includes 0. Options: evaluate on more queries (narrows the CI), check if the
   gain is concentrated in one query type, or treat it as no difference. Never ship it as "an improvement".

6. **Debugging exercise (`exercises/week01_buggy_dcg.py`) — the two planted bugs:**
   (a) `gains[: k - 1]` drops the k-th result (off-by-one; should be `[:k]`).
   (b) `sorted(gains)` sorts **low to high**, so the "ideal" is the worst order and nDCG goes above 1
   (it printed 16.85). Should be `sorted(gains, reverse=True)`. Rule: nDCG must always be in [0, 1].
