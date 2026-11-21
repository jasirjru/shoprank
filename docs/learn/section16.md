# ShopRank lessons: section 16, the one-time official test (built 21 Nov 2026, study after the build)

Official ESCI Task 1 test, US: 8,956 queries / 181,701 pairs, scored exactly once.
Answer the check questions first; the key is at the bottom.

| System | Test nDCG [95% CI] | dev |
|---|---|---|
| BM25 (title) | 0.8202 [0.8170, 0.8232] | 0.8163 |
| LambdaMART | 0.8337 [0.8305, 0.8368] | 0.8342 |
| MiniLM, Amazon's recipe | 0.8544 [0.8513, 0.8573] | 0.8534 |
| MiniLM distilled + bullets (served) | 0.8674 [0.8643, 0.8703] | 0.8667 |
| **DeBERTa-v3-base x5 mean (primary)** | **0.8877 [0.8849, 0.8905]** | 0.8879 |
| LightGBM stack (declared second) | 0.8883 [0.8855, 0.8910] | 0.8879 |
| Amazon's published MiniLM | 0.857 | |

---

## Part 1: Pre-registration: decide before you look

**Picture:** a cricket team names its playing XI before the toss. If it could pick the team after seeing the
pitch report AND the first over, the "best team" claim would mean nothing.

**What we did:** before any test score existed, the notebook named the primary system (DeBERTa x5 mean), the
one declared second system (the stack), the baselines, the metric, the CI method and a prediction for each
number. The script refuses a second test run (`check_once`).

```
dev     -> make choices (many looks)
dev2    -> check choices once (1,053 queries)
test    -> report once, choices frozen (8,956 queries)
```

**Interview answer:** "I pre-registered the systems, metric and predictions in a lab notebook before scoring
the official test set, scored it once with a script that refuses a second run, and reported the primary and
the declared second system with confidence intervals. Nothing was changed after seeing test."

## Part 2: Reading the result honestly

**Real numbers:** primary 0.8877 [0.8849, 0.8905]. Over BM25: +0.0676 [+0.0646, +0.0705]. Over LambdaMART:
+0.0541. The stack: +0.0006 [-0.0003, +0.0014] over the primary, so "no clear difference" even on 8,956
queries. Amazon's 0.857 lies below our whole interval.

- Careful (found in the 21 Nov audit): Amazon's 0.857 is the English part of the KDD Cup *public* test, a
  different sample of US queries from our 8,956, with max length 512 and the full train set. So it is a
  reference point, not a paired comparison. Our own run of their recipe on OUR test (0.8544) is the fair
  baseline: same queries, paired CI.
- Our run of Amazon's own MiniLM recipe gives 0.8544, consistent with their 0.857. That is the sanity check
  that we measure the same thing they did (we train on less data: dev and dev2 are held out).
- dev2 had suggested DeBERTa's dev score was ~0.007 optimistic. Test (8x more queries) says dev was right:
  0.8879 dev vs 0.8877 test. A 1,053-query split is noisier than it looks.

**Interview answer:** "On the official test set the DeBERTa ensemble scored 0.888 nDCG with a 95% interval of
0.885 to 0.891, 6.8 points above BM25 and 3.3 points above my own run of Amazon's cross-encoder recipe on
the same queries (0.854; Amazon published 0.857 on the public-test English queries)."

## Part 3: An independent metric check, and what ties do

**Picture:** you weigh a parcel on two different scales. If they agree, the scale is not the problem.

**What we did:** recomputed every nDCG with pytrec_eval, a Python wrapper of trec_eval (the C program used by
the TREC evaluations since the 1990s). Means agreed within 0.0003. Some single queries differed, up to 0.37.

**Why:** tied scores. 14.4% of the served model's pairs share their score with another product of the same
query (scores pass through bf16, which keeps only ~3 significant digits). trec_eval orders ties by document
id descending; our code ascending. With trec_eval's rule our numbers match exactly (difference 7.8e-16).

```
scores:  A 0.8125   B 0.8125   C 0.40      (A and B tied)
ours:    A, B, C        trec_eval:  B, A, C
if A is Exact and B Irrelevant, the two orders give different nDCG for this query
```

**Interview answer:** "I re-checked the metric with trec_eval. Averages matched to 3e-4; the per-query gaps came
only from tie-breaking, which I proved by switching our tie rule and getting an exact match. The ties come from
bf16 scores, a small effect (5e-5 on the mean) that fp32 score heads would remove."

---

## Check questions

1. (Part 1) Why name the primary system before scoring test? A) it is faster B) choosing after seeing test
   turns test into another dev set, so the number is no longer unbiased C) the GPU needs it D) to save memory
2. (Part 2) The stack is +0.0006 [-0.0003, +0.0014] over the primary on test. The right claim: A) the stack
   is better B) no clear difference C) the stack is worse D) test is broken
3. (Part 2) Why does our run of Amazon's MiniLM recipe (0.8544) matter? A) it is our best model B) it shows
   our metric and data match Amazon's, so comparing to their 0.857 is fair C) it is faster D) it uses bullets
4. (Part 3) Our nDCG and trec_eval differ on single queries but match on average. The cause was: A) a bug in
   trec_eval B) tied scores ordered differently C) different gains D) missing queries

## Answer key

1. B. 2. B. 3. B. 4. B (switching our tie rule to trec_eval's gives an exact match).
