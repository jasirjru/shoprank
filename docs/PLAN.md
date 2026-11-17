# ShopRank plan (from 17 Nov 2026)

Goal: finish ShopRank, understand every part well enough to explain it with no notes, and put it on the resume.

Rules:
- Learn first, then build. Each part gets a lesson before any code changes.
- Every experiment gets a lab-notebook entry: prediction first, then result with a 95% CI.
- The official test set is used **once per version**, at the end. Never tune on it.
- A version goes on the resume only after a no-notes check is passed.
- Every day: 1 hour of job applications and Deep Learning exam prep, alongside this plan.

## Phase A: v1, understand and ship what exists (about 7 days)

| Day | Lesson (format: real-life picture, worked example, diagram, words, interview answer, 5-question quiz) | Build / ship task |
|---|---|---|
| 1 | The problem and the data: ESCI labels (E/S/C/I), gains 1 / 0.1 / 0.01 / 0, train/dev/test split by query | Fix torch, run all tests |
| 2 | Metrics: DCG and nDCG by hand on a 4-product list; why random already scores 0.75; bootstrap CIs | Re-run the BM25 baseline and check the number matches the README |
| 3 | BM25: term frequency, IDF, k1 and b, worked on 3 product titles | – |
| 4 | Features and logistic regression: the 14 features, sigmoid, loss, gradient descent step by hand | – |
| 5 | LambdaMART: decision trees, boosting, why ranking by pairs and lists beats scoring one item at a time | – |
| 6 | Two-tower model: embeddings, InfoNCE loss, in-batch negatives, why it lost to BM25 | – |
| 7 | Mock interview (no notes), then fix weak topics | One official US test-set run; README with results and limits; push to GitHub (ask first); resume bullets |

## Phase B: v2, transformers (about 3 weeks). Updated 17 Nov from `docs/BEST_MODEL_RESEARCH.md`

| Step | Learn | Build | Target (estimate) |
|---|---|---|---|
| B1 | Embeddings and semantic similarity; negation | Add bge-small cosine, negation and numeric features to LambdaMART | about 0.85 (a same-setup public project reached 0.854) |
| B2 | Transformers, attention, cross-encoders, fine-tuning | **Fine-tune DeBERTa-v3-base cross-encoder**: 4-class E/S/C/I loss, rank by expected gain, input query + title + brand + colour, max_length 128 | about 0.87-0.89; beat Amazon's 0.857 |
| B3 | Stacking, error analysis | Cross-encoder score into LambdaMART; one official test run | +0.003-0.01 |
| B4 (optional) | Ensembles, adversarial training | DeBERTa-v3-large or a 2-3 model ensemble | about 0.89-0.90 (ceiling, because label agreement is 91%) |

## Phase C: make it real (after v2)

1. A small live demo (Gradio or a Hugging Face Space): type a query, see the ranked products.
2. Manglish check: type 50 real Manglish product queries into a default shop search and count failures; ask 5 Kerala shop owners.
3. Only if the check shows a real gap: label about 200 Manglish queries against one real shop catalogue and compare nDCG for default search versus ShopRank.

Optional later (plan weeks 7-10): QLoRA query parser, serving optimisation, MLOps.

Reference: `C:\Studio\Career\reports\shoprank_career_research_2026-11-17.md` (why ShopRank, honest score 7/10, sources).
