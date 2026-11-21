# Where ShopRank stands on ESCI Task 1 (21 Nov 2026)

Question: how good are ShopRank's scores compared with everything published on the same task, and what would it
take to reach the top? Sources were read in full (links at the bottom); nothing here re-runs a model on test.

## 1. A trap in the metric: two gain conventions exist
The ESCI paper defines the gains as Exact 1, Substitute 0.1, Complement 0.01, Irrelevant 0. Amazon's released
evaluation code (`esci-data/ranking/prepare_trec_eval_files.py`) maps S to grade 2 and C to grade 3 and runs
trec_eval with `ndcg.1=0,2=0.01,3=0.1,4=1`, i.e. **Substitute 0.01, Complement 0.1: swapped**. The SQID paper
(2024) found the same and reports Amazon's MiniLM baseline on the US test as 0.83 with the code's mapping and
0.8562 with the paper's mapping. ShopRank uses the paper's mapping. Frozen test scores, both conventions:

| US official test, 8,956 queries | paper gains (ShopRank) | code gains (Amazon README) |
|---|---|---|
| MiniLM, Amazon's recipe, our run | 0.8544 | 0.8354 |
| MiniLM, Amazon's recipe, SQID paper (independent) | 0.8562 | 0.83 |
| ShopRank served (distilled MiniLM) | 0.8674 | 0.8499 |
| ShopRank top (DeBERTa-v3-base x5) | 0.8877 | 0.8724 |

Our replication matches two independent sources within 0.002-0.005 under both conventions: strong outside
evidence that data, split and metric are right. Any comparison must use the same convention.

## 2. The scale
| Level | Score | Comparable to ShopRank? |
|---|---|---|
| Random order (SQID) | 0.748 | yes (US test) |
| BM25 (ours) | 0.8202 | yes |
| Amazon's MiniLM baseline | 0.852 all locales / 0.857 English, public test | roughly (different query sample) |
| Same baseline, US test (SQID, ours) | 0.856 / 0.854 | **yes** |
| Public LightGBM + MiniLM-L6 features project (GitHub, Anankin77) | 0.854 | yes (US test, paper gains; not peer-reviewed) |
| Public fine-tuned cross-encoder on Hugging Face (community model card) | 0.860 | yes (states US test, paper gains; not peer-reviewed) |
| **ShopRank served, 33M, CPU** | **0.867** | |
| **ShopRank top, DeBERTa-v3-base x5** | **0.888** | |
| KDD Cup 2022, 20th place | 0.893 | no: all 3 locales, private test subset |
| KDD Cup 2022, 9th: one mMonoT5 3.7B model | 0.901 | no (same reasons) |
| KDD Cup 2022, 1st: best single DeBERTa-v3-large | 0.9015 | no |
| KDD Cup 2022, 1st: ensemble | 0.9043 | no |

Why the KDD Cup numbers are not directly comparable: they average three locales (in Amazon's baseline English
scored highest, so the US-only score of the winners is probably a little above their all-locale score), they are
on the private part of the test, the leaderboard's gain convention is not documented in the papers, and those
teams trained on all locales plus the Task 2/3 data, used pseudo-labels from the public test and ensembles.

## 3. Verdict (tier)
- **Against every public baseline and replication on the same test: top.** +0.033 (top model) and +0.013 (served
  model) over our re-run of Amazon's recipe on the same queries; above the independent replication (SQID, 0.8562),
  a public LightGBM project (0.854) and the best public community model found (0.860).
- **Against the 2022 competition field: about 0.005 below 20th place and about 0.017 below the winners**
  (estimate; not the same test set or locales). The gap is explained by resources, not by a missing idea: the winners used 3-20x larger models
  (DeBERTa-v3-large, InfoXLM-large, mT5 3.7B), about 2-3x more labelled pairs (Task 2/3 data, all locales), and
  ensembles of 6+ large models. Their own ablations: Task 2 data "improved the model substantially" (9th place);
  DeBERTa-v3-large + adversarial training +0.003; self-distillation + pseudo-labels +0.002; label cleaning +0.003.
- **Served model: unusually efficient.** 0.867 at 33M parameters on a CPU is above every public baseline and above
  two 2025-26 ready-made ~600M rerankers measured zero-shot on dev (0.854-0.859).
- One-line rating: **strong, carefully validated single-GPU result; about 1-2 points below competition winners
  that used 10-100x more compute.** Label noise (ESCI: 91% annotator agreement) and the top-20 cluster at 0.89-0.90
  suggest the practical ceiling is close.

## 4. Is ShopRank finished?
Yes, as defined: test numbers with CIs, reproducible repo, served model with measured latency, lessons, CI. What is
not done is optional and listed in `project_status_2026-11-21.md`. If more score is wanted, the evidence above
points to more data and bigger models as the levers (section 5); a first measurement shows more data alone is
worth about +0.005 for the small model.

## 5. If the goal were to close the gap (not started; each needs a GPU budget decision)
1. **More labelled data: train on the large ESCI version (US).** Our models saw 68-77% of the small version's
   training queries; the large version has ~3.6x more US training queries (63,553 vs 17,823). **Measured since
   (dev):** MiniLM title-only, Amazon's recipe, large vs small data: +0.0052 [+0.0023, +0.0082] (0.8586 vs 0.8534),
   about what distillation alone gave (0.8575). Real but modest for this model; the effect on DeBERTa and on a
   distilled student trained on large data (needs ~7 GPU-hours of teacher folds) is untested.
2. DeBERTa-v3-large on 8 GB (bf16, gradient checkpointing, small batches): +0.003 in the winners' ablation, at
   ~3-4x the training time.
3. Cheap tricks with published gains: label cleaning (+0.003), self-distillation (+0.002). Pseudo-labelling on
   test is not allowed under ShopRank's rules.
Any new model can only be judged on dev and dev2: the official test set is spent.

## Sources
- ESCI paper: https://arxiv.org/abs/2206.06588 ; code: https://github.com/amazon-science/esci-data
- SQID (gain mapping, 0.8562): https://arxiv.org/abs/2405.15190
- 1st place, KDD Cup 2022 Task 1: https://arxiv.org/abs/2208.02958
- 9th place (mMonoT5): https://arxiv.org/abs/2208.06264
- 5th place (ZhichunRoad): https://arxiv.org/abs/2301.13455
- Community model card (0.8602): https://huggingface.co/albertobarnabo/ecommerce-product-search-reranker
- Public LightGBM project (0.854): https://github.com/Anankin77/esci-search-reranking
