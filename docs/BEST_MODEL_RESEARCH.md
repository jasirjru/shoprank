# Best model for ShopRank: research (17 Nov 2026)

Question: which model gives the highest accuracy on ESCI Task 1 (US), and what can run on an RTX 4060 (8 GB)?

## 1. Why "close to 100" is impossible

From the ESCI paper (Reddy et al. 2022, arXiv 2206.06588, section 2):
- "A minimum of three annotations were collected for each pair", and the majority vote became the label.
- Amazon sampled 100 cases per language and re-checked them: "The overall agreement is 91%."
- For Exact versus Not-Exact the agreement "increases to >96%".
- About half the disagreements were Irrelevant items that the judges thought were Substitutes.

So roughly 9% of fine-grained labels are debatable. A model that is "right" will still be marked wrong on those. **The realistic ceiling is about 0.90-0.91 full-list nDCG.** The best ever recorded is 0.9043.

## 2. What the best teams achieved (KDD Cup 2022, Task 1)

The numbers are averaged over US, ES and JP on the public/private test set, so they are not identical to ShopRank's US-only setup.

| System | nDCG | Source |
|---|---|---|
| BM25 (paper baseline, different setup) | 0.563 | ESCI paper Table |
| Amazon baseline: fine-tuned MS MARCO MiniLM cross-encoder (US) | 0.857 English / 0.852 overall | ESCI paper |
| Winner's single model: **mDeBERTa cross-encoder** | **0.8930** public | Winner paper (arXiv 2208.02958), Table 2 |
| + data cleaning and tuning | 0.8960 | same |
| + self-distillation and pseudo-labelling | 0.8982 / 0.8975 private | same |
| + label smoothing | 0.8990 | same |
| + **DeBERTa-v3-large** (English) + adversarial training (AWP/FGM) | 0.9022 / 0.9015 | same |
| + InfoXLM, XLM-RoBERTa, RemBERT ensemble, translation augmentation, bagging | **0.9043 private (1st place)** | same |

Winner's settings: `max_length=128`, batch 64, learning rate 3e-5. The model was trained as a **4-class classifier (E/S/C/I) with cross-entropy**, and the ranking score is the expected gain: `P(E)*1 + P(S)*0.1 + P(C)*0.01`.

Another top team (github.com/wufanyou/KDD-Cup-2022-Amazon) used 6 cross-encoders per language fed into **LightGBM** with extra features, and the same expected-gain ranking.

**Lesson:** about 95% of the winning score comes from **one fine-tuned cross-encoder** (0.893). Ensembles and tricks add the last 0.011.

## 3. A public project with the same setup as ShopRank

github.com/Anankin77/esci-search-reranking: US, small version (hard queries), about 20 candidates, official test set (8,956 queries), 95% CIs.

| Method | Test nDCG | ShopRank (dev) |
|---|---|---|
| Random | 0.747 | 0.750 |
| BM25 title | 0.818 | 0.819 |
| Zero-shot encoder cosine (all-MiniLM-L6-v2) | 0.828 | – |
| Logistic regression (with semantic and numeric/negation features) | 0.850 | 0.830 |
| LightGBM (pointwise and LambdaRank) | 0.854 | 0.836 |

The random and BM25 rows almost match ShopRank's, so the setups are comparable. Their extra **semantic and negation features** explain most of the gap: 0.854 against ShopRank's 0.836.

## 4. Newer research (2024-2026, arXiv)

- Several LLM-based rankers report ESCI gains, for example LRanker (an 8.9% relative MRR gain) and SimRec (a Qwen3-4B agent), but on **different setups and metrics**, so they can't be compared with Task 1 nDCG.
- One paper reports nDCG@10 of 0.938 on an **augmented subset** with LLM-generated annotations. That is not comparable either.
- **No 2024-2026 paper found beats 0.9043 on the standard Task 1 setup.** Fine-tuned cross-encoders remain the standard for this task.

## 5. Models that fit an RTX 4060 (8 GB)

Sizes are from the Hugging Face API, checked 17 Nov 2026. Training times are **estimates, not measured**.

| Model | Size | Use | Fits 8 GB? |
|---|---|---|---|
| all-MiniLM-L6-v2 | 23M | Semantic feature for LambdaMART | Yes, easily |
| bge-small-en-v1.5 | 33M | Semantic feature (same model as Jruva) | Yes, easily |
| bge-reranker-base | 278M | Zero-shot reranker score as a feature | Yes |
| bge-reranker-v2-m3 | 568M | Stronger zero-shot reranker | Inference yes; fine-tuning tight |
| Qwen3-Reranker-0.6B | 596M | Newer zero-shot reranker | Inference yes; fine-tuning tight |
| **DeBERTa-v3-base** | about 184M | **Fine-tuned cross-encoder (main model)** | **Yes**: fp16, max_length 128, batch about 32 |
| DeBERTa-v3-large | about 435M | Stretch | Only with fp16, gradient checkpointing, a small batch and accumulation; slow |

## 6. Recommendation (ranked)

| Step | What | Expected US nDCG (estimate) | Why |
|---|---|---|---|
| 1. Quick win (1-2 days) | Add semantic cosine (bge-small), negation and numeric features, and a zero-shot reranker score to LambdaMART | about 0.85 | The same-setup project reached 0.854 with semantic features |
| 2. **Main model (1-2 weeks)** | **Fine-tune DeBERTa-v3-base cross-encoder**: 4-class cross-entropy, expected-gain ranking, inputs query + title + brand + colour, max_length 128 | **about 0.87-0.89** | The winner's single cross-encoder scored 0.893 (multilingual, tuned) |
| 3. Stack (2-3 days) | Feed the cross-encoder score into LambdaMART with the step 1 features | +0.003-0.01 | The second-place team's design (cross-encoders into LightGBM) |
| 4. Stretch | DeBERTa-v3-large and/or a 2-3 model ensemble | about 0.89-0.90 | The winner's large model added +0.003-0.004 |

**Best single choice: a fine-tuned DeBERTa-v3-base cross-encoder, stacked with LambdaMART.** It is the best accuracy you can reach on this hardware, it is backed by the winners' published results, and it can be explained in an interview.

**Honest limits:**
- Every expected number is an estimate from published results on slightly different setups. ShopRank's real number comes only from running it.
- Beating Amazon's 0.857 is realistic; 0.90 would be top-tier. Anything near 1.0 is impossible because of the 91% label agreement.

## Sources

- ESCI paper: https://arxiv.org/abs/2206.06588 (label agreement and baselines)
- Winner paper: https://arxiv.org/abs/2208.02958 (Table 2 ablation, models, settings)
- KDD Cup 2022 papers and code: https://amazonkddcup.github.io/ ; https://github.com/wufanyou/KDD-Cup-2022-Amazon
- Same-setup project: https://github.com/Anankin77/esci-search-reranking
- arXiv search of 2024-2026 ESCI papers (export.arxiv.org API, 17 Nov 2026)
- Model sizes: Hugging Face API (huggingface.co/api/models/...)

## Primary-source digest, 20 Nov 2026 (research agent, every item read from the source)

**ESCI paper** (https://arxiv.org/pdf/2206.06588):
- Labels: >=3 annotators, majority vote.
- Agreement: author audit of 100 pairs per language gave 91%. 50% of disagreements are I judged as S; 15% are extreme (E vs I). Binary E vs not-E agreement is >96%.
- Baselines (public test): fine-tuned MS MARCO cross-encoder 0.857 (US) / 0.852 (all); BM25 (Terrier, all locales) 0.675 (US).

**Official scoring** (https://github.com/amazon-science/esci-data/blob/main/ranking/launch-predictions-task1.sh):
- `trec_eval -c -J -m 'ndcg.1=0,2=0.01,3=0.1,4=1'`: full-list nDCG, no cutoff.
- qrels come from `small_version==1 & split==test` (prepare_trec_eval_files.py).

**Amazon baseline** (ranking/train.py): MiniLM-L12 MS MARCO, title only, max_length 512, 1 epoch, lr 7e-6, 5,000 warmup steps, MSE on gain, batch 32.

**KDD Cup 2022 Task 1** (multi-locale; none of the papers report a US-only score):

1st, www (https://arxiv.org/pdf/2208.02958):
- Models: mDeBERTa, DeBERTa-v3-large (EN), InfoXLM, XLM-R, RemBERT.
- Input: `query [SEP] color: brand: description: <title+bullets+desc>`, max_len 128, lr 3e-5.
- Loss: 4-class CE with label smoothing, ranked by expected gain.
- Tricks: self-distillation (3-fold OOF soft labels, 0.7 hard / 0.3 soft), pseudo-labels, AWP.
- Ensemble: weighted average.
- Ablation (public test): base 0.8930 -> clean+tune 0.8960 -> +self-distil/pseudo 0.8982 -> +label smoothing 0.8990 -> +DeBERTa-large+AWP 0.9022 -> ensemble 0.9057.

2nd, qinpersevere (https://amazonkddcup.github.io/papers/8408.pdf):
- Models: DeBERTa-v3-large and RoBERTa-large (US).
- Input: title + brand + colour, max_len 400.
- Loss: MSE on gain (beat pairwise, converged faster).
- Training: lr 7e-6 (large), 5-fold CV, extra Task 2/3 data, domain-adaptive pretraining.
- Ablation: 1 fold 0.8946 -> 5 folds 0.9003 -> ensemble 0.9047.

3rd, day-day-up (https://amazonkddcup.github.io/papers/3782.pdf):
- Model: InfoXLM-large trained on Task 1+2+3 data; score is expected gain.
- Gains: query-level keyword context (>0.01 on Task 2, smaller on Task 1); 10-fold self-distillation (+0.003, Task 2).
- Ensemble: 8 models. Notes a fixed leak of Task 2/3 test samples into Task 1 train.

6th, ETS-Lab (https://amazonkddcup.github.io/papers/8572.pdf):
- Pipeline: cross-encoder class probabilities -> LightGBM.
- Gains (Task 2 only): per-query min/median/max of probabilities +0.008; other query/brand features +0.001 to +0.005.

Public US project, Anankin77 (https://github.com/Anankin77/esci-search-reranking):
- Method: LightGBM on BM25, attribute, numeric, negation features and MiniLM-L6 cosine.
- Result: 0.854 [0.851, 0.857] on US small test (8,956 queries).

**Consequences for ShopRank (single 8 GB GPU, US only):**
1. Keep MSE.
2. Bullets ablation at max_len 128 (1st place's format).
3. Section 14 stacking adds per-query score statistics.
4. OOF folds can double as self-distillation targets.
5. Extra large_version data only after a query-overlap check.
6. DeBERTa-large is the costliest item; decide after the seed spread is known.
