# ShopRank lessons: section 14, combining models (built 21 Nov 2026, study after the build)

Every number comes from real runs on this project's data: dev = 2,001 queries, dev2 = 1,053 queries.
Answer the check questions first, then look at the key at the bottom.

| System | dev nDCG | dev2 nDCG |
|---|---|---|
| LambdaMART (14 text features) | 0.8342 | 0.8364 |
| One DeBERTa + bullets | 0.8851 | |
| **Mean of 5 DeBERTa fold models** | **0.8879** | 0.8812 |
| z-score blend (DeBERTa, MiniLM, BM25) | 0.8876 | 0.8826 |
| LightGBM stack | 0.8879 | 0.8845 |

---

## Part 1: Out-of-fold (OOF) scores: honest scores for the training set

**Picture:** a teacher wants to know how well a student really understands. Asking questions the student
already saw the answers to tells nothing. So the class is split in 5 groups; each group's exam is marked by a
tutor who never saw that group's answers.

```
TRAIN queries split by hash into 5 folds:   [F0][F1][F2][F3][F4]
model 0 trains on F1-F4, scores F0     ->   honest scores for F0
model 1 trains on F0,F2-F4, scores F1  ->   honest scores for F1
...                                          every TRAIN pair gets exactly one honest score
dev / dev2: scored by all 5 models, then averaged
```

**Real numbers:** DeBERTa scores its own training pairs almost perfectly (it memorised them), but its OOF
nDCG on TRAIN is 0.8828, close to its dev nDCG. That closeness is the proof the scores are honest.
Each fold: ~14,250 queries / 286,000 pairs to train, ~3,560 queries / 71,000 pairs scored out-of-fold.

| Word | Meaning |
|---|---|
| fold | one of the 5 groups of queries |
| out-of-fold score | a score from a model that never trained on that query |
| leakage | the label sneaking into an input; here it would make the stacker trust the score too much |

**Interview answer:** "To stack models I need their scores on the training set, but a model's score on its
own training data is overconfident. I trained 5 models, each on 4 of 5 query folds, and scored only the
fold it had not seen. Folds are split by query so no query is in both halves, and I checked the OOF nDCG
(0.883) matches dev (0.885), so the scores behave like real held-out scores."

## Part 2: Ensembles: why 5 models beat 1

**Picture:** ask 5 doctors who studied from slightly different books and average their opinions. Each makes
small random mistakes; the mistakes partly cancel.

**Real numbers:** the 5 DeBERTa fold models score 0.8835-0.8858 on dev (mean 0.8847, each saw 80% of the
data). Their average scores 0.8879: +0.0028 [+0.0004, +0.0051] over the single model trained on 100%.
MiniLM gains almost nothing from averaging: its 5 models are too alike (seed spread 0.0003).

```
5 x 80% data, averaged   0.8879   <- best dev
1 x 100% data            0.8851
one 80% model (mean)     0.8847
```

**Interview answer:** "Averaging 5 cross-validation models gave +0.003 nDCG over one model trained on all the
data, with a confidence interval above zero. Ensembles help when the members make different mistakes, so it
helped DeBERTa more than MiniLM, whose runs were nearly identical."

## Part 3: z-score blend and LightGBM stacking

**Picture:** a blend is a fixed recipe (80% of A, 10% of B, 10% of C). A stacker is a chef who tastes the
ingredients and decides per dish.

- **z-score:** within one query, (score - query mean) / query std. It puts every model on the same scale.
- **Blend weights** were chosen by grid search on TRAIN OOF scores: 0.8 DeBERTa, 0.1 MiniLM, 0.1 BM25-title.
- **Stack:** LambdaMART on 14 text features + both scores + per-query z, percentile, gap to best, max,
  median, min. 81% of its learned importance went to one feature: DeBERTa's query z-score.

| System | dev vs mean-of-5 DeBERTa | dev2 vs mean-of-5 DeBERTa |
|---|---|---|
| blend | -0.0003 [-0.0019, +0.0013] | +0.0014 [-0.0006, +0.0034] |
| stack | +0.0000 [-0.0018, +0.0019] | +0.0033 [+0.0008, +0.0058] |
| stack, dev + dev2 pooled | +0.0012 [-0.0003, +0.0027] | |

**Why so little?** The KDD Cup 2022 winners stacked 5-10 strong and *different* cross-encoders. We have one
strong model and one weaker model of the same kind; the text features say things DeBERTa already reads.

**Interview answer:** "I fitted blend weights and a LightGBM stacker on out-of-fold scores only. Stacking gave
no gain on dev and +0.003 on a second held-out split; pooled it is +0.001 with an interval touching zero. The
stacker put 81% of its weight on the DeBERTa score, so with only one strong model there was little to
combine. Ensembling five DeBERTa models was the gain that held up."

## Part 4: A second held-out split, and selection bias

**Picture:** you try 10 shirts in one shop mirror and pick the one that looks best. In a different mirror it
looks a bit less good: part of "best" was the mirror. dev is the first mirror; dev2 is the second.

**Real numbers:** from dev to dev2, LambdaMART goes UP 0.0022, mean-of-5 DeBERTa goes DOWN 0.0067. DeBERTa's
settings (learning rate, bullets) were chosen on dev, so its dev score is a little optimistic. dev2 was
looked at once; nothing is tuned after that. The official test set gives the final, unbiased number.

**Interview answer:** "Every choice I made on dev makes dev slightly optimistic, so I kept a second held-out
split and looked at it once at the end. DeBERTa lost about 0.007 there while LambdaMART did not, which is the
size of the selection bias I should expect; the test set is scored once, only after all choices are frozen."

## Part 5: Distillation: a small model learns from a big one

**Picture:** a junior chef learns faster by copying a master chef's tasting notes ("this one is almost right")
than from the restaurant's pass/fail stamps alone. The master's scores carry more detail than 4 grades.

**Real numbers (MiniLM, 33M parameters; teacher = DeBERTa out-of-fold scores):**

| MiniLM run | dev nDCG | gain |
|---|---|---|
| title, human labels | 0.8534 | |
| + bullets | 0.8602 | +0.0068 [+0.0032, +0.0104] |
| + learn DeBERTa's scores instead of the labels | **0.8667** | +0.0064 [+0.0038, +0.0090] |
| the teacher (5 DeBERTa, 184M each) | 0.8879 | |

```
human label:     E=1.0   S=0.1   C=0.01   I=0     (4 values)
teacher score:   0.93    0.41    0.12     0.02    (illustrative: a smooth scale, "how Exact is it?")
```
The teacher's scores are out-of-fold: on a training query the teacher never saw it, so it cannot simply
hand back the memorised label.

**Interview answer:** "I distilled the DeBERTa ensemble into MiniLM by regressing its out-of-fold scores.
With a control run I split the student's +0.013 into +0.007 from giving it the bullet points and +0.006
from the teacher's soft scores, both with intervals above zero. The student is 5x smaller and keeps about
40% of the teacher's lead over plain MiniLM."

---

## Check questions

1. (Part 1) Why can't the stacker use DeBERTa's scores on the pairs it trained on? A) they are too slow
   B) the model memorised those labels, so the scores are over-confident C) they are in fp16 D) LightGBM
   cannot read them
2. (Part 1) Folds are split by query, not by pair, because: A) it is faster B) otherwise the same query
   is in training and scoring, leaking its other labels C) LightGBM needs it D) pairs have no id
3. (Part 2) Why did averaging help DeBERTa more than MiniLM? A) DeBERTa is larger B) DeBERTa's 5 models
   made more different mistakes; MiniLM's runs were nearly identical C) MiniLM had no bullets D) chance
4. (Part 3) The stack is +0.0000 on dev and +0.0033 on dev2. The best single statement is: A) the stack is
   clearly better B) the stack is useless C) pooled it is about +0.001 with an interval touching 0: at most a
   small gain D) dev2 is wrong
5. (Part 4) DeBERTa fell 0.0067 from dev to dev2 while LambdaMART rose. Most likely reason: A) dev2 has a bug
   B) DeBERTa's settings were chosen on dev, so dev flatters it C) DeBERTa overheated D) dev2 is in Spanish
6. (Part 5) Why was the control run `minilm_ft_bullets` needed? A) to warm up the GPU B) the student run
   changed two things (bullets and teacher); the control separates them C) to get a seed spread D) dev2

## Answer key

1. B. 2. B. 3. B. 4. C. 5. B. 6. B.
