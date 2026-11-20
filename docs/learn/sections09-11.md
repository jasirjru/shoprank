# ShopRank lessons: sections 9-11 (taught 20 Nov 2026)

Every number here comes from real runs on this project's data (dev = 2,001 queries, dev2 = 1,053).
The check-question answers are at the bottom. Answer first, then look.

---

# Section 9: Rigour fixes ("Is my exam fair?")

The whole section is one idea. Each fix closes one way the evaluation could cheat:

| Fix | The cheat it stops |
|---|---|
| 1. Split by query text | The model saw the exam question in homework, under a different ID number |
| 2. Graded logistic regression | Two models compared, but one got more information (an unfair race) |
| 3. Early-stopping metric | Practising for one exam, then being graded on a different one |
| 4. dev2 | Trying 17 setups and keeping the luckiest, so it needs checking on a fresh set |

## Part 1: Split by query text

**Picture:** a student sees "What is 5+5?" in homework, and the exam has "what is 5 + 5 ?". The exam
then measures memory, not skill.

**Real pairs in the OLD split (same search, different IDs, different splits):**

| Train had | Test had |
|---|---|
| `2.5 hdd` (id 2743) | `2.5" hdd` (id 2750) |
| `1 1/4 sink drain without overflow` (id 633) | `1 1/4” sink drain without overflow` (id 638) |
| `'freeze dryer' machine not dehydrator` (id 132) | `freeze dryer machine not dehydrator` (id 42526) |

**Fix (`src/shoprank/data/ingest.py`):**
1. Normalise the text: lowercase, turn runs of non letter/digit characters into one space, trim.
   `2.5" hdd` becomes `2 5 hdd`.
2. Hash the normalised text (not the ID) with MD5 into 20 buckets. Buckets 0-1 are dev, bucket 2 is
   dev2, buckets 3-19 are train.
3. If a train text also appears in test, drop the train copy (11 queries). The official test set
   can't move, because it is Amazon's benchmark.

**Counts:** Task 1 had 6 texts crossing splits and 12 texts with more than one query ID.

**Two bugs found and fixed:**
- **Lowercase must come before the regex.** Otherwise `[^a-z0-9]` deletes the capital letters.
  This gave a wrong early count of "103".
- **`[a-z0-9]` turned every Korean, Chinese or Arabic query into `''`.**
  - `가마솥` and `香奈儿` then counted as the same query. One of them was in test, so 4 unrelated
    train queries were wrongly dropped.
  - Fix: keep letters of any script with `[^\p{L}\p{N}]+`, and add a test.
  - Lesson: a cleaning rule can create false duplicates. Always look at real examples, not only counts.

```
raw query ─► lowercase ─► regex clean ─► MD5 % 20
                                          ├─ 0-1  → dev
                                          ├─ 2    → dev2
                                          └─ 3-19 → train
                   (text also in test? → dropped)
```

**Interview answer:** "I found near-duplicate queries, like '2.5 hdd' and '2.5\" hdd', with different IDs
in different splits. So I split by normalised query text: twins always share a split, and training
copies of test queries are dropped. I also caught my own normaliser merging all non-Latin queries into
an empty string, fixed it to keep letters of any script, and added a test."

**Check 9.1:** Why hash the normalised text instead of the query ID?
A. Text hashes are shorter.
B. So twin queries with different IDs always land in the same split.
C. IDs change on every download.
D. To make dev larger.

## Part 2: Graded logistic regression (a fair race)

**Picture:** student A got right/wrong notes, student B got 4-grade notes. To know whether B is smarter,
give A the same notes and race again.

**Problem:** the old LR learned only "Exact or not". LambdaMART learned all 4 labels.
**Fix:** a multinomial LR on 4 labels, ranked by expected gain:
`score = P(E)·1 + P(S)·0.1 + P(C)·0.01 + P(I)·0`

**Real example:** query `call of duty black ops. 2`

| Product | Label | Binary P(Exact) | Graded P(E) | Graded P(S) | Expected gain |
|---|---|---|---|---|---|
| Black Ops II PS3 (map pack) | E | 0.591 | 0.575 | 0.357 | 0.611 |
| Black Ops II Xbox 360 | E | 0.594 | 0.562 | 0.357 | 0.598 |
| Turtle Beach **headset** | **I** | 0.494 | 0.479 | 0.404 | 0.520 |
| Black Ops **III** PS4 | S | 0.507 | 0.473 | 0.430 | 0.516 |
| Black Ops II PS3 | E | 0.478 | 0.450 | 0.444 | 0.494 |

**Results:**
- Graded LR scored 0.8254 and binary LR 0.8256: a difference of −0.0002 [−0.0012, +0.0009].
- The two scores have a 0.9945 correlation across dev, so the ranking is almost identical.
- Both models put the **headset** above real games, because its title contains all the query words.

**Conclusion:**
- The extra labels don't change the linear model.
- LambdaMART's lead, +0.0088 [+0.0055, +0.0121], comes from **model type**:
  - trees learn combinations ("words match BUT the title adds 'headset'");
  - a listwise loss optimises the whole ranked list.
- **The prediction "graded LR closes the gap" was wrong.** That is recorded in the notebook.

```
same features, same 4 labels
   ├─ linear LR (adds features)        0.8254
   └─ LambdaMART (learns combinations) 0.8342  → gap = model type
```

**Interview answer:** "LambdaMART used graded labels but my logistic regression used only
Exact-or-not, so the comparison was unfair. I trained a multinomial logistic regression on all 4
labels, ranked by expected gain. It scored the same; its scores had a 0.99 correlation with the
binary model. So the gap comes from non-linear trees and a listwise loss, not label information."

**Check 9.2:** Graded LR got 4 labels but the same nDCG. What is the best conclusion?
A. The S and C labels are wrong.
B. LR needs more epochs.
C. With these features, label detail doesn't change the linear ranking, so LambdaMART's lead comes from non-linear combinations.
D. Expected gain is the wrong formula.

## Part 3: Early stopping on the reported metric

**Picture:** a coach judges your cricket practice on the first 10 balls only, but the match is the
whole over.

- LambdaMART adds trees one at a time.
- After each tree it scores 1,732 held-out *train* queries.
- It stops after 100 trees with no improvement (the "patience").

**Problem and fix:**
- Old: it stopped on nDCG@10 but reported full-list nDCG.
- New: `eval_at = longest list (136)`, so it stops on full-list nDCG.

**Real test (same data, same seed):**

| Stopping metric | Trees | dev nDCG |
|---|---|---|
| nDCG@10 | 259 | 0.8342 |
| Full list (136) | 259 | 0.8342 |

**No effect here.** The fix is still correct, and now it's proven rather than assumed: a null result,
recorded honestly. (The earlier claim "274 → 231 trees because of this fix" was wrong. That change came
from the new split.)

```
score ▲      best (259)
      │   ╭──●────────── flat for 100 trees → STOP
      │ ╭─╯
      └─┴──────────────► trees
```

**Interview answer:** "LambdaMART stopped on nDCG@10 while I reported full-list nDCG. I set eval_at to
the longest list so both match. Then I tested both settings: both picked 259 trees with identical
scores. A correct fix with no measured effect, and I recorded that."

**Check 9.3:** Why does the stopping metric matter?
A. Early stopping is only for neural networks.
B. Different metrics can peak at different tree counts, so nDCG@10 could pick a model that isn't best for the full-list nDCG you report.
C. nDCG@10 always overfits.
D. Full list trains faster.

## Part 4: dev2, the honesty check

**Picture:** try 17 shirts in one mirror and pick the best. Then ask one friend who didn't watch.
(Mirror = dev, friend = dev2.)

About 17 setups were compared on dev (7 BM25, 2 LR, 3 LambdaMART, 5 two-tower). The winner of many
tries is slightly lucky: that's **selection bias**.

| Model | dev | dev2 (scored once) |
|---|---|---|
| BM25 | 0.8163 | 0.8191 |
| Graded LR | 0.8254 | 0.8260 |
| LambdaMART | 0.8342 | 0.8364 |
| **LambdaMART − BM25** | **+0.0180** [+0.0140, +0.0219] | **+0.0173** [+0.0113, +0.0234] |

**How to read it:**
1. **Compare gaps, not raw scores** across sets. All dev2 scores are higher, so its queries are
   slightly easier.
2. **The gap held,** and the model order is the same, so the choice is real.
3. **The dev2 CI is wider,** because it has about half the queries.

| Set | Use | How often |
|---|---|---|
| dev | try many things, pick | many times |
| dev2 | confirm the pick | once |
| test | final number for the README and resume | once, at the very end |

**Interview answer:** "I compared about 17 setups on dev, so the winner carries selection bias. I held
out dev2 and scored it once: LambdaMART's lead over BM25 was +0.018 on dev and +0.017 on dev2, both
CIs above zero, so the choice holds. Test stays locked until the final run."

**Check 9.4:** dev2 scores are higher than dev for ALL models. What's the correct reading?
A. The models improved after dev.
B. dev2's queries are slightly easier, so compare gaps between models, not raw scores across sets.
C. dev2 has leakage.
D. Throw away dev.

### Section 9 full quiz (interview style)
1. Why change the split from query_id to normalised text? A) hashing text is faster B) near-duplicate queries with different IDs could sit in train and dev, inflating dev C) IDs not unique D) bigger dev
2. Why drop train copies of test texts instead of moving them to test? A) wrong labels B) test too large C) train must be smaller D) the official test split is fixed; changing it breaks comparison with published results
3. Graded LR = binary LR. What does it prove? A) the gap is model type, not the S/C labels B) S/C labels are noisy C) LR undertrained D) LambdaMART overfits dev
4. Why stop on full-list nDCG, not @10? A) @10 slower B) LightGBM lacks @10 C) the stopping metric should match the reported metric D) full list gives higher numbers
5. What is dev2 for? A) extra training data B) tuning BM25 C) replacing test D) checking selection bias after ~17 setups were compared on dev

---

# Section 10: Transformers

## Part 1: Tokens and embeddings

**Why:** word-match models ranked the **headset** above real games. They count shared words but can't
understand that "headset" makes the product an accessory.

**Picture:** LEGO. Any word is built from a fixed box of 30,522 bricks (tokens). Each brick has a
meaning card of 384 numbers (its embedding).

**Real MiniLM tokenizer output:**

| Text | Tokens |
|---|---|
| `call of duty black ops. 2` | `call` `of` `duty` `black` `ops` `.` `2` |
| `...Black Ops II KILO ... Headset` | ... `ops` `ii` `ki` `##lo` ... `heads` `##et` |
| `dehydrator` | `de` `##hy` `##dra` `##tor` |

- `##` means "glued to the previous piece". Rare words become known pieces, so there are no
  unknown words.
- `2` and `ii` are different tokens: the model must *learn* that 2 = II. BM25 never can.
- Embeddings are learned in **pretraining** on huge text. The from-scratch two-tower started from
  random numbers, which is part of why it scored only 0.785.

```
text ─► tokenizer ─► token IDs ─► embedding table ─► 384 numbers per token
```

**Interview answer:** "A transformer splits text into sub-word tokens, so rare words become known
pieces. Each token maps to a pretrained embedding that carries meaning learned from huge text. That's
how a pretrained model can learn that 'headset' means an accessory, while BM25 only counts words."

**Check 10.1:** Why did the from-scratch two-tower (0.785) lose to BM25, while a pretrained transformer is expected to beat both?
A. Transformers use more GPU memory.
B. The two-tower started from random embeddings with no language knowledge; pretrained embeddings already encode meaning.
C. BM25 uses embeddings.
D. Wrong vocabulary size.

## Part 2: Attention

**Picture:** a meeting of 27 people (tokens). Each glances at the others, decides whose words matter to
it, and updates its own understanding. This repeats 12 times (12 layers).

**Real example:** MiniLM on the query plus the headset title (27 tokens, 12 layers, 12 heads).
- In layer 1, `heads` attends to `gaming` 0.15, `[CLS]` 0.09, `##et` 0.07 and `edition` 0.05.
  It builds "gaming headset" from its neighbours.
- In the last layer, `[CLS]` attends to `turtle` 0.06 and `beach` 0.06: the extra brand words.

**Zero-shot scores (no ESCI training):**

| Product | Label | Score |
|---|---|---|
| Black Ops II PS3 | E | +7.45 |
| Black Ops III PS4 | S | +6.94 |
| Turtle Beach headset | I | +4.13 (last, correct) |

**Formula:** each token makes Q ("what am I looking for"), K ("what do I contain") and
V ("what do I pass on"). `weight(i→j) = softmax(Qᵢ·Kⱼ / √d)`, and the new token i = Σ weight × Vⱼ.
√d keeps the values stable, and softmax makes the weights sum to 1.

**Caveat:** attention weights are a hint about what the model uses, not proof.

**Interview answer:** "Attention lets each token build meaning from context. Each token makes query,
key and value vectors; the weights are softmax of query·key over √d, and the output is a weighted sum
of the values. In my data, 'heads' attends to 'gaming' and '##et', and zero-shot MiniLM ranked that
headset below the real games, which my word-match models couldn't do."

**Check 10.2:** Why did MiniLM rank the headset last while word-match features ranked it high?
A. MiniLM ignores titles.
B. Attention mixes context into each token, so "headset" is understood as an accessory; word-match features only count shared words.
C. MiniLM was trained on ESCI.
D. The title is too long for BM25.

**Image prompt (ChatGPT):** see the chat of 20 Nov. Panels: token chips → `heads` attention arrows
(gaming 0.15, ##et 0.07) → 12 layers → scoreboard +7.45 / +6.94 / +4.13.

## Part 3: Bi-encoder vs cross-encoder

**Picture:**
- **Bi-encoder = a dating app:** profiles are written once and compared fast, but they are summaries.
- **Cross-encoder = an interview:** the question and answer are read together. Accurate but slow.

```
BI:    query ─► model ─► vector ┐                 CROSS: [CLS] query [SEP] product [SEP]
       product ─► model ─► vector ┴─ cosine                       │ attention across both
       (product vectors stored in advance)                       one score
```

**Real comparison** (two different MiniLM models: an illustration, not a controlled test):

| Product | Label | Bi cosine (all-MiniLM-L6-v2) | Cross (ms-marco-MiniLM-L-12-v2) |
|---|---|---|---|
| Black Ops II PS3 | E | 0.832 | **+7.45 (1st, correct)** |
| Black Ops III PS4 | S | **0.868 (1st, wrong)** | +6.94 |
| Headset | I | 0.524 | +4.13 |

**Real speed (RTX 4060):**
- Bi: encode 2,000 products in 0.50 s (done once and stored).
- Bi: one query vs 1,000,000 stored vectors in 0.046 s (CPU).
- Cross: 2,000 pairs in 0.90 s, so 1,000,000 pairs would take about 450 s (estimated).

**Industry pattern:**
`1M products ─► BM25 / bi-encoder (fast) ─► top 100 ─► cross-encoder (precise) ─► top 10`
- Jruva = retrieval (BGE-small + BM25 hybrid).
- ShopRank = reranking (about 20 candidates per query).

**Interview answer:** "A bi-encoder encodes query and product separately, so product vectors can be
precomputed and searched over millions in milliseconds, but one vector loses detail: it ranked Black
Ops III above II in my data. A cross-encoder reads both together with full attention, which is more
accurate but costs one model pass per pair. So production retrieves with BM25 or a bi-encoder and
reranks the top candidates with a cross-encoder. Jruva does retrieval; ShopRank does reranking."

**Check 10.3:** Task 1 gives about 20 candidates per query. Why is a cross-encoder right here, but not for a full catalogue?
A. Cross-encoders can't handle more than 20 products.
B. Reranking 20 pairs is cheap, so full attention is affordable; millions of products need precomputed vectors.
C. Bi-encoders don't work on titles.
D. Cross-encoders need no GPU.

---

# Section 11: MiniLM zero-shot

**Picture:** a cricket expert judges a football match fairly well without football training.
That is zero-shot transfer.

**Run:**
- Model: `cross-encoder/ms-marco-MiniLM-L-12-v2`, trained on MS MARCO web search and never on ESCI.
- Input: `[CLS] query [SEP] title [SEP]`.
- Data: 2,001 dev queries, 40,304 pairs.
- Code: `src/shoprank/cross/zero_shot.py`.

| Model | Trained on ESCI? | dev nDCG |
|---|---|---|
| BM25 | no | 0.8163 |
| Logistic regression | 357k pairs | 0.8256 |
| **MiniLM zero-shot** | **0 pairs** | **0.8332** |
| LambdaMART | 357k pairs | 0.8342 |

**Results:**
- MiniLM vs BM25: +0.0169 [+0.0124, +0.0214].
- LambdaMART vs MiniLM: +0.0011 [−0.0037, +0.0058]: a **tie**.
- Adding the brand field: +0.0003, no gain.
- Speed: 2,534 pairs/s.
- **Complementary errors:** MiniLM is better on 897 queries, LambdaMART on 1,035, ties on 69.
  The per-query correlation is 0.734, so a blend should help (section 14).
- **The prediction "below LambdaMART" was half wrong:** it tied.

**Interview answer:** "With zero ESCI labels, a MiniLM cross-encoder trained on MS MARCO scored 0.833,
a statistical tie with my LambdaMART trained on 357,000 labelled pairs. The two won on different queries
(per-query correlation 0.73), which told me fine-tuning would help and that blending them should
help too."

**Check 11:** Zero-shot MiniLM ties LambdaMART, but the per-query correlation is only 0.73. What does that suggest?
A. A bug.
B. They make different mistakes, so combining them should beat either alone.
C. Dev is too small.
D. Replace LambdaMART now.

---

## Answer key (don't peek)

| Question | Answer |
|---|---|
| 9.1 | B |
| 9.2 | C |
| 9.3 | B |
| 9.4 | B |
| Section 9 quiz | 1 B, 2 D, 3 A, 4 C, 5 D |
| 10.1 | B |
| 10.2 | B |
| 10.3 | B |
| 11 | B (answered correctly on 20 Nov) |
