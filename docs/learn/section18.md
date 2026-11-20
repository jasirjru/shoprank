# ShopRank lessons: section 18, robustness, a failed fix, spell correction and shipping (built 21-22 Nov 2026, study after the build)

Files: `src/shoprank/cross/typos.py`, `src/shoprank/serve/spell.py`, `src/shoprank/serve/fetch.py`, `Dockerfile`, `CHANGELOG.md`.
Notebook entries: "Audit: robustness", "Phase A", "Phase B", "Phase D", "Phase D2".
Answer the check questions first; the key is at the bottom.

---

## Part 1: Behavioural tests, poke the model one way at a time

**Picture:** a car crash test. You do not only read the engine's power figure. You hit the car from one side,
then from the other, and see which part breaks.

**What we did (the "CheckList" idea):** a normal test score is one number on normal data. A behavioural test
changes the input in ONE controlled way, keeps the labels the same, and measures how much the score moves.
We did this on the served INT8 model, on all 2,001 dev queries, and wrote our predictions down first.

```
 same 2,001 queries  --change one thing-->  re-score  -->  compare nDCG with the untouched run
                                            (labels never change)
```

**Real numbers (dev nDCG, untouched = 0.8654):**

| Change to the input | nDCG change |
|---|---|
| UPPER CASE query | 0.0000 |
| one typo in the longest query word | -0.0262 |
| query words in reverse order | -0.0117 |
| product shown as title only | -0.0097 |
| negation word deleted | -0.0073 (slice: -0.0762) |

- Upper case costs nothing, because the tokenizer lower-cases the text. That is a check that the harness works.
- A typo cost about 0.026 to 0.028 (0.0283 for swapped letters, 0.0257 for neighbour-key typos in Phase D). That
  is bigger than distillation (+0.0064) and bullets (+0.0068) gained together. It is the model's biggest
  measured weakness. I predicted -0.010 to -0.025, so my prediction was too kind.
- Negation: I predicted the model ignores "without / not / no". I was wrong. Deleting the word reorders the
  list (median within-query Spearman 0.50) and costs 0.076 on the negation slice. The model DOES read negation.
  The slice is weak (0.8347 against 0.8654) mostly because ESCI built that query source to be hard, not because
  the model is blind to the word.

| Word | Meaning |
|---|---|
| behavioural test | change the input in one controlled way and watch the output |
| perturbation | the small controlled change you make to the input |
| slice | a group of queries that share a trait, such as negation queries |
| Spearman | a number from -1 to 1 that says how similar two orderings are |
| prediction written first | you commit to a guess before the run, so you cannot fool yourself afterwards |

**Interview answer:** "A single test score hides weak spots, so I ran CheckList-style behavioural tests on the
served model: I changed one thing at a time and kept the labels. Upper case cost nothing, reversed words cost
0.012, title-only products 0.010, and one typo cost about 0.026. I also found my own prediction was wrong: the
model does read negation, and the weak negation slice comes mostly from how that query source was built."

## Part 2: A failed fix, and why it is still useful

**Picture:** a student is told to practise exam questions with deliberate spelling mistakes. After a week the
marks go up a little, but not much. The test was fair, so the result is still worth knowing.

**What we did (Phase D, typo-consistency distillation):** in training, 30% of queries got one synthetic typo
(swap, delete, double a letter, or hit a neighbouring key). The target stayed the teacher's score for the
CLEAN query. The lesson for the student: a typo query should score like the clean one.

**Real numbers (dev, mean of 3 seeds):**

| | baseline | typo-trained | change [95% CI] |
|---|---|---|---|
| clean | 0.8657 | 0.8642 | -0.0015 [-0.0028, -0.0003] |
| typo (swapped letters) | 0.8374 | 0.8415 | +0.0041 [+0.0019, +0.0062] |

The typo gap was 0.0283, so +0.0041 recovers only 14% of the loss. The rule written BEFORE the run said: win
back at least 50%. It failed. So the model was not changed, and dev2 (the second check set, 1,053 queries) was
NOT used. It stays unspent for a better candidate.

```
 rule fixed before the run:   recover >= 50% of the typo loss
 result:                      14%   ->  rejected, dev2 untouched, served model stays v1.0
```

**Why it probably failed:** a typo splits the word into different sub-word pieces ("cbale" is not "cable").
The model is small (33M parameters) and saw each misspelled query only once, so it could not learn to undo
that. Part 3 shows a way that needs no learning at all.

**Why it is still useful:** a pre-registered rule stops us from calling a small gain a win. Keeping dev2
unspent keeps one honest check available. And the result tells us where NOT to spend more GPU time.

| Word | Meaning |
|---|---|
| distillation | a small student model learns from a bigger teacher model's scores |
| consistency training | train so that a changed input gives the same answer as the clean one |
| pre-registered rule | a pass/fail rule written down before the experiment |
| negative result | an honest "this did not work", recorded rather than hidden |
| unspent dev2 | a check set kept unused so it can still give an unbiased answer later |

**Interview answer:** "I tried to make the ranker robust to typos by training on misspelled queries with the
clean query's teacher score. It recovered only 14% of the typo loss, against a 50% rule I set beforehand, and
cost 0.0015 on clean queries, so I rejected it and left my second validation set unspent. I kept the result in
the notebook, because it told me that a small model cannot undo a sub-word split."

## Part 3: Fix the input, not the model (spell correction before ranking)

**Picture:** a post office that reads a badly written address and corrects "Lodnon" to "London" before the
letter goes on the van. The van driver does not need to become a handwriting expert.

**What we built (`serve/spell.py`):**
1. Vocabulary: count every word in the titles, brands and colours of the 1.2M US products. That gives 184,594
   words. Product text only: no labels, no queries.
2. A query word is a candidate only if it is letters only and 4+ characters. Digits and symbols ("usb-c",
   "512gb") are never touched.
3. If the word was seen fewer than 5 times in the catalogue (the "seen 5+ times = known word" rule), look for words exactly
   one edit away (Damerau-Levenshtein: insert, delete, swap two neighbours, or replace one letter).
4. Replace it by the most frequent such word, only if that word is at least 10 times more frequent (the "10x
   frequency" rule).
5. SymSpell idea: store words with one letter deleted in an index, so the search is a lookup, not a scan of
   184,594 words. That is why it takes about 0.01 ms per query.

```
 "cbale"  -> seen < 5 times in the catalogue?  yes
          -> one edit away: "cable" (very common), "cbale" itself
          -> "cable" is >= 10x more frequent  ->  query becomes "cable" -> ranker
```

**Results (3 baseline models averaged; settings 5, 10x, one edit were fixed before looking, not tuned):**

| | dev change [95% CI] | dev recovered | dev2 change [95% CI] | dev2 recovered |
|---|---|---|---|---|
| clean queries | +0.0003 [-0.0008, +0.0015] | | +0.0020 [-0.0002, +0.0042] | |
| typo (swapped letters) | +0.0252 [+0.0211, +0.0292] | 88-89% (89%) | +0.0287 [+0.0229, +0.0349] | 87% |
| typo (neighbour key) | +0.0226 [+0.0185, +0.0267] | 88% | +0.0263 [+0.0203, +0.0323] | 90% |

Compare with 14% from retraining. All three pre-registered rules passed on dev and once on dev2, so it went
into serving as v1.1. The corrector changed 5.9% of clean queries on dev (my guess was 1-4%), yet clean
nDCG did not drop.

**Known misses, stated honestly:** a misspelling that is common in titles stays, because it passes the "known
word" test ("wirless" appears 5+ times). A rare non-English word can be "corrected" into English ("llaves"
becomes "leaves").

**Why fixing the input beats retraining here:** the failure is a sub-word split the small model cannot undo.
A lookup against the catalogue removes the cause before the model sees it. It costs no GPU, no retraining, and
can be switched off (`SHOPRANK_SPELL=0`). The response also shows `query_used`, so a user can see the change.

| Word | Meaning |
|---|---|
| vocabulary | the list of words the catalogue contains, with counts |
| Damerau-Levenshtein | edit distance that counts insert, delete, replace and swap-two-neighbours as one edit each |
| SymSpell | a fast method: index each word with one letter removed, then look up |
| recovered | the share of the typo loss won back (gain divided by the gap) |
| false correction | changing a query word that was correct |

**Interview answer:** "Retraining recovered 14% of the typo loss, so I corrected the query before ranking
instead. I built a one-edit spell corrector from a 184,594-word catalogue vocabulary, with a rule that the
replacement must be 10 times more frequent and the original rare. It recovered 87-90% of the typo loss on two
held-out sets, left clean queries unchanged within noise, and costs 0.01 ms per query. Its known misses are
typos that are common in titles and rare foreign words."

## Part 4: Release engineering, what makes it "shippable"

**Picture:** a restaurant that cooks one good meal is not yet a restaurant. It also needs a clean kitchen
inspected, labelled ingredients, a menu with a date, and a way for a stranger to order the same dish.

**What we did:**
- **Vulnerability fix with re-verification.** osv-scanner found 3 known issues (torch 2.11 and setuptools 81).
  One upgrade (torch 2.13, setuptools 84.2.1) fixed all; the scanner then reported none. We re-scored: served
  dev nDCG identical (0.86668 before and after), DeBERTa fold 0 moved +0.00017. An upgrade is only done when
  the numbers are re-checked.
- **Pinned model revisions.** Hugging Face base models are fixed by commit (`hub.py`: DeBERTa 8ccc9b6, MiniLM
  7b02352), so "download the model" cannot change under us.
- **Checked download.** `shoprank.serve.fetch` pulls the served model from the GitHub release or Hugging Face
  and writes a file only after its SHA-256 matches the value in the code. A fingerprint that does not match
  means the file is refused. 5 of 5 files verified in the live test.
- **Torch-free Docker image.** Serving needs only ONNX Runtime, so the export has 45 packages (199 MB) instead
  of about 7 GB. The image is pinned by digest, runs as a non-root user, and has a health check.
- **CI with three jobs:** tests (ruff + pytest), docker (build the image and call `/rank`), security
  (osv-scanner and gitleaks, binaries checked by SHA-256). Tests went 124 to 138.
- **Versions and a CHANGELOG.** v1.0 is the baseline (official test scored once). v1.1 adds spell correction.
  Every number in the CHANGELOG names its split.

```
 code --> CI: tests + docker build + security scan --> tag --> release files + SHA256SUMS
                                                                  |
              stranger: fetch (checksum ok?) --> torch-free image --> /rank
```

**Two incidents, recorded in the notebook, not hidden:**
- **Memory.** The first Phase D run thrashed: a 10.3 GB process, because it loaded all 1.2M products. The fix
  was to read only the 336k products the training pairs use (texts identical, and tested).
- **CI 404.** The docker job ran before the GitHub release existed, so the model download returned 404. The
  release was created after the push. CI went green on a re-run. Lesson: create the release first, or make the
  job wait for it.

| Word | Meaning |
|---|---|
| SHA-256 | a fingerprint of a file; if one byte changes, the fingerprint changes |
| pin | fix a dependency to an exact version or commit |
| osv-scanner | a tool that lists known security holes in your dependencies |
| digest (Docker) | an exact fingerprint of an image, stronger than a tag like "3.12-slim" |
| CI | automatic checks that run on every push |
| 404 | web error: "that file is not there (yet)" |

**Interview answer:** "A model is shippable when a stranger can run it and trust it. I upgraded torch and
setuptools to clear three known vulnerabilities and re-scored to prove the served number stayed 0.86668. I
pinned base-model commits, made the model download refuse any file whose SHA-256 differs, built a 45-package
torch-free Docker image, and added CI jobs for tests, the image and security. I tagged v1.0 and v1.1 with a
changelog, and I recorded two incidents: a 10 GB memory blow-up I fixed by loading only 336k products, and a CI
404 because I created the release after the push."

---

## Check questions

**Q1.** Why change the input one way at a time in a behavioural test?
- A. It makes the model faster.
- B. So any drop in nDCG can be blamed on that one change.
- C. Labels are not needed that way.
- D. To avoid using dev data.

**Q2.** Deleting the negation word cost 0.0762 on the negation slice. What does that tell us?
- A. The model ignores negation.
- B. The model reads negation; the slice's low score has other causes, mostly how ESCI built those queries.
- C. The slice has no labels.
- D. Negation is solved.

**Q3.** Typo-consistency training recovered 14% of the typo loss. The pre-registered rule was 50%. What follows?
- A. Lower the rule to 10%, since it helped a little.
- B. Run on dev2 to see if it passes there.
- C. Reject it, keep the served model, and leave dev2 unspent.
- D. Train with a typo rate of 90% and report only that.

**Q4.** Which statement about the spell corrector is correct?
- A. It changes any word, including numbers and symbols.
- B. It only touches letters-only words of 4+ characters that are rare in the catalogue, and the replacement
  must be 10 times more frequent.
- C. It needs the labels to work.
- D. It scans all 184,594 words for each query word.

**Q5.** Why did spell correction recover 87-90% while retraining recovered 14%?
- A. It removes the typo before the model sees it, so the small model never needs to undo a sub-word split.
- B. It uses a bigger model.
- C. It was tuned on dev2.
- D. It changes the labels.

**Q6.** After upgrading torch, what showed the upgrade was safe?
- A. The scanner reporting zero issues, and nothing else.
- B. Re-scoring dev: served nDCG stayed 0.86668, and the tests passed.
- C. A new version number.
- D. The Docker image got smaller.

## Answer key
1. **B.** One change at a time makes the cause clear; the labels stay fixed so only the input effect is measured.
2. **B.** Deleting the word reorders the list (Spearman 0.50), so the model reads it; the weak slice comes from other causes.
3. **C.** The rule was set before the run. Changing it afterwards, or spending dev2 on a rejected model, defeats its purpose.
4. **B.** Digits and symbols are never touched, and the 5x and 10x rules guard against false corrections; the lookup uses a deletion index.
5. **A.** Fixing the input removes the cause; the 33M-parameter student cannot learn to undo a typo's sub-word split.
6. **B.** A clean scanner proves the vulnerabilities are gone, not that the results are unchanged; the re-score proves that.
