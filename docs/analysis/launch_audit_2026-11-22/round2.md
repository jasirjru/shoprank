# Launch audit, round 2 (22 Nov 2026)

Scope: check that round 1 (`code_repo.md`, `site.md`, `numbers.md`) was fixed by c4a5284 and 2cc430a, re-measure the
live page, and check the new content. Report only: nothing was edited, committed, pushed or posted. No test-set
scoring: test numbers were read from the saved `data/processed/runs/*_test.parquet` files. Raw outputs (Lighthouse
JSON, axe and puppeteer results, screenshots) are in the session scratch folder, not in the repo.

State audited: `main` = `origin/main` = `2cc430a`. The live HTML is byte-identical to `site/public/index.html` and the
live `og.png` is byte-identical to `site/public/og.png` (`cmp`, 22 Nov).

## Verdict

No BLOCKER. Every round 1 BLOCKER and SHOULD is fixed on the live page, with one regression: the round 1 reports
are committed and public, and they quote the private path that was removed from `docs/PLAN.md` (SHOULD-A). The new
known-item section has a wrong label, a wrong example and an unsupported heading (SHOULD-B). "Typical wins" should
be "clear wins" (SHOULD-C). All three are text changes. None touches the JS, so the CSP hash stays the same.

| ID | Rank | Finding | Where |
|---|---|---|---|
| SHOULD-A | SHOULD (regression) | Public round 1 reports quote the private career-file path, a private self-rating and a local user path | `docs/analysis/launch_audit_2026-11-22/code_repo.md:150-152`, `site.md:12` |
| SHOULD-B | SHOULD | Known-item table: "model number" row is really "any digit"; the "general browsing" example contains a digit; "Those searches gain the most" is false for brand; unsourced AI-assistant claim | `src/shoprank/site.py:40-43, 241-242` |
| SHOULD-C | SHOULD | "hand-picked to show typical wins": the wins are at the 90th-99.9th percentile of dev gain, so they are not typical | `src/shoprank/site.py:228` |
| N1-N8 | NICE | Desktop CLS went up, cancelled CI run, known-item run not saved, post-hoc headline metric, and others | below |

## 1. Round 1 items: fixed / not fixed / regressed

| Round 1 item | Status | Evidence (live page unless noted) |
|---|---|---|
| site B1: no og/twitter tags, no share image | **fixed** | 11 og:/twitter: tags plus canonical, all absolute `https://shoprank-jasirjru.vercel.app/...` (section 3) |
| site S1: jargon meta description | **fixed** | "A small AI model that re-orders shop search results so the right products come first: the top result is right in 75%..." |
| site S2: title said "in Amazon product search" | **fixed** | `<title>ShopRank: puts the right product first in shop search</title>` |
| site S3: chosen tab scrolled under the strip edge | **fixed** | 375 px, ArrowRight through tabs 1-5: tab left edge 46 px or more, strip left edge 30 px, never clipped (puppeteer) |
| site S4: fonts 95% of the weight | **fixed** | `opsz` axis dropped (`site.py:168-169`); total transfer 118-119 KiB (was 227 KiB) |
| site S5: tap targets under 44 px | **fixed** | at 320 px: Code 42x44, Model files 78x44, switch buttons 130x44 / 100x44, tabs 44 high |
| site S6: five recruiter phrases | **fixed** | all five rows in the new text (human reviewers, "scored only once...", "accuracy that typos cost", "Large teacher model (too slow for a laptop)", "Casual wording and a misspelling") |
| numbers F1: headline overstated, title read as Amazon | **fixed** | headline "in 3 of 4 shop searches" (test success@1 0.7468; see section 4) |
| numbers F2: hand-picked examples not disclosed | **fixed, wording issue** | disclosed with 65% better / 33% worse (recomputed: 65.5% / 32.8%); "typical wins" is new wording that is not accurate (SHOULD-C) |
| numbers F3: LambdaMART called "hand-made rules" | **fixed** | "Classic machine-learned ranking (LambdaMART)" |
| numbers F4: "normal laptop", "0.07 s" | **fixed** | "runs on a laptop CPU, no GPU needed"; "about 0.07 seconds (median) on a laptop CPU"; the engineers section keeps "16-thread" |
| numbers F5: synthetic typos | **fixed** | "On made-up one-letter typos it wins back almost 90%..." |
| numbers F6: headline number is pre-compression | **fixed in the table** | row "ShopRank, before compression"; the hero 75% has no such label (N5, NICE) |
| code SHOULD-1: "m23 for cars" / "See both lists" dead | **fixed** | section 5 |
| code SHOULD-2: private path in `docs/PLAN.md` | **fixed in PLAN.md, regressed elsewhere** | `docs/PLAN.md` is clean, but the same path is now public in `code_repo.md:150` (SHOULD-A) |
| code SHOULD-3: README first screen | **fixed** | `README.md:6-10` plain paragraph: 0.867 vs 0.820, 74.7% vs 62.7%, 32 MB, ~70 ms median, stack |

## 2. Lighthouse 13.6.0 and axe-core (live page)

The first pair of runs overlapped with `pytest` on the same machine, which inflated desktop TBT (630 ms, performance
66). Both modes were re-run alone, with desktop run twice. The table shows the clean runs.

| | Mobile (clean run; contended run in brackets) | Desktop (2 clean runs) |
|---|---|---|
| Performance | **98** (97) | **96 / 97** |
| Accessibility | **100** | **100** |
| Best Practices | **100** | **100** |
| SEO | **100** | **100** |
| LCP | 1.96 s (2.1 s) | 0.79 s / 0.66 s |
| CLS | 0 | **0.097 / 0.094** (round 1: 0.018; N1) |
| TBT | 0 ms | 0 ms / 0 ms |
| FCP | 1.96 s | 0.79 s / 0.66 s |

Mobile improved from 86 to 98 (round 1 TBT 340 ms, now 0). Still flagged: the render-blocking Google Fonts CSS
(estimated savings 590 ms on mobile), which round 1 accepted (N7). `errors-in-console` passes.

**axe-core 4.14.1** was run inside the live page through puppeteer-core, as in round 1. Tags: wcag2a/aa, wcag21a/aa,
wcag22aa, best-practice. It ran at 1350x940 and 375x812, with "For engineers" closed and open:
**0 violations in all 4 runs** (48-49 rules passed). The only "incomplete" result is color-contrast over the ruled
background gradient, as in round 1, where those pairs were checked by hand. The four console errors in that session
come from axe's own stylesheet fetch, which the CSP `connect-src` blocks. Lighthouse's console audit passes, so this is
a tool artefact, not a page error.

## 3. Social card

- Present in the live `<head>`: `og:type`, `og:site_name`, `og:url`, `og:title` ("ShopRank: puts the right product
  first in 3 of 4 shop searches"), `og:description`, `og:image`, `og:image:type`, `og:image:width` 1200,
  `og:image:height` 627, `og:image:alt`, `twitter:card` summary_large_image, `twitter:image`, and
  `<link rel="canonical">`. All URLs are absolute https. With no `twitter:title`/`twitter:description`, the og values
  are used (fine).
- `https://shoprank-jasirjru.vercel.app/og.png`: **HTTP 200, `Content-Type: image/png`, 66,321 bytes, PNG header
  1200x627**. Viewed: the text is legible, the numbers are 75% vs 63%, and the footer says "Amazon's public
  shopping-search benchmark · 8,956 searches · 32 MB model on a laptop CPU".
- Not done here (needs the owner's login): paste the URL into LinkedIn Post Inspector once before posting.

## 4. New numbers vs evidence

**Headline and "Right at #1" column.** These are the share of `rr_exact == 1` in the saved test runs (8,956 queries
each), recomputed today:

| Run | success@1 | Page shows | Notebook (`lab_notebook.md:927-928`) |
|---|---|---|---|
| bm25_title_test | 0.6268 | 63% (hero, table, og) | 0.6268 |
| lambdamart_test | 0.6580 | 66% | 0.6580 |
| minilm_ft_mse_fast_test | 0.7161 | 72% | 0.7161 |
| minilm_distil_w10_bullets_test | 0.7468 | 75% (hero, table, og) | 0.7468 |
| deberta_oof_avg5_test | 0.7903 | 79% | 0.7903 |

All match. "3 of 4" for 74.7% is honest rounding (judgement). Per-query nDCG is better on 65.5% and worse on 32.8%, so
"65% / 33%" is correct.

**Known-item table** (`site.py:40-43`; notebook `lab_notebook.md:929-931`). The page values 59/78, 67/77 and 62/74
with counts 413 / 338 / 1,318 match the notebook (0.593/0.782, 0.672/0.766, 0.618/0.737). Reproduction from the saved
dev runs, using the rule tags in `src/shoprank/eval/slices.py:33-44`:

| Row on page | Rule behind it | n | BM25 (saved `bm25_title.parquet`) | ShopRank fp32, no spell (saved run) | Page (API: INT8 + spell) |
|---|---|---|---|---|---|
| "With a model number" | `has_number`: any digit in the query | 413 | 0.593 | 0.768 | 0.782 |
| "With a brand name" | `brand`: a candidate's brand appears in the query | 338 | 0.672 | 0.746 | 0.766 |
| "General browsing" | neither of the above | 1,318 | 0.618 | 0.737 | 0.737 |
| (not shown) | both a digit and a brand | 68 | | | |

The counts and the BM25 column reproduce exactly. The ShopRank column comes from a run through the API that is not
saved (N3), and the saved fp32 run lands close to it. The numbers are fine, but the labels are not (SHOULD-B):

1. **"With a model number" is "any digit".** Only 137 of the 413 queries have a letter+digit token such as `fp30` or
   `a10`. A random sample of the rest: "0.5 gallon trash bags", "game of thrones season 8 dvd", "baby boy bandana bibs
   2 pack", "boy toys age 7".
2. **The "general browsing" example is in the wrong row.** "womens running shoes 10.5" is a dev query, and it
   contains a digit, so it is counted in row 1, not row 3.
3. **"Those searches gain the most" is false for brand searches.** Gains: number +18.9 points, brand +9.4, the rest
   +11.9. Brand searches, the clearest "already know what they want" case, gain the least. The heading "Best when
   shoppers already know what they want" is therefore not supported.
4. **"Many shoppers now arrive after asking an AI assistant or reading reviews"** has no source on the page or in the
   repo. Judgement: a skeptical reader will flag this as a market claim on a page that otherwise cites everything.
5. Also unstated: the ShopRank column includes the spell checker and BM25 gets the raw query, and the rows overlap
   (68 searches).

## 5. In-page links (live, puppeteer, reduced motion, 1350x940 and 375x812)

To test each link, I first selected another tab (ex-2), then scrolled to the link and clicked it:

- "See both lists" (proof box): the `ex-0` panel is shown, the `#ex-0` tab is `aria-selected`, and the panel top is
  at 0 px (scrolled into view). Works at both sizes.
- "m23 for cars" (Where it fails): the `ex-5` panel ("m23 for cars") is shown and selected, with its top at 0 px.
  Works at both sizes.
- The URL hash does not change, because the handler calls `preventDefault` (`site/src/app.js:38-47`). N6, harmless.

## 6. Tests, lint, CI

- `uv run pytest -q`: **200 passed**, 4 warnings (ONNX-export deprecations, as in round 1).
- `uv run ruff check src tests`: **All checks passed!**
- `gh run list --repo jasirjru/shoprank --limit 3`:
  - 44996996072, 2cc430a (HEAD): **success**
  - 44996975797, c4a5284: **cancelled**
  - 44936747382, b672e2e: **success**

  The cancelled run was superseded: 2cc430a was pushed 14 s later and its run tested a tree that contains c4a5284. So
  CI on HEAD is green, but not all three runs are "success" (N2).

## 7. What a skeptical ML engineer or recruiter would flag

Ranked findings with exact fixes. After any `site.py` change: `uv run python -m shoprank.site`, then `uv run pytest -q`
(`tests/test_site.py` checks that the page and the generator match), then deploy. No JS change, so the CSP hash in
`vercel.json` does not change.

### SHOULD-A (regression): private data in the public round 1 reports

c4a5284 committed `docs/analysis/launch_audit_2026-11-22/` to the public repo. `code_repo.md:150-152` quotes the exact
private path, a private self-rating and career notes, which is the content removed from
`docs/PLAN.md`. `site.md:12` contains a local Windows path with the user name. Both are on `origin/main`, and the
GitHub page for `code_repo.md` returns 200.
Fix (pick one):

1. **Simplest: redact.**
   - `code_repo.md:150-152`. Old: the three lines starting ``` `docs/PLAN.md:41`: a reference line (private note redacted on 22 Nov)
     and ending `(`docs/PLAN.md:3, 9, 22`).`. New:
     ``` `docs/PLAN.md:41` held a local path to a private file and a private note (removed in c4a5284). ```
   - `site.md:12`. Old: the line with the scratch path. New: `in the session scratch folder (not in the repo).`
2. **Or move the folder out of the public repo.** Move it to the git-ignored `docs/private/`
   (`git mv docs/analysis/launch_audit_2026-11-22 docs/private/`, keep it untracked), and change `lab_notebook.md:923`
   "Reports in `docs/analysis/launch_audit_2026-11-22/`" to "Audit reports kept private".

   Judgement: option 2 is cleaner. The reports talk about the LinkedIn post and about impressing recruiters, which
   reads oddly to a recruiter browsing the repo. The old text stays in git history, which round 1 judged acceptable.
   This report (`round2.md`) contains no local paths.

### SHOULD-B: known-item section labels (`src/shoprank/site.py`)

- Line 39 comment. Old: `# Right product at #1 on the dev split by kind of search (all 2,001 dev queries through the real API, 22 Nov 2026).`
  New: `# Right product at #1 on dev by rule tag from eval/slices.py (has_number = any digit; brand; neither). Rows 1-2 overlap (68). ShopRank = API (INT8 + spell), BM25 = raw query. 22 Nov 2026.`
- Lines 41-43. Old:
  ```python
      ("With a model number, like \u201cfp30 piano\u201d", 413, 0.593, 0.782),
      ("With a brand name", 338, 0.672, 0.766),
      ("General browsing, like \u201cwomens running shoes 10.5\u201d", 1318, 0.618, 0.737),
  ```
  New:
  ```python
      ("With a number, like \u201cfp30 piano\u201d or \u201c0.5 gallon trash bags\u201d", 413, 0.593, 0.782),
      ("With a brand name", 338, 0.672, 0.766),
      ("Neither, like \u201cwomens basketball shoes\u201d", 1318, 0.618, 0.737),
  ```
  All three example queries are dev queries in the right row (checked).
- Line 241. Old: `<h2 id="known-title">Best when shoppers already know what they want</h2>`
  New: `<h2 id="known-title">Biggest gain on searches with a number in them</h2>`
- Line 242. Old: `Many shoppers now arrive after asking an AI assistant or reading reviews, and search for one specific product. Those searches gain the most: how often the top result is exactly right, on 2,001 development searches.`
  New: `How often the top result is exactly right, by kind of search, on 2,001 development searches (ShopRank as served, with its spell checker). Searches with a number in them, such as a model number or a size, gain the most. 68 searches have both a number and a brand.`
- `docs/lab_notebook.md`: append a correction under the 22 Nov entry (do not rewrite it): `Correction: "known-item
  (model number)" is the has_number rule (any digit, eval/slices.py); only 137 of 413 have a model-like token. Brand
  gains +9.4 points, less than the rest (+11.9).`

### SHOULD-C: "typical wins" (`src/shoprank/site.py:228`)

The 5 wins sit at the 89.7th-99.9th percentile of per-query dev gain (`numbers.md` F2,
`data/processed/showcase_candidates.parquet`). Old: `hand-picked to show typical wins and one failure`
New: `hand-picked to show clear wins and one failure`

### NICE

- **N1. Desktop CLS 0.018 → 0.094-0.101 (3 runs).** Lighthouse blames the Newsreader web font swap on `h1#headline`.
  The headline is now two lines at 1350 px, so the font swap moves more text (round 1: one line). This is still at or
  just under Google's 0.1 "good" line, mobile CLS is 0, and the LinkedIn card is not affected. Fix (judgement; check
  with Lighthouse afterwards): add a metric-matched fallback face to the CSS, e.g. `@font-face { font-family:
  "Newsreader Fallback"; src: local("Georgia"); size-adjust: ...; ascent-override: ...; }`. Tune the values with a
  tool such as `fontaine`/`capsize`, and list that face second in `--serif` (`style.css:15`).
- **N2. CI run for c4a5284 was cancelled** (superseded). HEAD is green, so no action.
- **N3. The known-item API run is not saved.** No per-query file under `data/processed` was written on 22 Nov, and no
  script makes the table. The BM25 side reproduces exactly from saved files. Save the per-query dev API run (for
  example `data/processed/runs/api_int8_spell_dev.parquet`) and name the command in the notebook.
- **N4. The headline metric was chosen after the test was scored.** success@1 was read from the saved test runs on
  22 Nov with no prediction (the notebook says so, `lab_notebook.md:931`). `README.md:14` says "every run has a
  prediction written before it". The test was not re-scored, and nDCG had already been pre-registered and also
  improved, so the claim stands. But an ML engineer will ask. Add to the engineers' "Student" item (`site.py`, after
  the `(0.8544)` sentence): ` Right at #1 (success@1, read from the saved test run afterwards): 0.747 vs 0.627,
  +0.120 [+0.109, +0.131].`
- **N5. The hero 75% is the pre-compression model**, shown next to "runs on a laptop CPU" (also in the og.png
  footer). On dev, the served INT8 + spell scores 0.7496 against 0.7446 for fp32 without spell, so this is not
  misleading in substance. Optional: add "(full-precision model; the 32 MB version scores the same on dev)".
- **N6.** In-page example links do not update the URL hash. Harmless.
- **N7.** Google Fonts CSS still blocks rendering (estimated 590 ms on mobile). Round 1 accepted this; self-hosting
  the fonts is the later fix.
- **N8.** The known-item table is on dev, which was also used to choose the model, so it is slightly optimistic. The
  page says "development searches". Fine as is.
