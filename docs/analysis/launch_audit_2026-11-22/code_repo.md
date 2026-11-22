# Launch audit: code and repo (22 Nov 2026)

Scope: the brief at `_brief.md`, items 1-7. Report only; nothing was edited, committed or pushed.
Repo state audited: `main` at `b672e2e` (same as `origin/main`), plus the live page https://shoprank-jasirjru.vercel.app.

## Verdict

No BLOCKER found. Tests, lint, CI, secret scan and dependency scan are clean, every link returns 200, and the live
page runs its inline script under the CSP. Fix three SHOULD items today: the dead "m23 for cars" link on the live page,
a private local path in `docs/PLAN.md`, and the README first screen.

## 1. Tests and lint

| Check | Result | Source |
|---|---|---|
| `uv run pytest -q` | **200 passed**, 4 warnings (torch/transformers ONNX-export deprecation warnings, in `tests/test_onnx_export.py`), 36 s | local run, 22 Nov |
| `uv run ruff check src tests` | **All checks passed** | local run, 22 Nov |

## 2. GitHub Actions on main

`gh run list --repo jasirjru/shoprank --limit 5`: all 5 latest `ci` runs on `main` are **success** (newest run
44936747382, commit "docs: link the live project page...", 21 Nov 18:39 UTC). Jobs in the newest run: `security` (8 s),
`test` (1 m 30 s) and `docker` (32 s), all passed. No failures.

## 3. Security scans (both tools already installed: gitleaks 8.30.1, osv-scanner 2.7.0)

| Scan | Result |
|---|---|
| `gitleaks git .` (49 commits on local branches) | **no leaks found** |
| `gitleaks dir` on the committed tree (`git archive HEAD`, 1.74 MB) | **no leaks found** |
| `gitleaks dir .` on the whole working folder | 87 hits, all inside untracked/ignored folders (`.venv/`, `data/`, 13.4 GB scanned). None are in tracked files (the scan above proves it). Not a finding. |
| `osv-scanner scan source -L uv.lock` (111 packages) | **No issues found** |

The three SHA-256 values in `src/shoprank/serve/fetch.py:32-36` carry `# gitleaks:allow` with a reason: correct
(they are hashes of public files, not keys).

## 4. Code review

### SHOULD-1: the "m23 for cars" link in "Where it fails" does nothing on the live page (confirmed)

- Where: link built at `src/shoprank/site.py:221` (`<a href="#ex-{failure}">`), behaviour in `site/src/app.js:11-36`.
- What happens: with JavaScript on, `select()` sets `hidden` on every panel except the selected one
  (`app.js:16`). The only time the hash is read is on page load (`app.js:35`); there is no handler for in-page links.
  Clicking "m23 for cars" changes the URL to `#ex-5`, but panel `ex-5` stays hidden (height 0), so the page does not
  scroll and the example never appears. Verified on the live page on 22 Nov: after the click, panel states were
  `ex-0 visible, ex-1..ex-5 hidden`, `location.hash = #ex-5`. The same applies to "See both lists" (`#ex-0`,
  `site.py:133`) whenever the visitor has selected another tab first.
- Why it matters (judgement): it is the link in the honesty section that a sceptical engineer is most likely to click.
- Fix: in `site/src/app.js`, after the line `select(fromHash >= 0 ? fromHash : 0, false);` add:

```js
  // In-page links to an example (proof line, "Where it fails") open that tab and scroll to it.
  document.addEventListener("click", (e) => {
    const a = e.target.closest('a[href^="#ex-"]:not(.tab)');
    if (!a) return;
    const i = panels.findIndex((p) => "#" + p.id === a.getAttribute("href"));
    if (i < 0) return;
    e.preventDefault();
    select(i, false);
    panels[i].scrollIntoView({ behavior: reduce.matches ? "auto" : "smooth", block: "start" });
  });
```

  Then run `uv run python -m shoprank.site` (the script hash in `site/public/vercel.json` changes with `app.js`;
  `tests/test_site.py:65-71` fails until both generated files are committed) and `uv run pytest -q tests/test_site.py`.
  I injected this handler into the live page from the browser console and it selected tab 6 and then tab 1 as
  expected. Optional test: none exists for JS; a manual check is enough for one page.

### Checked and fine

- **XSS:** every data-derived string goes through `html.escape` (`site.py:57, 59, 72, 77, 79, 104, 113, 132`); numbers are
  formatted floats. `tests/test_site.py:48-53` checks `<script>` in a title is escaped. `app.js` never writes HTML
  (no `innerHTML`). The only unescaped values are constants in `site.py`.
- **CSP and line endings (the brief's specific worry):** the live response's CSP is
  `script-src 'sha256-OQI8WyyWsKGMOi/QCa5y+WvhWMYsEJ2kRTEfRBTTicE='`; the SHA-256 of the one inline `<script>` in the
  live HTML is the same value; the live HTML is byte-identical to the committed `site/public/index.html`, with LF
  endings; the browser console shows no CSP errors and `<html>` gets the `js` class (so the script ran). Git stores
  `index.html`, `app.js` and `vercel.json` with LF (`git ls-files --eol`: `i/lf`). `site.py:269-280` reads `app.js`
  with universal newlines and writes with `newline="\n"`, so the hash and the page always agree. Judgement: even a
  CRLF copy would not break the hash, because the HTML parser turns CRLF into LF before the browser hashes the
  script text. No action needed.
- **Other headers live:** `X-Content-Type-Options: nosniff`, `Referrer-Policy`, `Permissions-Policy`, HSTS (added by
  Vercel), `frame-ancestors 'none'`. `vercel.json` itself is not served (404).
- **Tabs ARIA:** `nav role="tablist"` with `a role="tab"`, `aria-selected`, `aria-controls`, roving `tabIndex`,
  arrow/Home/End keys (`app.js:26-33`); panels have `role="tabpanel"` and `aria-labelledby`. Both roles are allowed on
  those elements. Without JS every panel and both columns stay visible (progressive enhancement holds).
- **Switch:** `role="group"` with `aria-label`, two `button`s with `aria-pressed`; only shown under 900 px with JS
  (`style.css:142-148`). Correct pattern.
- **No failure example:** JS does not depend on it. In Python, `site.py:156` falls back to index 0 (see NICE-3).
- **app.py / fetch.py / spell.py for a stranger:** input limits and 411/413/422/503/500 paths are sound
  (`app.py:63-96, 169-199`); `fetch.py` verifies SHA-256 before writing and refuses to overwrite a different file
  (`fetch.py:63-92`); both download hosts return 200 for `serving.json` (GitHub release v1.1 and the pinned Hugging Face
  commit). The README's three run options (`README.md:68-79`) match the code; CI's `docker` job builds the image.

### NICE (later)

- **NICE-1: unused code.** `src/shoprank/serve/showcase.py:34-37` `class Ranking` is never used (only match in
  `src` and `tests`); `site/src/style.css:182` `.fine` and `style.css:159` `.intro h2` match nothing in the page.
  Fix: delete those lines (then regenerate the page, since the CSS is inlined).
- **NICE-2: the proof line cites a row the page does not show.** Live text: "In keyword search, the first right
  product is number 12" while each list shows 8 rows (`ROWS_SHOWN = 8`, `site.py:19`). Fix in `site.py:130`:
  old `f"the first right product is number {bm25_first}"`, new
  `f"the first right product is number {bm25_first}" + (f" (below the {ROWS_SHOWN} shown)" if bm25_first > ROWS_SHOWN else "")`.
- **NICE-3: "Where it fails" text is hard-coded** ("m23 for cars", `site.py:221`) while its link target is computed
  (`site.py:156`, falls back to `0`). If the examples change, the sentence can point at the wrong tab. Fine for the
  frozen v1.1.2 data (`tests/test_site.py:71` pins 6 examples).
- **NICE-4: `showcase.py:169`** writes `examples.json` without `newline="\n"` (CRLF on Windows; git normalises it, so
  harmless). Fix: add `newline="\n"` as in `site.py:279`.
- **NICE-5: results table** puts `aria-hidden="true"` on a header cell and data cells (`site.py:115, 118`), so the
  column count differs for screen readers. Minor; the bar column carries no extra information.
- **NICE-6: `app.py:167`** reports `version="0.1.0"` in OpenAPI while the package is 1.1.2 (`pyproject.toml:3`).

## 5. Links (curl -sIL, 22 Nov)

All return **200**: the live page and `favicon.svg`; github.com/jasirjru, /shoprank, /shoprank#readme,
/shoprank/releases/tag/v1.1, /blob/main/docs/MODEL_CARD.md, /blob/main/docs/lab_notebook.md;
huggingface.co/jasirjru/shoprank-reranker-minilm-int8; github.com/amazon-science/esci-data;
github.com/Anankin77/esci-search-reranking; jasirjru-portfolio.vercel.app; the Google Fonts CSS URL. Also the two
model download URLs used by `fetch.py`. `README.md` and `docs/MODEL_CARD.md` contain no relative Markdown links
(file paths are given as code, e.g. `docs/lab_notebook.md`). `http://127.0.0.1:7860` in README is the local demo
address (expected). **No broken links.**

## 6. README first screen (judgement)

### SHOULD-3: the first screen is written for ML engineers, not a LinkedIn visitor

After the live-page link, `README.md:6-9` opens with "Product search re-ranking on the Amazon Shopping Queries
dataset (ESCI, Task 1, US English): from BM25 to a DeBERTa-v3 cross-encoder ensemble...". The first number a visitor
sees in bold is the offline teacher's 0.8877 (`README.md:24`), while the live page's headline is 0.867 vs 0.820. A
non-specialist cannot tell what it does, which model is served, or what stack was used.
Fix: insert one plain paragraph between line 4 and line 6 of `README.md`:

```
ShopRank re-orders product search results so the right product comes first. On Amazon's official Shopping Queries
test (8,956 searches, scored once) the served 32 MB model scores nDCG 0.867, against 0.820 for keyword search
(BM25), and ranks a typical search in about 70 ms on a laptop CPU. Built with Python, PyTorch, ONNX Runtime,
FastAPI and Docker.
```

(Numbers from `README.md:24, 52` and `site.py:28-34`.) Optional: link the four plain paths under
"## Documentation" (`README.md:171-176`) so they are clickable on GitHub.

## 7. Anything that must not be public

Commands: `git ls-files` (160 files) + `git grep` for emails, phones, Windows/home paths, secret assignments; file
sizes; full `git log -p origin/main`; PDF metadata (ExifTool).

### SHOULD-2: private local path and career notes in `docs/PLAN.md`

`docs/PLAN.md:41`: a reference line (private note redacted on 22 Nov) This is a local Windows path to a private career file, plus a private self-rating; it
is the only local path in any tracked file. The same file says career notes (`docs/PLAN.md:3, 9, 22`).
Fix: delete line 41 of `docs/PLAN.md` (old text above, new text: nothing). Judgement: consider moving the whole
`docs/PLAN.md` to the git-ignored `docs/private/`. The line stays in git history (added in an earlier public commit);
that is acceptable for a folder name, not worth a history rewrite.

### Checked and fine

- **Emails:** only the GitHub noreply address (`pyproject.toml:6`). Public commit authors: the noreply address and
  `jasirjru@example.com` (33 older commits; a placeholder, not personal data; see NICE-7).
- **Phone numbers:** none. The one regex hit is an Amazon product id (ASIN `0525657622`) in
  `docs/analysis/label_audit100_deberta_dev.csv:53`.
- **Private files:** `docs/HANDOFF.md` and `docs/private/` are git-ignored and appear in **no** commit of
  `origin/main` or of any public tag (v1.0, v1.1, v1.1.1, v1.1.2; all are ancestors of `origin/main`). They exist only
  on the local branch `backup/pre-scrub-2026-11-21`, which is not on GitHub (`git ls-remote`: only `main` and tags).
- **Secrets:** no secret-like assignments; gitleaks clean (section 3).
- **Large files:** biggest tracked file is `docs/learn/study-guide/ShopRank_Study_Guide.pdf` (0.99 MB); no model
  weights or data files are tracked. PDF metadata: Creator Chromium, Title "ShopRank Study Guide", no author or path.
- **Owner name** on the page and in `pyproject.toml` is intended (`PRODUCT.md:35`).

### NICE

- **NICE-7: older commits use `jasirjru@example.com`** (33 commits on `origin/main`), so GitHub does not link them to
  the profile. Not worth rewriting public history; just keep the noreply address for new commits.
- **NICE-8: never push `backup/pre-scrub-2026-11-21`.** It holds the pre-scrub history with `docs/HANDOFF.md`. Delete
  it once the scrubbed history is confirmed good (`git branch -D backup/pre-scrub-2026-11-21`), or keep it local only.
- **NICE-9: internal notes are public.** `PRODUCT.md:16-17` ("Job: decide whether this candidate is worth a call") and
  `.claude/` (35 files, including third-party Hugging Face skills, Apache-2.0, source in `.claude/skills/SOURCES.txt`)
  are visible to visitors. Judgement: harmless, but they add clutter to the repo root for a recruiter; the copied
  Apache-2.0 skills should carry the licence text or be removed.

## Summary of findings

| ID | Rank | Finding | File |
|---|---|---|---|
| SHOULD-1 | SHOULD | "m23 for cars" and "See both lists" links do nothing when the target tab is hidden (live, confirmed) | `site/src/app.js:35`, `src/shoprank/site.py:133, 221` |
| SHOULD-2 | SHOULD | Private local path + career self-rating in a public doc | `docs/PLAN.md:41` |
| SHOULD-3 | SHOULD | README first screen is jargon-first; bold number differs from the page headline | `README.md:6-9, 24` |
| NICE-1..9 | NICE | Unused code/CSS, proof line cites a hidden row, hard-coded failure text, CRLF export, table aria-hidden, API version, placeholder commit email, local backup branch, internal notes public | see above |
