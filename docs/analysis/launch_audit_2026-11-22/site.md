# Live page audit: https://shoprank-jasirjru.vercel.app (22 Nov 2026)

Scope: the live page before the LinkedIn post. Report only; nothing was edited, committed or deployed.
The live HTML is byte-identical to `site/public/index.html` in the repo (70,072 bytes, `cmp` equal), so every
`site/public/index.html` finding maps to `src/shoprank/site.py`, `site/src/style.css`, `site/src/app.js` or
`site/data/examples.json`. After any fix: `uv run python -m shoprank.site` (regenerates index.html and the CSP hash in
vercel.json; `tests/test_site.py` checks they match), then deploy.

Tools: Lighthouse 13.6.0 (npx, headless Chrome 157); axe-core 4.14.1 (see section 2 for how); puppeteer-core from the
Lighthouse npx cache for 320/375/390 px, no-JS, focus and reduced-motion checks; Python for contrast ratios.
Raw outputs (scratch, not in repo): `lh_mobile.json`, `lh_desktop.json`, `axe_results.json`, screenshots, in
`[local scratch folder]`.

## Verdict

One BLOCKER: the page has no Open Graph / Twitter tags and no share image, so the LinkedIn card will have no picture.
Accessibility is clean (Lighthouse 100, axe 0 violations, all text pairs pass WCAG AA). Six SHOULD items, all small.

## Findings ranked

| # | Rank | Finding | Where |
|---|------|---------|-------|
| B1 | BLOCKER | No og:/twitter: tags and no 1200x627 share image | site.py:161-168 |
| S1 | SHOULD | Meta description (LinkedIn's fallback card text) is jargon: "re-ranking model", "nDCG" | site.py:163 |
| S2 | SHOULD | `<title>` "...in Amazon product search" can read as running in Amazon / affiliated (judgement) | site.py:162 |
| S3 | SHOULD | Bug: selecting an off-screen example tab scrolls it 14 px under the strip's left edge | style.css:100, app.js:20-22 |
| S4 | SHOULD | Fonts are 95% of page weight; dropping Newsreader's `opsz` axis saves about 110 KB | site.py:149-150 |
| S5 | SHOULD | Tap targets under 44 px: top-bar Code/Model links, phone list switch | style.css:52-54, 144 |
| S6 | SHOULD | Five first-layer phrases a recruiter will not follow | section 5 |
| N1-N8 | NICE | Listed at the end | |

---

## 1. Lighthouse (lh_mobile.json, lh_desktop.json)

| | Mobile (Moto G Power, 4x CPU, slow 4G) | Desktop |
|---|---|---|
| Performance | **86** | **99** |
| Accessibility | **100** | **100** |
| Best Practices | **100** | **100** |
| SEO | **100** | **100** |
| FCP | 2.4 s | 0.7 s |
| LCP | 2.4 s (element: `p.lede`; 1.63 s of it is render delay) | 0.7 s |
| CLS | 0 | 0.018 (h1 shifts when the web font swaps in) |
| TBT | 340 ms | 0 ms |
| Speed Index | 3.7 s | 0.9 s |

Failed or partial audits, with fix:

| Audit | Result | Cause | Fix |
|---|---|---|---|
| render-blocking-insight (both) | Est. savings 160 ms mobile / 270 ms desktop | The Google Fonts stylesheet `<link>` (site.py:168) blocks first paint | S4 below. The usual `media="print" onload=...` trick is **blocked by this page's CSP** (`script-src` allows one hash; inline event handlers are refused), so either self-host the fonts (CSS goes inline, `font-src 'self'`) or accept the 160 ms. |
| network-dependency-tree-insight (both) | fail | Chain: HTML, then fonts.googleapis.com CSS, then fonts.gstatic.com woff2 (3 hops) | Same as above; self-hosting with `<link rel="preload" as="font" type="font/woff2" crossorigin>` removes two hops. |
| first-contentful-paint (mobile) | 2.4 s, score 0.71 | Render-blocking font CSS | Same. |
| total-blocking-time / max-potential-fid / mainthread-work-breakdown (mobile) | 340 ms / 710 ms / 3.3 s | Main thread is Style & Layout 1.5 s, "Other" 1.1 s; one 706 ms task at about 1.9 s. Script evaluation is only 0.2 s. 857 DOM elements (six panels x two lists rendered up front). | Judgement: lab-only with 4x CPU slowdown; acceptable for launch. Cheap win is N3 (forced reflow). |
| forced-reflow-insight (desktop) | 146 ms at index.html line 445 | app.js:21-22 reads `offsetLeft`/`offsetWidth` right after toggling `hidden` on six panels in `select()` | N3. |
| speed-index (mobile) | 3.7 s, 0.86 | Font swap and late paint | S4. |
| bf-cache (mobile only) | 1 failure | `CacheLimitPrunedOnCriticalMemoryPressure`, Lighthouse marks it "Not actionable" | None (environment, not the page). |

## 2. axe

`npx -y @axe-core/cli https://shoprank-jasirjru.vercel.app --exit` ran axe-core 4.14.1 but failed: its bundled
ChromeDriver supports Chrome 152 and the installed Chrome is 157 (it suggests `npx browser-driver-manager install chrome`,
which would download a new Chrome; not done). Instead I ran **the same axe-core 4.14.1 file from that npx package**
inside the live page via puppeteer-core (already in the Lighthouse npx cache), tags wcag2a, wcag2aa, wcag21a, wcag21aa,
wcag22aa, best-practice, at 1350x940 and 320x700, each with the "For engineers" section closed and open.
Lighthouse's Accessibility category (axe-based) also scored 100.

| Run | Violations | Incomplete (needs review) | Passed rules |
|---|---|---|---|
| Desktop, details closed | 0 | color-contrast, 90 nodes | 48 |
| Desktop, details open | 0 | color-contrast, 113 nodes | 48 |
| 320 px, details closed | 0 | color-contrast, 90 nodes | 49 |
| 320 px, details open | 0 | color-contrast, 110 nodes | 49 |

By impact: critical 0, serious 0, moderate 0, minor 0. The "incomplete" items all say "background color could not be
determined due to a background gradient / pseudo element": the ruled-paper `background-image` on body (style.css:31)
and the `.sheet::before` margin rule. I computed those pairs by hand in section 3; all pass.

## 3. Manual accessibility (style.css, app.js, site.py)

### Colour contrast (WCAG 2 relative luminance; AA needs 4.5:1 for text, 3:1 for large text and UI parts)

| Foreground on background | Ratio | Used for | AA |
|---|---|---|---|
| ink #1d1b16 on paper #fbf8f1 | 16.22 | body text | pass |
| ink-soft #3d3a33 on paper | 10.69 | lede, section ledes, notes, legend, footer | pass |
| pencil #5d5952 on paper | 6.56 | skills line, byline, "Search", positions, "+ N more", table head, keyword-search grade | pass |
| red #b3261e on paper | 6.16 | 0.867 score, grades, spell-check line, 12 px "fails" flag, link hover | pass |
| blue #1f3a8a on paper | 9.75 | links | pass |
| ink on highlight #fbe8b8 | 14.20 | proof box, "right product" rows | pass |
| blue on highlight | 8.54 | "See both lists" link | pass |
| ink-soft on highlight | 9.36 | keyword-list exact rows | pass |
| pencil on highlight | 5.75 | position numbers in exact rows | pass |
| red on highlight | 5.40 | marks in exact rows | pass |
| paper on ink | 16.22 | primary button, selected tab, skip link, pressed switch | pass |
| white on red | 6.54 | primary button hover | pass |
| #ffb4a9 on ink | 10.12 | "fails" flag on the selected tab | pass |
| pencil on rule line #e4ddcc (worst case: text sitting on a ruled line) | 5.14 | | pass |
| red on rule line (worst case) | 4.83 | | pass |
| mark at 0.85 opacity on paper | 4.79 | complement / irrelevant marks (non-text, needs 3:1) | pass |
| red focus outline on paper | 6.16 | focus ring (non-text) | pass |
| rule #e4ddcc on paper | **1.28** | outline of unselected example tabs | see N1 |

### Focus visibility
- Global `:focus-visible { outline: 2px solid var(--red); outline-offset: 3px }` (style.css:42), 6.16:1. Seen in
  screenshots on the example tabs and on the "For engineers" summary. Skip link appears on focus (style.css:45-46).
- The tab strip is a scroll container (style.css:100) with only 2-4 px padding, so the 5 px ring is shaved by 1 px at
  the top and 3 px on the outer side of the first and last tab (N2). The S3 bug can hide 14 px of the left side.

### Tap targets (measured at 320 px; brief threshold 44 px; WCAG 2.2 AA 2.5.8 minimum is 24 px)
| Target | Size (px) | Note |
|---|---|---|
| Top-bar "Code" / "Model" | 34x28 / 42x28 | below 44 (S5) |
| Brand "ShopRank" | 78x18 | below 44; passes 2.5.8 by the spacing exception (judgement) |
| Phone switch "Keyword search" / "ShopRank" | 129x40 / 101x40 | below 44 (S5); "Keyword search" wraps to two lines (N4) |
| Skip link | 132x36 | below 44, keyboard-only, acceptable |
| Inline links in sentences, footer, engineers list | about 15-17 high | exempt as inline text links under 2.5.8 |
| Primary/ghost buttons, example tabs | 44 high | pass |
| "For engineers" summary | 52 high | pass |

### Reduced motion
- `scroll-behavior: smooth` switched off under `prefers-reduced-motion` (style.css:25); summary "+" rotation off
  (style.css:205); tab-strip `scrollTo` uses `auto` (app.js:22); FLIP row animation returns early (app.js:48).
  Verified: with reduced motion emulated, clicking the switch created 0 animations. Colour transitions (0.2 s) remain,
  which is fine (not motion).

### No-JavaScript fallback (verified with JS disabled at 375 px)
- All 6 examples visible, both columns shown (12 lists), switch hidden, no horizontal scroll; page is 11,173 px tall
  (vs 3,865 px with JS). Tabs still work as in-page anchor links. Works. Minor ARIA mismatch in N5.

### 320 px width
- No horizontal page scroll at 320, 375 or 390 (scrollWidth equals viewport). The tab strip scrolls inside itself
  (272 px visible of 1,306 px), and the next tab is visibly cut, which hints at scrolling.
- First screen: h1 ends at 149 px, scores at 251, lede at 446, proof box at 616. The GitHub button ends at 684 px,
  so on a 320x568 phone it is below the fold; on 375x667 and 390x844 it is visible (ends at 628).

## 4. Social sharing (BLOCKER B1)

Live `<head>` (site.py:159-169) has `title`, `description`, `theme-color`, favicon. It has **none** of: `og:title`,
`og:description`, `og:image`, `og:url`, `og:type`, `twitter:card`, nor `<link rel="canonical">`. The page has no
`<img>` at all and the only icon is an SVG, so LinkedIn has nothing to use as a picture: the post will show a bare
text card. LinkedIn reads `og:title`, `og:description`, `og:image`, `og:url` and recommends a 1200x627 image
(LinkedIn Help, "Make your website shareable on LinkedIn"; size from the brief). Judgement: blocker because the card is
the first thing a recruiter sees in the feed.

Fix, two parts:

1. Make `site/public/og.png`, 1200x627 PNG, under 5 MB. Simplest honest image: a headless-Chrome screenshot of the
   page's own first screen at a 1200x627 viewport (headline, the circled 0.867 vs 0.820, one line of plain text such as
   "Small AI model, beats keyword search on Amazon's public shopping benchmark"). Keep text large; LinkedIn shows the
   card small on phones.
2. In `src/shoprank/site.py`, after line 164 (`<meta name="theme-color" ...>`) insert:

```html
<link rel="canonical" href="https://shoprank-jasirjru.vercel.app/">
<meta property="og:type" content="website">
<meta property="og:site_name" content="ShopRank">
<meta property="og:url" content="https://shoprank-jasirjru.vercel.app/">
<meta property="og:title" content="ShopRank: a small AI model that puts the right product first in shop search">
<meta property="og:description" content="Re-orders shop search results so the products people want come first. Scores 0.867 vs 0.820 for keyword search on Amazon's public shopping benchmark. Runs on a laptop. Real examples and code.">
<meta property="og:image" content="https://shoprank-jasirjru.vercel.app/og.png">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="627">
<meta property="og:image:alt" content="ShopRank scores 0.867 against 0.820 for keyword search on Amazon's public shopping benchmark">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="ShopRank: a small AI model that puts the right product first in shop search">
<meta name="twitter:description" content="Scores 0.867 vs 0.820 for keyword search on Amazon's public shopping benchmark. Real examples and code.">
<meta name="twitter:image" content="https://shoprank-jasirjru.vercel.app/og.png">
```

The `og:image` URL must be absolute. The CSP (`img-src 'self'`) does not affect crawlers. After deploying, paste the
URL into LinkedIn Post Inspector once before posting, which also refreshes LinkedIn's cache of the bare card.

**S1** (meta description, site.py:163). LinkedIn and Google fall back to it. Old:
`A 32 MB re-ranking model that beats keyword search on Amazon's Shopping Queries benchmark: nDCG 0.867 vs 0.820 on the official test set. Real examples, numbers and code.`
New: `A small AI model that re-orders shop search results so the right products come first. Scores 0.867 vs 0.820 for keyword search on Amazon's public shopping benchmark. Real examples and code.`

**S2** (title, site.py:162; judgement). "in Amazon product search" can read as "runs in Amazon's search"; the footer
disclaimer is far below. Old: `ShopRank: puts the right product first in Amazon product search`
New: `ShopRank: puts the right product first in shop search`

## 5. Copy check for a non-technical recruiter (first layer only)

Judgement throughout. S6 = the first five rows; the rest are NICE.

| # | Where | Old | New |
|---|---|---|---|
| 1 | site.py:196 | `Same products, two orders. The marks are Amazon's human labels.` | `Both lists hold the same products in a different order. The ticks and crosses come from Amazon's human reviewers, not from me.` |
| 2 | site.py:204 | `Amazon's official test: 8,956 searches, checked once at the end. Higher is better.` | `Amazon's official test: 8,956 searches, scored only once, at the very end, so the result could not be tuned to the test. Higher is better.` |
| 3 | site.py:214 | `It wins back almost 90% of the loss.` | `It wins back almost 90% of the accuracy that typos cost.` ("loss" is an ML term.) |
| 4 | site.py:33 | `Big teacher model (needs a GPU)` | `Large teacher model (too slow for a laptop)` |
| 5 | examples.json:217 | `Casual words. Keyword search shows women's jeans; ShopRank finds ripped jeans for babies.` | `Casual wording and a misspelling. Keyword search shows women's jeans; ShopRank finds ripped jeans for babies.` |
| 6 | site.py:183 | `Ranking score on Amazon's official test, where 1.0 is a perfect order.` | `Ranking score on Amazon's public shopping-search test, where 1.0 is a perfect order.` ("official test" of what?) |
| 7 | site.py:182 | `keyword search` (score label) | `plain keyword search` (the lede could add: "the matching most shop search boxes use") |
| 8 | site.py:223 | `It only re-orders a list it is given, and only for English (US) searches.` | `It does not search the whole catalogue: it re-orders the results another search returns, and only for English (US) searches.` |
| 9 | site.py:176 | top-bar link `Model` | `Model files` (a recruiter may not know what "Model" links to) |
| 10 | site.py:67 | grade number has only a `title` tooltip, invisible on phones | no change needed if row 6 lands (the lede already says 1.0 is perfect) |
| 11 | examples.json:835 | `...ShopRank shows fridges instead of spare parts.` | `...ShopRank shows fridges instead of mostly spare parts.` Keyword search's #1 row is a refrigerator shelf that Amazon labelled "Right product", which contradicts "instead of spare parts" for a careful reader. |

Plain enough already: the headline, "small AI model ... runs on a normal laptop", the legend words, "Big teacher /
Small student / Compressed / Typo fix", "Where it fails". The skills line (`Python · PyTorch · FastAPI · Docker · ML
ranking`) is useful for recruiter keyword matching; consider adding `Hugging Face · ONNX` (NICE).

Example notes live in `site/data/examples.json` (written by `shoprank.serve.showcase` from CLI notes); edit the JSON
note text and regenerate the page. That does not touch the model or any scoring.

## 6. Page weight and Google Fonts

7 requests, 227 KiB transferred (Lighthouse `network-requests`, identical mobile and desktop):

| Request | Transfer | Share |
|---|---|---|
| HTML (Brotli; 70,072 bytes raw, CSS and JS inline) | 15.8 KB | 7% |
| Google Fonts CSS | 1.3 KB | 1% |
| Newsreader roman woff2 | 131.9 KB | 58% |
| Newsreader italic woff2 | 63.2 KB | 28% |
| IBM Plex Mono 400 + 500 woff2 | 10.1 + 10.0 KB | 9% |
| favicon.svg | 0.4 KB | 0% |

Google Fonts is 216.6 KB, 95% of the bytes, 2 extra origins (preconnects are present, site.py:166-167), and the only
render-blocking resource. `display=swap` is set, so text is never invisible; the swap causes the desktop CLS of 0.018.

**S4 fix** (site.py:149-150). The font request asks for the optical-size axis (`opsz 6..72`), which makes the
variable font much larger. Measured on fonts.googleapis.com with a Chrome UA, latin subset: with `opsz` the files were
103 KB + 63 KB; without it 58 KB + 24 KB. Old:
```python
FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
         "&family=Newsreader:ital,opsz,wght@0,6..72,400..600;1,6..72,400&display=swap")
```
New:
```python
FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
         "&family=Newsreader:ital,wght@0,400..600;1,400&display=swap")
```
Trade-off (judgement): the h1 loses optical sizing; `font-optical-sizing: auto` (style.css:35) then does nothing. Check
the headline by eye after the change. Larger option for later: self-host subsetted woff2 in `site/public`, inline the
`@font-face` rules, preload the roman file, and change the CSP to `font-src 'self'`; removes the render-blocking request.

---

## SHOULD details not covered above

**S3. Tab strip scrolls the chosen tab under its own edge** (style.css:100, app.js:20-22). `select()` compares
`t.offsetLeft` with `strip.scrollLeft`, but `.tabs` is not positioned, so `offsetLeft` is measured from `.sheet`
(`position: relative`, style.css:49), about 30 px further left. Measured at 375 px with reduced motion: after
ArrowRight to tabs 1-4 the strip starts at x=30 but the selected tab starts at x=16, so 14 px of the tab and its focus
ring are cut off. Same when a partly visible tab is tapped. Fix, style.css:100, old:
`.tabs { display: flex; gap: 0.5rem; overflow-x: auto;`
new:
`.tabs { position: relative; display: flex; gap: 0.5rem; overflow-x: auto;`
(This makes the strip the `offsetParent`; no JS change, so the CSP hash stays the same.)

**S5. Tap targets** (style.css:52-54 and 144). Old `.toplinks { display: flex; gap: 1.25rem; font-size: 0.9375rem; }`
New:
```css
.toplinks { display: flex; gap: 0.75rem; font-size: 0.9375rem; }
.toplinks a { display: inline-flex; align-items: center; min-height: 2.75rem; padding: 0 0.25rem; }
```
and in `.topbar` (style.css:52) change `align-items: baseline` to `align-items: center`. In style.css:144 change
`min-height: 2.5rem` to `min-height: 2.75rem` for `.js .switch button`.

## NICE

- **N1.** Unselected example tabs are outlined in #e4ddcc on #fbf8f1 (1.28:1, style.css:103). The tab text and the
  selected dark pill make them readable as controls, so this is not a clear WCAG 1.4.11 failure (judgement). Fix:
  `box-shadow: inset 0 0 0 1px var(--pencil)` on `.tab` (5.75:1 or better), keep the hover style.
- **N2.** Focus ring clipped inside the tab strip. style.css:100 old `padding: 0.25rem 0.125rem 0.75rem; margin: 0 -0.125rem;`
  new `padding: 0.375rem 0.375rem 0.75rem; margin: 0 -0.375rem;`.
- **N3.** Forced reflow on load (146 ms desktop, part of mobile TBT): app.js:36 calls `select()` at start-up, which
  reads layout right after hiding five panels. Fix: skip the strip-scroll check on the initial call, e.g. give
  `select(i, focus, scroll = true)` a third argument and call `select(..., false, fromHash >= 0)` at app.js:36. This
  changes app.js, so regenerate to update the CSP hash.
- **N4.** At 320-375 px the "Keyword search" switch label wraps onto two lines inside a 40 px pill. Add
  `white-space: nowrap;` to `.js .switch button` (style.css:144) and reduce its `padding: 0 1rem` to `0 0.875rem`.
- **N5.** Without JS, all six panels show but the markup still says `role="tablist"`/`role="tab"`/`role="tabpanel"`
  with only the first tab `aria-selected="true"` (site.py:75, 103-105). Screen readers then announce a tab widget that
  does not behave like one. Fix: emit plain links/sections from site.py and add the roles in app.js. Low impact.
- **N6.** On a 320x568 phone the GitHub button sits just below the fold (ends at 684 px). Acceptable; shortening the
  lede (row 6/7 wording) or the proof box would lift it.
- **N7.** Desktop CLS 0.018 from the font swap on the h1. Within "good" (<0.1). A `size-adjust`ed local fallback
  `@font-face` would remove it.
- **N8.** No `<link rel="canonical">` (included in the B1 block). SEO already scores 100.
