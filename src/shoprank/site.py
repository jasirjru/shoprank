"""The public project page (one static HTML file), generated from the project's real data.

    uv run python -m shoprank.site            # site/data/examples.json -> site/public/index.html

The page needs no server and no framework: Vercel serves site/public as static files. Every example comes from
`shoprank.serve.showcase` (the real API on dev queries); every number below is from README.md / MODEL_CARD.md.
All text from the data is HTML-escaped. tests/test_site.py checks the committed page matches this generator.
"""
from __future__ import annotations

import base64
import hashlib
import json
from html import escape
from pathlib import Path

EXAMPLES_PATH = Path("site/data/examples.json")
OUT_PATH = Path("site/public/index.html")
ROWS_SHOWN = 8
REPO = "https://github.com/jasirjru/shoprank"
MODEL = "https://huggingface.co/jasirjru/shoprank-reranker-minilm-int8"
PORTFOLIO = "https://jasirjru-portfolio.vercel.app"
GITHUB_PROFILE = "https://github.com/jasirjru"
DATASET = "https://github.com/amazon-science/esci-data"
OWNER = "Jasir Abdul Hameed"
SITE = "https://shoprank-jasirjru.vercel.app"

# Official ESCI Task 1 test (US, 8,956 queries), scored once: full-list nDCG (README.md) and the share of
# queries whose top result is an Exact match, read from the saved test runs (rr_exact == 1).
RESULTS = [
    ("Keyword search", 0.8202, 0.627, "baseline"),
    ("Classic machine-learned ranking (LambdaMART)", 0.8337, 0.658, ""),
    ("Amazon's MiniLM method, re-run by me", 0.8544, 0.716, "published baseline"),
    ("ShopRank, before compression", 0.8674, 0.747, "this page"),
    ("Large teacher model (too slow for a laptop)", 0.8877, 0.790, "offline"),
]
SCALE_MIN, SCALE_MAX = 0.80, 0.90

# Right product at #1 on dev by rule tag from eval/slices.py (has_number = any digit; brand; neither). Rows 1-2
# overlap (68 queries). ShopRank = the API (INT8 + spell), BM25 = the raw query. 22 Nov 2026.
KNOWN_ITEM = [
    ("With a number, like \u201cfp30 piano\u201d or a size", 413, 0.593, 0.782),
    ("With a brand name", 338, 0.672, 0.766),
    ("Neither, like \u201ccheap spy camera\u201d", 1318, 0.618, 0.737),
]

MARKS = {  # authored SVG paths, one stroke family; shape (not colour) carries the label
    "Exact": '<path d="M4 12.5l5 5L20 6"/>',
    "Substitute": '<path d="M4 12.5l5 5L20 6"/><path d="M12 20.5h8"/>',
    "Complement": '<path d="M12 6v12M6 12h12"/>',
    "Irrelevant": '<path d="M6 6l12 12M18 6L6 18"/>',
}
MARK_TEXT = {"Exact": "Right product", "Substitute": "Good alternative", "Complement": "Goes with it",
             "Irrelevant": "Wrong product"}


def mark_svg(label: str) -> str:
    return (f'<svg class="mark mark-{label.lower()}" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
            f'{MARKS[label]}</svg>')


def ranked_list(items: list[dict], side: str) -> str:
    rows = []
    for pos, item in enumerate(items[:ROWS_SHOWN], start=1):
        label = item["label"]
        rows.append(
            f'<li class="row is-{label.lower()}" data-pid="{escape(item["product_id"])}">'
            f'<span class="pos">{pos}</span>{mark_svg(label)}'
            f'<span class="title">{escape(item["title"])}</span>'
            f'<span class="sr">, {MARK_TEXT[label]}</span></li>')
    more = len(items) - ROWS_SHOWN
    tail = f'<p class="more">+ {more} more products</p>' if more > 0 else ""
    return f'<ol class="list list-{side}">{"".join(rows)}</ol>{tail}'


def grade(value: float, side: str) -> str:
    return f'<span class="grade grade-{side}" title="Ranking score for this search (1.0 = perfect)">{value:.2f}</span>'


BOARD = [  # (key, label, size) in the order shown; "bm25" and "shoprank" come from the example's own grades
    ("bm25", "Keyword search", "word matching"),
    ("bge", "BGE reranker", "568M, ready-made"),
    ("qwen3", "Qwen3 reranker", "596M, ready-made"),
    ("teacher", "Teacher", "5 DeBERTa models"),
    ("shoprank", "ShopRank", "33M, 32 MB"),
]
TAB_CHARS = 26


def scoreboard(ex: dict) -> str:
    scores = {**ex.get("others", {}), **ex["ndcg"]}
    if not ex.get("others"):
        return ""
    best = max(scores.values())
    items = []
    for key, label, size in BOARD:
        value = scores[key]
        cls = " is-ours" if key == "shoprank" else ""
        cls += " is-best" if value == best else ""
        sub = f'<span class="board-size">{size}</span>' if size else ""
        items.append(f'<li class="board-item{cls}"><span class="board-name">{label}{sub}</span>'
                     f'<span class="board-score">{value:.2f}</span></li>')
    return ('<div class="board"><p class="board-title">Same search, five systems (ranking score, 1.0 = perfect)</p>'
            f'<ul>{"".join(items)}</ul></div>')


def example_panel(ex: dict, index: int) -> str:
    corrected = ex["query_used"] != ex["query"].strip()
    fix = (f'<p class="fix">Spell checker changed it to <q class="q">{escape(ex["query_used"])}</q></p>' if corrected else "")
    failure = " is-failure" if ex["ndcg"]["shoprank"] < ex["ndcg"]["bm25"] else ""
    return f"""
<section class="example{failure}" id="ex-{index}" role="tabpanel" aria-labelledby="tab-{index}" data-mode="shoprank">
  <div class="example-head">
    <p class="query">Search <q class="q">{escape(ex["query"])}</q></p>
    {fix}
    <p class="note">{escape(ex["note"])}</p>
  </div>
  {scoreboard(ex)}
  <div class="switch" role="group" aria-label="Show ranking by">
    <button type="button" data-show="bm25" aria-pressed="false">Keyword search</button>
    <button type="button" data-show="shoprank" aria-pressed="true">ShopRank</button>
  </div>
  <div class="columns">
    <div class="col col-bm25">
      <h3 class="col-title">Keyword search {grade(ex["ndcg"]["bm25"], "bm25")}</h3>
      {ranked_list(ex["bm25"], "bm25")}
    </div>
    <div class="col col-shoprank">
      <h3 class="col-title">ShopRank {grade(ex["ndcg"]["shoprank"], "shoprank")}</h3>
      {ranked_list(ex["shoprank"], "shoprank")}
    </div>
  </div>
</section>"""


def short(query: str) -> str:
    query = query.strip()
    return query if len(query) <= TAB_CHARS else query[:TAB_CHARS].rsplit(" ", 1)[0] + "\u2026"


SUIT_NAMES = {"phones": "Phones", "consoles": "Consoles", "tablets": "Tablets & laptops", "home": "Audio & home"}


def tabs(examples: list[dict]) -> str:
    """Answer cards (paper slips with depth): every example at a glance. Each card is a tab that opens the
    full side-by-side comparison below."""
    cards = []
    for i, ex in enumerate(examples):
        fails = ex["ndcg"]["shoprank"] < ex["ndcg"]["bm25"]
        suit = ex.get("suit", "phones")
        ours, base = ex["ndcg"]["shoprank"], ex["ndcg"]["bm25"]
        flag = '<span class="card-flag">fails</span>' if fails else ""
        selected = "true" if i == 0 else "false"
        cards.append(
            f'<a class="tab card{" is-failure" if fails else ""}" id="tab-{i}" href="#ex-{i}" role="tab" '
            f'aria-selected="{selected}" aria-controls="ex-{i}">'
            f'<span class="card-top"><span class="card-suit">{SUIT_NAMES[suit]}</span>{flag}</span>'
            f'<span class="card-query">{escape(ex.get("label") or short(ex["query"]))}</span>'
            f'<span class="card-grades"><span class="card-from">{base:.2f}</span>'
            f'<span class="card-arrow" aria-hidden="true">\u2192</span>'
            f'<span class="sr"> keyword search, then ShopRank </span>'
            f'<span class="card-to">{ours:.2f}</span></span>'
            f'<span class="card-story">{escape(ex.get("headline", ""))}</span></a>')
    return f'<nav class="tabs cards" role="tablist" aria-label="Example searches">{"".join(cards)}</nav>'


def results_table() -> str:
    rows = []
    for name, value, top1, note in RESULTS:
        width = (value - SCALE_MIN) / (SCALE_MAX - SCALE_MIN) * 100
        cls = " is-ours" if note == "this page" else ""
        rows.append(f'<tr class="result{cls}"><th scope="row">{escape(name)}</th>'
                    f'<td class="value">{value:.3f}</td><td class="value">{top1:.0%}</td>'
                    f'<td class="track" aria-hidden="true"><span class="bar" style="--w:{width:.1f}%"></span></td></tr>')
    return ('<table class="results"><caption class="sr">Ranking score on Amazon&#39;s official test</caption>'
            '<thead><tr><th scope="col">Method</th><th scope="col">Score</th><th scope="col">Right at #1</th>'
            '<th scope="col" class="track-head" aria-hidden="true"><span>0.80</span><span>0.85</span><span>0.90</span>'
            f'</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')


def proof_line(ex: dict) -> str:
    """One-sentence version of the first example, built from its data, for the first screen."""
    bm25_first = next((i for i, x in enumerate(ex["bm25"], 1) if x["label"] == "Exact"), None)
    lead = 0
    for item in ex["shoprank"]:
        if item["label"] != "Exact":
            break
        lead += 1
    where = (f"the first right product is number {bm25_first}"
             + (f" (below the {ROWS_SHOWN} shown)" if bm25_first > ROWS_SHOWN else "") if bm25_first
             else "no right product appears at all")
    return (f'<p class="proof">Example: search <q class="q">{escape(ex["query"])}</q>. In keyword search, {where}. '
            f'ShopRank puts {lead} right product{"" if lead == 1 else "s"} first. <a class="nowrap" href="#ex-0">See both lists</a></p>')


def known_item_table() -> str:
    rows = "".join(f'<tr><th scope="row">{label}<span class="count">{n:,} searches</span></th>'
                   f'<td class="value">{bm:.0%}</td><td class="value ours">{sr:.0%}</td></tr>'
                   for label, n, bm, sr in KNOWN_ITEM)
    return ('<table class="results known"><caption class="sr">Right product at number 1, by kind of search</caption>'
            '<thead><tr><th scope="col">Kind of search</th><th scope="col">Keyword search</th>'
            f'<th scope="col">ShopRank</th></tr></thead><tbody>{rows}</tbody></table>')


def legend() -> str:
    items = "".join(f'<li>{mark_svg(k)}<span>{v}</span></li>' for k, v in MARK_TEXT.items())
    return f'<ul class="legend" aria-label="Marks are Amazon\'s human labels">{items}</ul>'


GITHUB_ICON = ('<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><path d="M8 0C3.58 0 0 3.58 0 8c0 '
               '3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-'
               '.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51'
               '-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18'
               ' 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15'
               ' 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 '
               '0016 8c0-4.42-3.58-8-8-8z"/></svg>')

FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
         "&family=Newsreader:ital,wght@0,400..600;1,400&display=swap")


def page(data: dict, css: str, js: str) -> str:
    examples = data["examples"]
    panels = "".join(example_panel(ex, i) for i, ex in enumerate(examples))
    failure = next((i for i, ex in enumerate(examples) if ex["ndcg"]["shoprank"] < ex["ndcg"]["bm25"]), 0)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ShopRank: puts the right product first in shop search</title>
<meta name="description" content="A small AI model that re-orders shop search results so the right products come first: the top result is right in 75% of searches, against 63% for keyword search, on Amazon's public shopping benchmark. Real examples and code.">
<meta name="theme-color" content="#fbf8f1">
<link rel="canonical" href="{SITE}/">
<meta property="og:type" content="website">
<meta property="og:site_name" content="ShopRank">
<meta property="og:url" content="{SITE}/">
<meta property="og:title" content="ShopRank: puts the right product first in 3 of 4 shop searches">
<meta property="og:description" content="A small AI model that re-orders shop search results. Right product at #1 in 75% of searches vs 63% for keyword search, on Amazon's public shopping benchmark. Real examples and code.">
<meta property="og:image" content="{SITE}/og.png">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="627">
<meta property="og:image:alt" content="ShopRank: right product first in 75% of searches, against 63% for keyword search">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{SITE}/og.png">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>{css}</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="sheet">
<header class="topbar">
  <a class="brand" href="#top" id="top">ShopRank</a>
  <nav class="toplinks" aria-label="Project links"><a href="{REPO}">Code</a><a href="{MODEL}">Model files</a></nav>
</header>
<main id="main">
<section class="intro" aria-labelledby="headline">
  <h1 id="headline">Puts the right product first in 3 of 4 shop searches</h1>
  <div class="lede-block">
  <p class="score"><span class="pair"><span class="num num-ours">75%</span><span class="score-label">ShopRank</span></span><span class="pair"><span class="num num-base">63%</span><span class="score-label">plain keyword search</span></span></p>
  <p class="lede">Share of searches where the top result is exactly what was asked for, on Amazon's public shopping-search test (8,956 searches). ShopRank is a small AI model that re-orders search results so the products people want come first. It runs on a laptop CPU, no GPU needed.</p>
  {proof_line(examples[0])}
  </div>
  <div class="intro-side">
  <div class="actions">
      <a class="btn btn-primary" href="{REPO}">{GITHUB_ICON}See the code on GitHub</a>
      <a class="btn btn-ghost" href="{MODEL}">Get the model on Hugging Face</a>
    </div>
    <p class="skills">Python · PyTorch · FastAPI · Docker · ML ranking</p>
    <p class="byline">Built by <a href="{PORTFOLIO}">{OWNER}</a></p>
  </div>
</section>

<section aria-labelledby="examples-title">
  <h2 id="examples-title">See it work</h2>
  <p class="section-lede">Real searches for phones, consoles, laptops, earbuds and fridges that the model never saw while learning, hand-picked: clear wins and one honest failure. Each also shows two big ready-made AI rerankers on the same search. Over all 8,956 test searches, ShopRank ranks better than keyword search on 65% and worse on 33%. Ticks and crosses come from Amazon&#39;s human reviewers.</p>
  {legend()}
  {tabs(examples)}
  {panels}
</section>

<section aria-labelledby="results-title">
  <h2 id="results-title">The scores</h2>
  <p class="section-lede">Amazon's public shopping-search test: 8,956 searches, scored only once, at the very end, so the result could not be tuned to the test. Score is the ranking score (1.0 = perfect order); &ldquo;Right at #1&rdquo; is how often the top result is exactly right.</p>
  {results_table()}
</section>

<section aria-labelledby="known-title">
  <h2 id="known-title">Biggest gain on searches with a number in them</h2>
  <p class="section-lede">How often the top result is exactly right, by kind of search, on 2,001 development searches (ShopRank as served, with its spell checker). Searches with a number in them, such as a model number or a size, gain the most (a search can be in both of the first two rows).</p>
  {known_item_table()}
</section>

<section aria-labelledby="how-title">
  <h2 id="how-title">How it works</h2>
  <ol class="steps">
    <li><p><strong>Big teacher.</strong> Five DeBERTa models learn from Amazon's labelled searches. Accurate, but too slow for a normal computer.</p></li>
    <li><p><strong>Small student.</strong> A small model learns to copy the teachers and keeps most of their skill.</p></li>
    <li><p><strong>Compressed.</strong> Shrunk to 32&nbsp;MB with almost no loss, it ranks a typical list of 16 products in about 0.07 seconds (median) on a laptop CPU.</p></li>
    <li><p><strong>Typo fix.</strong> Typos hurt the ranking, so a spell checker fixes the search first. On made-up one-letter typos it wins back almost 90% of the accuracy that typos cost.</p></li>
  </ol>
</section>

<section aria-labelledby="limits-title">
  <h2 id="limits-title">Where it fails</h2>
  <ul class="limits">
    <li>Keyword search still ranks better on about a third of searches, though mostly by a little: in over half of those the difference is under 0.05 on the 1.0 scale (development set).</li>
    <li>Exact product names can trip it up: for <a href="#ex-{failure}">nintendo switch console</a> it put the cheaper Switch Lite first, while keyword search found the console.</li>
    <li>On long, chat-style requests the big ready-made models can rank better (see the Chromebook example).</li>
    <li>The spell checker sometimes &ldquo;fixes&rdquo; real names: <q>nivda</q> (Nvidia) became <q>nivea</q>.</li>
    <li>It does not search the whole catalogue: it re-orders the results another search returns, and only for English (US) searches.</li>
  </ul>
</section>

<details class="engineers">
  <summary>For engineers: method and exact numbers</summary>
  <ul>
    <li><strong>Task and metric.</strong> Amazon Shopping Queries (ESCI) Task 1, US. Full-list nDCG with gains Exact 1, Substitute 0.1, Complement 0.01, Irrelevant 0. Keyword baseline: BM25 on product titles.</li>
    <li><strong>Test discipline.</strong> Official test (8,956 queries) scored once after pre-registration; 95% query-bootstrap intervals and a trec_eval cross-check in the <a href="{REPO}#readme">README</a>. The examples above come from the dev split.</li>
    <li><strong>Teacher.</strong> 5-fold DeBERTa-v3-base cross-encoders, mean score: 0.8877 [0.8849, 0.8905].</li>
    <li><strong>Student.</strong> MiniLM-L12 (33M parameters) distilled with MSE on each pair's out-of-fold teacher score: 0.8674 [0.8643, 0.8703], +0.013 [+0.011, +0.015] over Amazon's MiniLM recipe re-run on the same queries (0.8544).</li>
    <li><strong>Serving.</strong> ONNX Runtime dynamic INT8 (32 MB, dev -0.0008 vs fp32), FastAPI in Docker; p50 69 ms / p95 232 ms per request (median 16 candidates) on a 16-thread laptop CPU.</li>
    <li><strong>Typos.</strong> One synthetic typo costs 0.028 dev nDCG. A one-edit spell corrector over the catalogue vocabulary recovers 87-90% on dev and dev2 with no measurable loss on clean queries; typo-augmented training recovered 14% and was rejected.</li>
    <li><strong>Ready-made rerankers.</strong> Zero-shot BGE-reranker-v2-m3 (568M) 0.8585 and Qwen3-Reranker-0.6B 0.8540 on dev, both below the served model (paired CIs exclude 0).</li>
    <li><strong>Read more.</strong> <a href="{REPO}/blob/main/docs/MODEL_CARD.md">Model card</a> · <a href="{REPO}/blob/main/docs/lab_notebook.md">Lab notebook</a> · <a href="{REPO}">Code</a></li>
  </ul>
</details>
</main>

<footer>
  <p>Built by <a href="{PORTFOLIO}">{OWNER}</a> · <a href="{GITHUB_PROFILE}">GitHub</a> · <a href="{REPO}">ShopRank code</a> · <a href="{MODEL}">Model</a></p>
  <p>Data: <a href="{DATASET}">Shopping Queries Dataset</a> (Reddy et al., 2022), &copy; Amazon.com, Inc. or its affiliates, Apache-2.0. Not affiliated with or endorsed by Amazon. Examples come from the development split; the official test set was scored once.</p>
</footer>
</div>
<script>{js}</script>
</body>
</html>
"""


def csp_headers(js: str) -> dict:
    """Vercel headers: a strict Content-Security-Policy that allows exactly this page's inline script."""
    digest = base64.b64encode(hashlib.sha256(js.encode("utf-8")).digest()).decode("ascii")
    csp = ("default-src 'self'; "
           f"script-src 'sha256-{digest}'; "
           "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
           "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
           "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
    return {"headers": [{"source": "/(.*)", "headers": [
        {"key": "Content-Security-Policy", "value": csp},
        {"key": "X-Content-Type-Options", "value": "nosniff"},
        {"key": "Referrer-Policy", "value": "strict-origin-when-cross-origin"},
        {"key": "Permissions-Policy", "value": "camera=(), microphone=(), geolocation=()"},
    ]}]}


def build(examples_path: Path = EXAMPLES_PATH, src: Path = Path("site/src")) -> tuple[str, dict]:
    data = json.loads(examples_path.read_text(encoding="utf-8"))
    css = (src / "style.css").read_text(encoding="utf-8")
    js = (src / "app.js").read_text(encoding="utf-8")
    return page(data, css, js), csp_headers(js)


def main() -> None:
    html, headers = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8", newline="\n")
    (OUT_PATH.parent / "vercel.json").write_text(json.dumps(headers, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{OUT_PATH} ({len(html.encode('utf-8')) / 1024:.0f} KB) and vercel.json written")


if __name__ == "__main__":
    main()
