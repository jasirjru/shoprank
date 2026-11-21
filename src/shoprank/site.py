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

# Official ESCI Task 1 test (US, 8,956 queries), full-list nDCG, scored once (README.md).
RESULTS = [
    ("Keyword search", 0.8202, "baseline"),
    ("Classic ranking with hand-made rules", 0.8337, ""),
    ("Amazon's published method (my re-run)", 0.8544, "published baseline"),
    ("ShopRank, this model", 0.8674, "this page"),
    ("Big teacher model (needs a GPU)", 0.8877, "offline"),
]
SCALE_MIN, SCALE_MAX = 0.80, 0.90

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


def tabs(examples: list[dict]) -> str:
    buttons = []
    for i, ex in enumerate(examples):
        flag = ' <span class="tab-flag">fails</span>' if ex["ndcg"]["shoprank"] < ex["ndcg"]["bm25"] else ""
        selected = "true" if i == 0 else "false"
        buttons.append(f'<a class="tab" id="tab-{i}" href="#ex-{i}" role="tab" aria-selected="{selected}" '
                       f'aria-controls="ex-{i}">{escape(ex["query"])}{flag}</a>')
    return f'<nav class="tabs" role="tablist" aria-label="Example searches">{"".join(buttons)}</nav>'


def results_table() -> str:
    rows = []
    for name, value, note in RESULTS:
        width = (value - SCALE_MIN) / (SCALE_MAX - SCALE_MIN) * 100
        cls = " is-ours" if note == "this page" else ""
        rows.append(f'<tr class="result{cls}"><th scope="row">{escape(name)}</th>'
                    f'<td class="value">{value:.3f}</td>'
                    f'<td class="track" aria-hidden="true"><span class="bar" style="--w:{width:.1f}%"></span></td></tr>')
    return ('<table class="results"><caption class="sr">Ranking score on Amazon&#39;s official test</caption>'
            '<thead><tr><th scope="col">Method</th><th scope="col">Score</th>'
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
    where = (f"the first right product is number {bm25_first}" if bm25_first
             else "no right product appears at all")
    return (f'<p class="proof">Example: search <q class="q">{escape(ex["query"])}</q>. In keyword search, {where}. '
            f'ShopRank puts {lead} right product{"" if lead == 1 else "s"} first. <a href="#ex-0">See both lists</a></p>')


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
         "&family=Newsreader:ital,opsz,wght@0,6..72,400..600;1,6..72,400&display=swap")


def page(data: dict, css: str, js: str) -> str:
    examples = data["examples"]
    panels = "".join(example_panel(ex, i) for i, ex in enumerate(examples))
    failure = next((i for i, ex in enumerate(examples) if ex["ndcg"]["shoprank"] < ex["ndcg"]["bm25"]), 0)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ShopRank: puts the right product first in Amazon product search</title>
<meta name="description" content="A 32 MB re-ranking model that beats keyword search on Amazon's Shopping Queries benchmark: nDCG 0.867 vs 0.820 on the official test set. Real examples, numbers and code.">
<meta name="theme-color" content="#fbf8f1">
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
  <nav class="toplinks" aria-label="Project links"><a href="{REPO}">Code</a><a href="{MODEL}">Model</a></nav>
</header>
<main id="main">
<section class="intro" aria-labelledby="headline">
  <h1 id="headline">Puts the right product first in shop search</h1>
  <div class="lede-block">
  <p class="score"><span class="pair"><span class="num num-ours">0.867</span><span class="score-label">ShopRank</span></span><span class="pair"><span class="num num-base">0.820</span><span class="score-label">keyword search</span></span></p>
  <p class="lede">Ranking score on Amazon's official test, where 1.0 is a perfect order. ShopRank is a small AI model that re-orders search results so the products people want come first. It runs on a normal laptop.</p>
  {proof_line(examples[0])}
  </div>
  <div class="actions">
    <a class="btn btn-primary" href="{REPO}">{GITHUB_ICON}See the code on GitHub</a>
    <a class="btn btn-ghost" href="{MODEL}">Get the model on Hugging Face</a>
  </div>
  <p class="skills">Python · PyTorch · FastAPI · Docker · ML ranking</p>
  <p class="byline">Built by <a href="{PORTFOLIO}">{OWNER}</a></p>
</section>

<section aria-labelledby="examples-title">
  <h2 id="examples-title">See it work</h2>
  <p class="section-lede">Real searches the model never saw while learning. Same products, two orders. The marks are Amazon's human labels.</p>
  {legend()}
  {tabs(examples)}
  {panels}
</section>

<section aria-labelledby="results-title">
  <h2 id="results-title">The scores</h2>
  <p class="section-lede">Amazon's official test: 8,956 searches, checked once at the end. Higher is better.</p>
  {results_table()}
</section>

<section aria-labelledby="how-title">
  <h2 id="how-title">How it works</h2>
  <ol class="steps">
    <li><p><strong>Big teacher.</strong> Five large models learn from Amazon's labelled searches. Accurate, but too slow for a normal computer.</p></li>
    <li><p><strong>Small student.</strong> A small model learns to copy the teachers and keeps most of their skill.</p></li>
    <li><p><strong>Compressed.</strong> Shrunk to 32&nbsp;MB, it ranks 16 products in 0.07 seconds on a laptop.</p></li>
    <li><p><strong>Typo fix.</strong> Typos hurt the ranking, so a spell checker fixes the search first. It wins back almost 90% of the loss.</p></li>
  </ol>
</section>

<section aria-labelledby="limits-title">
  <h2 id="limits-title">Where it fails</h2>
  <ul class="limits">
    <li>Exact model names can trip it up: for <a href="#ex-{failure}">m23 for cars</a> it showed stickers, not McLaren M23 model cars.</li>
    <li>The spell checker sometimes &ldquo;fixes&rdquo; real names: <q>nivda</q> (Nvidia) became <q>nivea</q>.</li>
    <li>It only re-orders a list it is given, and only for English (US) searches.</li>
  </ul>
</section>

<details class="engineers">
  <summary>For engineers: method and exact numbers</summary>
  <ul>
    <li><strong>Task and metric.</strong> Amazon Shopping Queries (ESCI) Task 1, US. Full-list nDCG with gains Exact 1, Substitute 0.1, Complement 0.01, Irrelevant 0. Keyword baseline: BM25 on product titles.</li>
    <li><strong>Test discipline.</strong> Official test (8,956 queries) scored once after pre-registration; 95% query-bootstrap intervals and a trec_eval cross-check in the <a href="{REPO}#readme">README</a>. The examples above come from the dev split.</li>
    <li><strong>Teacher.</strong> 5-fold DeBERTa-v3-base cross-encoders, mean score: 0.8877 [0.8849, 0.8905].</li>
    <li><strong>Student.</strong> MiniLM-L12 (33M parameters) distilled with MSE on each pair's out-of-fold teacher score: 0.8674 [0.8643, 0.8703], +0.013 [+0.011, +0.015] over Amazon's MiniLM recipe re-run on the same queries (0.8544).</li>
    <li><strong>Serving.</strong> ONNX Runtime dynamic INT8 (32 MB, dev -0.0008 vs fp32), FastAPI in Docker; p50 69 ms / p95 232 ms per 16-candidate request on a 16-thread laptop CPU.</li>
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
