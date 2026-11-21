"""Section 17: Gradio demo of the served re-ranker (the same INT8 ONNX model and input checks as POST /rank).

Run (needs the demo dependency group):
    uv sync --group demo
    uv run --group demo python -m shoprank.serve.demo        # http://127.0.0.1:7860

Type a query and one product per line. A line is `title`, or `title | brand | colour | bullets` (any later
part may be left out). The demo checks the input with the API's own RankRequest rules, scores every product
with the model, and shows the list best first. Only local use is intended: launch() binds to 127.0.0.1 and
does not create a public share link.
"""
from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from pydantic import ValidationError

from shoprank.serve.app import RankRequest, Scorer, scorer_from_env
from shoprank.serve.ranker import serving_text

LINE_FIELDS = ("title", "brand", "color", "bullets")
SEPARATOR = "|"
HOST = "127.0.0.1"
PORT = 7860
EXAMPLES = [
    ["usb c cable 6ft", ("Anker USB C to USB C cable, 6 ft, 100W fast charging | Anker | Black\n"
                         "HDMI cable 10 ft, 4K | AmazonBasics\n"
                         "USB C cable 3 ft, 2 pack | Amazon Basics | White\n"
                         "USB C wall charger 20W | Apple | White")],
    ["running shoes without laces", ("Slip-on running shoes, no laces, breathable mesh | Skechers | Black\n"
                                     "Running shoes with reflective laces | Nike | Blue\n"
                                     "Elastic no-tie shoelaces for sneakers | Lock Laces\n"
                                     "Trail running shoes, quick-lace system | Salomon | Grey")],
]


def parse_products(text: str) -> list[dict]:
    """One product per non-empty line: 'title | brand | colour | bullets'; ids are p1, p2, ... in line order."""
    products = []
    for line in (ln for ln in text.splitlines() if ln.strip()):
        parts = [p.strip() for p in line.split(SEPARATOR, len(LINE_FIELDS) - 1)]
        product = {f: (parts[i] or None) if i < len(parts) else None for i, f in enumerate(LINE_FIELDS)}
        product["title"] = product["title"] or ""  # blank title is rejected by RankRequest with a clear message
        products.append({"product_id": f"p{len(products) + 1}", **product})
    return products


def validation_message(err: ValidationError) -> str:
    first = err.errors()[0]
    where = ".".join(str(p) for p in first["loc"])
    return f"{where}: {first['msg']}"


def rank_rows(scorer: Scorer, query: str, products_text: str) -> tuple[list[list], str]:
    """Rows [rank, score, title, brand, colour] best first, and a one-line status. Raises ValueError on bad input."""
    try:
        req = RankRequest(query=query, products=parse_products(products_text))
    except ValidationError as err:
        raise ValueError(validation_message(err)) from None
    start = time.perf_counter()
    texts = [serving_text(p.model_dump(), scorer.fields) for p in req.products]
    scores = np.asarray(scorer.score([req.query] * len(texts), texts), dtype=float)
    ms = (time.perf_counter() - start) * 1000
    order = sorted(range(len(texts)), key=lambda i: (-scores[i], req.products[i].product_id))
    rows = [[rank, round(float(scores[i]), 3), req.products[i].title, req.products[i].brand or "",
             req.products[i].color or ""] for rank, i in enumerate(order, start=1)]
    return rows, f"{len(rows)} products scored in {ms:.0f} ms by {scorer.name} (CPU)"


def build_demo(scorer: Scorer, examples: Sequence[Sequence[str]] = EXAMPLES):
    import gradio as gr

    def run(query: str, products_text: str) -> tuple[list[list], str]:
        try:
            return rank_rows(scorer, query, products_text)
        except ValueError as err:
            raise gr.Error(str(err)) from None

    with gr.Blocks(title="ShopRank") as demo:
        gr.Markdown("# ShopRank: product search re-ranking\n"
                    "A cross-encoder (MiniLM distilled from DeBERTa, INT8 on CPU) scores how well each product "
                    "matches the query. Official ESCI test nDCG 0.867. Scores rank products within one query; "
                    "they are not probabilities.")
        query = gr.Textbox(label="Query", placeholder="e.g. usb c cable 6ft", max_length=200)
        products = gr.Textbox(label="Products, one per line: title | brand | colour | bullets", lines=8)
        button = gr.Button("Rank", variant="primary")
        table = gr.Dataframe(headers=["rank", "score", "title", "brand", "colour"], label="Ranked, best first",
                             interactive=False, wrap=True)
        status = gr.Markdown()
        gr.Examples(examples=[list(e) for e in examples], inputs=[query, products])
        button.click(run, inputs=[query, products], outputs=[table, status], concurrency_limit=1)
        query.submit(run, inputs=[query, products], outputs=[table, status], concurrency_limit=1)
    return demo


def main() -> None:
    build_demo(scorer_from_env()).launch(server_name=HOST, server_port=PORT, share=False)


if __name__ == "__main__":
    main()
