# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack
Delegated (21 Nov 2026: the owner asked for "the best engine"). Chosen: static HTML and CSS, generated from the
project's real data by a small Python script in this repo (`site/`), hosted on Vercel. Reason: one page needs no
framework; zero runtime dependencies gives the fastest load, nothing to sleep or break, and code the owner can
explain. A later live demo may run the INT8 ONNX model in the browser (ONNX Runtime Web).

## Users
1. A recruiter or HR screener with about 30 seconds, usually arriving from a resume or LinkedIn link, often on
   a phone. Job: decide whether this candidate is worth a call.
2. An ML hiring manager or engineer with about 3 minutes, usually on a laptop. Job: judge whether the work is
   real and sound (baselines, held-out evaluation, honest limits) before an interview.

## Product Purpose
ShopRank re-ranks Amazon product-search results so the right products come first. The public page proves that
it works on real data and shows the engineering judgement behind it. Success: a recruiter can say what it does
and that it beats keyword search after one screen; an engineer finds the evidence (numbers, code, model) within
three minutes.

## Positioning
A small model (33M parameters, 32 MB INT8, CPU) distilled from a larger DeBERTa ensemble, measured honestly
against keyword search (BM25) and against a re-run of Amazon's published baseline, with a spell-correction fix
found by stress tests. Every number comes from a held-out split; the official test set was scored once.

## Operating Context
Visitors arrive from a resume, LinkedIn, GitHub or the portfolio site. The page links to the GitHub repo
(github.com/jasirjru/shoprank), the Hugging Face model (huggingface.co/jasirjru/shoprank-reranker-minilm-int8)
and the owner's portfolio. Owner's name may appear; no phone or email (inferred default; the owner did not
answer the privacy question).

## Capabilities and Constraints
- Six real examples from the dev split, exported by `src/shoprank/serve/showcase.py` from the real API
  (`site/data/examples.json`), parity-checked against the API.
- Test numbers (official ESCI Task 1 test, US, 8,956 queries, full-list nDCG): DeBERTa x5 0.8877, served
  MiniLM 0.8674, Amazon recipe re-run 0.8544, LambdaMART 0.8337, BM25 0.8202. Dev: INT8 0.8654.
- Serving: p50 69 ms per 16-product query on a laptop CPU, 32 MB.
- No live model on the page yet (no server); a live demo is a later step.
- No Amazon images or branding; the dataset (Shopping Queries Dataset, Apache-2.0) has no images. Keep the
  licence link and the Amazon copyright notice; do not imply endorsement.

## Evidence on Hand
README.md, docs/MODEL_CARD.md, docs/lab_notebook.md, site/data/examples.json, docs/website/research.md.
No testimonials, users or traffic exist; never invent them.

## Product Principles
1. Show, then claim: a real side-by-side ranking comes before any number.
2. Every number names its split and its baseline.
3. Show one honest failure on purpose.
4. Plain words first; jargon gets a one-line meaning.
5. Nothing on the page may wait on a server.

## Accessibility & Inclusion
WCAG 2.2 AA. Relevance labels must not rely on colour alone. Phone-first.
