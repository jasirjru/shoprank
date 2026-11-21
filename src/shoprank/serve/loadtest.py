"""Section 15: load test of the ranking API with real dev queries and their real candidate lists.

Usage (server already running, e.g. `uv run uvicorn shoprank.serve.app:app --port 8000`):
    uv run python -m shoprank.serve.loadtest --label int8 --concurrency 1 4 16 --requests 300

Each request = one dev query with all its candidates (median ~20 products), exactly what the API is for.
Per concurrency level: 20 warm-up requests (not timed), then N timed requests sent by C parallel clients.
Reports client-side latency p50 / p95 / p99 (successful requests only), requests/s and pairs/s; writes
data/processed/serving/loadtest_<label>.json.
Limits of this measurement: it is closed-loop (each client sends its next request only after the reply), so at
high concurrency the percentiles describe a saturated server, not latency at a fixed arrival rate; client and
server share the laptop's CPU; with 300 requests, p99 rests on the 3 slowest.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np
import pandas as pd

from shoprank.features.build import PROCESSED, load_split

OUT = PROCESSED / "serving"
WARMUP = 20
FIELDS = ("title", "brand", "color", "bullets")
TIMEOUT_S = 60.0


def build_requests(split: str = "dev", seed: int = 0) -> list[dict]:
    """One request body per query, products in a shuffled order (a server must not rely on input order)."""
    pairs, _ = load_split(split)
    products = pd.read_parquet(PROCESSED / "products.parquet", columns=["product_id", *FIELDS])
    merged = pairs.merge(products, on="product_id", how="left", validate="many_to_one")
    merged = merged.astype(object).where(merged.notna(), None)
    rng = np.random.default_rng(seed)
    bodies = []
    for _, g in merged.groupby("query_id", sort=True):
        rows = g.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        bodies.append({"query": rows["query"].iloc[0], "products": [
            {"product_id": r["product_id"], "title": r["title"] or "(no title)",
             **{f: (str(r[f])[:5000] if r[f] is not None else None) for f in FIELDS[1:]}}
            for r in rows.to_dict("records")]})
    order = rng.permutation(len(bodies))
    return [bodies[i] for i in order]


def percentiles(latencies_ms: list[float]) -> dict:
    a = np.asarray(latencies_ms)
    return {"p50_ms": round(float(np.percentile(a, 50)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1),
            "p99_ms": round(float(np.percentile(a, 99)), 1), "mean_ms": round(float(a.mean()), 1)}


async def run_level(url: str, bodies: list[dict], concurrency: int, n: int) -> dict:
    import httpx

    async with httpx.AsyncClient(base_url=url, timeout=TIMEOUT_S) as client:
        for b in bodies[:WARMUP]:
            (await client.post("/rank", json=b)).raise_for_status()
        queue = list(bodies[WARMUP:WARMUP + n])
        latencies, errors, pairs = [], 0, 0

        async def worker() -> None:
            nonlocal errors, pairs
            while queue:
                b = queue.pop()
                start = time.perf_counter()
                try:
                    r = await client.post("/rank", json=b)
                except httpx.HTTPError:
                    errors += 1
                    continue
                if r.status_code != 200:  # fast errors must not flatter the latency or throughput
                    errors += 1
                    continue
                latencies.append((time.perf_counter() - start) * 1000)
                pairs += len(b["products"])

        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - start
    if not latencies:
        raise RuntimeError(f"all {errors} requests failed at concurrency {concurrency}")
    return {"concurrency": concurrency, "requests": len(latencies), "errors": errors, **percentiles(latencies),
            "requests_per_s": round(len(latencies) / wall, 1), "pairs_per_s": round(pairs / wall)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Load test POST /rank with real dev queries")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--requests", type=int, default=300)
    args = ap.parse_args()

    bodies = build_requests()
    sizes = [len(b["products"]) for b in bodies]
    print(f"{len(bodies):,} dev queries; candidates per request: median {int(np.median(sizes))}, max {max(sizes)}")
    results = []
    for c in args.concurrency:
        res = asyncio.run(run_level(args.url, bodies, c, args.requests))
        results.append(res)
        print(f"{args.label} c={c:<3} p50 {res['p50_ms']:>7.1f} ms  p95 {res['p95_ms']:>7.1f}  p99 {res['p99_ms']:>7.1f}"
              f"  {res['requests_per_s']:>6.1f} req/s  {res['pairs_per_s']:>6,} pairs/s  errors {res['errors']}",
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"loadtest_{args.label}.json").write_text(json.dumps(
        {"label": args.label, "url": args.url, "median_candidates": int(np.median(sizes)), "levels": results},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
