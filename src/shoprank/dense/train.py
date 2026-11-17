"""Train the two-tower model on TRAIN (query, Exact product) pairs; evaluate on DEV. Week 3.

Usage:
    uv run python -m shoprank.dense.train --name twotower_b256
    uv run python -m shoprank.dense.train --batch-size 32 --name twotower_b32
    uv run python -m shoprank.dense.train --overfit-one-batch       # sanity check: loss must go ~0
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import duckdb
import torch

from shoprank.dense.hashing import pack, text_ids
from shoprank.dense.model import TwoTower, info_nce
from shoprank.eval.run_task1 import save
from shoprank.features.build import PROCESSED, load_split
from shoprank.ltr.train import per_query_results

MODELS = PROCESSED / "models"
SEED = 0


def product_text(p: dict) -> str:
    return f"{p.get('title') or ''} {p.get('brand') or ''}"


def load_positives() -> tuple[list[int], list[list[int]], list[list[int]]]:
    """(query_id, query ids, product ids) for every TRAIN Task-1 pair labelled Exact."""
    rows = duckdb.sql(f"""
        SELECT j.query_id, j.query, p.title, p.brand
        FROM '{(PROCESSED / "judgments.parquet").as_posix()}' j
        JOIN '{(PROCESSED / "products.parquet").as_posix()}' p USING (product_id)
        WHERE j.small_version = 1 AND j.split = 'train' AND j.esci_label = 'E'
        ORDER BY j.query_id, j.product_id
    """).fetchall()
    cache: dict[str, list[int]] = {}

    def ids(text: str) -> list[int]:
        if text not in cache:
            cache[text] = text_ids(text)
        return cache[text]
    qids = [r[0] for r in rows]
    queries = [ids(r[1]) for r in rows]
    products = [ids(product_text({"title": r[2], "brand": r[3]})) for r in rows]
    return qids, queries, products


def train(args: argparse.Namespace, device: torch.device) -> TwoTower:
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    qids, queries, products = load_positives()
    order = list(range(len(qids)))
    if args.overfit_one_batch:
        order = order[: args.batch_size]
    print(f"{len(order):,} positive pairs | batch {args.batch_size} | temperature {args.temperature}")

    model = TwoTower(args.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    epochs = 200 if args.overfit_one_batch else args.epochs
    start = time.time()
    for epoch in range(epochs):
        rng.shuffle(order)
        total, steps = 0.0, 0
        for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
            batch = order[i:i + args.batch_size]
            q = model.query(*pack([queries[j] for j in batch], device))
            p = model.product(*pack([products[j] for j in batch], device))
            bq = torch.tensor([qids[j] for j in batch], device=device)
            loss = info_nce(q, p, args.temperature, same_query=bq[:, None] == bq[None, :])
            opt.zero_grad()      # forget the previous step's gradients
            loss.backward()      # autograd: same idea as our scratch engine, on GPU tensors
            opt.step()           # AdamW update
            total += loss.item()
            steps += 1
        if not args.overfit_one_batch or epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch + 1}: mean loss {total / steps:.4f}")
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    print(f"trained in {time.time() - start:.0f}s | peak GPU memory {peak:.0f} MB")
    return model


@torch.no_grad()
def evaluate(model: TwoTower, device: torch.device) -> list[dict]:
    model.eval()
    pairs, products = load_split("dev")
    pids = sorted(products)
    pvec = {}
    for i in range(0, len(pids), 4096):
        chunk = pids[i:i + 4096]
        out = model.product(*pack([text_ids(product_text(products[p])) for p in chunk], device))
        pvec.update(zip(chunk, out, strict=True))
    queries = pairs.drop_duplicates("query_id")
    qvec = dict(zip(queries["query_id"],
                    model.query(*pack([text_ids(t) for t in queries["query"]], device)), strict=True))
    scores = [float(qvec[q] @ pvec[p]) for q, p in zip(pairs["query_id"], pairs["product_id"], strict=True)]
    return per_query_results(pairs.assign(score=scores), "score")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the two-tower model")
    ap.add_argument("--name", default="twotower")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--overfit-one-batch", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train(args, device)
    if args.overfit_one_batch:
        return
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(MODELS) / f"{args.name}.pt")
    config = {k: v for k, v in vars(args).items() if k != "overfit_one_batch"}
    summary = save(args.name, "dev", evaluate(model, device), {"model": "two_tower", **config})
    print(f"{args.name}: nDCG full {summary['ndcg_full']:.4f}   nDCG@10 {summary['ndcg10']:.4f}")


if __name__ == "__main__":
    main()
