"""Zero-shot modern rerankers (BGE-reranker-v2-m3, Qwen3-Reranker-0.6B) on the DEV split only.

Usage:
    uv run python -m shoprank.eval.baseline_rerankers --model bge
    uv run python -m shoprank.eval.baseline_rerankers --model qwen3

Neither model sees ESCI labels. Input is the product text the served model uses (title, brand, color,
bullets). Output per model: data/processed/runs/zeroshot_<short>_dev.json and .parquet (per-query nDCG).
Only DEV is read; test and dev2 are never touched.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from shoprank.eval.bootstrap import paired_bootstrap_ci
from shoprank.features.build import PROCESSED

RUNS = PROCESSED / "runs"
FIELDS = ["title", "brand", "color", "bullets"]
SERVED_RUN = RUNS / "minilm_distil_w10_bullets.parquet"
TEACHER_RUN = RUNS / "deberta_oof_avg5.parquet"

QWEN_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
QWEN_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
               'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN_DOC_CHARS = 1200  # cut the product text ourselves so tokenizer truncation can never remove QWEN_SUFFIX

MODELS = {
    "bge": {"name": "BAAI/bge-reranker-v2-m3", "short": "bge_v2m3", "max_length": 256, "batch": 64},
    "qwen3": {"name": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls", "short": "qwen3_06b", "max_length": 512,
              "batch": 32},
}


def qwen_query(query: str, instruction: str = QWEN_INSTRUCTION) -> str:
    return f"{QWEN_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n"


def qwen_document(text: str, max_chars: int = QWEN_DOC_CHARS) -> str:
    return f"<Document>: {text[:max_chars]}{QWEN_SUFFIX}"


def length_order(pairs: Sequence[tuple[str, str]]) -> np.ndarray:
    """Indices sorting pairs by total character length, so batches hold similar lengths (little padding)."""
    return np.argsort([len(q) + len(t) for q, t in pairs], kind="stable")


def score_sorted(pairs: Sequence[tuple[str, str]], predict: Callable[[list, int], np.ndarray],
                 batch_size: int, chunk: int = 2048, oom: type[BaseException] = RuntimeError) -> np.ndarray:
    """Score pairs in length order, return scores in the original order.
    `predict(chunk_pairs, batch_size)` gives one score per pair. On `oom` the batch size is halved and
    the chunk retried; at batch size 1 the error is raised."""
    order = length_order(pairs)
    out = np.empty(len(pairs), dtype=float)
    for start in range(0, len(pairs), chunk):
        idx = order[start:start + chunk]
        part = [pairs[i] for i in idx]
        while True:
            try:
                out[idx] = np.asarray(predict(part, batch_size), dtype=float)
                break
            except oom:
                if batch_size <= 1:
                    raise
                batch_size //= 2
    return out


def compare(candidate: pd.DataFrame, reference: pd.DataFrame, column: str = "ndcg_full") -> dict:
    """Paired bootstrap of candidate - reference on the queries both have, aligned on query_id."""
    merged = candidate[["query_id", column]].merge(reference[["query_id", column]], on="query_id",
                                                   suffixes=("_c", "_r")).sort_values("query_id")
    if len(merged) != len(candidate) or len(merged) != len(reference):
        raise ValueError(f"query_id sets differ: {len(candidate)} vs {len(reference)}, {len(merged)} shared")
    mean, low, high = paired_bootstrap_ci(merged[f"{column}_r"], merged[f"{column}_c"])
    return {"delta": mean, "low": low, "high": high}


def pinned_revision(model_name: str) -> str:
    """Commit of the cached snapshot (refs/main), i.e. what was actually downloaded."""
    ref = Path.home() / ".cache/huggingface/hub" / f"models--{model_name.replace('/', '--')}" / "refs/main"
    return ref.read_text().strip()


def build_inputs(key: str, pairs: pd.DataFrame, texts: list[str]) -> list[tuple[str, str]]:
    queries = pairs["query"].tolist()
    if key == "qwen3":
        return [(qwen_query(q), qwen_document(t)) for q, t in zip(queries, texts, strict=True)]
    return list(zip(queries, texts, strict=True))


def run(key: str) -> dict:
    import torch
    from sentence_transformers import CrossEncoder

    from shoprank.cross.zero_shot import product_texts, read_products
    from shoprank.features.build import load_split
    from shoprank.ltr.train import per_query_results

    cfg = MODELS[key]
    pairs, _ = load_split("dev")
    products = read_products(FIELDS, pairs["product_id"])
    inputs = build_inputs(key, pairs, product_texts(pairs, products, FIELDS))
    del products

    model = CrossEncoder(cfg["name"], max_length=cfg["max_length"], device="cuda",
                         model_kwargs={"torch_dtype": torch.float16})
    n_params = sum(p.numel() for p in model.model.parameters())
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    scores = score_sorted(inputs, lambda part, bs: model.predict(part, batch_size=bs, show_progress_bar=False,
                                                                 convert_to_numpy=True),
                          cfg["batch"], oom=torch.cuda.OutOfMemoryError)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    RUNS.mkdir(parents=True, exist_ok=True)
    np.save(RUNS / f"zeroshot_{cfg['short']}_dev_scores.npy", scores)  # keep the GPU work if a later step fails
    per_q = pd.DataFrame(per_query_results(pairs.assign(score=scores), "score")).sort_values("query_id")
    per_q = per_q.reset_index(drop=True)
    result = {"model": cfg["name"], "revision": pinned_revision(cfg["name"]), "params": int(n_params),
              "max_length": cfg["max_length"], "fields": FIELDS, "split": "dev", "pairs": len(pairs),
              "queries": len(per_q), "ndcg_full": float(per_q["ndcg_full"].mean()),
              "ndcg10": float(per_q["ndcg10"].mean()), "seconds": round(seconds, 1),
              "pairs_per_s": round(len(pairs) / seconds, 1), "peak_vram_gb": round(peak_gb, 2)}
    if key == "qwen3":
        result["instruction"] = QWEN_INSTRUCTION
        result["doc_chars"] = QWEN_DOC_CHARS
    for label, path in (("vs_served", SERVED_RUN), ("vs_teacher", TEACHER_RUN)):
        result[label] = compare(per_q, pd.read_parquet(path))
    per_q[["query_id", "ndcg10", "ndcg_full"]].to_parquet(RUNS / f"zeroshot_{cfg['short']}_dev.parquet", index=False)
    (RUNS / f"zeroshot_{cfg['short']}_dev.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Zero-shot rerankers on DEV")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    print(json.dumps(run(p.parse_args().model), indent=2))


if __name__ == "__main__":
    main()
