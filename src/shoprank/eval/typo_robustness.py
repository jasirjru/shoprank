"""Section 18: typo robustness of two groups of models ("arms"), each trained with several seeds.

Usage:
    uv run python -m shoprank.eval.typo_robustness --split dev \
        --base minilm_distil_w10_bullets minilm_distil_w10_bullets_s43 minilm_distil_w10_bullets_s44 \
        --treat minilm_distil_typo30_s42 minilm_distil_typo30_s43 minilm_distil_typo30_s44

Every model scores the split three times: clean queries and two fixed typo versions (`cross.typos.eval_typo`,
one mistake in the longest word: "transpose" and "neighbour"). Per query, nDCG is averaged over the seeds of an
arm; arms are compared with a paired bootstrap over queries. The test split is refused (it is spent).
Writes data/processed/runs/typo_robustness_<split>.json.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from shoprank.cross.typos import eval_typo
from shoprank.eval.bootstrap import paired_bootstrap_ci
from shoprank.features.build import PROCESSED, load_split
from shoprank.ltr.train import per_query_results

CONDITIONS = ("clean", "typo_transpose", "typo_neighbour")
ALLOWED_SPLITS = ("dev", "dev2")


def condition_queries(pairs: pd.DataFrame, condition: str) -> list[str]:
    """The query text for each pair under `condition`; the typo depends only on (query, query_id)."""
    if condition == "clean":
        return pairs["query"].tolist()
    kind = condition.removeprefix("typo_")
    per_query = {qid: eval_typo(q, kind, key=qid) for qid, q in zip(pairs["query_id"], pairs["query"], strict=True)}
    return [per_query[qid] for qid in pairs["query_id"]]


def per_query_ndcg(pairs: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Full-list nDCG per query, ordered by query_id (same order for every model and condition)."""
    return np.array([r["ndcg_full"] for r in per_query_results(pairs.assign(score=scores), "score")])


def arm_means(table: dict[str, dict[str, np.ndarray]], names: Sequence[str]) -> dict[str, np.ndarray]:
    """Per-query nDCG averaged over the seeds (models) of one arm, for every condition."""
    return {c: np.mean([table[n][c] for n in names], axis=0) for c in CONDITIONS}


def seed_spread(table: dict[str, dict[str, np.ndarray]], names: Sequence[str], condition: str) -> float:
    means = [float(table[n][condition].mean()) for n in names]
    return max(means) - min(means)


def score_models(names: Sequence[str], pairs: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from shoprank.cross.finetune import MODELS, Prepared4DMask, predict
    from shoprank.cross.zero_shot import product_texts, read_products

    device = "cuda" if torch.cuda.is_available() else "cpu"
    queries = {c: condition_queries(pairs, c) for c in CONDITIONS}
    table, texts_by_fields = {}, {}
    for name in names:
        cfg = json.loads((PROCESSED / "runs" / f"{name}.json").read_text(encoding="utf-8"))
        fields = tuple(cfg["fields"])
        if fields not in texts_by_fields:
            products = read_products(list(fields), pairs["product_id"])
            texts_by_fields[fields] = product_texts(pairs, products, list(fields))
            del products
        tokenizer = AutoTokenizer.from_pretrained(MODELS / name)
        model = AutoModelForSequenceClassification.from_pretrained(MODELS / name, dtype=torch.float32).to(device)
        net = Prepared4DMask(model) if model.config._attn_implementation == "sdpa" else model
        table[name] = {c: per_query_ndcg(pairs, predict(net, tokenizer, queries[c], texts_by_fields[fields], 128,
                                                         cfg.get("max_length", 128), device, bf16=True))
                       for c in CONDITIONS}
        print(name, {c: round(float(v.mean()), 4) for c, v in table[name].items()}, flush=True)
        del model, net
        if device == "cuda":
            torch.cuda.empty_cache()
    return table


def compare(table: dict[str, dict[str, np.ndarray]], base: Sequence[str], treat: Sequence[str]) -> dict:
    b, t = arm_means(table, base), arm_means(table, treat)
    out = {"per_model": {n: {c: round(float(v.mean()), 5) for c, v in table[n].items()} for n in table}}
    for c in CONDITIONS:
        delta, lo, hi = paired_bootstrap_ci(b[c], t[c])
        out[c] = {"base": round(float(b[c].mean()), 5), "treat": round(float(t[c].mean()), 5),
                  "treat_minus_base": [round(delta, 5), round(lo, 5), round(hi, 5)],
                  "base_seed_spread": round(seed_spread(table, base, c), 5),
                  "treat_seed_spread": round(seed_spread(table, treat, c), 5)}
    for c in CONDITIONS[1:]:
        gap = out["clean"]["base"] - out[c]["base"]
        out[c]["base_gap"] = round(gap, 5)
        out[c]["recovered_share"] = round((out[c]["treat"] - out[c]["base"]) / gap, 3) if gap > 0 else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Typo robustness of two model arms")
    ap.add_argument("--split", choices=ALLOWED_SPLITS, default="dev")
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--treat", nargs="+", required=True)
    args = ap.parse_args()
    pairs, _ = load_split(args.split)
    pairs = pairs.sort_values(["query_id", "product_id"]).reset_index(drop=True)
    result = {"split": args.split, "queries": int(pairs["query_id"].nunique()), "base": args.base,
              "treat": args.treat, **compare(score_models([*args.base, *args.treat], pairs), args.base, args.treat)}
    path = PROCESSED / "runs" / f"typo_robustness_{args.split}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_model"}, indent=2))


if __name__ == "__main__":
    main()
