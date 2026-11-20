"""Section 18: does catalogue spell correction before ranking undo the typo loss, without hurting clean queries?

Usage:
    uv run python -m shoprank.eval.spell_eval --split dev \
        --models minilm_distil_w10_bullets minilm_distil_w10_bullets_s43 minilm_distil_w10_bullets_s44

For each query set (clean, typo_transpose, typo_neighbour) the models score it as typed and after
`serve.spell` correction. Per-query nDCG is averaged over the models (seeds); "+spell vs as typed" is a paired
bootstrap over queries. Also reports how many queries the corrector changes and how many typo queries it
restores exactly. Writes data/processed/runs/spell_eval_<split>.json. The test split is refused.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from shoprank.eval.bootstrap import paired_bootstrap_ci
from shoprank.eval.typo_robustness import (
    ALLOWED_SPLITS,
    CONDITIONS,
    condition_queries,
    score_models,
)
from shoprank.features.build import PROCESSED, load_split
from shoprank.serve.spell import SpellCorrector, load_corrector

SPELL = "+spell"


def query_sets(pairs: pd.DataFrame, corrector: SpellCorrector) -> tuple[dict[str, list[str]], dict]:
    """Each condition as typed and corrected, plus correction statistics per condition (one count per query)."""
    sets, stats = {}, {}
    clean = dict(zip(pairs["query_id"], pairs["query"], strict=False))
    for c in CONDITIONS:
        typed = condition_queries(pairs, c)
        per_query = dict(zip(pairs["query_id"], typed, strict=True))
        start = time.perf_counter()
        fixed = {qid: corrector.correct(q) for qid, q in per_query.items()}
        ms = (time.perf_counter() - start) * 1000 / len(per_query)
        sets[c] = typed
        sets[c + SPELL] = [fixed[qid] for qid in pairs["query_id"]]
        changed = sum(fixed[q] != per_query[q] for q in per_query)
        restored = sum(fixed[q].lower() == clean[q].lower() for q in per_query if per_query[q] != clean[q])
        typo_queries = sum(per_query[q] != clean[q] for q in per_query)
        stats[c] = {"queries": len(per_query), "changed_by_spell": changed,
                    "changed_share": round(changed / len(per_query), 4), "ms_per_query": round(ms, 3)}
        if c != "clean":
            stats[c] |= {"typo_queries": typo_queries, "restored_exactly": restored,
                         "restored_share": round(restored / max(1, typo_queries), 4)}
    return sets, stats


def summarise(table: dict[str, dict[str, np.ndarray]], models: Sequence[str]) -> dict:
    mean = {k: np.mean([table[m][k] for m in models], axis=0) for k in table[models[0]]}
    out = {}
    for c in CONDITIONS:
        delta, lo, hi = paired_bootstrap_ci(mean[c], mean[c + SPELL])
        out[c] = {"as_typed": round(float(mean[c].mean()), 5), "spell": round(float(mean[c + SPELL].mean()), 5),
                  "spell_minus_typed": [round(delta, 5), round(lo, 5), round(hi, 5)]}
    clean = out["clean"]["as_typed"]
    for c in CONDITIONS[1:]:
        gap = clean - out[c]["as_typed"]
        out[c]["gap"] = round(gap, 5)
        out[c]["recovered_share"] = round((out[c]["spell"] - out[c]["as_typed"]) / gap, 3) if gap > 0 else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Spell correction before ranking: clean and typo queries")
    ap.add_argument("--split", choices=ALLOWED_SPLITS, default="dev")
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    pairs, _ = load_split(args.split)
    pairs = pairs.sort_values(["query_id", "product_id"]).reset_index(drop=True)
    sets, stats = query_sets(pairs, load_corrector())
    print(json.dumps(stats, indent=2), flush=True)
    table = score_models(args.models, pairs, sets)
    result = {"split": args.split, "queries": int(pairs["query_id"].nunique()), "models": args.models,
              "correction_stats": stats, **summarise(table, args.models),
              "per_model": {m: {k: round(float(v.mean()), 5) for k, v in table[m].items()} for m in table}}
    (PROCESSED / "runs" / f"spell_eval_{args.split}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_model", "correction_stats")}, indent=2))


if __name__ == "__main__":
    main()
