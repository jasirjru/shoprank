"""Label-noise estimate from a random sample judged blind by two independent judges (section 18 follow-up).

Usage:
    uv run python -m shoprank.eval.label_audit data/processed/label_audit judge_sonnet judge_opus

Reads key.csv (item, query_id, product_id, esci_label) and <judge>.csv (item, label, confidence, reason).
Reports, for each judge, agreement with the ESCI label (4 classes and Exact vs not-Exact), Cohen's kappa between
the two judges, and the "likely label error" rate: both judges agree with each other and disagree with the ESCI
label, with a Wilson 95% interval. Writes label_audit_summary.json in the same folder.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

LABELS = ("E", "S", "C", "I")
Z95 = 1.959964


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion (stays inside [0, 1] for small counts, unlike p +- 1.96 se)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def load_judge(folder: Path, name: str) -> pd.Series:
    df = pd.read_csv(folder / f"{name}.csv")
    if sorted(df["item"]) != list(range(1, len(df) + 1)) or not df["label"].isin(LABELS).all():
        raise ValueError(f"{name}: need one valid E/S/C/I label per item 1..n")
    return df.set_index("item")["label"]


def share(mask: pd.Series) -> dict:
    k, n = int(mask.sum()), len(mask)
    lo, hi = wilson(k, n)
    return {"count": k, "n": n, "share": round(k / n, 4), "wilson95": [round(lo, 4), round(hi, 4)]}


def summarise(gold: pd.Series, a: pd.Series, b: pd.Series) -> dict:
    gold, a, b = gold.sort_index(), a.sort_index(), b.sort_index()
    if not (gold.index.equals(a.index) and gold.index.equals(b.index)):
        raise ValueError("judges and key must cover the same items")
    out = {}
    for name, j in (("judge_a", a), ("judge_b", b)):
        out[name] = {"agree_4class": share(j == gold), "agree_exact_vs_not": share((j == "E") == (gold == "E")),
                     "kappa_vs_label": round(float(cohen_kappa_score(j, gold)), 3)}
    out["judges_agree"] = share(a == b)
    out["kappa_judges"] = round(float(cohen_kappa_score(a, b)), 3)
    out["likely_label_error"] = share((a == b) & (a != gold))
    out["likely_error_exact_vs_not"] = share(((a == "E") == (b == "E")) & ((a == "E") != (gold == "E")))
    both = (a == b) & (a != gold)
    out["likely_error_by_label"] = {lab: int((both & (gold == lab)).sum()) for lab in LABELS}
    out["label_counts"] = {lab: int((gold == lab).sum()) for lab in LABELS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Label-noise estimate from two blind judges")
    ap.add_argument("folder", type=Path)
    ap.add_argument("judge_a")
    ap.add_argument("judge_b")
    args = ap.parse_args()
    gold = pd.read_csv(args.folder / "key.csv").set_index("item")["esci_label"]
    result = {"judge_a_file": args.judge_a, "judge_b_file": args.judge_b,
              **summarise(gold, load_judge(args.folder, args.judge_a), load_judge(args.folder, args.judge_b))}
    (args.folder / "label_audit_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
