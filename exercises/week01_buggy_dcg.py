"""Week 1 no-AI debugging exercise. Do NOT ask an AI to fix this. Do NOT look at your metrics.py.

This nDCG has TWO planted bugs. Find both by reasoning and small hand-checked examples.
Run: uv run python exercises/week01_buggy_dcg.py
Write down: (1) each bug, (2) a 2-3 item example that exposes it, (3) the fix. Then tell Claude.
"""
import math


def buggy_dcg(gains, k):
    total = 0.0
    for i, g in enumerate(gains[: k - 1], start=1):
        total += g / math.log2(i + 1)
    return total


def buggy_ndcg(gains, k):
    ideal = buggy_dcg(sorted(gains), k)
    return buggy_dcg(gains, k) / ideal if ideal else 0.0


if __name__ == "__main__":
    for gains, k in [([1.0, 0.1, 0.0], 3), ([0.0, 1.0], 2), ([0.1, 1.0, 0.01], 3)]:
        print(gains, k, "->", round(buggy_ndcg(gains, k), 4))
    print("Hint: nDCG must always be between 0 and 1. Is it here? What should each line print?")
