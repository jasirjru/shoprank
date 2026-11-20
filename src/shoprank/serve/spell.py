"""Query spell correction against the catalogue vocabulary (section 18): fix a typo BEFORE the ranker sees it.

Build the vocabulary once (word counts from the titles, brands and colours of the 1.2M US products):
    uv run python -m shoprank.serve.spell build          # -> data/processed/spell_vocab.json
Try it:
    uv run python -m shoprank.serve.spell try "usb c cbale 6ft"

Rule for each query word (lower-case letters only, 4+ characters):
  1. A word seen KNOWN_MIN_FREQ or more times in the catalogue is spelled right: keep it. Brands, model
     names and Spanish words that occur in titles survive this way.
  2. Otherwise look for catalogue words exactly ONE edit away (Damerau-Levenshtein: one letter deleted,
     inserted, replaced, or two neighbours swapped). This is the SymSpell idea: index every catalogue word by
     its one-letter deletions, so candidates are found by dictionary look-ups, not by comparing against
     every word.
  3. Take the most frequent candidate, but only if it is at least CANDIDATE_RATIO times more frequent than
     the typed word: a rare word is replaced only by a much more common one.
Words with digits or symbols ("6ft", "usb-c") are never touched. Nothing here knows how the evaluation typos
are generated (for example it may change the first letter), so the test is fair.
The vocabulary uses product text only, never labels or queries, so it does not leak evaluation answers.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

VOCAB_PATH = Path("data/processed/spell_vocab.json")
PRODUCTS_PATH = Path("data/processed/products.parquet")
VOCAB_FIELDS = ("title", "brand", "color")
WORD = re.compile(r"[a-z]+")
MIN_LEN = 4
KNOWN_MIN_FREQ = 5
CANDIDATE_RATIO = 10
STORE_MIN_FREQ = 2  # words seen once are mostly noise; dropping them keeps the file small
BATCH_ROWS = 100_000


def deletions(word: str) -> set[str]:
    return {word[:i] + word[i + 1:] for i in range(len(word))}


def one_edit_apart(a: str, b: str) -> bool:
    """True if b is exactly one Damerau-Levenshtein edit from a (delete, insert, replace, adjacent swap)."""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    short, long_ = (a, b) if la < lb else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


class SpellCorrector:
    def __init__(self, freq: Mapping[str, int]) -> None:
        self.freq = dict(freq)
        self._by_deletion: dict[str, list[str]] = defaultdict(list)
        for word, n in self.freq.items():
            if n >= KNOWN_MIN_FREQ:
                for d in deletions(word):
                    self._by_deletion[d].append(word)

    def _known(self, word: str) -> bool:
        return self.freq.get(word, 0) >= KNOWN_MIN_FREQ

    def candidates(self, word: str) -> set[str]:
        found = set(self._by_deletion.get(word, ()))  # typed word is missing a letter
        for d in deletions(word):
            if self._known(d):
                found.add(d)  # typed word has an extra letter
            found.update(self._by_deletion.get(d, ()))  # a replaced letter or a swap
        return {w for w in found if one_edit_apart(word, w)}

    def correct_word(self, word: str) -> str:
        if len(word) < MIN_LEN or not word.isalpha() or not word.isascii() or self._known(word.lower()):
            return word
        low = word.lower()
        options = self.candidates(low)
        if not options:
            return word
        best = max(options, key=lambda w: (self.freq[w], w))
        return best if self.freq[best] >= CANDIDATE_RATIO * max(1, self.freq.get(low, 0)) else word

    def correct(self, query: str) -> str:
        """The query with misspelled words replaced; the input unchanged (spacing kept) if no word changed."""
        words = query.split()
        fixed = [self.correct_word(w) for w in words]
        return query if fixed == words else " ".join(fixed)


def count_words(texts: Iterable[str]) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        counts.update(WORD.findall(text.lower()))
    return counts


def build_vocab(products: Path = PRODUCTS_PATH, out: Path = VOCAB_PATH) -> dict[str, int]:
    """Word counts over the product titles, brands and colours, read in batches to keep memory low."""
    import pyarrow.parquet as pq

    counts: Counter = Counter()
    for batch in pq.ParquetFile(products).iter_batches(batch_size=BATCH_ROWS, columns=list(VOCAB_FIELDS)):
        for field in VOCAB_FIELDS:
            counts.update(count_words(v for v in batch.column(field).to_pylist() if v))
    vocab = {w: n for w, n in counts.most_common() if n >= STORE_MIN_FREQ}
    out.write_text(json.dumps(vocab), encoding="utf-8")
    return vocab


def load_corrector(path: Path = VOCAB_PATH) -> SpellCorrector:
    return SpellCorrector(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    ap = argparse.ArgumentParser(description="Catalogue-based query spell correction")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    t = sub.add_parser("try")
    t.add_argument("query")
    args = ap.parse_args()
    if args.cmd == "build":
        start = time.perf_counter()
        vocab = build_vocab()
        print(f"{len(vocab):,} words (count >= {STORE_MIN_FREQ}) -> {VOCAB_PATH} in {time.perf_counter() - start:.0f}s")
        return
    corrector = load_corrector()
    print(corrector.correct(args.query))


if __name__ == "__main__":
    main()
