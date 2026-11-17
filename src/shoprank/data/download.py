"""Download the raw ESCI files from Amazon's official repo and record their hashes.

Usage:
    uv run python -m shoprank.data.download

Resumable: re-running skips finished files and continues partial ones.
Writes data/raw/manifest.json (bytes + SHA-256 per file) so every result is tied to exact inputs.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path("data/raw")
LFS = "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/"
RAW = "https://raw.githubusercontent.com/amazon-science/esci-data/main/shopping_queries_dataset/"
FILES = {
    "shopping_queries_dataset_examples.parquet": LFS,   # ~51 MB: query-product judgments
    "shopping_queries_dataset_products.parquet": LFS,   # ~1.1 GB: product catalog, 3 locales
    "shopping_queries_dataset_sources.csv": RAW,        # ~1.7 MB: query source info
}
CHUNK = 1 << 20
USER_AGENT = "shoprank-download"


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def download(url: str, dest: Path) -> None:
    total = remote_size(url)
    done = dest.stat().st_size if dest.exists() else 0
    if done == total:
        print(f"  already complete: {dest.name}")
        return
    headers = {"User-Agent": USER_AGENT, "Range": f"bytes={done}-"} if done else {"User-Agent": USER_AGENT}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r, dest.open("ab" if done else "wb") as f:
        while chunk := r.read(CHUNK):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {dest.name}: {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    if done != total:
        raise SystemExit(f"Incomplete download for {dest.name}: {done} of {total} bytes. Re-run to resume.")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, base in FILES.items():
        dest = RAW_DIR / name
        print(f"{name}")
        try:
            download(base + name, dest)
        except OSError as e:
            sys.exit(f"Download failed for {name}: {e}. Re-run to resume.")
        manifest[name] = {"bytes": dest.stat().st_size, "sha256": sha256_of(dest), "source": base + name}
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest -> {RAW_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
