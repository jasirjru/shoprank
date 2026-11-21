"""Download the served model (INT8 ONNX + tokenizer) published with v1.0 and verify every file's SHA-256.

Usage:
    uv run python -m shoprank.serve.fetch                      # from the GitHub release (default)
    uv run python -m shoprank.serve.fetch --source huggingface

Writes into data/processed/models/onnx/minilm_distil_w10_bullets/, the folder the API and demo read, so a
fresh clone can serve without a GPU or any training. A file is written only after its checksum matches; an
existing file with a different checksum (for example a model you trained yourself) is never overwritten
unless you pass --force.
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

DEST = Path("data/processed/models/onnx/minilm_distil_w10_bullets")
SOURCES = {
    "github": "https://github.com/jasirjru/shoprank/releases/download/v1.0/",
    # pinned to the commit of the v1.0 upload, not the moving main branch
    "huggingface": "https://huggingface.co/jasirjru/shoprank-reranker-minilm-int8/resolve/"
                   "ab467f4119cfb6c6acaecc4fbdcf9a52dbacbcd6/",
}
# SHA-256 of the v1.0 files (also published as SHA256SUMS next to them). The checksum, not the URL, is the trust
# anchor: a changed file on either host is rejected.
FILES: dict[str, str] = {
    "model_int8.onnx": "14d506b276a74efa42259977df8741e3d153c1160b71b751d6219ea3a06692b8",
    # gitleaks reads "token" + 64 hex characters as an API key; these are SHA-256 hashes of public files.
    "tokenizer.json": "79ea220416a57ca8acdf069d71875d5d2ddadef66021aeb6c120a7c546c04344",  # gitleaks:allow
    "tokenizer_config.json": "eb1312d29a13ef3bf49cab31c4d2460ba215c4b53a3fc85afb6773ada597174d",  # gitleaks:allow
    "serving.json": "f668c64a98d7646f7bf1eed3794d28965995f10ca72c638676b50a732bd3b9cf",
    "quality.json": "c3c2b11c686262ecf702ac00c5c753ee1ac145a49eab10f20f536a45030fe904",
}
CHUNK = 1 << 20
TIMEOUT_S = 60
USER_AGENT = "shoprank-fetch"

Opener = Callable[[str], bytes]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return response.read()


def fetch(dest: Path = DEST, base_url: str = SOURCES["github"], files: Mapping[str, str] = FILES,
          opener: Opener = http_get, force: bool = False) -> list[str]:
    """Download missing files, verify each SHA-256, return the names written. Raises SystemExit on a mismatch."""
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name, expected in files.items():
        target = dest / name
        if target.exists():
            if sha256_file(target) == expected:
                print(f"  ok (already present): {name}")
                continue
            if not force:
                raise SystemExit(f"{target} exists with a different checksum; not overwriting (use --force).")
        data = opener(base_url + name)
        actual = sha256_bytes(data)
        if actual != expected:
            raise SystemExit(f"checksum mismatch for {name}: expected {expected}, got {actual}; nothing written.")
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(data)
        partial.replace(target)  # appears only once complete and verified
        written.append(name)
        print(f"  downloaded and verified: {name} ({len(data) / 2**20:.1f} MB)")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and verify the served ShopRank model (v1.0)")
    ap.add_argument("--source", choices=sorted(SOURCES), default="github")
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--force", action="store_true", help="replace existing files whose checksum differs")
    args = ap.parse_args()
    print(f"source: {SOURCES[args.source]}\ndest:   {args.dest}")
    fetch(args.dest, SOURCES[args.source], force=args.force)
    print("model ready: uv run uvicorn shoprank.serve.app:app --port 8000")


if __name__ == "__main__":
    main()
