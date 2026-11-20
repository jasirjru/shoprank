#!/usr/bin/env sh
# Write the serving-only requirements (the ~45 packages the API imports) from uv.lock, with hashes.
# The training stack (torch, LightGBM, pandas, ...) is pruned, so serving needs ~200 MB instead of several GB.
# Used by the Dockerfile and by the README "serving only" install. Usage: sh scripts/serve_requirements.sh [out]
set -eu
uv export --frozen --no-dev --no-emit-project \
    --prune torch --prune sentence-transformers --prune accelerate --prune lightgbm --prune datasets \
    --prune scikit-learn --prune pytrec-eval-terrier --prune duckdb --prune pandas --prune pyarrow \
    --prune onnx --prune sentencepiece \
    -o "${1:-requirements-serve.txt}"
