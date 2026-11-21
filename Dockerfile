# CPU serving image: FastAPI + ONNX Runtime + the INT8 re-ranker published with v1.0. No torch, no GPU:
# only the ~45 packages the API imports (about 200 MB instead of ~7 GB for the training environment).
#   docker build -t shoprank .
#   docker run --rm -p 8000:8000 shoprank
# Base images are pinned by digest; packages come from uv.lock with hashes; the model is checked by SHA-256.
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
COPY --from=ghcr.io/astral-sh/uv:0.12.17@sha256:10787c682e4184e4f290de1171fd4703dc63de99221f10fe1c99002ce7fa9acc /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
# Serving subset of the lock: drop the training stack (torch, LightGBM, pandas, ...) and its dependencies.
RUN uv export --frozen --no-dev --no-emit-project \
        --prune torch --prune sentence-transformers --prune accelerate --prune lightgbm --prune datasets \
        --prune scikit-learn --prune pytrec-eval-terrier --prune duckdb --prune pandas --prune pyarrow \
        --prune onnx --prune sentencepiece \
        -o requirements-serve.txt \
    && uv pip install --system --no-cache --require-hashes -r requirements-serve.txt

COPY src ./src
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
# Model files from the v1.0 GitHub release; the build fails if any checksum differs.
RUN python -m shoprank.serve.fetch

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "shoprank.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
