"""Section 15: the ranking API. POST /rank re-ranks a query's candidate products with the INT8 cross-encoder.

Run:
    uv run uvicorn shoprank.serve.app:app --port 8000
Settings (environment variables):
    SHOPRANK_MODEL_DIR    default data/processed/models/onnx/minilm_distil_w10_bullets
    SHOPRANK_MODEL_FILE   default model_int8.onnx (model_opt.onnx = fp32, for the comparison)
    SHOPRANK_THREADS      ONNX Runtime threads per model call (default: ONNX Runtime's choice = all logical CPUs)
    SHOPRANK_MAX_CONCURRENT  model calls allowed at the same time (default 1: others wait in line; parallel
                          calls on one CPU make threads fight, measured in the section 15 load test)

Design:
- Input is validated at the boundary (sizes, required title, unique ids); a bad request gets a 422 with the
  reason, never a model call.
- One request = one query and its candidates (the ESCI Task 1 setting: re-rank, not retrieve). All candidates
  are scored in length-sorted batches inside the request. Requests are not merged across users ("dynamic
  batching"): on CPU, ONNX Runtime already spreads one batch over all cores, so merging mainly adds waiting.
- The endpoint is a plain `def`: FastAPI runs it in a worker thread, so the event loop never blocks on the
  model. Model errors become a 500 with a generic message; the detail goes to the log.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Protocol

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from shoprank.serve.ranker import serving_text

LOG = logging.getLogger("shoprank.serve")
MAX_PRODUCTS = 200          # ESCI Task 1 lists have at most ~40 candidates; 200 leaves room
MAX_QUERY_CHARS = 200
MAX_ID_CHARS = 64
MAX_SHORT_FIELD_CHARS = 500
MAX_LONG_FIELD_CHARS = 5000  # long text is truncated to 128 tokens by the tokenizer anyway
MAX_BODY_BYTES = 1_000_000   # rejected before JSON parsing (a full valid request is far below this)
QUEUE_TIMEOUT_S = 10.0       # a request that waits longer for the model gets 503 instead of piling up
DEFAULT_MODEL_DIR = Path("data/processed/models/onnx/minilm_distil_w10_bullets")


class Scorer(Protocol):
    fields: Sequence[str]
    name: str

    def score(self, queries: Sequence[str], texts: Sequence[str]) -> np.ndarray: ...


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    title: str = Field(min_length=1, max_length=MAX_LONG_FIELD_CHARS)
    brand: str | None = Field(default=None, max_length=MAX_SHORT_FIELD_CHARS)
    color: str | None = Field(default=None, max_length=MAX_SHORT_FIELD_CHARS)
    bullets: str | None = Field(default=None, max_length=MAX_LONG_FIELD_CHARS)

    @field_validator("title")
    @classmethod
    def title_has_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title is blank")
        return v


class RankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    products: list[Product] = Field(min_length=1, max_length=MAX_PRODUCTS)

    @field_validator("query")
    @classmethod
    def query_has_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query is blank")
        return v.strip()

    @field_validator("products")
    @classmethod
    def ids_are_unique(cls, v: list[Product]) -> list[Product]:
        if len({p.product_id for p in v}) != len(v):
            raise ValueError("product_id values must be unique")
        return v


class Ranked(BaseModel):
    product_id: str
    score: float


class RankResponse(BaseModel):
    model: str
    results: list[Ranked]
    latency_ms: float


class OnnxScorer:
    """The served model: ONNX Runtime INT8 (or fp32) cross-encoder with its training-time product fields."""

    def __init__(self, model_dir: Path, model_file: str, threads: int | None) -> None:
        from shoprank.serve.ranker import OnnxCrossEncoder

        meta = json.loads((model_dir / "serving.json").read_text(encoding="utf-8"))
        self.fields, self.name = meta["fields"], f"{meta['run']}/{model_file}"
        self._encoder = OnnxCrossEncoder(model_dir / model_file, model_dir, meta["max_length"], threads)

    def score(self, queries: Sequence[str], texts: Sequence[str]) -> np.ndarray:
        return self._encoder.score(queries, texts)


def scorer_from_env() -> Scorer:
    threads = os.environ.get("SHOPRANK_THREADS")
    return OnnxScorer(Path(os.environ.get("SHOPRANK_MODEL_DIR", DEFAULT_MODEL_DIR)),
                      os.environ.get("SHOPRANK_MODEL_FILE", "model_int8.onnx"), int(threads) if threads else None)


def get_scorer(request: Request) -> Scorer:
    return request.app.state.scorer


ScorerDep = Annotated[Scorer, Depends(get_scorer)]  # module level: FastAPI resolves string annotations here


def create_app(make_scorer: Callable[[], Scorer] = scorer_from_env, max_concurrent: int | None = None) -> FastAPI:
    limit = threading.BoundedSemaphore(max_concurrent or int(os.environ.get("SHOPRANK_MAX_CONCURRENT", "1")))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.scorer = make_scorer()  # load once at start-up, not per request
        yield

    app = FastAPI(title="ShopRank", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        size = request.headers.get("content-length")
        if size is None and request.method == "POST":
            return JSONResponse({"detail": "content-length required"}, status_code=411)
        if size is not None and (not size.isdigit() or int(size) > MAX_BODY_BYTES):
            return JSONResponse({"detail": f"request body over {MAX_BODY_BYTES:,} bytes"}, status_code=413)
        return await call_next(request)

    @app.get("/health")
    def health(scorer: ScorerDep) -> dict:
        return {"status": "ok", "model": scorer.name}

    @app.post("/rank", response_model=RankResponse)
    def rank(req: RankRequest, scorer: ScorerDep) -> RankResponse:
        start = time.perf_counter()
        texts = [serving_text(p.model_dump(), scorer.fields) for p in req.products]
        if not limit.acquire(timeout=QUEUE_TIMEOUT_S):
            raise HTTPException(status_code=503, detail="server busy, retry later")
        try:
            scores = np.asarray(scorer.score([req.query] * len(texts), texts), dtype=float)
        except Exception:
            LOG.exception("scoring failed for a request with %d products", len(texts))
            raise HTTPException(status_code=500, detail="scoring failed") from None
        finally:
            limit.release()
        if scores.shape != (len(texts),) or not np.isfinite(scores).all():
            LOG.error("scorer returned shape %s / non-finite values", scores.shape)
            raise HTTPException(status_code=500, detail="scoring failed")
        order = sorted(range(len(texts)), key=lambda i: (-scores[i], req.products[i].product_id))
        return RankResponse(model=scorer.name,
                            results=[Ranked(product_id=req.products[i].product_id, score=float(scores[i]))
                                     for i in order],
                            latency_ms=round((time.perf_counter() - start) * 1000, 2))

    return app


app = create_app()
