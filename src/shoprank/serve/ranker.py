"""Section 15: CPU cross-encoder scoring with ONNX Runtime, and the product text used at serving time.

The product text is built exactly as in training (shoprank.cross.zero_shot.product_texts): the chosen fields
joined by single spaces, missing fields as empty strings, then stripped. A different join at serving time
would silently change the model's input ("training/serving skew").
"""
from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

DEFAULT_BATCH = 32


def serving_text(product: Mapping[str, object], fields: Sequence[str]) -> str:
    """Same text as training: ' '.join of the fields, None/missing -> '', then strip."""
    return " ".join("" if product.get(f) is None else str(product.get(f)) for f in fields).strip()


def length_sorted_batches(lengths: Sequence[int], batch_size: int) -> list[np.ndarray]:
    """Indices grouped into batches of similar length (less padding = less wasted compute)."""
    order = np.argsort(np.asarray(lengths), kind="stable")
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


class OnnxCrossEncoder:
    """Scores (query, product text) pairs with an ONNX cross-encoder on CPU. One instance per process."""

    def __init__(self, model_path: Path, tokenizer_dir: Path, max_length: int = 128, threads: int | None = None,
                 batch_size: int = DEFAULT_BATCH) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        out_shape = self.session.get_outputs()[0].shape
        if out_shape[-1] != 1:  # a 4-class head would need softmax @ gains (score_from_logits), not this path
            raise ValueError(f"expected a 1-logit ranking head, got output shape {out_shape}")
        self._tokenizer_lock = threading.Lock()  # HF fast tokenizers are not safe for concurrent calls
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.max_length, self.batch_size = max_length, batch_size

    def score(self, queries: Sequence[str], texts: Sequence[str]) -> np.ndarray:
        if len(queries) != len(texts):
            raise ValueError(f"{len(queries)} queries for {len(texts)} texts")
        out = np.empty(len(queries), dtype=np.float32)
        lengths = [len(q) + len(t) for q, t in zip(queries, texts, strict=True)]  # cheap proxy for token count
        for idx in length_sorted_batches(lengths, self.batch_size):
            with self._tokenizer_lock:
                enc = self.tokenizer([queries[i] for i in idx], [texts[i] for i in idx], truncation=True,
                                     max_length=self.max_length, padding=True, return_tensors="np")
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.inputs}
            out[idx] = self.session.run(["logits"], feed)[0].reshape(-1)
        return out
