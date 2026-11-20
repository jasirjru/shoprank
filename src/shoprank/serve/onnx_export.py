"""Section 15: export a fine-tuned cross-encoder to ONNX, fuse it, quantise to INT8, and check quality on dev.

Usage:
    uv run python -m shoprank.serve.onnx_export minilm_distil_w10_bullets

Writes data/processed/models/onnx/<run>/ with model_fp32.onnx, model_int8.onnx and the tokenizer.
Steps, following the ONNX Runtime docs for BERT-style models:
  1. torch.onnx.export (TorchScript exporter, dynamo=False: the dynamo exporter needs the extra `onnxscript`
     package), eager attention, dynamic batch and sequence axes.
  2. onnxruntime.transformers.optimizer (model_type "bert"): fuses attention, LayerNorm and GELU kernels.
  3. quantize_dynamic: INT8 weights, activations quantised on the fly per batch (no calibration data needed).
Quality gate: dev nDCG of the INT8 model within 0.002 of the PyTorch model (written as a prediction first).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from shoprank.cross.finetune import MODELS
from shoprank.cross.zero_shot import product_texts
from shoprank.features.build import PROCESSED, load_split
from shoprank.ltr.stack import mean_ndcg

ONNX_DIR = MODELS / "onnx"
OPSET = 17
PARITY_PAIRS = 512


def export(model_dir: Path, out: Path) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out)
    sample = tokenizer(["usb c cable"], ["Anker USB C cable 6 ft"], return_tensors="pt")
    path = out / "model_fp32.onnx"
    torch.onnx.export(
        model, (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]), path,
        input_names=["input_ids", "attention_mask", "token_type_ids"], output_names=["logits"],
        dynamic_axes={n: {0: "batch", 1: "seq"} for n in ("input_ids", "attention_mask", "token_type_ids")}
        | {"logits": {0: "batch"}},
        opset_version=OPSET, dynamo=False,
    )
    return path


def optimise_and_quantise(fp32: Path, hidden: int, heads: int) -> tuple[Path, Path]:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.transformers import optimizer

    opt_path = fp32.with_name("model_opt.onnx")
    optimizer.optimize_model(str(fp32), model_type="bert", num_heads=heads, hidden_size=hidden).save_model_to_file(
        str(opt_path))
    int8 = fp32.with_name("model_int8.onnx")
    # The fused graph has ONNX Runtime-only ops (com.microsoft domain), which ONNX shape inference cannot type;
    # their outputs are float, so say so (the error message of quantize_dynamic suggests exactly this).
    quantize_dynamic(str(opt_path), str(int8), weight_type=QuantType.QInt8, per_channel=True,
                     extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
    return opt_path, int8


def torch_reference(model_dir: Path, queries: list[str], texts: list[str], max_length: int) -> np.ndarray:
    """PyTorch fp32 scores on CPU for a parity check (the ONNX graph must compute the same function)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    with torch.inference_mode():
        enc = tokenizer(queries, texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        return model(**enc).logits.reshape(-1).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX export + INT8 quantisation + dev quality check")
    ap.add_argument("run")
    args = ap.parse_args()

    from shoprank.serve.ranker import OnnxCrossEncoder

    config = json.loads((PROCESSED / "runs" / f"{args.run}.json").read_text(encoding="utf-8"))
    fields, max_length = config["fields"], config.get("max_length", 128)  # same input length as training
    out = ONNX_DIR / args.run
    fp32 = export(MODELS / args.run, out)
    from transformers import AutoConfig

    mc = AutoConfig.from_pretrained(MODELS / args.run)
    opt, int8 = optimise_and_quantise(fp32, mc.hidden_size, mc.num_attention_heads)
    vocab = PROCESSED / "spell_vocab.json"  # `serve.spell build`; the API reads it from the model folder
    if vocab.exists():
        shutil.copyfile(vocab, out / vocab.name)
    else:
        print(f"note: no {vocab}; run `python -m shoprank.serve.spell build` to enable spell correction", flush=True)
    (out / "serving.json").write_text(json.dumps({"run": args.run, "fields": fields,
                                                  "max_length": max_length}, indent=2), encoding="utf-8")
    for p in (fp32, opt, int8):
        print(f"{p.name}: {p.stat().st_size / 2**20:.1f} MB", flush=True)

    pairs, _ = load_split("dev")
    products = pd.read_parquet(PROCESSED / "products.parquet", columns=["product_id", *fields])
    texts = product_texts(pairs, products, fields)
    queries = pairs["query"].tolist()
    torch_scores = torch_reference(MODELS / args.run, queries[:PARITY_PAIRS], texts[:PARITY_PAIRS], max_length)

    results = {}
    for label, path in (("onnx_fp32", opt), ("onnx_int8", int8)):
        enc = OnnxCrossEncoder(path, out, max_length)
        start = time.perf_counter()
        s = enc.score(queries, texts)
        secs = time.perf_counter() - start
        nd = mean_ndcg(pairs.assign(s=s), "s")
        results[label] = {"dev_ndcg_full": round(nd, 4), "cpu_seconds_dev": round(secs, 1),
                          "pairs_per_second": round(len(pairs) / secs)}
        results[label]["max_abs_diff_vs_torch_fp32"] = float(np.abs(s[:PARITY_PAIRS] - torch_scores).max())
        print(f"{label}: dev nDCG full {nd:.4f}  {secs:.1f} s on CPU ({len(pairs) / secs:,.0f} pairs/s); max |diff| "
              f"vs PyTorch fp32 on {PARITY_PAIRS} pairs {results[label]['max_abs_diff_vs_torch_fp32']:.2e}", flush=True)
    (out / "quality.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
