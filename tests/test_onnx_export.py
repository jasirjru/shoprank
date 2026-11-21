"""ONNX export path of the served model: export -> fuse -> INT8, on a tiny random BERT (runs on CPU in seconds)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from shoprank.serve.onnx_export import export, optimise_and_quantise, torch_reference
from shoprank.serve.ranker import OnnxCrossEncoder

HIDDEN, HEADS = 32, 2
WORDS = ["usb", "c", "cable", "anker", "6", "ft", "hdmi", "red", "mug", "lamp", "with", "for"]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tiny")
    vocab = folder / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]), encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab))
    torch.manual_seed(0)
    config = transformers.BertConfig(vocab_size=len(WORDS) + 5, hidden_size=HIDDEN, num_attention_heads=HEADS,
                                     num_hidden_layers=2, intermediate_size=64, num_labels=1)
    model = transformers.BertForSequenceClassification(config).eval()
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return folder


def test_export_fuse_quantise_keeps_the_scores(tiny_model, tmp_path):
    fp32 = export(tiny_model, tmp_path)
    opt, int8 = optimise_and_quantise(fp32, HIDDEN, HEADS)
    queries = ["usb c cable", "red mug", "lamp for mug", "hdmi"]
    texts = ["anker usb c cable 6 ft", "red mug", "usb lamp", "hdmi cable with c"]
    reference = torch_reference(tiny_model, queries, texts, max_length=32)

    fused = OnnxCrossEncoder(opt, tmp_path, max_length=32).score(queries, texts)
    np.testing.assert_allclose(fused, reference, atol=1e-4)  # same function after kernel fusion

    quantised = OnnxCrossEncoder(int8, tmp_path, max_length=32).score(queries, texts)
    assert quantised.shape == (4,) and np.isfinite(quantised).all()
    assert int8.stat().st_size < opt.stat().st_size  # INT8 weights are smaller
