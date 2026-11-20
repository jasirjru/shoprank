"""Model fetch: write only verified files, never overwrite a different local model, skip files already correct."""
from __future__ import annotations

import hashlib

import pytest

from shoprank.serve import fetch as fetch_mod
from shoprank.serve.fetch import FILES, fetch

PAYLOADS = {"model.onnx": b"weights" * 100, "tokenizer.json": b'{"vocab": 1}'}
SUMS = {name: hashlib.sha256(data).hexdigest() for name, data in PAYLOADS.items()}


def server(payloads, calls):
    def opener(url):
        name = url.rsplit("/", 1)[1]
        calls.append(name)
        return payloads[name]
    return opener


def test_downloads_and_verifies_every_file(tmp_path):
    calls = []
    written = fetch(tmp_path, "https://host/", SUMS, server(PAYLOADS, calls))
    assert sorted(written) == sorted(PAYLOADS) and sorted(calls) == sorted(PAYLOADS)
    assert all((tmp_path / n).read_bytes() == d for n, d in PAYLOADS.items())
    assert not list(tmp_path.glob("*.part"))


def test_tampered_file_is_rejected_and_not_written(tmp_path):
    bad = {**PAYLOADS, "model.onnx": b"evil"}
    with pytest.raises(SystemExit, match="checksum mismatch for model.onnx"):
        fetch(tmp_path, "https://host/", SUMS, server(bad, []))
    assert not (tmp_path / "model.onnx").exists()


def test_correct_local_files_are_not_downloaded_again(tmp_path):
    for name, data in PAYLOADS.items():
        (tmp_path / name).write_bytes(data)
    calls = []
    assert fetch(tmp_path, "https://host/", SUMS, server(PAYLOADS, calls)) == []
    assert calls == []


def test_different_local_model_is_kept_unless_forced(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"my own trained model")
    with pytest.raises(SystemExit, match="not overwriting"):
        fetch(tmp_path, "https://host/", SUMS, server(PAYLOADS, []))
    assert (tmp_path / "model.onnx").read_bytes() == b"my own trained model"
    fetch(tmp_path, "https://host/", SUMS, server(PAYLOADS, []), force=True)
    assert (tmp_path / "model.onnx").read_bytes() == PAYLOADS["model.onnx"]


def test_published_manifest_covers_what_the_server_reads():
    assert {"model_int8.onnx", "tokenizer.json", "tokenizer_config.json", "serving.json", "spell_vocab.json"} <= set(FILES)
    assert all(len(s) == 64 for s in FILES.values())
    assert fetch_mod.SOURCES["github"].endswith("/v1.1/")
