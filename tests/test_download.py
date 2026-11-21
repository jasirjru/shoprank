"""Raw-data download: resume from a partial file, skip a finished one, fail loudly on a short read."""
from __future__ import annotations

import hashlib
import io

import pytest

from shoprank.data import download as dl

PAYLOAD = bytes(range(256)) * 40  # 10,240 bytes


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def fake_server(payload: bytes, served: list):
    def urlopen(request, timeout=None):
        rng = request.get_header("Range")
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        served.append(start)
        return FakeResponse(payload[start:])
    return urlopen


def test_resumes_a_partial_file(tmp_path, monkeypatch):
    dest = tmp_path / "f.bin"
    dest.write_bytes(PAYLOAD[:3000])
    served: list = []
    monkeypatch.setattr(dl, "remote_size", lambda url: len(PAYLOAD))
    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_server(PAYLOAD, served))
    dl.download("http://x/f.bin", dest)
    assert served == [3000]
    assert dest.read_bytes() == PAYLOAD


def test_skips_a_complete_file(tmp_path, monkeypatch):
    dest = tmp_path / "f.bin"
    dest.write_bytes(PAYLOAD)
    monkeypatch.setattr(dl, "remote_size", lambda url: len(PAYLOAD))
    monkeypatch.setattr(dl.urllib.request, "urlopen", lambda *a, **k: pytest.fail("no request expected"))
    dl.download("http://x/f.bin", dest)


def test_short_read_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "remote_size", lambda url: len(PAYLOAD) + 10)
    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_server(PAYLOAD, []))
    with pytest.raises(SystemExit, match="Incomplete download"):
        dl.download("http://x/f.bin", tmp_path / "f.bin")


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(PAYLOAD)
    assert dl.sha256_of(path) == hashlib.sha256(PAYLOAD).hexdigest()
