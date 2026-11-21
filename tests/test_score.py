"""Per-pair scorer: keeps the test lock and writes one row per (query, product)."""
import numpy as np
import pandas as pd
import pytest

from shoprank.cross.score import check_split, score_frame


def test_test_split_is_locked_unless_explicitly_allowed():
    with pytest.raises(SystemExit):
        check_split("test", allow_test=False)
    check_split("test", allow_test=True)  # section 16: once, on purpose
    check_split("dev", allow_test=False)


def test_score_frame_keeps_one_row_per_pair_with_labels():
    pairs = pd.DataFrame({"query_id": [1, 1], "query": ["q", "q"], "product_id": ["a", "b"],
                          "esci_label": ["E", "I"], "gain": [1.0, 0.0]})
    out = score_frame(pairs, np.array([0.9, 0.1], dtype=np.float32))
    assert out.columns.tolist() == ["query_id", "product_id", "esci_label", "gain", "score"]
    assert out["score"].tolist() == pytest.approx([0.9, 0.1])
    with pytest.raises(ValueError):
        score_frame(pairs, np.array([0.5]))


def test_existing_test_scores_are_never_overwritten(tmp_path, monkeypatch):
    import sys

    from shoprank.cross import score as score_mod

    monkeypatch.setattr(score_mod, "SCORES", tmp_path)
    (tmp_path / "m_test.parquet").write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", ["score", "m", "--split", "test", "--allow-test"])
    with pytest.raises(SystemExit, match="write-once"):
        score_mod.main()
