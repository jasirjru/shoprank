import duckdb
import pandas as pd
import pytest

from shoprank.features.build import FEATURES, build, pair_features

PRODUCT = {"title": "Samsung 55 Inch Smart TV Black", "brand": "Samsung", "color": "Black",
           "bullets": None, "description": "Great TV"}


def test_pair_features_hand_checked():
    f = pair_features("samsung tv 55 black", PRODUCT)
    assert f["title_coverage"] == 1.0                  # all 4 query words are in the title
    assert f["title_jaccard"] == pytest.approx(4 / 6)  # 4 shared / 6 distinct words overall
    assert f["brand_in_query"] == 1.0
    assert f["color_in_query"] == 1.0
    assert f["numbers_matched"] == 1.0 and f["numbers_missing"] == 0.0
    assert f["phrase_in_title"] == 0.0                 # words are not in this exact order in the title
    assert f["has_description"] == 1.0


def test_number_mismatch_is_counted():
    f = pair_features("samsung tv 65", PRODUCT)
    assert f["numbers_matched"] == 0.0 and f["numbers_missing"] == 1.0


def test_empty_fields_do_not_crash():
    f = pair_features("", {"title": None, "brand": None, "color": None})
    assert f["title_coverage"] == 0.0 and f["brand_in_query"] == 0.0


@pytest.fixture
def processed(tmp_path):
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE j AS SELECT * REPLACE (CAST(gain AS DOUBLE) AS gain) FROM (VALUES
            (1, 'red shoe', 'A', 'E', 1.0, 1, 'train'), (1, 'red shoe', 'B', 'I', 0.0, 1, 'train'),
            (2, 'blue dress', 'B', 'S', 0.1, 1, 'train'), (2, 'blue dress', 'C', 'E', 1.0, 1, 'train'))
        t(query_id, query, product_id, esci_label, gain, small_version, split)
    """)
    con.execute("""
        CREATE TABLE p AS SELECT * FROM (VALUES
            ('A', 'red running shoe', 'nike', 'great', NULL, 'red'),
            ('B', 'blue cotton dress', 'zara', NULL, 'nice', 'blue'),
            ('C', 'blue evening dress long', 'zara', NULL, NULL, 'blue'))
        t(product_id, title, brand, bullets, description, color)
    """)
    con.execute(f"COPY j TO '{(tmp_path / 'judgments.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY p TO '{(tmp_path / 'products.parquet').as_posix()}' (FORMAT parquet)")
    return tmp_path


def test_build_rows_are_grouped_by_query_and_labels_mapped(processed):
    df = build("train", processed)
    assert list(df["query_id"]) == sorted(df["query_id"])  # each query's rows are contiguous
    assert dict(zip(df["esci_label"], df["label"], strict=True)) == {"E": 3, "I": 0, "S": 2}
    assert set(FEATURES) <= set(df.columns)


def test_features_do_not_depend_on_labels(processed):
    """Leakage guard: flipping every label must not change a single feature value."""
    before = build("train", processed)
    flipped = pd.read_parquet(processed / "judgments.parquet")
    flipped["esci_label"] = flipped["esci_label"].map({"E": "I", "I": "E", "S": "C", "C": "S"})
    flipped["gain"] = 1.0 - flipped["gain"]
    flipped.to_parquet(processed / "judgments.parquet", index=False)
    after = build("train", processed)
    pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES])
