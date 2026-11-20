"""Leakage and integrity tests for the ingest step, on a tiny fake ESCI dataset."""
import duckdb
import pytest

from shoprank.data.ingest import ESCI_GAINS, NORM_SQL, build


@pytest.fixture
def built(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    con = duckdb.connect()
    # 40 train queries + 5 test queries, 2 judged products each, plus one Spanish row to be filtered out.
    con.execute("""
        CREATE TABLE ex AS
        SELECT i AS example_id, 'query ' || q AS query, q AS query_id, 'P' || i AS product_id,
               'us' AS product_locale, ['E','S','C','I'][1 + i % 4] AS esci_label,
               1 AS small_version, 1 AS large_version,
               CASE WHEN q >= 40 THEN 'test' ELSE 'train' END AS split
        FROM (SELECT i, i // 2 AS q FROM range(90) t(i))
        UNION ALL SELECT 999, 'zapato', 999, 'PES', 'es', 'E', 1, 1, 'train'
        -- near-duplicate of 'query 1' under a different id, and a train copy of test query 'query 41'
        UNION ALL SELECT 500, 'QUERY 1 !', 500, 'P1', 'us', 'E', 1, 1, 'train'
        UNION ALL SELECT 501, 'Query  41', 501, 'P2', 'us', 'E', 1, 1, 'train'
        -- two unrelated non-Latin queries; one is in test, the train one must NOT be dropped with it
        UNION ALL SELECT 502, '가마솥', 502, 'P3', 'us', 'E', 1, 1, 'train'
        UNION ALL SELECT 503, '香奈儿', 503, 'P4', 'us', 'E', 1, 1, 'test'
    """)
    con.execute("""
        CREATE TABLE pr AS
        SELECT 'P' || i AS product_id, 'title ' || i AS product_title, NULL AS product_description,
               NULL AS product_bullet_point, 'brand' AS product_brand, 'red' AS product_color, 'us' AS product_locale
        FROM range(90) t(i)
        UNION ALL SELECT 'PES', 'zapato', NULL, NULL, 'x', 'rojo', 'es'
    """)
    con.execute(f"COPY ex TO '{(raw / 'shopping_queries_dataset_examples.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY pr TO '{(raw / 'shopping_queries_dataset_products.parquet').as_posix()}' (FORMAT parquet)")
    stats = build(duckdb.connect(), raw, out)
    return out, stats


def test_no_query_in_two_splits(built):
    out, _ = built
    n = duckdb.sql(f"""
        SELECT count(*) FROM (SELECT query_id FROM '{(out / 'judgments.parquet').as_posix()}'
                              GROUP BY query_id HAVING count(DISTINCT split) > 1)
    """).fetchall()[0][0]
    assert n == 0


def test_official_test_split_is_preserved_and_dev_comes_from_train(built):
    _, stats = built
    q = stats["queries_by_split"]
    assert q["test"] == 6
    # 40 original + 'QUERY 1 !' + the Korean query; only the copy of a test query is dropped.
    assert q["train"] + q.get("dev", 0) + q.get("dev2", 0) == 42
    assert q.get("dev", 0) > 0 and q.get("dev2", 0) > 0
    assert q.get("dropped", 0) == 1


def test_split_is_deterministic(built, tmp_path):
    out, _ = built
    again = tmp_path / "again"
    build(duckdb.connect(), out.parent / "raw", again)
    q = "SELECT query_id, split FROM '{}' ORDER BY query_id"
    first = duckdb.sql(q.format((out / "queries.parquet").as_posix())).fetchall()
    second = duckdb.sql(q.format((again / "queries.parquet").as_posix())).fetchall()
    assert first == second


def test_only_us_locale_and_gains_match_labels(built):
    out, _ = built
    rows = duckdb.sql(f"SELECT DISTINCT esci_label, gain FROM '{(out / 'judgments.parquet').as_posix()}'").fetchall()
    assert all(ESCI_GAINS[label] == gain for label, gain in rows)
    ids = duckdb.sql(f"SELECT product_id FROM '{(out / 'products.parquet').as_posix()}'").fetchall()
    assert ("PES",) not in ids


def test_every_judged_product_has_catalog_entry(built):
    out, _ = built
    missing = duckdb.sql(f"""
        SELECT count(*) FROM '{(out / 'judgments.parquet').as_posix()}' j
        ANTI JOIN '{(out / 'products.parquet').as_posix()}' p USING (product_id)
    """).fetchall()[0][0]
    assert missing == 0


def test_near_duplicate_texts_share_one_split(built):
    out, _ = built
    n = duckdb.sql(f"""
        SELECT count(*) FROM (
            SELECT {NORM_SQL} AS t
            FROM '{(out / 'judgments.parquet').as_posix()}' WHERE split <> 'dropped'
            GROUP BY t HAVING count(DISTINCT split) > 1)
    """).fetchall()[0][0]
    assert n == 0


def test_train_copies_of_test_queries_are_dropped(built):
    out, _ = built
    rows = duckdb.sql(f"SELECT DISTINCT split FROM '{(out / 'judgments.parquet').as_posix()}' WHERE query_id = 501").fetchall()
    assert rows == [("dropped",)]


def test_non_latin_queries_keep_their_text(built):
    out, _ = built
    rows = duckdb.sql(f"SELECT DISTINCT split FROM '{(out / 'judgments.parquet').as_posix()}' WHERE query_id = 502").fetchall()
    assert rows[0][0] != "dropped"
