"""Turn raw ESCI files into clean US-English tables with a train/dev/test split.

Usage:
    uv run python -m shoprank.data.ingest

Outputs (data/processed/):
    judgments.parquet  one row per (query, product) human label, with gain and split
    queries.parquet    one row per query: query_id, query, split
    products.parquet   US products that appear in any judgment

Split rules (see docs/learn/week01.md):
    - test  = ESCI official test split (never used for tuning)
    - dev   = ~10% of official train queries, chosen by a stable hash of the NORMALISED query text
    - dev2  = another ~5% bucket, touched only once to confirm the final model choice
    - train = the rest of official train
    - dropped = official-train rows whose normalised text also occurs in official test
    Normalised text = lowercase, non [a-z0-9] runs -> one space, trimmed. Splitting on it means
    near-duplicates ("iPhone Case!" / "iphone case") under different query_ids share one split.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
LOCALE = "us"
SPLIT_BUCKETS = 20  # buckets 0-1 -> dev (~10%), bucket 2 -> dev2 (~5%), rest -> train
NORM_SQL = "trim(regexp_replace(lower(query), '[^a-z0-9]+', ' ', 'g'))"  # lower BEFORE the regex

# Gain per label, as used for ESCI Task 1 nDCG (Exact > Substitute > Complement > Irrelevant).
ESCI_GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def build(con: duckdb.DuckDBPyConnection, raw_dir: Path, out_dir: Path) -> dict:
    examples = (raw_dir / "shopping_queries_dataset_examples.parquet").as_posix()
    products = (raw_dir / "shopping_queries_dataset_products.parquet").as_posix()
    out_dir.mkdir(parents=True, exist_ok=True)
    gains_sql = " ".join(f"WHEN '{k}' THEN {v}" for k, v in ESCI_GAINS.items())

    # MD5 of the normalised text: same text -> same bucket on every run, machine and DuckDB version.
    # (DuckDB's own hash() is not promised to stay identical across versions.)
    # The official test split cannot move, so train rows whose text also appears in test are dropped.
    con.execute(f"""
        CREATE OR REPLACE TABLE judgments AS
        WITH ex AS (
            SELECT *, {NORM_SQL} AS qnorm
            FROM read_parquet('{examples}')
            WHERE product_locale = '{LOCALE}'
        ),
        test_texts AS (SELECT DISTINCT qnorm FROM ex WHERE split = 'test')
        SELECT
            query_id, query, product_id, esci_label,
            CAST(CASE esci_label {gains_sql} END AS DOUBLE) AS gain,  -- literal 0.1 is DECIMAL otherwise
            small_version, large_version,
            CASE
                WHEN split = 'test' THEN 'test'
                WHEN qnorm IN (SELECT qnorm FROM test_texts) THEN 'dropped'
                WHEN md5_number(qnorm) % {SPLIT_BUCKETS} < 2 THEN 'dev'
                WHEN md5_number(qnorm) % {SPLIT_BUCKETS} = 2 THEN 'dev2'
                ELSE 'train'
            END AS split
        FROM ex
    """)
    con.execute("""
        CREATE OR REPLACE TABLE queries AS
        SELECT query_id, any_value(query) AS query, any_value(split) AS split
        FROM judgments GROUP BY query_id
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE products AS
        SELECT p.product_id, p.product_title AS title, p.product_description AS description,
               p.product_bullet_point AS bullets, p.product_brand AS brand, p.product_color AS color
        FROM read_parquet('{products}') p
        WHERE p.product_locale = '{LOCALE}'
          AND p.product_id IN (SELECT DISTINCT product_id FROM judgments)
    """)
    for table in ("judgments", "queries", "products"):
        con.execute(f"COPY {table} TO '{(out_dir / f'{table}.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)")

    stats = {
        "judgments_by_split": dict(con.execute("SELECT split, count(*) FROM judgments GROUP BY 1 ORDER BY 1").fetchall()),
        "queries_by_split": dict(con.execute("SELECT split, count(*) FROM queries GROUP BY 1 ORDER BY 1").fetchall()),
        "labels": dict(con.execute("SELECT esci_label, count(*) FROM judgments GROUP BY 1 ORDER BY 1").fetchall()),
        "products": con.execute("SELECT count(*) FROM products").fetchall()[0][0],
        "task1_queries_test": con.execute(
            "SELECT count(DISTINCT query_id) FROM judgments WHERE small_version = 1 AND split = 'test'"
        ).fetchall()[0][0],
    }
    (out_dir / "ingest_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    for name in ("shopping_queries_dataset_examples.parquet", "shopping_queries_dataset_products.parquet"):
        if not (RAW_DIR / name).is_file():
            raise SystemExit(f"Missing {RAW_DIR / name}. Run: uv run python -m shoprank.data.download")
    stats = build(duckdb.connect(), RAW_DIR, OUT_DIR)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
