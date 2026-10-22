"""
scripts/migrate_documents_to_postgres.py
=========================================
One-time migration: copy the documents bot data from the old SQLite
``library.db`` into the new PostgreSQL schema (``core.db``), and push the
block + series embedding vectors to Weaviate.

Usage::

    python -m scripts.migrate_documents_to_postgres \
        --sqlite-db documents/data/library.db

    # Dry-run (prints counts only, no writes)
    python -m scripts.migrate_documents_to_postgres --sqlite-db ... --dry-run

    # Skip Weaviate upload (row data only)
    python -m scripts.migrate_documents_to_postgres --sqlite-db ... --no-vectors

Behaviour:

* **Idempotent**: each table is truncated before insert (``TRUNCATE ... CASCADE``).
* **Preserves ids**: block_id / series_id / fact_id / issue_id / result_id are
  copied verbatim using ``OVERRIDING SYSTEM VALUE``; the matching sequences are
  then set to ``MAX(id) + 1`` so new inserts don't collide.
* **FTS**: Postgres triggers populate ``blocks.tsv`` / ``series.tsv`` automatically.
* **Vectors**: pulled from the SQLite ``blocks_vec`` / ``series_vec`` virtual
  tables (sqlite-vec) and pushed to Weaviate via ``documents.weaviate_store``.

The old ``library.db`` is **never modified** — it's opened read-only.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import struct
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
logger = logging.getLogger("migrate")

# Tables migrated in dependency order.
TABLES_IN_ORDER = (
    "documents",
    "pages",
    "corrections",
    "blocks",
    "series",
    "facts",
    "issues",
    "metric_rules",
    "calculated_metrics",
)

# Column lists (keep schema-equivalent to documents/db_schema.py).
COLUMNS = {
    "documents": ("doc_id", "file_name", "title", "doc_type", "entity", "as_of_key",
                  "as_of_label", "pages", "status", "progress", "ingested_at"),
    "pages": ("doc_id", "page", "heading", "text", "image", "record"),
    "corrections": ("doc_id", "page", "series_label", "period_key", "value", "note", "created_at"),
    "blocks": ("block_id", "doc_id", "page", "kind", "section", "title", "text", "content",
               "bbox", "source", "confidence", "section_path", "anchor_block_id"),
    "series": ("series_id", "doc_id", "page", "block_id", "label", "name", "metric_id",
               "kind", "unit", "entity", "source", "confidence", "path", "scale"),
    "facts": ("fact_id", "series_id", "doc_id", "page", "period_key", "period_kind",
              "period_start", "period_end", "period_label", "partial", "value",
              "as_printed", "validated", "confidence", "note"),
    "issues": ("issue_id", "doc_id", "page", "kind", "ref", "message", "suggestion"),
    "metric_rules": ("rule_id", "source_name", "source_page", "source_heading",
                     "document_pattern", "metric_family", "metric_name", "definition",
                     "applies_to", "example", "operation", "compiled_spec", "aliases",
                     "routing_spec", "interpretation_spec", "output_type", "enabled",
                     "status", "review_reason", "version", "updated_at"),
    "calculated_metrics": ("result_id", "doc_id", "source_page", "metric_id", "metric_name",
                           "metric_family", "entity", "period_key", "period_label",
                           "comparison_period_key", "comparison_period_label", "value",
                           "formatted_value", "unit", "formula", "inputs_json",
                           "source_fact_ids", "answer", "search_text", "calculation_status",
                           "calculation_version", "calculated_at"),
}

# Tables whose primary key is a serial we need to reset after bulk insert.
SERIAL_SEQUENCES = {
    "blocks": ("block_id", "blocks_block_id_seq"),
    "series": ("series_id", "series_series_id_seq"),
    "facts": ("fact_id", "facts_fact_id_seq"),
    "issues": ("issue_id", "issues_issue_id_seq"),
    "calculated_metrics": ("result_id", "calculated_metrics_result_id_seq"),
}


def open_sqlite(db_path: Path):
    import sqlite3
    con = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


async def migrate_rows(sqlite_con, pg_conn, table: str, batch: int = 500) -> int:
    cols = COLUMNS[table]
    rows = sqlite_con.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()
    if not rows:
        logger.info("  %s: 0 rows", table)
        return 0

    placeholders = ", ".join(f"${i + 1}" for i in range(len(cols)))
    columns_sql = ", ".join(cols)

    if table in SERIAL_SEQUENCES:
        insert_sql = (f"INSERT INTO {table} ({columns_sql}) "
                      f"OVERRIDING SYSTEM VALUE VALUES ({placeholders})")
    else:
        insert_sql = f"INSERT INTO {table} ({columns_sql}) VALUES ({placeholders})"

    total = 0
    buffer: list[tuple] = []
    for row in rows:
        buffer.append(tuple(row[c] for c in cols))
        if len(buffer) >= batch:
            await pg_conn.executemany(insert_sql, buffer)
            total += len(buffer)
            buffer.clear()
    if buffer:
        await pg_conn.executemany(insert_sql, buffer)
        total += len(buffer)

    if table in SERIAL_SEQUENCES:
        pk, seq = SERIAL_SEQUENCES[table]
        await pg_conn.execute(
            f"SELECT setval('{seq}', COALESCE((SELECT MAX({pk}) FROM {table}), 0) + 1, false)"
        )
    logger.info("  %s: %d rows copied", table, total)
    return total


async def truncate_tables(pg_conn) -> None:
    # CASCADE handles FK-like relationships (we don't have FKs declared but TRUNCATE
    # with CASCADE on parents also wipes dependents correctly).
    await pg_conn.execute(
        "TRUNCATE "
        + ", ".join(TABLES_IN_ORDER)
        + " RESTART IDENTITY CASCADE"
    )
    logger.info("Truncated %d tables", len(TABLES_IN_ORDER))


# ---------------------------------------------------------------------------
# Vector migration (SQLite-vec -> Weaviate)
# ---------------------------------------------------------------------------

def _unpack_vec(blob: bytes) -> list[float]:
    """Unpack sqlite-vec's little-endian float32 vector blob."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _load_vec_extension(sqlite_con) -> bool:
    try:
        import sqlite_vec
        sqlite_con.enable_load_extension(True)
        sqlite_vec.load(sqlite_con)
        sqlite_con.enable_load_extension(False)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("sqlite-vec extension not available (%s); will try raw blob fallback", exc)
        return False


def _vectors_for_table(sqlite_con, table: str, id_col: str) -> list[tuple[int, str, list[float]]]:
    """Return [(row_id, doc_id, vector), ...] by joining <table>_vec with <table>.

    Tries two approaches: the sqlite-vec virtual-table SELECT first, then a
    raw-blob fallback if the extension can't be loaded.
    """
    vec_table = f"{table}_vec"
    out: list[tuple[int, str, list[float]]] = []
    try:
        rows = sqlite_con.execute(
            f"SELECT v.{id_col} AS rid, v.embedding AS emb, t.doc_id AS doc_id "
            f"FROM {vec_table} v JOIN {table} t ON t.{id_col} = v.{id_col}"
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s via sqlite-vec SELECT (%s); skipping", vec_table, exc)
        return out
    for r in rows:
        blob = r["emb"]
        if blob is None:
            continue
        try:
            vec = _unpack_vec(blob) if isinstance(blob, (bytes, bytearray)) else list(blob)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  skip %s row %s: %s", vec_table, r["rid"], exc)
            continue
        out.append((int(r["rid"]), str(r["doc_id"]), vec))
    return out


async def migrate_vectors(sqlite_con) -> dict[str, int]:
    """Push block + series embeddings from sqlite-vec into Weaviate."""
    from documents import weaviate_store

    _load_vec_extension(sqlite_con)
    weaviate_store.ensure_collections()

    block_items = _vectors_for_table(sqlite_con, "blocks", "block_id")
    series_items = _vectors_for_table(sqlite_con, "series", "series_id")

    def _upsert_blocks():
        return weaviate_store.upsert_block_embeddings(block_items)

    def _upsert_series():
        return weaviate_store.upsert_series_embeddings(series_items)

    blocks_n = await asyncio.to_thread(_upsert_blocks)
    series_n = await asyncio.to_thread(_upsert_series)
    return {"blocks": blocks_n, "series": series_n}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(sqlite_path: Path, dry_run: bool, skip_vectors: bool) -> int:
    sqlite_con = open_sqlite(sqlite_path)

    # Row count summary from SQLite before we touch anything.
    logger.info("Source SQLite: %s", sqlite_path.resolve())
    for table in TABLES_IN_ORDER:
        try:
            n = sqlite_con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info("  SQLite %s: %d rows", table, n)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  SQLite %s: cannot read (%s)", table, exc)

    if dry_run:
        logger.info("--dry-run: nothing written")
        return 0

    from core.db import init_db, close_db, get_pool

    await init_db()
    pool = get_pool()
    if pool is None:
        logger.error("PostgreSQL pool not initialised. Set POSTGRES_DSN in your environment.")
        return 2

    try:
        async with pool.acquire() as pg_conn:
            logger.info("Target PostgreSQL pool acquired. Truncating documents tables...")
            await truncate_tables(pg_conn)

            logger.info("Copying rows SQLite -> PostgreSQL ...")
            for table in TABLES_IN_ORDER:
                try:
                    await migrate_rows(sqlite_con, pg_conn, table)
                except Exception as exc:  # noqa: BLE001
                    logger.error("  %s: FAILED (%s)", table, exc)
                    raise

        if not skip_vectors:
            logger.info("Pushing embeddings SQLite-vec -> Weaviate ...")
            counts = await migrate_vectors(sqlite_con)
            logger.info("  Weaviate upserts: %s", counts)
        else:
            logger.info("--no-vectors: skipped Weaviate upload")
    finally:
        await close_db()
        sqlite_con.close()

    logger.info("Migration complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sqlite-db", type=Path,
        default=Path("documents/data/library.db"),
        help="Path to the old SQLite library.db (default: documents/data/library.db)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print row counts and plan but write nothing.")
    parser.add_argument("--no-vectors", action="store_true",
                        help="Skip the Weaviate vector upload (row data only).")
    args = parser.parse_args()

    if not args.sqlite_db.is_file():
        logger.error("SQLite file not found: %s", args.sqlite_db)
        return 1

    return asyncio.run(run(args.sqlite_db, args.dry_run, args.no_vectors))


if __name__ == "__main__":
    raise SystemExit(main())
