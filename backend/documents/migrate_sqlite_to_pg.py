"""Migrate the other developer's SQLite library.db into your PostgreSQL database.

Copies: documents, pages, blocks, series, facts, issues, corrections,
        metric_rules, calculated_metrics.

Existing PG rows are UPSERTED (matched by primary key) so the script is
safe to re-run.  New SQLite rows are added; conflicting rows are updated.

Use --clean to TRUNCATE all target tables before importing (recommended for
a fresh start so you don't end up with orphan rows from a prior ingestion).

Usage:
  python -m documents.migrate_sqlite_to_pg "C:\\path\\to\\library.db" --dry-run
  python -m documents.migrate_sqlite_to_pg "C:\\path\\to\\library.db" --clean
"""
from __future__ import annotations

import argparse
import asyncio
import sqlite3
from pathlib import Path

from . import store


# ---------------------------------------------------------------------------
# Table migration specs: (table, pk_columns, all_columns)
# Each spec drives a generic INSERT ... ON CONFLICT (pk) DO UPDATE.
# Column order must match the SQLite SELECT and PG INSERT.
# ---------------------------------------------------------------------------

TABLES = [
    # --- documents ---
    ("documents", ["doc_id"], [
        "doc_id", "file_name", "title", "doc_type", "entity",
        "as_of_key", "as_of_label", "pages", "status", "progress",
    ]),
    # --- pages ---
    ("pages", ["doc_id", "page"], [
        "doc_id", "page", "heading", "text", "image", "record",
    ]),
    # --- blocks ---
    ("blocks", ["block_id"], [
        "block_id", "doc_id", "page", "kind", "section", "title", "text",
        "content", "bbox", "source", "confidence", "section_path", "anchor_block_id",
    ]),
    # --- series ---
    ("series", ["series_id"], [
        "series_id", "doc_id", "page", "block_id", "label", "name",
        "metric_id", "kind", "unit", "entity", "source", "confidence", "path", "scale",
    ]),
    # --- facts ---
    ("facts", ["fact_id"], [
        "fact_id", "series_id", "doc_id", "page", "period_key", "period_kind",
        "period_start", "period_end", "period_label", "partial",
        "value", "as_printed", "validated", "confidence", "note",
    ]),
    # --- issues ---
    ("issues", ["issue_id"], [
        "issue_id", "doc_id", "page", "kind", "ref", "message", "suggestion",
    ]),
    # --- corrections ---
    ("corrections", ["doc_id", "page", "series_label", "period_key"], [
        "doc_id", "page", "series_label", "period_key", "value", "note",
    ]),
    # --- metric_rules ---
    ("metric_rules", ["rule_id"], [
        "rule_id", "source_name", "source_page", "source_heading", "document_pattern",
        "metric_family", "metric_name", "definition", "applies_to", "example",
        "operation", "compiled_spec", "aliases", "routing_spec", "interpretation_spec",
        "output_type", "enabled", "status", "review_reason", "version",
    ]),
    # --- calculated_metrics (PK = composite unique, use result_id for identity) ---
    ("calculated_metrics",
     ["doc_id", "source_page", "metric_id", "entity", "period_key",
      "comparison_period_key", "calculation_version"],
     [
         "doc_id", "source_page", "metric_id", "metric_name", "metric_family",
         "entity", "period_key", "period_label", "comparison_period_key",
         "comparison_period_label", "value", "formatted_value", "unit", "formula",
         "inputs_json", "source_fact_ids", "answer", "search_text",
         "calculation_status", "calculation_version",
     ]),
]


def _sqlite_has_column(cursor, table: str, column: str) -> bool:
    """Check whether a column exists in the SQLite table."""
    info = cursor.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in info)


def _safe_columns(cursor, table: str, wanted: list[str]) -> list[str]:
    """Return only the columns that actually exist in the SQLite table."""
    info = cursor.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {row[1] for row in info}
    return [col for col in wanted if col in existing]


def _build_upsert_sql(table: str, pk_cols: list[str], all_cols: list[str]) -> str:
    """Build a PostgreSQL INSERT ... ON CONFLICT DO UPDATE statement."""
    placeholders = ", ".join(f"${i+1}" for i in range(len(all_cols)))
    col_list = ", ".join(all_cols)
    pk_list = ", ".join(pk_cols)
    update_cols = [c for c in all_cols if c not in pk_cols]
    if update_cols:
        set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
        conflict = f"ON CONFLICT ({pk_list}) DO UPDATE SET {set_clause}"
    else:
        conflict = f"ON CONFLICT ({pk_list}) DO NOTHING"
    return f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) {conflict}"


async def migrate(sqlite_path: Path, *, dry_run: bool = False, clean: bool = False) -> dict:
    from core.db import init_db, close_db

    # --- Open SQLite ---
    lite = sqlite3.connect(str(sqlite_path))
    lite.row_factory = sqlite3.Row

    await init_db()
    report = {}

    try:
        async with store.acquire() as conn:
            # --- Optional: truncate all target tables first ---
            if clean and not dry_run:
                print("CLEANING: truncating all target tables...\n")
                await conn.execute(
                    "TRUNCATE calculated_metrics, metric_rules, issues, corrections, "
                    "facts, series, blocks, pages, documents CASCADE"
                )
                print("  All tables truncated.\n")
            elif clean and dry_run:
                print("CLEANING: would truncate all tables (dry run — skipped)\n")

            for table, pk_cols, wanted_cols in TABLES:
                # Check which columns actually exist in SQLite
                available = _safe_columns(lite.cursor(), table, wanted_cols)
                if not available:
                    print(f"  SKIP {table}: not found in SQLite")
                    report[table] = {"status": "skipped", "rows": 0}
                    continue

                # For PG insert we need all wanted columns; missing ones get defaults
                missing_cols = set(wanted_cols) - set(available)

                # Read from SQLite (only available columns)
                select_cols = ", ".join(available)
                rows = lite.execute(f"SELECT {select_cols} FROM {table}").fetchall()
                print(f"  {table}: {len(rows)} rows from SQLite", end="")

                if dry_run:
                    print(" (dry run — not written)")
                    report[table] = {"status": "dry_run", "rows": len(rows)}
                    continue

                if not rows:
                    print(" (empty)")
                    report[table] = {"status": "ok", "rows": 0}
                    continue

                # Build tuples for PG insert (fill missing columns with defaults)
                defaults = {
                    "doc_type": "", "entity": "", "as_of_key": "", "as_of_label": "",
                    "progress": "", "image": "", "record": "", "content": None,
                    "bbox": None, "section_path": None, "anchor_block_id": None,
                    "block_id": None, "metric_id": "", "scale": None,
                    "source_heading": "", "document_pattern": "",
                    "interpretation_spec": "{}", "review_reason": "",
                    "note": "", "suggestion": "",
                }

                pg_rows = []
                for row in rows:
                    values = []
                    for col in wanted_cols:
                        if col in available:
                            val = row[col]
                            # Coerce None text columns to empty string for NOT NULL PG columns
                            if val is None and col not in (
                                "value", "content", "bbox", "section_path",
                                "anchor_block_id", "block_id", "scale",
                            ):
                                val = defaults.get(col, "")
                            values.append(val)
                        else:
                            values.append(defaults.get(col, ""))
                    pg_rows.append(tuple(values))

                # For tables with SERIAL/IDENTITY PKs (blocks, series, facts, issues),
                # we need to reset the sequence after inserting explicit IDs.
                serial_tables = {
                    "blocks": "blocks_block_id_seq",
                    "series": "series_series_id_seq",
                    "facts": "facts_fact_id_seq",
                    "issues": "issues_issue_id_seq",
                    "calculated_metrics": "calculated_metrics_result_id_seq",
                }

                # Use the actual wanted_cols for the PG upsert (not just available)
                sql = _build_upsert_sql(table, pk_cols, wanted_cols)

                # Insert in batches inside a transaction
                async with conn.transaction():
                    # For SERIAL columns, temporarily allow explicit ID insertion
                    if table in serial_tables:
                        # Override the GENERATED BY DEFAULT by just inserting — PG allows it
                        pass

                    batch_size = 500
                    for i in range(0, len(pg_rows), batch_size):
                        batch = pg_rows[i:i + batch_size]
                        await conn.executemany(sql, batch)

                    # Reset sequence to max+1 so future inserts don't collide
                    if table in serial_tables:
                        id_col = pk_cols[0] if len(pk_cols) == 1 else None
                        if id_col:
                            max_val = await conn.fetchval(
                                f"SELECT COALESCE(MAX({id_col}), 0) FROM {table}"
                            )
                            seq_name = serial_tables[table]
                            await conn.execute(
                                f"SELECT setval('{seq_name}', $1, true)", max_val
                            )

                print(f" -> inserted/updated in PG")
                report[table] = {"status": "ok", "rows": len(pg_rows)}

    finally:
        lite.close()
        await close_db()

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite_db", type=Path, help="Path to the SQLite library.db file")
    parser.add_argument("--dry-run", action="store_true", help="Read SQLite but don't write to PG")
    parser.add_argument("--clean", action="store_true",
                        help="TRUNCATE all target tables before importing (recommended for fresh start)")
    args = parser.parse_args()

    if not args.sqlite_db.is_file():
        parser.error(f"File not found: {args.sqlite_db}")

    print(f"Source: {args.sqlite_db}")
    print(f"Mode:   {'DRY RUN' if args.dry_run else 'LIVE — will write to PostgreSQL'}\n")

    report = asyncio.run(migrate(args.sqlite_db, dry_run=args.dry_run, clean=args.clean))

    print("\n=== SUMMARY ===")
    total = 0
    for table, info in report.items():
        print(f"  {table:25s}  {info['rows']:>8,} rows  [{info['status']}]")
        total += info["rows"]
    print(f"  {'TOTAL':25s}  {total:>8,} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
