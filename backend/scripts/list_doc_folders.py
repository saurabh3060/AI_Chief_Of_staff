"""
scripts/list_doc_folders.py
============================
Show all documents stored in the documents module PostgreSQL tables,
with a per-document breakdown of row counts across every table.

Usage:
    cd backend
    python -m scripts.list_doc_folders
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


async def main() -> None:
    from core.db import init_db, close_db, get_pool

    await init_db()
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            # 1. All documents
            docs = await conn.fetch(
                "SELECT doc_id, file_name, title, doc_type, status, pages, "
                "       as_of_label, ingested_at "
                "FROM documents ORDER BY file_name"
            )

            if not docs:
                print("No documents found in the documents table.")
                return

            print(f"\n{'═' * 90}")
            print(f"  Documents in PostgreSQL  ({len(docs)} total)")
            print(f"{'═' * 90}\n")

            for i, d in enumerate(docs, 1):
                doc_id = d["doc_id"]

                # Count rows in each related table for this doc
                n_pages = await conn.fetchval(
                    "SELECT count(*) FROM pages WHERE doc_id = $1", doc_id)
                n_blocks = await conn.fetchval(
                    "SELECT count(*) FROM blocks WHERE doc_id = $1", doc_id)
                n_series = await conn.fetchval(
                    "SELECT count(*) FROM series WHERE doc_id = $1", doc_id)
                n_facts = await conn.fetchval(
                    "SELECT count(*) FROM facts WHERE doc_id = $1", doc_id)
                n_issues = await conn.fetchval(
                    "SELECT count(*) FROM issues WHERE doc_id = $1", doc_id)
                n_corrections = await conn.fetchval(
                    "SELECT count(*) FROM corrections WHERE doc_id = $1", doc_id)
                n_calculated = await conn.fetchval(
                    "SELECT count(*) FROM calculated_metrics WHERE doc_id = $1", doc_id)

                ingested = str(d["ingested_at"] or "")[:19]
                print(f"  {i}. {d['file_name']}")
                print(f"     doc_id  : {doc_id}")
                print(f"     title   : {(d['title'] or '')[:70]}")
                print(f"     type    : {d['doc_type'] or '-'}    status: {d['status']}    as_of: {d['as_of_label'] or '-'}")
                print(f"     ingested: {ingested}")
                print(f"     rows    : {n_pages} pages, {n_blocks} blocks, {n_series} series, "
                      f"{n_facts} facts, {n_issues} issues, {n_corrections} corrections, "
                      f"{n_calculated} calculated_metrics")
                print()

            # 2. Global metric_rules count (not per-doc)
            n_rules = await conn.fetchval("SELECT count(*) FROM metric_rules")
            print(f"{'─' * 90}")
            print(f"  metric_rules (global, not per-document): {n_rules} rule(s)")
            print(f"{'═' * 90}")

    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
