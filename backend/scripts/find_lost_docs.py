"""
scripts/find_lost_docs.py
==========================
Search for any trace of the two Credit Cards documents across all
documents module tables, plus check if their doc_ids were reused.

Usage:
    cd backend
    python -m scripts.find_lost_docs
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

            # 1. Search documents table for anything with "credit" or "bikram" or "cc"
            print("═" * 80)
            print("  Searching documents table for credit card related entries...")
            print("═" * 80)
            rows = await conn.fetch(
                "SELECT doc_id, file_name, title, status, pages, ingested_at "
                "FROM documents "
                "WHERE LOWER(file_name) LIKE '%credit%' "
                "   OR LOWER(file_name) LIKE '%bikram%' "
                "   OR LOWER(file_name) LIKE '%cc%' "
                "   OR LOWER(title) LIKE '%credit card%' "
                "ORDER BY ingested_at DESC"
            )
            if rows:
                for r in rows:
                    print(f"\n  file_name  : {r['file_name']}")
                    print(f"  doc_id     : {r['doc_id']}")
                    print(f"  title      : {(r['title'] or '')[:80]}")
                    print(f"  status     : {r['status']}  pages={r['pages']}")
                    print(f"  ingested_at: {r['ingested_at']}")
            else:
                print("  (none found)")

            # 2. Check blocks/series for orphaned data from those files
            print(f"\n{'═' * 80}")
            print("  Searching blocks for orphaned credit card content...")
            print("═" * 80)
            orphan_blocks = await conn.fetch(
                "SELECT DISTINCT b.doc_id, LEFT(b.text, 100) AS preview "
                "FROM blocks b "
                "WHERE (LOWER(b.text) LIKE '%credit cards%business update dec%' "
                "   OR  LOWER(b.text) LIKE '%bikram%') "
                "LIMIT 10"
            )
            if orphan_blocks:
                for ob in orphan_blocks:
                    exists = await conn.fetchval(
                        "SELECT file_name FROM documents WHERE doc_id = $1",
                        ob['doc_id'],
                    )
                    print(f"\n  doc_id  : {ob['doc_id']}")
                    print(f"  in docs?: {exists or 'NOT FOUND (orphaned!)'}")
                    print(f"  preview : {ob['preview']}")
            else:
                print("  (no orphaned blocks found)")

            # 3. Check onedrive_sync_files for those file names
            print(f"\n{'═' * 80}")
            print("  Searching onedrive_sync_files for credit card entries...")
            print("═" * 80)
            sync_rows = await conn.fetch(
                "SELECT item_id, name, relative_path, doc_id, synced_at, last_modified "
                "FROM onedrive_sync_files "
                "WHERE LOWER(name) LIKE '%credit%' "
                "   OR LOWER(name) LIKE '%bikram%' "
                "   OR LOWER(name) LIKE '%cc12%' "
                "ORDER BY synced_at DESC"
            )
            if sync_rows:
                for sr in sync_rows:
                    print(f"\n  name          : {sr['name']}")
                    print(f"  relative_path : {sr['relative_path']}")
                    print(f"  doc_id        : {sr['doc_id']}")
                    print(f"  synced_at     : {sr['synced_at']}")
                    print(f"  last_modified : {sr['last_modified']}")
                    # Check if this doc_id still exists in documents
                    doc = await conn.fetchrow(
                        "SELECT file_name, status FROM documents WHERE doc_id = $1",
                        sr['doc_id'],
                    )
                    if doc:
                        print(f"  → documents table: {doc['file_name']} ({doc['status']})")
                    else:
                        print(f"  → documents table: MISSING (deleted!)")
            else:
                print("  (none found)")

            # 4. Show ALL distinct doc_ids in pages/blocks/series that have no parent doc
            print(f"\n{'═' * 80}")
            print("  Checking for orphaned doc_ids (in series but not in documents)...")
            print("═" * 80)
            orphans = await conn.fetch("""
                SELECT DISTINCT s.doc_id,
                       (SELECT COUNT(*) FROM series WHERE doc_id = s.doc_id) AS n_series,
                       (SELECT COUNT(*) FROM facts f JOIN series ss ON ss.series_id = f.series_id WHERE ss.doc_id = s.doc_id) AS n_facts
                FROM series s
                WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.doc_id = s.doc_id)
            """)
            if orphans:
                for o in orphans:
                    print(f"  doc_id={o['doc_id']}  series={o['n_series']}  facts={o['n_facts']}")
            else:
                print("  (no orphaned series — all doc_ids have a parent document)")

    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
