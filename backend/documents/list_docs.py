"""List all documents in the PostgreSQL database.

Usage:
  python -m documents.list_docs
"""
from __future__ import annotations
import asyncio
from . import store


async def run():
    from core.db import init_db, close_db
    await init_db()
    try:
        async with store.acquire() as conn:
            rows = await conn.fetch(
                "SELECT doc_id, file_name, status, pages FROM documents ORDER BY file_name"
            )
            print(f"\n{'doc_id':>20s}  {'pages':>5s}  {'status':>8s}  file_name")
            print("-" * 90)
            for r in rows:
                print(f"{r['doc_id']:>20s}  {r['pages'] or 0:>5d}  {r['status']:>8s}  {r['file_name']}")
            print(f"\nTotal: {len(rows)} documents")
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(run())
