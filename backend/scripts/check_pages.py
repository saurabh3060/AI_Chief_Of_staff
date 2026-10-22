"""Quick check: is the pages table also empty?"""
import asyncio
from core.db import init_db, get_pool, close_db

async def main():
    await init_db()
    pool = get_pool()
    async with pool.acquire() as conn:
        pages = await conn.fetchval("SELECT COUNT(*) FROM pages")
        sync_files = await conn.fetchval("SELECT COUNT(*) FROM onedrive_sync_files")
        sync_runs = await conn.fetch("SELECT status, summary, created_at FROM onedrive_sync_runs ORDER BY created_at DESC LIMIT 3")
        print(f"pages table: {pages} rows")
        print(f"onedrive_sync_files: {sync_files} rows")
        print(f"onedrive_sync_runs (last 3):")
        for r in sync_runs:
            print(f"  status={r['status']}  created_at={r['created_at']}  summary={r['summary']}")
        if not sync_runs:
            print("  (no sync runs recorded)")
    await close_db()

if __name__ == "__main__":
    asyncio.run(main())
