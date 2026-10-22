"""
check_sync.py — Quick script to see which files were synced vs skipped.

Usage:
    python check_sync.py              # show all tracked files + latest run
    python check_sync.py --recent     # only files synced in the last run
"""

import asyncio
import os
import sys

import asyncpg
from dotenv import load_dotenv

load_dotenv()

DSN = os.getenv("POSTGRES_DSN", "")


async def main():
    if not DSN:
        print("ERROR: POSTGRES_DSN not set in .env")
        sys.exit(1)

    conn = await asyncpg.connect(DSN)

    # ---- Latest sync run ----
    run = await conn.fetchrow(
        "SELECT status, summary, created_at "
        "FROM onedrive_sync_runs ORDER BY created_at DESC LIMIT 1"
    )
    if run:
        print("=" * 60)
        print(f"  LAST SYNC RUN")
        print(f"  Status : {run['status']}")
        print(f"  Time   : {run['created_at']}")
        if run["summary"]:
            import json
            summary = run["summary"] if isinstance(run["summary"], dict) else json.loads(run["summary"])
            for k, v in summary.items():
                print(f"  {k:10s}: {v}")
        print("=" * 60)
    else:
        print("No sync runs found.")
        await conn.close()
        return

    # ---- Determine filter ----
    recent_only = "--recent" in sys.argv
    if recent_only and run:
        # Files synced within 10 minutes of the last run
        rows = await conn.fetch(
            """
            SELECT name, relative_path, doc_id, last_modified, size, synced_at
            FROM onedrive_sync_files
            WHERE synced_at >= $1 - INTERVAL '10 minutes'
            ORDER BY synced_at DESC
            """,
            run["created_at"],
        )
        print(f"\n  FILES SYNCED IN LAST RUN ({len(rows)}):\n")
    else:
        rows = await conn.fetch(
            """
            SELECT name, relative_path, doc_id, last_modified, size, synced_at
            FROM onedrive_sync_files
            ORDER BY synced_at DESC NULLS LAST
            """,
        )
        print(f"\n  ALL TRACKED FILES ({len(rows)}):\n")

    # ---- Print table ----
    if not rows:
        print("  (none)")
    else:
        print(f"  {'#':<4} {'File Name':<45} {'Synced At':<22} {'Size':>10}")
        print(f"  {'-'*4} {'-'*45} {'-'*22} {'-'*10}")
        for i, r in enumerate(rows, 1):
            synced = str(r["synced_at"])[:19] if r["synced_at"] else "never"
            size_kb = f"{(r['size'] or 0) / 1024:.0f} KB"
            print(f"  {i:<4} {r['name']:<45} {synced:<22} {size_kb:>10}")

    print()
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
