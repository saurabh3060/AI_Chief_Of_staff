"""
scripts/show_sync_status.py
============================
Show which files were synced vs skipped in the last sync run,
and for synced files show what changed (last_modified / size diff).

Usage:
    cd backend
    python -m scripts.show_sync_status
"""
from __future__ import annotations

import asyncio
import json
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
            # ── Last two sync runs ──
            runs = await conn.fetch(
                "SELECT status, summary, created_at "
                "FROM onedrive_sync_runs ORDER BY created_at DESC LIMIT 2"
            )
            if not runs:
                print("  No sync runs recorded.")
                return

            latest_run = runs[0]
            prev_run = runs[1] if len(runs) > 1 else None
            run_ts = latest_run['created_at']

            print(f"\n  Latest sync : {latest_run['status']}  at {str(run_ts)[:19]}")
            s = latest_run['summary']
            if isinstance(s, str):
                s = json.loads(s)
            if s:
                print(f"  Summary     : synced={s.get('synced',0)} skipped={s.get('skipped',0)} "
                      f"failed={s.get('failed',0)} found={s.get('files_found',0)} "
                      f"duration={s.get('duration_s',0)}s")
            if prev_run:
                print(f"  Previous run: {prev_run['status']}  at {str(prev_run['created_at'])[:19]}")

            # ── All tracked files ──
            rows = await conn.fetch(
                "SELECT item_id, name, relative_path, last_modified, "
                "       size, doc_id, synced_at "
                "FROM onedrive_sync_files ORDER BY synced_at DESC NULLS LAST, name"
            )

            if not rows:
                print("\n  No files tracked.")
                return

            synced_files = []
            skipped_files = []

            for r in rows:
                ts = r['synced_at']
                if run_ts and ts and abs((ts - run_ts).total_seconds()) < 900:
                    synced_files.append(r)
                else:
                    skipped_files.append(r)

            # ── SYNCED files ──
            print(f"\n{'═' * 90}")
            print(f"  SYNCED in last run ({len(synced_files)} files)")
            print(f"{'═' * 90}")
            for i, f in enumerate(synced_files, 1):
                path = f['relative_path']
                display = f"{path}/{f['name']}" if path else f['name']
                size_kb = (f['size'] or 0) / 1024
                print(f"\n  {i:>3}. {display}")
                print(f"       size         : {size_kb:,.0f} KB")
                print(f"       last_modified : {f['last_modified']}")
                print(f"       synced_at     : {str(f['synced_at'])[:19]}")
                print(f"       doc_id        : {f['doc_id']}")

            # ── SKIPPED files ──
            print(f"\n{'═' * 90}")
            print(f"  SKIPPED / unchanged ({len(skipped_files)} files)")
            print(f"{'═' * 90}")
            for i, f in enumerate(skipped_files, 1):
                path = f['relative_path']
                display = f"{path}/{f['name']}" if path else f['name']
                size_kb = (f['size'] or 0) / 1024
                print(f"\n  {i:>3}. {display}")
                print(f"       size         : {size_kb:,.0f} KB")
                print(f"       last_modified : {f['last_modified']}")
                print(f"       synced_at     : {str(f['synced_at'] or 'never')[:19]}")

            print(f"\n{'═' * 90}")
            print(f"  Total tracked: {len(rows)} files  "
                  f"(synced={len(synced_files)}, skipped={len(skipped_files)})")
            print(f"{'═' * 90}")

    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
