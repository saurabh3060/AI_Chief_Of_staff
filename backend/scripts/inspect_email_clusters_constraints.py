"""
scripts/inspect_email_clusters_constraints.py
===============================================
Diagnostic, READ-ONLY script: lists every constraint and index on the
`email_clusters` table so we can find the real name/definition of
whatever is enforcing "one row per meeting_id" (the thing raising
`duplicate key value violates unique constraint "uq_email_clusters_active_meeting"`
in the pipeline logs).

Does NOT modify the database in any way - only runs SELECT queries
against Postgres system catalogs (pg_constraint / pg_indexes).

Connects using the exact same POSTGRES_DSN the app uses (via
core.config.get_settings()), so there is no risk of accidentally
inspecting the wrong database/environment.

Usage (from the `backend/` directory, with the project venv active):
    python scripts/inspect_email_clusters_constraints.py
"""

import asyncio
import sys
from pathlib import Path

# Allow running this script directly (python scripts\inspect_....py)
# without the backend/ root being on sys.path - add it ourselves so
# `core.config` resolves the same way it does for main.py/uvicorn.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg

from core.config import get_settings


async def main() -> None:
    settings = get_settings()
    dsn = settings.postgres_dsn
    if not dsn:
        print("POSTGRES_DSN is empty in settings/.env - nothing to inspect.")
        return

    # Don't print the DSN itself (contains credentials) - just confirm
    # which host/db we're hitting, parsed out safely.
    safe_target = dsn.split("@")[-1] if "@" in dsn else "(unparsed dsn)"
    print(f"Connecting to: {safe_target}\n")

    conn = await asyncpg.connect(dsn=dsn)
    try:
        print("=" * 70)
        print("CONSTRAINTS on email_clusters (pg_constraint)")
        print("=" * 70)
        constraint_rows = await conn.fetch(
            """
            SELECT conname, contype, pg_get_constraintdef(oid) AS definition
            FROM pg_constraint
            WHERE conrelid = 'email_clusters'::regclass
            ORDER BY conname;
            """
        )
        if not constraint_rows:
            print("(no constraints found)")
        for row in constraint_rows:
            print(f"- name={row['conname']!r} type={row['contype']!r}")
            print(f"    def: {row['definition']}")

        print()
        print("=" * 70)
        print("INDEXES on email_clusters (pg_indexes)")
        print("=" * 70)
        index_rows = await conn.fetch(
            """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'email_clusters'
            ORDER BY indexname;
            """
        )
        if not index_rows:
            print("(no indexes found)")
        for row in index_rows:
            print(f"- {row['indexname']}")
            print(f"    def: {row['indexdef']}")

        print()
        print("=" * 70)
        print("Rows currently sharing a meeting_id (would collide on insert)")
        print("=" * 70)
        dup_rows = await conn.fetch(
            """
            SELECT meeting_id, COUNT(*) AS cnt, array_agg(cluster_id) AS cluster_ids
            FROM email_clusters
            WHERE meeting_id IS NOT NULL AND meeting_id <> ''
            GROUP BY meeting_id
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC;
            """
        )
        if not dup_rows:
            print("(no meeting_id currently has more than one row)")
        for row in dup_rows:
            print(f"- meeting_id={row['meeting_id']!r} count={row['cnt']} clusters={row['cluster_ids']}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
