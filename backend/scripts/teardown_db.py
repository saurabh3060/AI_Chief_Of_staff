"""
scripts/teardown_db.py
======================
Destructive teardown of the backend's PostgreSQL tables.

It does TWO things, in order, inside one transaction:

  1. TRUNCATE every managed table (empties all rows, resets identity
     sequences via RESTART IDENTITY).
  2. Strips STRUCTURE off those tables:
       * DROPs every secondary index.
       * DROPs every constraint (PRIMARY KEY, UNIQUE, FOREIGN KEY, CHECK).
     The empty table shells are left behind — ``scripts/rebuild_db.py``
     puts the constraints and indexes back from the single source of truth
     (``agent_email.db_schema`` + the DDL mirrored from ``core/db.py``).

Why not just DROP TABLE?
------------------------
This is the "clear the data and remove the plumbing but keep the table
names" variant you asked for. If you want the full drop-and-recreate flow
instead, ``scripts/reset_db.py`` already does that.

Managed tables
--------------
Everything the app creates in PostgreSQL:

    email_clusters, email_topics, meeting_embeddings, meeting_history,
    meeting_notes, email_summaries          (agent_email/db_schema.py)
    bu_emails (or CHATBOT_PG_TABLE)          (agent_email/db_schema.py)
    tracker_item_overrides,
    tracker_completed_items,
    tracker_deleted_items                    (core/db.py)
    chatbot_history                          (core/db.py)

(Delegated Graph OAuth tokens are no longer stored in PostgreSQL — both
the access and refresh tokens live in AWS Secrets Manager.)

Preserving the trackers
-----------------------
By default the Open (Action) Status Tracker tables ARE included. Pass
``--preserve-trackers`` to leave their data and structure untouched.

Safety
------
  * Dry-run by default. Without ``--confirm`` it only PRINTS the plan and
    changes nothing.
  * Only ever touches the explicit managed-table list below — it never
    discovers/drops arbitrary tables.

Usage (run from the backend/ directory)
---------------------------------------
    # Dry-run (prints the plan, changes nothing):
    python scripts/teardown_db.py

    # Actually truncate + strip constraints/indexes on ALL managed tables:
    python scripts/teardown_db.py --confirm

    # Same, but leave the Open Status Tracker tables alone:
    python scripts/teardown_db.py --confirm --preserve-trackers
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from core.config import get_settings  # noqa: E402
from core.db import init_db, close_db, get_pool  # noqa: E402


# Open (Action) Status Tracker tables — optionally preserved.
TRACKER_TABLES: tuple[str, ...] = (
    "tracker_item_overrides",
    "tracker_completed_items",
    "tracker_deleted_items",
)

# Email-agent + core tables (the chatbot source table name is appended at
# runtime from settings.chatbot_pg_table, since it is configurable).
EMAIL_TABLES: tuple[str, ...] = (
    "email_clusters",
    "email_topics",
    "meeting_embeddings",
    "meeting_history",
    "meeting_notes",
    "email_summaries",
)

CORE_TABLES: tuple[str, ...] = (
    "chatbot_history",
)


def managed_tables(preserve_trackers: bool) -> list[str]:
    """Full ordered list of tables this script operates on."""
    settings = get_settings()
    tables: list[str] = [*EMAIL_TABLES, *CORE_TABLES, settings.chatbot_pg_table]
    if not preserve_trackers:
        tables.extend(TRACKER_TABLES)
    # De-dupe while keeping order (chatbot_pg_table could collide in theory).
    seen: set[str] = set()
    ordered: list[str] = []
    for t in tables:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


async def _existing_tables(conn, candidates: list[str]) -> list[str]:
    """Subset of `candidates` that actually exist in the public schema."""
    rows = await conn.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
        "AND tablename = ANY($1::text[])",
        candidates,
    )
    present = {r["tablename"] for r in rows}
    return [t for t in candidates if t in present]


async def _constraints_for(conn, table: str) -> list[tuple[str, str]]:
    """Return (constraint_name, constraint_type) for a table.

    contype: p=primary key, u=unique, f=foreign key, c=check.
    """
    rows = await conn.fetch(
        """
        SELECT con.conname AS name, con.contype AS kind
          FROM pg_constraint con
          JOIN pg_class rel ON rel.oid = con.conrelid
          JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
         WHERE nsp.nspname = 'public'
           AND rel.relname = $1
        """,
        table,
    )
    return [(r["name"], r["kind"]) for r in rows]


async def _indexes_for(conn, table: str) -> list[str]:
    """Return names of indexes NOT backing a constraint (those drop with it)."""
    rows = await conn.fetch(
        """
        SELECT i.relname AS index_name
          FROM pg_index idx
          JOIN pg_class i   ON i.oid = idx.indexrelid
          JOIN pg_class t   ON t.oid = idx.indrelid
          JOIN pg_namespace n ON n.oid = t.relnamespace
         WHERE n.nspname = 'public'
           AND t.relname = $1
           AND NOT EXISTS (
               SELECT 1 FROM pg_constraint c WHERE c.conindid = idx.indexrelid
           )
        """,
        table,
    )
    return [r["index_name"] for r in rows]


async def _print_plan(preserve_trackers: bool) -> None:
    pool = get_pool()
    tables = managed_tables(preserve_trackers)
    print("=== teardown_db plan ===")
    print(f"preserve-trackers: {preserve_trackers}")
    async with pool.acquire() as conn:
        present = await _existing_tables(conn, tables)
        missing = [t for t in tables if t not in present]

        print("\nWill TRUNCATE (RESTART IDENTITY CASCADE) and strip structure from:")
        for t in present:
            cons = await _constraints_for(conn, t)
            idxs = await _indexes_for(conn, t)
            print(
                f"  - {t:28s} [{len(cons)} constraint(s), "
                f"{len(idxs)} standalone index(es)]"
            )
        if missing:
            print("\nNot present (skipped):")
            for t in missing:
                print(f"  - {t}")
        if preserve_trackers:
            print("\nPRESERVED (never touched):")
            for t in TRACKER_TABLES:
                print(f"  - {t}")
    print("========================")


async def _teardown(preserve_trackers: bool) -> None:
    pool = get_pool()
    if pool is None:
        raise RuntimeError(
            "_teardown: PostgreSQL pool unavailable (POSTGRES_DSN empty?)"
        )
    tables = managed_tables(preserve_trackers)

    async with pool.acquire() as conn:
        present = await _existing_tables(conn, tables)
        if not present:
            print("[PG] no managed tables present — nothing to do")
            return

        async with conn.transaction():
            # 1) Empty the data. One TRUNCATE for all tables so FK order
            #    doesn't matter; CASCADE + RESTART IDENTITY resets serials.
            quoted = ", ".join(f'"{t}"' for t in present)
            await conn.execute(
                f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE"
            )
            print(f"[PG] truncated {len(present)} table(s): {', '.join(present)}")

            # 2) Drop constraints. FOREIGN KEY first, then the rest, so a
            #    referenced PK/UNIQUE is never blocked by a dependent FK.
            for pass_kinds in (("f",), ("p", "u", "c")):
                for table in present:
                    for name, kind in await _constraints_for(conn, table):
                        if kind not in pass_kinds:
                            continue
                        await conn.execute(
                            f'ALTER TABLE "{table}" DROP CONSTRAINT '
                            f'IF EXISTS "{name}" CASCADE'
                        )
                        print(f"[PG] dropped constraint {table}.{name} ({kind})")

            # 3) Drop remaining standalone indexes (constraint-backed ones
            #    already went away with their constraint above).
            for table in present:
                for index_name in await _indexes_for(conn, table):
                    await conn.execute(f'DROP INDEX IF EXISTS "{index_name}"')
                    print(f"[PG] dropped index {index_name}")

    print("[PG] teardown complete — empty table shells remain")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Truncate the backend's PostgreSQL tables and strip their "
            "constraints + indexes, leaving empty table shells. Dry-run "
            "unless --confirm is passed."
        )
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually perform the teardown (otherwise dry-run).",
    )
    parser.add_argument(
        "--preserve-trackers",
        action="store_true",
        help="Leave the Open Status Tracker tables untouched.",
    )
    args = parser.parse_args()

    await init_db()
    try:
        if get_pool() is None:
            print("PostgreSQL is not configured (POSTGRES_DSN empty). Nothing to do.")
            return

        await _print_plan(args.preserve_trackers)

        if not args.confirm:
            print("\nDRY-RUN: nothing changed. Re-run with --confirm to apply.")
            return

        await _teardown(args.preserve_trackers)
        print("\nTeardown finished. Run scripts/rebuild_db.py to restore structure.")
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
