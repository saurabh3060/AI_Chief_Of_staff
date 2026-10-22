"""
scripts/rebuild_db.py
=====================
Rebuild the backend's PostgreSQL tables, constraints and indexes to the
current, known-good schema.

This is the companion to ``scripts/teardown_db.py``. After a teardown has
truncated the tables and stripped their constraints/indexes (leaving empty
shells), this script puts the full structure back:

  * email_clusters, email_topics, meeting_embeddings, meeting_history,
    meeting_notes, email_summaries  — rebuilt from the single source of
    truth ``agent_email.db_schema.create_email_schema`` (primary keys,
    the ``is_active`` flag, every secondary index, and all additive
    column migrations).
  * bu_emails (or ``CHATBOT_PG_TABLE``) — rebuilt from
    ``agent_email.db_schema.create_bu_emails_schema`` (including the
    load-bearing UNIQUE index on ``message_id``).
  * chatbot_history and the three tracker tables — rebuilt from DDL
    mirrored here from ``core/db.py`` (the place those tables are normally
    created at app startup). NOTE: delegated Graph OAuth tokens are no
    longer stored in PostgreSQL (both tokens live in AWS Secrets Manager),
    so there is no ``graph_oauth_tokens`` table to rebuild.

Because every statement in those builders is ``CREATE TABLE/INDEX IF NOT
EXISTS`` (and column adds are ``ADD COLUMN IF NOT EXISTS``), this script is
idempotent and safe to run even on a database that already has some or all
of the structure in place — it only fills in what is missing. It does NOT
drop or truncate anything.

Usage (run from the backend/ directory)
---------------------------------------
    # Dry-run (prints the plan, changes nothing):
    python scripts/rebuild_db.py

    # Actually (re)create all tables, constraints and indexes:
    python scripts/rebuild_db.py --confirm
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
from agent_email.db_schema import (  # noqa: E402
    create_email_schema,
    create_bu_emails_schema,
)


# ---------------------------------------------------------------------------
# Core tables created ad-hoc in core/db.py (NOT in agent_email/db_schema.py).
# Mirrored here verbatim so a rebuild restores them identically. Keep in
# sync with core/db.py::init_db.
# ---------------------------------------------------------------------------
CORE_TABLE_DDL: list[tuple[str, str]] = [
    (
        "tracker_item_overrides",
        """
        CREATE TABLE IF NOT EXISTS tracker_item_overrides (
            id          VARCHAR(16) PRIMARY KEY,
            text        TEXT,
            owner       TEXT,
            business_unit TEXT,
            due_date    TEXT,
            updated_at  TIMESTAMPTZ DEFAULT now()
        );
        """,
    ),
    (
        "tracker_completed_items",
        """
        CREATE TABLE IF NOT EXISTS tracker_completed_items (
            id          VARCHAR(16) PRIMARY KEY,
            text        TEXT NOT NULL,
            owner       TEXT NOT NULL,
            business_unit TEXT NOT NULL,
            due_date    TEXT,
            source_topic TEXT,
            completed_at TIMESTAMPTZ DEFAULT now()
        );
        """,
    ),
    (
        "tracker_deleted_items",
        """
        CREATE TABLE IF NOT EXISTS tracker_deleted_items (
            id          VARCHAR(16) PRIMARY KEY,
            deleted_at  TIMESTAMPTZ DEFAULT now()
        );
        """,
    ),
    (
        "chatbot_history",
        """
        CREATE TABLE IF NOT EXISTS chatbot_history (
            id          BIGSERIAL PRIMARY KEY,
            question    TEXT NOT NULL,
            answer      TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
    ),
]

CORE_INDEX_DDL: list[tuple[str, str]] = [
    (
        "idx_chatbot_history_created_at",
        "CREATE INDEX IF NOT EXISTS idx_chatbot_history_created_at "
        "ON chatbot_history (created_at);",
    ),
]


def _print_plan() -> None:
    settings = get_settings()
    print("=== rebuild_db plan ===")
    print("Will CREATE IF NOT EXISTS (idempotent — fills in what is missing):")
    print("\n  Core tables (from core/db.py DDL):")
    for name, _ in CORE_TABLE_DDL:
        print(f"    - {name}")
    for name, _ in CORE_INDEX_DDL:
        print(f"    - index {name}")
    print("\n  Email-agent tables + indexes (from create_email_schema):")
    for name in (
        "email_clusters", "email_topics", "meeting_embeddings",
        "meeting_history", "meeting_notes", "email_summaries",
    ):
        print(f"    - {name}")
    print("\n  Chatbot source table + indexes (from create_bu_emails_schema):")
    print(f"    - {settings.chatbot_pg_table}")
    print("=======================")


async def _rebuild() -> None:
    pool = get_pool()
    if pool is None:
        raise RuntimeError(
            "_rebuild: PostgreSQL pool unavailable (POSTGRES_DSN empty?)"
        )
    settings = get_settings()

    async with pool.acquire() as conn:
        # 1) Core tables + indexes (mirrored from core/db.py).
        async with conn.transaction():
            for name, ddl in CORE_TABLE_DDL:
                await conn.execute(ddl)
                print(f"[PG] ensured core table {name}")
            for name, ddl in CORE_INDEX_DDL:
                await conn.execute(ddl)
                print(f"[PG] ensured core index {name}")

        # 2) Email-agent tables + constraints + indexes (source of truth).
        async with conn.transaction():
            await create_email_schema(conn)
        print("[PG] rebuilt email-agent tables via create_email_schema")

        # 3) Chatbot raw-message source table + its unique/secondary indexes.
        #    create_bu_emails_schema manages its own per-index savepoints,
        #    so it is not wrapped in an outer transaction here.
        await create_bu_emails_schema(conn, settings.chatbot_pg_table)
        print(
            f"[PG] rebuilt chatbot source table '{settings.chatbot_pg_table}' "
            f"via create_bu_emails_schema"
        )

    print("[PG] rebuild complete")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the backend's PostgreSQL tables, constraints and "
            "indexes from the current schema (idempotent, non-destructive). "
            "Dry-run unless --confirm is passed."
        )
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually perform the rebuild (otherwise dry-run).",
    )
    args = parser.parse_args()

    await init_db()
    try:
        if get_pool() is None:
            print("PostgreSQL is not configured (POSTGRES_DSN empty). Nothing to do.")
            return

        _print_plan()

        if not args.confirm:
            print("\nDRY-RUN: nothing changed. Re-run with --confirm to apply.")
            return

        await _rebuild()
        print("\nRebuild finished.")
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
