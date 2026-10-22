"""
scripts/reset_db.py
====================
Destructive reset of the email-agent data stores to a clean, known-good
structure — in BOTH PostgreSQL and Weaviate — WITHOUT touching the Open
(Action) Status Tracker tables.

What it does
------------
PostgreSQL:
  * DROPs and recreates the email-agent tables so they come back with the
    exact current schema (including the ``is_active`` flag on
    ``email_clusters`` and all indexes). The schema itself is NOT
    duplicated here — it is rebuilt from the single source of truth,
    ``agent_email.db_schema.create_email_schema``, so this script can never
    drift from the real schema.
  * Delegated Graph OAuth tokens are NOT touched here — they are no longer
    stored in PostgreSQL (both the access and refresh tokens live in AWS
    Secrets Manager). There is no ``graph_oauth_tokens`` table to reset.

Weaviate:
  * Deletes and recreates THIS project's owned collections (``clusters``,
    ``meetings``) so the vector copy starts empty and matches the fresh,
    all-active PostgreSQL state.

What it NEVER touches
---------------------
The Open Status Tracker tables are explicitly preserved:
    tracker_item_overrides
    tracker_completed_items
    tracker_deleted_items
These are listed in ``PRESERVE_TABLES`` and the script refuses to drop
them. If a future change adds a tracker table, add it there too.

Safety
------
This is destructive and HARD to misuse by accident:
  * Dry-run by default. Without ``--confirm`` it only PRINTS what it would
    drop/recreate and changes nothing.
  * Weaviate deletes go through the same gated ``_admin_delete_collection``
    the app uses, which refuses to touch any collection that is not one of
    this project's own (guards a shared Weaviate server).
  * ``--scope`` limits the blast radius to ``postgres``, ``weaviate``, or
    ``all`` (default ``all``).

Usage (run from the backend/ directory)
---------------------------------------
    # Dry-run (prints the plan, changes nothing):
    python scripts/reset_db.py

    # Actually reset BOTH PostgreSQL and Weaviate:
    python scripts/reset_db.py --confirm

    # Only PostgreSQL, or only Weaviate:
    python scripts/reset_db.py --scope postgres --confirm && python scripts/reset_db.py --scope weaviate --confirm
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
from agent_email.db_schema import create_email_schema  # noqa: E402
from agent_email import weaviate_vector_store as wvs  # noqa: E402
from agent_email.weaviate_vector_store import (  # noqa: E402
    OWNED_NAMESPACES,
    owned_collection_names,
)


# Open (Action) Status Tracker tables — NEVER dropped by this script.
PRESERVE_TABLES: tuple[str, ...] = (
    "tracker_item_overrides",
    "tracker_completed_items",
    "tracker_deleted_items",
)

# Email-agent tables that this script resets. Dropped (CASCADE) then
# recreated from create_email_schema so the fresh structure always matches
# the single source of truth in agent_email/db_schema.py.
EMAIL_TABLES: tuple[str, ...] = (
    "email_clusters",
    "email_topics",
    "meeting_embeddings",
    "meeting_history",
    "meeting_notes",
    "email_summaries",
)

# Tables this script is allowed to drop. A guard below cross-checks this
# against PRESERVE_TABLES so a preserved table can never slip in.
# NOTE: delegated Graph OAuth tokens are no longer in PostgreSQL (they live
# in AWS Secrets Manager), so no auth-token table is reset here.
RESET_TABLES: tuple[str, ...] = tuple(EMAIL_TABLES)


def _assert_safe_table_set() -> None:
    """Fail loudly if the reset set ever overlaps the preserve set."""
    overlap = set(RESET_TABLES) & set(PRESERVE_TABLES)
    if overlap:
        raise RuntimeError(
            f"refusing to run: reset set overlaps preserved tracker tables: {sorted(overlap)}"
        )


def _print_plan(scope: str) -> None:
    print("=== reset_db plan ===")
    print(f"scope: {scope}")
    if scope in ("postgres", "all"):
        print("\nPostgreSQL — will DROP and recreate (empty):")
        for t in RESET_TABLES:
            print(f"  - {t}")
        print("\nPostgreSQL — PRESERVED (never touched):")
        for t in PRESERVE_TABLES:
            print(f"  - {t}")
    if scope in ("weaviate", "all"):
        settings = get_settings()
        print("\nWeaviate — will DELETE and recreate (empty):")
        for namespace, name in owned_collection_names(settings).items():
            cnt = wvs.object_count(namespace, settings)
            state = "missing" if cnt < 0 else f"{cnt} objects"
            print(f"  - {namespace:10s} -> {name:30s} [{state}]")
    print("=====================")


async def _reset_postgres() -> None:
    pool = get_pool()
    if pool is None:
        raise RuntimeError(
            "_reset_postgres: PostgreSQL pool unavailable (POSTGRES_DSN empty?)"
        )
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Drop only the reset set; preserved tracker tables are left
            # exactly as-is. CASCADE clears dependent indexes/constraints.
            for table in RESET_TABLES:
                await conn.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
                print(f"[PG] dropped {table}")
            # Recreate the email-agent tables + indexes from the single
            # source of truth (gets is_active, all indexes, migrations).
            await create_email_schema(conn)
            print("[PG] recreated email-agent tables via create_email_schema")
    print("[PG] reset complete")


def _reset_weaviate() -> None:
    settings = get_settings()
    # Delete then recreate each owned collection so the vector copy starts
    # empty. Delete is gated: it refuses anything not owned by this project.
    for namespace in OWNED_NAMESPACES:
        deleted = wvs._admin_delete_collection(namespace, settings)
        name = owned_collection_names(settings)[namespace]
        print(f"[WEAVIATE] {'deleted' if deleted else 'nothing to delete for'} {name!r}")
    # ensure_schema recreates every owned collection that is now missing.
    created = wvs.ensure_schema(settings)
    for name, was_created in created.items():
        print(f"[WEAVIATE] {'created' if was_created else 'already present'} {name!r}")
    print("[WEAVIATE] reset complete")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Destructive reset of email-agent PostgreSQL tables + Weaviate "
            "collections to a clean structure. NEVER touches the Open Status "
            "Tracker tables. Dry-run unless --confirm is passed."
        )
    )
    parser.add_argument(
        "--scope",
        choices=("postgres", "weaviate", "all"),
        default="all",
        help="Which store(s) to reset (default: all).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually perform the reset (otherwise dry-run).",
    )
    args = parser.parse_args()

    _assert_safe_table_set()
    settings = get_settings()

    # Weaviate status in the plan needs a live client; PostgreSQL needs the
    # pool only when we actually touch it (and for a live object count the
    # plan does not require the DB).
    needs_pg = args.scope in ("postgres", "all")

    if needs_pg:
        await init_db()

    try:
        _print_plan(args.scope)

        if not args.confirm:
            print("\nDRY-RUN: nothing changed. Re-run with --confirm to apply.")
            return

        if args.scope in ("postgres", "all"):
            await _reset_postgres()
        if args.scope in ("weaviate", "all"):
            _reset_weaviate()

        print("\nReset finished.")
    finally:
        if needs_pg:
            await close_db()
        from core.weaviate.client import close_client
        close_client()


if __name__ == "__main__":
    asyncio.run(main())
