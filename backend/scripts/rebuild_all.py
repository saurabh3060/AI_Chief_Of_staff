"""
scripts/rebuild_all.py
======================
Rebuild every storage layer back into a consistent state. This is the
companion to ``wipe_all_data.py`` and the normal recovery path after a
Weaviate/Redis loss.

What it does, in order
----------------------
1. PostgreSQL schema — ``init_db()`` creates/verifies every table + index
   (trackers, email agent, chatbot source, chatbot history). Idempotent:
   safe whether the tables are empty, partial, or already present.

2. Email vector store (email namespaces) — goes through the email agent's
   vector-store dispatcher (``agent_email.vector_store_backend``), so it
   targets whichever backend is configured via ``VECTOR_STORE_BACKEND``:
   Weaviate (default) or the local on-disk FAISS indexes.
   ``ensure_schema()`` creates any missing owned collection/index
   (``clusters`` / ``meetings``), then ``ensure_weaviate_consistent()``
   repopulates vectors from PostgreSQL's STORED embeddings (no re-embedding,
   no Ollama/LLM calls) into that same backend. If PostgreSQL is empty there
   is nothing to restore — that is expected right after a full wipe; the
   actual cluster data then comes from step 5.

3. Weaviate (chatbot collection) + index — recreates the
   ``ChatbotEmailMessage`` collection and rebuilds its message-level index
   from the ``bu_emails`` rows in PostgreSQL (``build_weaviate_index`` with
   ``recreate=True``), then rebuilds the in-process BM25 index via
   ``trigger_reindex``. This DOES embed, so it needs Ollama reachable; it
   is skipped with ``--skip-chatbot`` and is a no-op when bu_emails is empty.

4. Redis — scoped clear + refill of this app's cache keys from PostgreSQL
   (``clear_and_refill_from_source``). Disposable cache, best-effort.

5. (optional) Full mailbox pipeline — with ``--run-pipeline`` the script
   runs ``run_email_cluster_pipeline`` (fetch -> relevance+extract ->
   reuse-semantics/topic-key -> embed -> cluster -> synthesize -> persist).
   Structured semantics are extracted ONCE at the relevance stage and
   persisted to ``email_topics`` (business_objective/topics/entities/actions/
   subtopics/decisions/context); clustering reuses those rows instead of a
   second topic-tag LLM call. This is the ONLY step that creates NEW cluster
   rows (and ``email_topics`` rows) from Microsoft Graph, so it is required
   to repopulate a PostgreSQL that was just truncated. It needs Graph
   credentials exactly like ``run_pipeline.py`` (delegated sign-in /
   app-only / local .eml).

Usage (run from the backend/ directory, venv active)
----------------------------------------------------
    # Rebuild the derived layers (Weaviate + chatbot index + Redis) from
    # whatever PostgreSQL currently holds — no mailbox fetch:
    python scripts/rebuild_all.py

    # Full rebuild from scratch after a wipe (also fetch + synthesize from
    # the mailbox to repopulate PostgreSQL first):
    python scripts/rebuild_all.py --run-pipeline

    # Skip the chatbot embedding step (e.g. Ollama is down):
    python scripts/rebuild_all.py --skip-chatbot

Exit code 0 on success, non-zero if a required step failed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from core.config import get_settings  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402


def _section(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


# ---------------------------------------------------------------------------
# Step 1 — PostgreSQL schema (init_db creates/verifies all tables + indexes)
# ---------------------------------------------------------------------------
async def rebuild_postgres_schema(settings) -> None:
    _section("Step 1 — PostgreSQL schema")
    if not settings.postgres_dsn:
        print("SKIPPED: POSTGRES_DSN empty — Postgres disabled.")
        return

    from core.db import get_pool

    # init_db() is called once in main() so the pool stays open for the
    # Weaviate/Redis restore steps that read from it. Here we just report.
    pool = get_pool()
    if pool is None:
        print("WARNING: pool unavailable after init_db(); schema not verified.")
        return

    async with pool.acquire() as conn:
        tables = [
            r["tablename"]
            for r in await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
                "ORDER BY tablename;"
            )
        ]
    print(f"Schema verified by init_db(). Public tables ({len(tables)}):")
    print("  " + (", ".join(tables) if tables else "(none)"))


# ---------------------------------------------------------------------------
# Step 2 — Weaviate email namespaces (ensure + restore from Postgres vectors)
# ---------------------------------------------------------------------------
async def rebuild_weaviate_email(settings) -> None:
    _section("Step 2 — email vector store (email namespaces) from PostgreSQL embeddings")
    if not getattr(settings, "email_backend_is_postgres", False):
        print("SKIPPED: email backend is not PostgreSQL (POSTGRES_DSN empty).")
        return

    # Go through the vector-store dispatcher (vector_store_backend) rather
    # than importing weaviate_vector_store directly, so this step operates
    # on whichever backend the email agent is configured to use:
    #   VECTOR_STORE_BACKEND=weaviate -> owned Weaviate collections
    #   VECTOR_STORE_BACKEND=faiss    -> the local on-disk FAISS indexes
    # ensure_schema is create-if-missing on both; ensure_weaviate_consistent
    # (in weaviate_sync) already dispatches through the same seam, so the
    # restore repopulates the active backend from PostgreSQL's stored vectors.
    from agent_email import vector_store_backend as wvs
    from agent_email import postgres_store

    print(f"Active email vector backend: {wvs.active_backend()}")

    try:
        created = await asyncio.to_thread(wvs.ensure_schema, settings)
        for name, was_created in created.items():
            print(f"  collection {name:28s} "
                  f"{'created' if was_created else 'already present'}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR ensuring email vector-store schema: {type(exc).__name__}: {exc}")
        return

    try:
        restored = await postgres_store.ensure_weaviate_consistent(settings)
        if restored:
            print(f"Restored vectors from PostgreSQL: {restored}")
        else:
            print("Nothing to restore (Weaviate already matches PostgreSQL, "
                  "or PostgreSQL has no stored embeddings yet).")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR restoring vectors: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Step 3 — Chatbot collection + index (recreate, embed from bu_emails)
# ---------------------------------------------------------------------------
async def rebuild_chatbot_index(settings, skip: bool) -> None:
    _section("Step 3 — Chatbot Weaviate collection + index")
    if skip:
        print("SKIPPED: --skip-chatbot given.")
        return
    if not settings.postgres_dsn:
        print("SKIPPED: POSTGRES_DSN empty — no bu_emails source to index.")
        return

    from core.weaviate.client import get_client
    from chatbot.weaviate_store import recreate_collection, collection_count
    from chatbot.indexer import build_weaviate_index

    try:
        client = await asyncio.to_thread(get_client, settings)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Weaviate not reachable: {type(exc).__name__}: {exc}")
        return

    try:
        # recreate=True inside build_weaviate_index drops+recreates the
        # collection for a clean rebuild, but call recreate_collection first
        # so an empty bu_emails still leaves a fresh, valid empty collection.
        await asyncio.to_thread(recreate_collection, client, settings)
        print(f"Recreated collection {settings.chatbot_weaviate_collection!r}.")

        written = await build_weaviate_index(client, recreate=True)
        total = await asyncio.to_thread(collection_count, client, settings)
        print(f"Indexed {written} message chunk(s); collection now holds "
              f"{total} object(s).")

        # Rebuild the BM25 side + flip chatbot state to ready.
        from chatbot.refresh import trigger_reindex

        reindexed = await trigger_reindex(reason="rebuild_all.py")
        print(f"BM25 / chatbot state refreshed ({reindexed} chunk(s)).")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR building chatbot index "
              f"(is Ollama reachable for embeddings?): "
              f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Step 4 — Redis cache (scoped clear + refill from Postgres)
# ---------------------------------------------------------------------------
async def rebuild_redis(settings) -> None:
    _section("Step 4 — Redis cache (clear + refill from PostgreSQL)")
    if not settings.redis_enabled:
        print("SKIPPED: REDIS_HOST empty — Redis disabled.")
        return

    from core.redis.client import ping as redis_ping
    from agent_email import redis_cache

    try:
        if not await redis_ping(settings):
            print("SKIPPED: Redis configured but not reachable (PING failed).")
            return
        result = await redis_cache.clear_and_refill_from_source(settings)
        print(f"Redis cache refilled from PostgreSQL: {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR (non-fatal, cache is disposable): "
              f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Step 5 — Optional full mailbox pipeline (creates NEW cluster rows)
# ---------------------------------------------------------------------------
async def run_full_pipeline(settings) -> int:
    _section("Step 5 — Full mailbox pipeline (fetch -> cluster -> synthesize)")
    if not settings.bu_email_pipeline_enabled:
        print("SKIPPED: BU_EMAIL_PIPELINE_ENABLED=false.")
        return 0

    mode = settings.graph_auth_mode.lower()
    bearer_token = None
    if mode == "delegated":
        from auth.token_store import refresh_access_token

        bearer_token = await refresh_access_token(settings=settings)
        if not bearer_token:
            print("ERROR: no delegated Graph credential stored. A user must "
                  "sign in to the backend once before the pipeline can run.")
            return 2

    from agent_email.service import run_email_cluster_pipeline

    try:
        written = await run_email_cluster_pipeline(
            bearer_token=bearer_token, settings=settings
        )
        print(f"Pipeline complete: persisted {written} cluster record(s).")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: pipeline run failed: {type(exc).__name__}: {exc}")
        return 1


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild PostgreSQL schema + Weaviate collections/indexes "
                    "+ chatbot index + Redis cache into a consistent state."
    )
    parser.add_argument(
        "--run-pipeline", action="store_true",
        help="Also run the full mailbox pipeline to repopulate PostgreSQL "
             "(required after a full wipe; needs Graph credentials).",
    )
    parser.add_argument(
        "--skip-chatbot", action="store_true",
        help="Skip rebuilding the chatbot Weaviate index (e.g. Ollama down).",
    )
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    print("rebuild_all — rebuilding storage layers from backend/.env config.")

    from core.db import init_db, close_db

    # Open the shared pool once; the restore steps read through it.
    await init_db()
    rc = 0
    try:
        await rebuild_postgres_schema(settings)

        # If asked, repopulate PostgreSQL FIRST so the derived layers below
        # rebuild against fresh data in a single pass.
        if args.run_pipeline:
            rc = await run_full_pipeline(settings)

        await rebuild_weaviate_email(settings)
        await rebuild_chatbot_index(settings, skip=args.skip_chatbot)
        await rebuild_redis(settings)
    finally:
        await close_db()
        try:
            from core.weaviate.client import close_client

            close_client()
        except Exception:  # noqa: BLE001
            pass
        try:
            from core.redis.client import close_redis

            await close_redis()
        except Exception:  # noqa: BLE001
            pass

    _section("DONE")
    if rc == 0:
        print("Rebuild complete. Run scripts/diagnostics_precheck.py to verify.")
    else:
        print("Rebuild finished with errors (see Step 5 above).")
    return rc


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
