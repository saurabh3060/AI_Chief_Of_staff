"""
core/db.py
============
Shared PostgreSQL connection pool for the entire backend.

Provides two lifecycle functions that main.py calls during startup and
shutdown, plus a ``get_pool()`` accessor that any store module can use
to acquire a connection.

Usage in a store module::

    from core.db import get_pool

    async def load_something():
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT ...")

The pool is created lazily on first ``init_db()`` call (during the
FastAPI lifespan startup) and torn down on ``close_db()`` (during
shutdown).  If ``POSTGRES_DSN`` is empty the pool is never created and
``get_pool()`` raises immediately — the JSON file stores remain the
fallback in that case.
"""

import asyncio

import asyncpg
from core.config import get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("db")

_pool: asyncpg.Pool | None = None
# The event loop the pool (and all its connections) were created on -
# asyncpg connections are bound to a single loop, so any code that needs
# to run a query from a *different* thread (e.g. a sync generator driven
# by FastAPI's threadpool, like the chatbot's SSE stream) must schedule
# its coroutine onto *this* loop via asyncio.run_coroutine_threadsafe
# rather than asyncio.run() - a fresh loop from asyncio.run() cannot use
# connections that belong to this loop, which surfaces as asyncpg's
# "another operation is in progress" error.
_main_loop: asyncio.AbstractEventLoop | None = None


async def init_db() -> None:
    """
    Create the asyncpg connection pool using ``POSTGRES_DSN`` from settings.

    Called once during FastAPI lifespan startup.  Also runs
    ``CREATE TABLE IF NOT EXISTS`` for every tracker table so the app
    is self-healing — if the tables don't exist yet they are created
    automatically without needing the migration script.
    """
    global _pool, _main_loop
    settings = get_settings()

    if not settings.postgres_dsn:
        logger.info("POSTGRES_DSN is empty — PostgreSQL disabled, using JSON file stores")
        return

    _pool = await asyncpg.create_pool(
        dsn=settings.postgres_dsn,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    # Remember the loop the pool lives on so sync code running in other
    # threads can schedule DB work onto it (see get_main_loop()).
    _main_loop = asyncio.get_running_loop()
    logger.info("PostgreSQL connection pool created")

    # Auto-create tables if they don't exist (safe to run every startup).
    async with _pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tracker_item_overrides (
                id          VARCHAR(16) PRIMARY KEY,
                text        TEXT,
                owner       TEXT,
                business_unit TEXT,
                due_date    TEXT,
                updated_at  TIMESTAMPTZ DEFAULT now()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tracker_completed_items (
                id          VARCHAR(16) PRIMARY KEY,
                text        TEXT NOT NULL,
                owner       TEXT NOT NULL,
                business_unit TEXT NOT NULL,
                due_date    TEXT,
                source_topic TEXT,
                completed_at TIMESTAMPTZ DEFAULT now()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tracker_deleted_items (
                id          VARCHAR(16) PRIMARY KEY,
                deleted_at  TIMESTAMPTZ DEFAULT now()
            );
        """)
        # NOTE: delegated Microsoft Graph OAuth tokens are NO LONGER stored
        # in PostgreSQL. BOTH the short-lived access token (+ expiry) and the
        # long-lived refresh token now live exclusively in AWS Secrets
        # Manager (see core/secrets_manager.py and auth/token_store.py), which
        # is the single source of truth. The former `graph_oauth_tokens` table
        # is intentionally not created; any pre-existing table on an
        # already-provisioned database is simply left unused (no destructive
        # DROP is run here).
        logger.info("Tracker tables verified/created")

        # Permanent store for every chatbot Q&A pair. Redis (see
        # chatbot/redis_history.py) still holds a short, fast-access
        # window of recent turns, but this table is the durable record
        # that GET /chatbot/history reads from, so history survives a
        # Redis restart/eviction indefinitely.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chatbot_history (
                id          BIGSERIAL PRIMARY KEY,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chatbot_history_created_at
            ON chatbot_history (created_at);
        """)
        logger.info("Chatbot history table verified/created")

        # ── OneDrive sync state (replaces the local JSON file) ──
        # onedrive_sync_files: one row per tracked SharePoint file.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS onedrive_sync_files (
                item_id         TEXT PRIMARY KEY,
                name            TEXT NOT NULL DEFAULT '',
                relative_path   TEXT NOT NULL DEFAULT '',
                last_modified   TEXT NOT NULL DEFAULT '',
                size            BIGINT NOT NULL DEFAULT 0,
                doc_id          TEXT NOT NULL DEFAULT '',
                synced_at       TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        # onedrive_sync_runs: one row per sync run (latest = current state).
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS onedrive_sync_runs (
                id              BIGSERIAL PRIMARY KEY,
                status          TEXT NOT NULL DEFAULT 'unknown',
                summary         JSONB,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        logger.info("OneDrive sync state tables verified/created")

        # The chatbot's raw-message RAG source table (default "bu_emails").
        # Created unconditionally whenever PostgreSQL is configured — unlike
        # the email agent tables below, this is NOT gated on
        # EMAIL_STORAGE_BACKEND, because the chatbot reads from this table
        # regardless of how the cluster pipeline persists its own data.
        # The email pipeline writes to it (see
        # agent_email.postgres_store.save_emails_to_bu_emails).
        from agent_email.db_schema import create_bu_emails_schema

        try:
            async with conn.transaction():
                await create_bu_emails_schema(conn, settings.chatbot_pg_table)
            logger.info(
                "Chatbot source table '%s' verified/created",
                settings.chatbot_pg_table,
            )
        except Exception as e:
            # Never block startup on this: the chatbot degrades to
            # not-ready and the email pipeline skips its bu_emails write,
            # but every other agent keeps working.
            logger.error(
                "Could not verify/create chatbot source table '%s': %s",
                settings.chatbot_pg_table, e,
            )

        # ── Documents bot (ex-SQLite library.db) ──
        # documents, pages, blocks (+tsvector), series (+tsvector), facts,
        # issues, corrections, metric_rules, calculated_metrics. Vectors
        # for blocks/series live in Weaviate (see documents/weaviate_store).
        from documents.db_schema import create_documents_schema

        try:
            async with conn.transaction():
                await create_documents_schema(conn)
            logger.info("Documents bot tables verified/created")
        except Exception as e:
            logger.error(
                "Could not verify/create documents bot tables: %s", e,
            )

        # Cached chatbot embedding vectors (Postgres is the master,
        # Weaviate is rebuilt from here).
        from agent_email.db_schema import create_chatbot_embeddings_schema

        try:
            async with conn.transaction():
                await create_chatbot_embeddings_schema(conn)
            logger.info("Chatbot embeddings table verified/created")
        except Exception as e:
            logger.error(
                "Could not verify/create chatbot_embeddings table: %s", e,
            )

    # Auto-create the email agent tables + indexes when the email storage
    # backend is PostgreSQL (row data + Weaviate object id). Kept out of
    # the tracker block above so it only runs when actually needed.
    # Wrapped in a transaction so all DDL applies atomically.
    if getattr(settings, "email_backend_is_postgres", False):
        from agent_email.db_schema import create_email_schema

        async with _pool.acquire() as conn:
            async with conn.transaction():
                await create_email_schema(conn)
        logger.info("Email agent tables verified/created")

        # Also surface this on the agent_email logger so it shows up in
        # email_agent.log (not just app.log), making the active backend
        # obvious wherever the email pipeline logs are being watched.
        email_logger = get_agent_logger("agent_email")
        email_logger.info(
            "Email storage backend=postgres: verified/created tables "
            "email_clusters, email_topics, meeting_embeddings, meeting_history "
            "(embeddings stored in Weaviate at http=%s:%s grpc=%s:%s)",
            settings.weaviate_http_host, settings.weaviate_http_port,
            settings.weaviate_grpc_host, settings.weaviate_grpc_port,
        )


async def close_db() -> None:
    """Close the connection pool.  Called during FastAPI lifespan shutdown."""
    global _pool, _main_loop
    if _pool is not None:
        await _pool.close()
        _pool = None
        _main_loop = None
        logger.info("PostgreSQL connection pool closed")


def get_pool() -> asyncpg.Pool | None:
    """
    Return the active connection pool, or ``None`` if PostgreSQL is not
    configured (POSTGRES_DSN is empty).

    Store modules check this: if None, they fall back to JSON files.
    """
    return _pool


def get_main_loop() -> asyncio.AbstractEventLoop | None:
    """
    Return the event loop the connection pool was created on, or ``None``
    if the pool was never initialised.

    Code running outside that loop (e.g. a sync generator iterated in
    FastAPI's threadpool) must use
    ``asyncio.run_coroutine_threadsafe(coro, get_main_loop())`` to run a
    query against the shared pool - calling ``asyncio.run()`` instead
    creates a second loop that cannot use connections bound to this one.
    """
    return _main_loop
