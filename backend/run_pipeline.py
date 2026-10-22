"""
run_pipeline.py
================
Standalone, one-shot runner for the email clustering pipeline.

The backend server no longer runs this pipeline (see the lifespan hook in
`main.py`): the API is read-only and only serves cluster/topic records
that already exist in PostgreSQL. This script is the ONLY thing that
produces those records. Run it on demand, or wire it into an external
scheduler (cron, systemd timer, Windows Task Scheduler, CI job):

    # from the backend/ directory, with the same .env the server uses:
    python run_pipeline.py

It performs one full pipeline run and exits. Exit code 0 means the run
completed (even if it legitimately found zero emails); non-zero means the
run could not be performed (e.g. no delegated credential available, or an
unhandled error).

Authentication
--------------
The pipeline talks to Microsoft Graph. How it gets a token depends on
GRAPH_AUTH_MODE (identical to the server's own behaviour):

- delegated (default): a human must have signed in to the backend at least
  once so a refresh token was captured and persisted to AWS Secrets Manager
  (see auth/token_store + /auth/callback). This script reads the stored
  tokens and mints a FRESH access token from the stored refresh token every
  run - no browser interaction, survives restarts, and renews itself for as
  long as the refresh token stays valid. If nothing is stored (nobody has
  signed in) the script exits non-zero and asks for a sign-in.
- app_only: no stored token needed; GraphClient acquires its own token.
- local: reads .eml files; no token needed.

The script initialises the same storage layers the server does at startup
(PostgreSQL pool, Weaviate consistency, Redis cache) so the records it
writes are immediately visible to the running backend's read routes.
"""

from __future__ import annotations

import asyncio
import sys


async def _async_main() -> int:
    # Imports are inside the function so logging is configured first and so
    # importing this module (e.g. for a --help style check) is cheap.
    from core.config import get_settings
    from core.logging_config import get_agent_logger, setup_logging

    setup_logging()
    logger = get_agent_logger("pipeline_runner")
    settings = get_settings()

    logger.info(
        "External pipeline runner starting (env=%s, graph_auth_mode=%s)",
        settings.app_env,
        settings.graph_auth_mode,
    )

    # Respect the same master switch the in-process cron used to honour, so
    # there is a single place to disable all pipeline runs.
    if not settings.bu_email_pipeline_enabled:
        logger.warning(
            "BU_EMAIL_PIPELINE_ENABLED=false; refusing to run the pipeline. "
            "Set it to true to allow external runs."
        )
        return 0

    from core.db import init_db, close_db

    await init_db()

    try:
        # Resolve the Graph credential FIRST, before opening/churning the
        # Weaviate + Redis connections. In delegated mode a missing token
        # means the run cannot happen at all — bailing out here avoids the
        # pointless connect -> self-heal -> refill -> close cycle (and the
        # extra client teardown that can leave stale collection handles) that
        # happened when this check ran only after the storage self-heal.
        bearer_token = await _resolve_bearer_token(settings, logger)
        if settings.graph_auth_mode.lower() == "delegated" and not bearer_token:
            logger.error(
                "No delegated Graph credential available. A user must sign in to the "
                "backend at least once (so a refresh token is stored) before the "
                "external pipeline can run. Aborting."
            )
            return 2

        # Mirror the backend's startup storage self-heal so an external run
        # writes into a consistent three-layer stack (PostgreSQL system of
        # record + Weaviate searchable copy + Redis cache). All best-effort:
        # PostgreSQL alone is sufficient for correctness. Only reached once we
        # know the run will actually proceed.
        if getattr(settings, "email_backend_is_postgres", False):
            try:
                from agent_email import vector_store_backend as wvs

                # Pre-run readiness gate: ensure every owned collection exists
                # AND is queryable over gRPC (retrying through any transient
                # REST/gRPC schema desync) BEFORE the pipeline touches vectors,
                # so a run never starts against a half-resolved schema.
                readiness = await asyncio.to_thread(wvs.ensure_schema_ready, settings)
                not_ready = [n for n, r in readiness.items() if not r.get("ready")]
                if not_ready:
                    logger.warning(
                        "[WEAVIATE] pre-run: collection(s) not yet queryable %s "
                        "(continuing; writes will re-mirror vectors)", not_ready
                    )
                wvs.startup_report(settings)
                # NOTE: the Weaviate consistency RESTORE and the Redis INPUT
                # refill that used to run here were redundant: the pipeline's
                # first action is a scoped clear + rebuild
                # (clear_and_restore_weaviate_from_postgres +
                # clear_and_refill_from_source in run_email_cluster_pipeline),
                # which authoritatively repopulates both layers milliseconds
                # later. Only the readiness GATE above is kept here, because it
                # blocks starting a run against a half-resolved schema — it is a
                # gate, not a rebuild. A Redis ping for the log line is kept so
                # pre-run cache availability is still visible.
            except Exception as exc:  # noqa: BLE001
                logger.error("Weaviate readiness check failed (continuing): %s", exc)

            try:
                from core.redis.client import ping as redis_ping

                connected = await redis_ping(settings)
                logger.info("[REDIS] ping connected=%s", connected)
                if not connected:
                    logger.warning(
                        "[REDIS] pre-run: cache unavailable — reads fall through to the "
                        "system of record (continuing)"
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("Redis check failed (continuing): %s", exc)

        # Clear chatbot Q&A history from Redis so stale conversations
        # from the previous cycle don't show up after the new run.
        try:
            from chatbot.redis_history import clear_history as clear_chatbot_history
            clear_chatbot_history()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not clear chatbot Redis history (continuing): %s", exc)

        # Clear the cached question→answer entries too, so answers computed
        # from the previous cycle's data are never served after the new run.
        try:
            from chatbot.redis_qa_cache import clear_qa_cache
            clear_qa_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not clear chatbot QA cache (continuing): %s", exc)

        from agent_email.service import run_email_cluster_pipeline

        # In delegated mode, hand the pipeline a refresh-capable token
        # provider so a 401 encountered PART-WAY through a long run triggers
        # an in-flight refresh + retry (mint a new access token from the
        # stored refresh token) instead of failing the whole run. In
        # app_only/local modes there is no delegated token to refresh, so no
        # provider is wired (behaviour unchanged).
        token_provider = None
        if settings.graph_auth_mode.lower() == "delegated":
            token_provider = _make_delegated_token_provider(settings)

        logger.info("Starting email cluster pipeline run")
        written = await run_email_cluster_pipeline(
            bearer_token=bearer_token,
            settings=settings,
            token_provider=token_provider,
        )
        logger.info("Pipeline run complete: persisted %d cluster record(s)", written)

        # ── Safety net: ensure the chatbot's Weaviate collection is
        #    fully caught up with PostgreSQL. trigger_reindex() inside
        #    the pipeline may have failed silently (Ollama down, timeout,
        #    etc.), leaving Weaviate with fewer objects than bu_emails
        #    has active rows. This check runs a full rebuild only when
        #    there is a genuine gap, so it's a no-op when everything
        #    already succeeded above.
        try:
            await _ensure_chatbot_weaviate_caught_up(settings, logger)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Chatbot Weaviate safety-net failed (non-fatal): %s", exc
            )

        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline run failed: %s", exc)
        return 1
    finally:
        # Tear down the resources opened above, mirroring the server's
        # shutdown so no connections are leaked when the process exits.
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
        try:
            # Final Ollama sweep (per-call ChatOllama sockets are already
            # closed after each call); mirrors the server's shutdown so the
            # one-shot runner exits without the ":11434" unclosed-socket /
            # unclosed-transport ResourceWarning.
            from core.ollama_client.client import close_ollama

            await close_ollama()
        except Exception:  # noqa: BLE001
            pass


def _make_delegated_token_provider(settings):
    """
    Build the async token provider handed to `GraphClient` for in-flight
    re-authentication (delegated mode only).

    The returned callable matches `core.graph_client.client.TokenProvider`:
    it takes a keyword-only `force_refresh` flag and returns a currently-valid
    delegated access token (or `None` when none can be produced). It is backed
    by `auth.token_store.refresh_access_token`, which reads the latest stored
    tokens from AWS Secrets Manager and, when needed, exchanges the stored
    refresh token for a new access token (persisting the rotated tokens back).

    `GraphClient` calls it with `force_refresh=True` only after a 401, so the
    known-dead cached token is bypassed and a brand-new one is minted.
    """
    from auth.token_store import refresh_access_token

    async def _provider(*, force_refresh: bool = False) -> str | None:
        return await refresh_access_token(settings=settings, force_refresh=force_refresh)

    return _provider


async def _resolve_bearer_token(settings, logger) -> str | None:
    """
    Return the Graph bearer token to drive the pipeline with, per auth mode.

    - local: None (reads .eml files).
    - app_only: None (GraphClient acquires its own app-only token).
    - delegated: a fresh access token refreshed from the stored refresh
      token in AWS Secrets Manager; None if nothing has been stored yet.
    """
    mode = settings.graph_auth_mode.lower()
    if mode in ("local", "app_only"):
        return None

    # delegated
    from auth.token_store import refresh_access_token

    return await refresh_access_token(settings=settings)


async def _ensure_chatbot_weaviate_caught_up(settings, logger) -> None:
    """Safety net: rebuild the chatbot Weaviate index if it has fewer
    objects than PostgreSQL has active bu_emails rows.

    This catches the case where ``trigger_reindex()`` inside the email
    pipeline failed silently (Ollama unreachable, timeout, etc.) and
    left Weaviate stale. When counts already match, this is a cheap
    no-op (two COUNT queries).
    """
    from core.db import get_pool
    from chatbot.indexer import build_weaviate_index, weaviate_collection_count
    from core.weaviate.client import get_client

    pool = get_pool()
    if pool is None:
        logger.debug("[SAFETY-NET] no PG pool — skipping chatbot Weaviate check")
        return

    table = settings.chatbot_pg_table

    async with pool.acquire() as conn:
        pg_count = await conn.fetchval(
            f"SELECT COUNT(*) FROM {table} "
            f"WHERE is_active = 1 "
            f"  AND NULLIF(BTRIM(conversation_id), '') IS NOT NULL "
            f"  AND NULLIF(BTRIM(message_id), '') IS NOT NULL"
        )

    try:
        client = get_client()
    except Exception as e:
        logger.warning("[SAFETY-NET] cannot connect to Weaviate: %s", e)
        return

    wv_count = weaviate_collection_count(client)

    if wv_count >= pg_count:
        logger.info(
            "[SAFETY-NET] chatbot Weaviate is caught up "
            "(weaviate=%d, postgres_active=%d) — nothing to do",
            wv_count, pg_count,
        )
        return

    logger.warning(
        "[SAFETY-NET] chatbot Weaviate is behind "
        "(weaviate=%d, postgres_active=%d) — rebuilding index...",
        wv_count, pg_count,
    )

    inserted = await build_weaviate_index(client, recreate=True)
    logger.info(
        "[SAFETY-NET] chatbot Weaviate rebuild complete: %d object(s) ingested",
        inserted,
    )


def main() -> None:
    raise SystemExit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
