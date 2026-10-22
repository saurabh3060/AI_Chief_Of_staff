"""
agent_email/health_monitor.py
=============================
Optional light background consistency check for the three-layer storage.

This is the ONLY extra timer the storage layer adds (besides the existing
email pipeline cron). Every ``interval`` seconds (default 5 minutes) it
runs the same self-healing primitives used at startup and before write
batches, across all rebuildable layers:

  1. Weaviate (vectors): ensure the owned collections exist and, if any is
     missing/empty while PostgreSQL has rows, rebuild it from PostgreSQL's
     stored embeddings (no embedding/LLM calls).
  2. Redis (read cache): detect an emptied cache (the ``clusters:all``
     snapshot absent while PostgreSQL has cluster rows) and refill every key
     from the system of record. This gives Redis the same active
     emptiness-detection Weaviate already had, instead of only refilling at
     startup / on a pipeline run / lazily on a cache miss.
  3. Pre-meeting brief matching (the derived ``meeting_clusters:*`` cache):
     when (and only when) the Redis cache was just refilled, re-warm the
     per-meeting matches via pure matching over existing PostgreSQL data (no
     re-clustering, no summarization), since ``refill_from_source`` leaves
     that derived cache empty by design.

It is deliberately cheap (a couple of count/exists queries + create-if-
missing) and failure-tolerant: an exception in one tick is logged and the
loop keeps running; each layer is healed independently so a failure in one
never blocks the others. It only runs when a PostgreSQL DSN is configured
(the Redis/brief steps additionally require a configured Redis).
"""

from __future__ import annotations

import asyncio

from core.config import get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_email")

_monitor_task: asyncio.Task | None = None
_DEFAULT_INTERVAL_SECONDS = 300.0  # 5 minutes


async def _tick() -> None:
    from agent_email import postgres_store

    settings = get_settings()

    # 1) Weaviate (vectors): rebuild any owned collection that is missing/
    #    empty while PostgreSQL holds vectors for it. Unchanged behaviour.
    try:
        restored = await postgres_store.ensure_weaviate_consistent(settings)
        if restored:
            logger.info("[HEALTH-MONITOR] restored namespaces this tick: %s", restored)
    except Exception as exc:  # noqa: BLE001
        logger.error("[HEALTH-MONITOR] Weaviate consistency check failed: %s", exc)

    # 2) Redis (read cache): detect an emptied cache (the clusters:all
    #    snapshot absent while PostgreSQL has cluster rows) and refill it from
    #    the system of record. Previously Redis was only refilled at startup /
    #    on a pipeline run / lazily on a cache miss, with no active detection
    #    between restarts - this closes that gap so an out-of-band FLUSH is
    #    recovered within one monitor interval. Copies persisted records only;
    #    no embedding/LLM work, no re-clustering.
    refilled = False
    if getattr(settings, "redis_enabled", False):
        try:
            from agent_email import redis_cache

            healed = await redis_cache.heal_if_empty(settings)
            if healed is not None:
                refilled = True
                logger.info(
                    "[HEALTH-MONITOR] Redis cache was empty -> refilled from source: %s",
                    healed,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("[HEALTH-MONITOR] Redis consistency check failed: %s", exc)

    # 3) Pre-meeting cluster matching (derived meeting_clusters:* cache):
    #    refill_from_source deliberately does NOT repopulate this derived
    #    cache, so after a Redis refill the per-meeting matches would stay
    #    empty until each meeting is next requested. Re-warm them here with
    #    prepare_stored_meeting_matches, which is PURE MATCHING over data
    #    already in PostgreSQL (clusters + meeting embeddings) - it never
    #    re-runs clustering or summarization. Only done when we actually
    #    refilled the cache this tick, so a healthy cache pays no extra cost.
    if refilled:
        try:
            from agent_email.service import prepare_stored_meeting_matches

            prepared = await prepare_stored_meeting_matches(settings)
            logger.info(
                "[HEALTH-MONITOR] re-warmed pre-meeting brief match cache for %d meeting(s)",
                prepared,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[HEALTH-MONITOR] pre-meeting brief match re-warm failed: %s", exc
            )


async def _monitor_loop(interval_seconds: float) -> None:
    while True:
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise
        await _tick()


def start_storage_health_monitor(interval_seconds: float = _DEFAULT_INTERVAL_SECONDS) -> None:
    """
    Start the 5-minute storage consistency monitor (idempotent). No-op
    unless the PostgreSQL email backend is active — the three-layer model
    (and therefore Weaviate restore) only applies in that mode.
    """
    global _monitor_task
    if _monitor_task is not None and not _monitor_task.done():
        return

    settings = get_settings()
    if not getattr(settings, "email_backend_is_postgres", False):
        logger.info("[HEALTH-MONITOR] not started (POSTGRES_DSN not set)")
        return

    logger.info(
        "[HEALTH-MONITOR] starting storage consistency monitor (every %.0f s)",
        interval_seconds,
    )
    _monitor_task = asyncio.create_task(_monitor_loop(interval_seconds))


async def stop_storage_health_monitor() -> None:
    """Cancel the background monitor task cleanly on app shutdown."""
    global _monitor_task
    if _monitor_task is None:
        return
    _monitor_task.cancel()
    try:
        await _monitor_task
    except asyncio.CancelledError:
        pass
    _monitor_task = None
