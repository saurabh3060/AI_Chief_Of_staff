"""
agent_email/store.py
=====================
Storage facade for the email cluster synthesis pipeline.

This is the single seam the business logic (``service.py``,
``meeting_anchor.py``, ``agent_open_tracker/service.py``) calls to read and
write cluster / topic / meeting records. It never talks to a database or a
vector store directly — it delegates to the three-layer stack:

    read  : Redis cache  -> PostgreSQL (system of record), self-healing
                            Weaviate from PostgreSQL on an empty read.
    write : PostgreSQL (system of record) -> Weaviate (rebuildable copy)
                            -> Redis cache.

PostgreSQL holds the rows AND the embeddings (JSONB) + model/dim +
content_hash; Weaviate is a rebuildable searchable copy; Redis is a
disposable no-TTL read cache. There is no CSV / flat-file path — the
database is the only backend.
"""

from __future__ import annotations

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger

from agent_email import postgres_store
from agent_email.schema import EmailCandidate, EmailClusterRecord

logger = get_agent_logger("agent_email")


def _redis_enabled(settings: Settings) -> bool:
    return bool(getattr(settings, "redis_enabled", False))


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------

async def read_all_cluster_records(settings: Settings | None = None) -> list[EmailClusterRecord]:
    """
    Read the ACTIVE cluster records. Read order: Redis cache -> PostgreSQL.

    This returns only the live, one-per-meeting view (``is_active = 1``) —
    the same set Weaviate and the Redis cache hold and that the UI /
    meeting-matching path consumes. Superseded historical clusters live in
    PostgreSQL only and are reachable via
    ``postgres_store.read_all_cluster_records(include_inactive=True)``.

    A cache hit returns exactly what a store read would. On miss we read
    PostgreSQL (the system of record) and refill the cache. If the read is
    empty while PostgreSQL has rows, Weaviate is rebuilt from PostgreSQL and
    the read is retried (no re-embedding). Redis down/absent = transparent
    miss.
    """
    active = settings or get_settings()

    if _redis_enabled(active):
        from agent_email import redis_cache
        cached = await redis_cache.get_clusters(active)
        if cached is not None:
            logger.debug("[REDIS][CACHE-HIT] clusters records=%d", len(cached))
            return [EmailClusterRecord.model_validate(r) for r in cached]
        logger.debug("[REDIS][CACHE-MISS] clusters -> reading PostgreSQL")

    records = await postgres_store.read_all_cluster_records(settings=active)
    # Self-heal: empty read while PostgreSQL has rows => rebuild Weaviate
    # from PostgreSQL and re-read (no re-embedding). No-op when genuinely
    # empty, so we never fabricate data.
    records = await postgres_store.heal_clusters_if_empty(records, settings=active)

    if _redis_enabled(active):
        from agent_email import redis_cache
        await redis_cache.set_clusters([r.model_dump() for r in records], active)
    return records


async def write_cluster_records(
    records: list[EmailClusterRecord],
    settings: Settings | None = None,
) -> int:
    """
    Upsert cluster records. Write order: PostgreSQL + Weaviate (inside the
    PostgreSQL store) THEN refresh the Redis cache from the full snapshot so
    the cache reflects exactly what the system of record now holds.
    """
    active = settings or get_settings()
    written = await postgres_store.write_cluster_records(records, settings=active)
    if _redis_enabled(active):
        from agent_email import redis_cache
        snapshot = await postgres_store.read_all_cluster_records(settings=active)
        await redis_cache.set_clusters([r.model_dump() for r in snapshot], active)
        # Cluster rows changed => every per-meeting matched-cluster result is
        # potentially stale. Invalidate the derived meeting_clusters cache so
        # the next meeting-list / brief request recomputes against new data.
        await redis_cache.clear_meeting_clusters(active)
    return written


# ---------------------------------------------------------------------------
# Topics (no embedding, no cache — written straight to PostgreSQL)
# ---------------------------------------------------------------------------

async def write_topic_records(
    candidates: list[EmailCandidate],
    run_id: str = "",
    settings: Settings | None = None,
) -> int:
    """Persist the per-mail structured semantics (single-extractor output) to PostgreSQL."""
    active = settings or get_settings()
    return await postgres_store.write_topic_records(candidates, run_id=run_id, settings=active)


async def read_email_semantics(
    message_ids: list[str],
    run_id: str = "",
    settings: Settings | None = None,
) -> dict[str, dict]:
    """Bulk-read the structured semantics extracted at the relevance stage.

    One query over all message_ids (no N+1). Returns
    ``{message_id: semantics_dict}`` for clustering to reuse instead of
    re-running an LLM extraction.
    """
    active = settings or get_settings()
    return await postgres_store.read_email_semantics(message_ids, run_id=run_id, settings=active)


# ---------------------------------------------------------------------------
# Meeting embeddings
# ---------------------------------------------------------------------------

async def read_meeting_embeddings(settings: Settings | None = None) -> dict[str, tuple[str, str]]:
    """Return meeting_id -> (subject, embedding_json). Redis -> PostgreSQL."""
    active = settings or get_settings()

    if _redis_enabled(active):
        from agent_email import redis_cache
        cached = await redis_cache.get_meeting_embeddings(active)
        if cached is not None:
            logger.debug("[REDIS][CACHE-HIT] meeting_embeddings records=%d", len(cached))
            # JSON has no tuples; restore the (subject, embedding) tuple shape.
            return {mid: tuple(val) for mid, val in cached.items()}
        logger.debug("[REDIS][CACHE-MISS] meeting_embeddings -> reading PostgreSQL")

    result = await postgres_store.read_meeting_embeddings(settings=active)
    if _redis_enabled(active):
        from agent_email import redis_cache
        await redis_cache.set_meeting_embeddings(
            {mid: list(val) for mid, val in result.items()}, active
        )
    return result


async def read_meeting_departments(settings: Settings | None = None) -> dict[str, str]:
    """Return meeting_id -> identified department from the meeting_embeddings
    table. Reads PostgreSQL directly (the system of record); this is only used
    on the match-recompute path, not per UI request, so it is not cached."""
    active = settings or get_settings()
    return await postgres_store.read_meeting_departments(settings=active)


async def write_meeting_embeddings(
    records: list[dict[str, str]],
    settings: Settings | None = None,
) -> int:
    active = settings or get_settings()
    written = await postgres_store.write_meeting_embeddings(records, settings=active)
    if _redis_enabled(active):
        from agent_email import redis_cache
        snapshot = await postgres_store.read_meeting_embeddings(settings=active)
        await redis_cache.set_meeting_embeddings(
            {mid: list(val) for mid, val in snapshot.items()}, active
        )
        # The meeting query vector feeds the matched-cluster computation, so a
        # changed meeting embedding can change the match. Invalidate the
        # derived per-meeting cache.
        await redis_cache.clear_meeting_clusters(active)
    return written


# ---------------------------------------------------------------------------
# Meeting history
# ---------------------------------------------------------------------------

async def read_meeting_history(settings: Settings | None = None) -> dict[str, dict[str, str]]:
    """Return meeting_id -> {subject, start_datetime, embedding, ...}. Redis -> PostgreSQL.

    Hardening (see RCA of run 20261014T125700Z, where today's meetings
    vanished from the UI):

    * An EMPTY cached snapshot (``{}``) is treated as a cache MISS, not a
      hit, so the read falls through to PostgreSQL (the system of record)
      and self-heals. Previously a ``{}`` left in Redis — e.g. written when
      a transient "pool unavailable" read returned empty mid-pipeline — was
      a valid no-TTL hit that pinned the calendar to "0 meetings" forever.
    * The read-through cache write is skipped when PostgreSQL returns an
      empty result, so a transient empty read can never POISON the cache
      with an authoritative-looking empty snapshot. A genuinely empty
      meeting history simply stays a (cheap) cache miss until real rows
      exist, which is the safe direction.
    """
    active = settings or get_settings()

    if _redis_enabled(active):
        from agent_email import redis_cache
        cached = await redis_cache.get_meeting_history(active)
        # Only a NON-EMPTY cached snapshot is a real hit. An empty dict is
        # ambiguous (could be a poisoned write), so fall through to the
        # system of record instead of trusting it.
        if cached:
            logger.debug("[REDIS][CACHE-HIT] meeting_history records=%d", len(cached))
            return cached
        logger.debug("[REDIS][CACHE-MISS] meeting_history -> reading PostgreSQL")

    result = await postgres_store.read_meeting_history(settings=active)
    # Never cache an empty read: that is exactly what poisoned the no-TTL
    # bulk key during the incident. Populate the cache only when there is
    # real data to serve.
    if _redis_enabled(active) and result:
        from agent_email import redis_cache
        await redis_cache.set_meeting_history(result, active)
    return result


async def write_meeting_history(
    records: list[dict[str, str]],
    settings: Settings | None = None,
) -> int:
    active = settings or get_settings()
    written = await postgres_store.write_meeting_history(records, settings=active)
    if _redis_enabled(active):
        from agent_email import redis_cache
        snapshot = await postgres_store.read_meeting_history(settings=active)
        # Mirror the read-path hardening: never push an empty read-back into
        # the cache. A transient empty read right after a successful write
        # (e.g. a pool race) must not overwrite a good snapshot with {}.
        if snapshot:
            await redis_cache.set_meeting_history(snapshot, active)
        # Meeting history feeds anchoring/recurrence inputs; invalidate the
        # derived per-meeting matched-cluster cache so refreshes stay fresh.
        await redis_cache.clear_meeting_clusters(active)
    return written


async def delete_meeting_occurrences(
    meeting_ids: list[str],
    settings: Settings | None = None,
) -> int:
    """Cascade-delete occurrences from history + embeddings + vectors.

    Thin wrapper over the PostgreSQL primitive that also refreshes the Redis
    layer: the bulk meeting-history / meeting-embedding snapshots are
    re-read from the system of record (so the deleted ids drop out), and the
    derived per-meeting matched-cluster cache is invalidated. Mirrors the
    cache maintenance the ``write_*`` wrappers already perform.
    """
    active = settings or get_settings()
    deleted = await postgres_store.delete_meeting_occurrences(meeting_ids, settings=active)
    if deleted and _redis_enabled(active):
        from agent_email import redis_cache

        history_snapshot = await postgres_store.read_meeting_history(settings=active)
        if history_snapshot:
            await redis_cache.set_meeting_history(history_snapshot, active)
        emb_snapshot = await postgres_store.read_meeting_embeddings(settings=active)
        await redis_cache.set_meeting_embeddings(
            {mid: list(val) for mid, val in emb_snapshot.items()}, active
        )
        await redis_cache.clear_meeting_clusters(active)
    return deleted
