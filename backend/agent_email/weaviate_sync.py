"""
agent_email/weaviate_sync.py
============================
Weaviate sync / self-healing restore for the email agent.

Weaviate is a rebuildable copy of PostgreSQL, never the master. These
helpers (re)build Weaviate from the embeddings ALREADY STORED in
PostgreSQL — they never call the embedding model or an LLM. The vector
ids are re-derived deterministically from each row's natural key, exactly
as the write path does, so the restored objects line up with the
``weaviate_id`` each row already carries.

Split out of ``postgres_store`` so that module stays focused on the row
read/write path. The ``read_all_cluster_records`` dependency is imported
lazily inside the one function that needs it, so there is no import cycle.
"""

from __future__ import annotations

import asyncio

import numpy as np

from core.config import Settings, get_settings
from core.db import get_pool
from core.logging_config import get_agent_logger

# Vector-store access goes through the backend dispatcher (Weaviate by
# default, local FAISS when VECTOR_STORE_BACKEND=faiss). TEMPORARY seam —
# revert to `weaviate_vector_store as wvs` to drop the FAISS toggle.
from agent_email import vector_store_backend as wvs
from agent_email.vector_store_backend import NAMESPACE_CLUSTERS, NAMESPACE_MEETINGS
from agent_email.storage_ids import stable_object_id
from agent_email.embedding_json import json_to_vec, jsonb_to_embedding_str
from agent_email.schema import EmailClusterRecord

logger = get_agent_logger("agent_email")


async def _to_thread(func, *args, **kwargs):
    """Run a blocking Weaviate call off the event loop."""
    return await asyncio.to_thread(func, *args, **kwargs)


async def _load_cluster_vectors_from_pg(conn) -> list[tuple[str, np.ndarray]]:
    # Only ACTIVE clusters live in Weaviate (the live, one-per-meeting
    # view). Superseded rows are kept in PostgreSQL for history but must
    # not be restored into the searchable copy.
    rows = await conn.fetch(
        "SELECT cluster_id, embedding FROM email_clusters "
        "WHERE embedding IS NOT NULL AND is_active = 1"
    )
    out: list[tuple[str, np.ndarray]] = []
    for r in rows:
        vec = json_to_vec(jsonb_to_embedding_str(r["embedding"]))
        if vec is not None:
            out.append((stable_object_id(r["cluster_id"]), vec))
    return out


async def _load_meeting_vectors_from_pg(conn) -> list[tuple[str, np.ndarray]]:
    """Both meeting_embeddings and meeting_history live in the meetings namespace."""
    out: list[tuple[str, np.ndarray]] = []
    me_rows = await conn.fetch(
        "SELECT meeting_id, embedding FROM meeting_embeddings WHERE embedding IS NOT NULL"
    )
    for r in me_rows:
        vec = json_to_vec(jsonb_to_embedding_str(r["embedding"]))
        if vec is not None:
            out.append((stable_object_id(f"meeting:{r['meeting_id']}"), vec))
    mh_rows = await conn.fetch(
        "SELECT meeting_id, embedding FROM meeting_history WHERE embedding IS NOT NULL"
    )
    for r in mh_rows:
        vec = json_to_vec(jsonb_to_embedding_str(r["embedding"]))
        if vec is not None:
            out.append((stable_object_id(f"history:{r['meeting_id']}"), vec))
    return out


async def _pg_vector_counts(conn) -> dict[str, int]:
    """How many stored vectors PostgreSQL holds per Weaviate namespace."""
    clusters = await conn.fetchval(
        "SELECT count(*) FROM email_clusters "
        "WHERE embedding IS NOT NULL AND is_active = 1"
    )
    meetings = await conn.fetchval(
        "SELECT (SELECT count(*) FROM meeting_embeddings WHERE embedding IS NOT NULL) "
        "+ (SELECT count(*) FROM meeting_history WHERE embedding IS NOT NULL)"
    )
    return {NAMESPACE_CLUSTERS: int(clusters or 0), NAMESPACE_MEETINGS: int(meetings or 0)}


async def restore_weaviate_from_postgres(
    namespaces: list[str] | None = None,
    settings: Settings | None = None,
) -> dict[str, int]:
    """
    Rebuild the given Weaviate namespaces from PostgreSQL's STORED
    embeddings. No embedding/LLM calls. Returns ``{namespace: restored}``.

    This is the self-healing path invoked when a collection is missing/
    empty while PostgreSQL has rows. Logs ``[RESTORE] namespace=.. count=N``.
    """
    active = settings or get_settings()
    pool = get_pool()
    if pool is None:
        raise RuntimeError("restore_weaviate_from_postgres: PostgreSQL pool unavailable")

    targets = namespaces or [NAMESPACE_CLUSTERS, NAMESPACE_MEETINGS]
    restored: dict[str, int] = {}
    async with pool.acquire() as conn:
        for namespace in targets:
            if namespace == NAMESPACE_CLUSTERS:
                vectors = await _load_cluster_vectors_from_pg(conn)
            elif namespace == NAMESPACE_MEETINGS:
                vectors = await _load_meeting_vectors_from_pg(conn)
            else:
                logger.warning("[PG][RESTORE] unknown namespace=%s, skipping", namespace)
                continue
            written = await _to_thread(wvs.upsert_vectors, namespace, vectors, active)
            restored[namespace] = written
            logger.info("[RESTORE] namespace=%s count=%d", namespace, written)
    return restored


async def clear_and_restore_weaviate_from_postgres(
    settings: Settings | None = None,
) -> dict[str, dict[str, int]]:
    """
    Scoped clear + rebuild of this project's Weaviate vectors from
    PostgreSQL. Used at the START of a pipeline run so each run begins from
    a clean, authoritative copy of the system of record.

    Strictly scoped to this application's own data:
      * Only this project's owned namespaces/collections are touched (via
        ``wvs.clear_namespace_objects``, which refuses any other namespace).
      * Collections are kept (objects deleted, schema preserved) so no
        other reader sees a missing class.
      * PostgreSQL is NEVER modified — it is only read to repopulate.

    After clearing, only the CURRENT contents of PostgreSQL are restored:
    for the clusters namespace that is exactly the ACTIVE clusters
    (``is_active = 1`` with a stored embedding; see
    ``_load_cluster_vectors_from_pg``), preserving the existing
    vector-search contract that only live, one-per-meeting clusters are
    searchable. Superseded/historical clusters stay in PostgreSQL only.

    Returns ``{namespace: {"cleared": n, "restored": m}}``. Best-effort and
    non-fatal: a Weaviate hiccup is logged and does not raise, because
    Weaviate is a rebuildable copy and the pipeline's writes later in the
    run will re-mirror vectors anyway.
    """
    active = settings or get_settings()
    result: dict[str, dict[str, int]] = {}
    try:
        # Make sure the owned collections exist AND are queryable before
        # clearing/restoring (create-if-missing, never delete; retries
        # through any transient REST/gRPC schema desync).
        await _to_thread(wvs.ensure_schema_ready, active)

        cleared: dict[str, int] = {}
        for namespace in (NAMESPACE_CLUSTERS, NAMESPACE_MEETINGS):
            cleared[namespace] = await _to_thread(
                wvs.clear_namespace_objects, namespace, active
            )

        restored = await restore_weaviate_from_postgres(
            [NAMESPACE_CLUSTERS, NAMESPACE_MEETINGS], active
        )

        for namespace in (NAMESPACE_CLUSTERS, NAMESPACE_MEETINGS):
            result[namespace] = {
                "cleared": int(cleared.get(namespace, 0)),
                "restored": int(restored.get(namespace, 0)),
            }
        logger.info(
            "[WEAVIATE-REBUILD] scoped clear+restore from PostgreSQL: %s", result
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[WEAVIATE-REBUILD] scoped clear+restore failed (continuing; "
            "pipeline writes will re-mirror vectors): %s",
            exc,
        )
    return result


async def clusters_row_count(settings: Settings | None = None) -> int:
    """Number of rows in email_clusters (0 if the pool is unavailable)."""
    pool = get_pool()
    if pool is None:
        return 0
    async with pool.acquire() as conn:
        return int(await conn.fetchval("SELECT count(*) FROM email_clusters") or 0)


async def heal_clusters_if_empty(
    records: list[EmailClusterRecord],
    settings: Settings | None = None,
) -> list[EmailClusterRecord]:
    """
    Self-heal hook for the cluster read path.

    If a read came back EMPTY but PostgreSQL actually has cluster rows,
    that points at a lost/empty Weaviate index (the searchable copy) while
    the system of record is intact. Rebuild Weaviate from PostgreSQL's
    stored embeddings (no re-embedding), then re-read PostgreSQL and return
    the fresh records. If PostgreSQL is genuinely empty too, there is
    nothing to heal and the empty list is returned unchanged — we never
    fabricate data.
    """
    if records:
        return records
    active = settings or get_settings()
    pg_count = await clusters_row_count(active)
    if pg_count <= 0:
        return records  # genuinely empty — nothing to heal
    logger.warning(
        "[SCHEMA-LOST] cluster read returned empty while PostgreSQL has %d row(s) "
        "-> restoring Weaviate from PostgreSQL and re-reading",
        pg_count,
    )
    await ensure_weaviate_consistent(active)
    # Imported lazily to avoid an import cycle with postgres_store.
    from agent_email.postgres_store import read_all_cluster_records

    return await read_all_cluster_records(settings=active)


async def ensure_weaviate_consistent(settings: Settings | None = None) -> dict[str, int]:
    """
    Ensure each owned Weaviate collection exists and, when a collection is
    missing or empty while PostgreSQL holds vectors for it, restore it from
    PostgreSQL. Safe no-op when everything already lines up.

    Returns ``{namespace: restored_count}`` for whatever was rebuilt (empty
    dict when nothing needed restoring). Logs ``[SCHEMA-LOST]`` for any
    collection that was gone/empty but had PostgreSQL rows behind it.
    """
    active = settings or get_settings()
    pool = get_pool()
    if pool is None:
        logger.info("[PG] ensure_weaviate_consistent: pool unavailable, skipping")
        return {}

    # Readiness gate: create any missing collections (never deletes) and
    # BLOCK until each is actually queryable over gRPC, retrying through any
    # transient REST/gRPC schema desync. This is what stops a momentary
    # "could not find class" from being mistaken for a missing collection
    # below.
    await _to_thread(wvs.ensure_schema_ready, active)

    async with pool.acquire() as conn:
        pg_counts = await _pg_vector_counts(conn)

    to_restore: list[str] = []
    for namespace, pg_count in pg_counts.items():
        if pg_count <= 0:
            continue
        wv_count = await _to_thread(wvs.object_count, namespace, active)
        # Three states (see object_count sentinels):
        #   COUNT_UNRESOLVABLE (-2): REST-present but gRPC can't resolve it
        #       yet — a TRANSIENT desync, NOT a lost collection. Do NOT
        #       rebuild (that caused the create/verify/restore thrash); skip
        #       and let the next pass (or the background monitor) retry once
        #       the server converges.
        #   COUNT_MISSING (-1) or 0: genuinely gone/empty while PostgreSQL
        #       has rows -> rebuild from the system of record.
        if wv_count == wvs.COUNT_UNRESOLVABLE:
            logger.warning(
                "[SCHEMA-DESYNC] namespace=%s is REST-present but gRPC-unresolvable "
                "(postgres_vectors=%d) -> transient, NOT rebuilding; will retry next pass",
                namespace, pg_count,
            )
            continue
        if wv_count <= 0:
            logger.warning(
                "[SCHEMA-LOST] namespace=%s weaviate_objects=%d postgres_vectors=%d "
                "-> rebuilding from PostgreSQL",
                namespace, max(wv_count, 0), pg_count,
            )
            to_restore.append(namespace)

    if not to_restore:
        return {}
    return await restore_weaviate_from_postgres(to_restore, active)
