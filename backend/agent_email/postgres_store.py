"""
agent_email/postgres_store.py
=============================
PostgreSQL-backed persistence for the email cluster synthesis pipeline.

This is the low-level PostgreSQL + Weaviate layer. The business logic does
not call it directly — it goes through the ``agent_email/store.py`` facade,
which adds the Redis read cache and self-healing on top of these functions.

Three-layer storage model
--------------------------
PostgreSQL is the permanent **system of record** and now holds BOTH the
row data AND the embedding vector (as a JSONB float array), together with
its provenance (``embedding_model``, ``embedding_dim``) and a
``content_hash`` of the row's stable fields.

* **PostgreSQL** (``agent_email/db_schema.py``) — master copy. Reads come
  from here; the embedding is read straight out of the ``embedding`` JSONB
  column, no vector-store round-trip required.
* **Weaviate** (``agent_email/weaviate_vector_store.py``) — a *rebuildable*
  searchable copy of the vectors, never the master. Each row still stores
  a deterministic ``weaviate_id`` so the Weaviate object can be
  (re)written and (re)found. If Weaviate loses a collection it is restored
  FROM PostgreSQL's stored embeddings with no re-embedding.

Write order is PostgreSQL first, then Weaviate (then Redis in the cache
layer). A row is only (re)written to Weaviate when its ``content_hash``
shows it is new or changed — unchanged rows are skipped, so no embedding
or LLM work is repeated for them.

Embedding string contract
-------------------------
The callers expect embeddings as a JSON float-array **string**
(``EmailClusterRecord.cluster_embedding``, ``read_meeting_embeddings ->
(subject, embedding_json)``, etc.). This module stores the embedding as a
JSON array in PostgreSQL and returns the same JSON string form on read —
the stored JSON *is* the string the callers want, so there is no vector
round-trip to Weaviate on the read path.

Transactionality
----------------
Every write is wrapped in ``async with conn.transaction()`` so the row
changes (including the embedding JSONB) are all-or-nothing. The Weaviate
copy is written only AFTER the DB transaction commits successfully, and
only for rows whose ``content_hash`` changed — so a DB failure never
leaves orphan vectors, and a crash between commit and the Weaviate write
simply leaves Weaviate one row behind PostgreSQL, which the self-healing
restore (rebuild Weaviate from PostgreSQL's stored embeddings) repairs
with no re-embedding.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import numpy as np

from core.config import Settings, get_settings
from core.db import get_pool
from core.logging_config import get_agent_logger

from agent_email import weaviate_vector_store as wvs
from agent_email.weaviate_vector_store import NAMESPACE_CLUSTERS, NAMESPACE_MEETINGS
from agent_email.schema import EmailCandidate, EmailClusterRecord, EmailTopicRecord
from agent_email.storage_ids import stable_object_id
from agent_email.embedding_json import vec_to_json, json_to_vec, jsonb_to_embedding_str
from agent_email.db_schema import validate_table_name, bu_emails_existing_columns, has_unique_single_column_index
from agent_email.content_hash import (
    cluster_content_hash,
    meeting_embedding_content_hash,
    meeting_history_content_hash,
)

logger = get_agent_logger("agent_email")


# ---------------------------------------------------------------------------
# Embedding <-> JSON string helpers (shared via agent_email.embedding_json)
# ---------------------------------------------------------------------------

# Private aliases kept for the existing call sites in this module; the
# implementations live in embedding_json so weaviate_sync can share them
# without importing this module (avoids an import cycle).
_vec_to_json = vec_to_json
_json_to_vec = json_to_vec
_jsonb_to_embedding_str = jsonb_to_embedding_str


async def _to_thread(func, *args, **kwargs):
    """Run a blocking Weaviate call off the event loop."""
    return await asyncio.to_thread(func, *args, **kwargs)


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------

async def read_all_cluster_records(
    settings: Settings | None = None,
    include_inactive: bool = False,
) -> list[EmailClusterRecord]:
    """
    Read cluster rows from PostgreSQL.

    By default this returns ONLY active rows (``is_active = 1``) — the
    live, one-per-meeting view that Weaviate and the Redis cache also hold
    and that the UI / meeting-matching path consumes. Pass
    ``include_inactive=True`` to read the full history (every cluster ever
    synthesized, active and superseded alike) for audit / admin use.
    """
    active = settings or get_settings()
    pool = get_pool()
    if pool is None:
        logger.warning("[PG] read_all_cluster_records: pool unavailable, returning []")
        return []

    where = "" if include_inactive else "WHERE is_active = 1"
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT cluster_id, department, topic, summary, top_questions,
                   action_items, email_refs, email_count, embedding,
                   meeting_id, anchored_meeting_id, meeting_start,
                   is_active, updated_at
            FROM email_clusters
            {where}
            ORDER BY cluster_id
            """
        )

    records: list[EmailClusterRecord] = []
    for r in rows:
        email_refs = r["email_refs"]
        if isinstance(email_refs, (dict, list)):
            email_refs = json.dumps(email_refs)
        # Embedding is the system-of-record JSONB column. asyncpg returns
        # JSONB as a str unless a codec is registered, and the callers want
        # exactly that JSON-array string, so pass it straight through
        # (normalizing the None / already-decoded cases).
        cluster_embedding = _jsonb_to_embedding_str(r["embedding"])
        try:
            records.append(EmailClusterRecord(
                cluster_id=r["cluster_id"],
                department=r["department"] or "",
                topic=r["topic"] or "",
                summary=r["summary"] or "",
                top_questions=r["top_questions"] or "",
                action_items=r["action_items"] or "",
                email_refs=email_refs or "",
                email_count=int(r["email_count"] or 0),
                cluster_embedding=cluster_embedding,
                meeting_id=r["meeting_id"] or "",
                anchored_meeting_id=r["anchored_meeting_id"] or "",
                meeting_start=r["meeting_start"] or "",
                is_active=int(r["is_active"] if r["is_active"] is not None else 1),
                updated_at=r["updated_at"].isoformat() if r["updated_at"] else "",
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PG] skipping malformed cluster row (cluster_id=%s): %s", r["cluster_id"], exc)
    return records


async def write_cluster_records(records: list[EmailClusterRecord], settings: Settings | None = None) -> int:
    """
    Upsert cluster rows into PostgreSQL (system of record) then mirror the
    changed vectors into Weaviate.

    Only rows whose ``content_hash`` differs from what is already stored
    are written — unchanged rows are skipped entirely, so no embedding is
    re-derived and no vector is re-sent to Weaviate for them. Write order
    is PostgreSQL first, Weaviate second (never the reverse).
    """
    active = settings or get_settings()
    pool = get_pool()
    if pool is None:
        raise RuntimeError("write_cluster_records: PostgreSQL pool unavailable")

    emb_model = active.embedding_model
    emb_dim = int(active.embedding_dim)

    if not records:
        logger.info("[PG][CLUSTER-WRITE] no records, nothing to write")
        return 0

    # Intra-batch guard for the "one active cluster per meeting" invariant.
    #
    # The upstream pipeline (run_email_cluster_pipeline) can emit TWO
    # different clusters (different cluster_id) that resolve to the SAME
    # non-empty meeting_id — e.g. the clustering stage splits emails of one
    # meeting across groups, and each group independently condenses to that
    # meeting_id. Both are created with is_active=1. The partial unique
    # index uq_email_clusters_active_meeting forbids two active rows for the
    # same meeting_id, so persisting both in one batch fails on INSERT.
    #
    # The deactivation step further below only reconciles the incoming batch
    # against ALREADY-PERSISTED rows; it cannot resolve two active siblings
    # inside this same batch (both land in the "keep" set). So resolve it
    # here first: for each meeting_id with multiple incoming active clusters,
    # keep exactly one as active (the most representative: highest
    # email_count, tie-broken by cluster_id for determinism) and demote the
    # rest to is_active=0. Demoted clusters still persist in PostgreSQL as
    # history (same treatment as superseded clusters) but are not mirrored
    # into Weaviate as live.
    demote_cluster_ids: set[str] = set()
    _active_by_meeting: dict[str, list[EmailClusterRecord]] = {}
    for rec in records:
        if int(getattr(rec, "is_active", 1) or 0) and (rec.meeting_id or "").strip():
            _active_by_meeting.setdefault(rec.meeting_id, []).append(rec)
    for meeting_id, recs in _active_by_meeting.items():
        if len(recs) <= 1:
            continue
        winner = max(recs, key=lambda r: (int(r.email_count or 0), r.cluster_id or ""))
        for r in recs:
            if r.cluster_id != winner.cluster_id:
                demote_cluster_ids.add(r.cluster_id)
        logger.warning(
            "[PG][CLUSTER-WRITE] meeting_id=%s had %d active incoming clusters; "
            "keeping %s active, demoting %d to inactive (history)",
            meeting_id, len(recs), winner.cluster_id, len(recs) - 1,
        )

    # Before a write batch, make sure the owned Weaviate collections exist
    # (create-if-missing, never delete). Best-effort: a Weaviate hiccup must
    # not block the PostgreSQL write, which is the system of record.
    try:
        await _to_thread(wvs.ensure_schema, active)
    except Exception as exc:  # noqa: BLE001
        logger.error("[PG][CLUSTER-WRITE] ensure_schema before write failed (continuing): %s", exc)

    async with pool.acquire() as conn:
        # Read existing hashes so we can skip unchanged rows (no re-embed,
        # no Weaviate re-write for them).
        existing_hashes = {
            r["cluster_id"]: (r["content_hash"] or "")
            for r in await conn.fetch("SELECT cluster_id, content_hash FROM email_clusters")
        }

        vectors: list[tuple[str, np.ndarray]] = []
        prepared: list[tuple] = []
        skipped = 0
        # Meetings for which this batch introduces an active cluster. Any
        # OTHER active cluster for the same meeting must be flipped to
        # inactive (and its vector pulled from Weaviate) so the "one active
        # cluster per meeting" invariant holds and Weaviate/Redis keep only
        # the live view.
        incoming_active_by_meeting: dict[str, set[str]] = {}
        for rec in records:
            is_active = int(getattr(rec, "is_active", 1) or 0)
            # Demote intra-batch duplicate-meeting losers (see pre-pass above)
            # before this value feeds the content hash, the Weaviate mirror,
            # and the is_active column — all must agree the row is inactive.
            if rec.cluster_id in demote_cluster_ids:
                is_active = 0
            new_hash = cluster_content_hash(
                cluster_id=rec.cluster_id,
                department=rec.department,
                topic=rec.topic,
                summary=rec.summary,
                top_questions=rec.top_questions,
                action_items=rec.action_items,
                email_refs=rec.email_refs or "[]",
                email_count=int(rec.email_count or 0),
                meeting_id=rec.meeting_id,
                meeting_start=rec.meeting_start,
                is_active=is_active,
            )
            if is_active and (rec.meeting_id or "").strip():
                incoming_active_by_meeting.setdefault(rec.meeting_id, set()).add(rec.cluster_id)
            if existing_hashes.get(rec.cluster_id) == new_hash:
                skipped += 1
                continue

            weaviate_id = None
            embedding_json = None
            vec = _json_to_vec(rec.cluster_embedding or "")
            if vec is not None:
                weaviate_id = stable_object_id(rec.cluster_id)
                embedding_json = json.dumps([float(x) for x in np.asarray(vec).ravel()])
                # Only ACTIVE clusters are mirrored into Weaviate — an
                # inactive (superseded) row stays in PostgreSQL for history
                # but must not appear in the searchable copy.
                if is_active:
                    vectors.append((weaviate_id, vec))
            prepared.append((
                rec.cluster_id, rec.department, rec.topic, rec.summary,
                rec.top_questions, rec.action_items, rec.email_refs or "[]",
                int(rec.email_count or 0), weaviate_id,
                embedding_json, emb_model if embedding_json else "",
                emb_dim if embedding_json else 0, new_hash,
                rec.meeting_id, getattr(rec, "anchored_meeting_id", "") or "",
                rec.meeting_start, is_active,
            ))

        superseded_weaviate_ids: list[str] = []
        async with conn.transaction():
            # Deactivate any OTHER active cluster for a meeting that is about
            # to receive a new active cluster. This MUST run BEFORE the
            # INSERT/upsert below: the partial unique index
            # uq_email_clusters_active_meeting (one active cluster per
            # meeting_id) is checked per-statement, so if the new active row
            # were inserted while the previous meeting's active row still had
            # is_active = 1, the INSERT would collide before we ever got the
            # chance to flip the old one off. Deactivating first clears the
            # old row out of the partial index so the INSERT lands cleanly.
            # The superseded rows stay in PostgreSQL (history) but are
            # collected here so their vectors can be pulled from Weaviate
            # after commit.
            for meeting_id, keep_cluster_ids in incoming_active_by_meeting.items():
                rows = await conn.fetch(
                    """
                    UPDATE email_clusters
                       SET is_active = 0, updated_at = now()
                     WHERE meeting_id = $1
                       AND is_active = 1
                       AND cluster_id <> ALL($2::text[])
                    RETURNING weaviate_id
                    """,
                    meeting_id, list(keep_cluster_ids),
                )
                for row in rows:
                    if row["weaviate_id"]:
                        superseded_weaviate_ids.append(row["weaviate_id"])

            if prepared:
                await conn.executemany(
                    """
                    INSERT INTO email_clusters
                        (cluster_id, department, topic, summary, top_questions,
                         action_items, email_refs, email_count, weaviate_id,
                         embedding, embedding_model, embedding_dim, content_hash,
                         meeting_id, anchored_meeting_id, meeting_start,
                         is_active, updated_at)
                    VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,
                            $10::jsonb,$11,$12,$13,$14,$15,$16,$17, now())
                    ON CONFLICT (cluster_id) DO UPDATE SET
                        department = EXCLUDED.department,
                        topic = EXCLUDED.topic,
                        summary = EXCLUDED.summary,
                        top_questions = EXCLUDED.top_questions,
                        action_items = EXCLUDED.action_items,
                        email_refs = EXCLUDED.email_refs,
                        email_count = EXCLUDED.email_count,
                        weaviate_id = EXCLUDED.weaviate_id,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_dim = EXCLUDED.embedding_dim,
                        content_hash = EXCLUDED.content_hash,
                        meeting_id = EXCLUDED.meeting_id,
                        anchored_meeting_id = EXCLUDED.anchored_meeting_id,
                        meeting_start = EXCLUDED.meeting_start,
                        is_active = EXCLUDED.is_active,
                        updated_at = now()
                    """,
                    prepared,
                )

    # DB committed — mirror the active vectors into Weaviate (the
    # rebuildable copy) and remove the vectors of any cluster that was just
    # superseded, so Weaviate holds exactly the active one-per-meeting set.
    if vectors:
        await _to_thread(wvs.upsert_vectors, NAMESPACE_CLUSTERS, vectors, active)
    if superseded_weaviate_ids:
        await _to_thread(
            wvs.remove_vectors, NAMESPACE_CLUSTERS, superseded_weaviate_ids, active
        )

    logger.info(
        "[PG][CLUSTER-WRITE] upserted %d row(s), %d active vector(s), "
        "deactivated %d superseded, skipped %d unchanged",
        len(prepared), len(vectors), len(superseded_weaviate_ids), skipped,
    )
    return len(prepared)


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------

async def write_topic_records(
    candidates: list[EmailCandidate],
    run_id: str = "",
    settings: Settings | None = None,
) -> int:
    pool = get_pool()
    if pool is None:
        raise RuntimeError("write_topic_records: PostgreSQL pool unavailable")

    rows: list[tuple] = []
    for candidate in candidates:
        msg = candidate.message
        rec = EmailTopicRecord(
            message_id=msg.id or "",
            subject=msg.subject or "",
            from_address=msg.from_address or "",
            received_at=msg.received_at.isoformat() if msg.received_at else "",
            topic_key=candidate.topic_key or "",
            run_id=run_id,
            business_objective=candidate.business_objective or "",
            topics=candidate.topics or [],
            entities=candidate.entities or [],
            actions=candidate.actions or [],
            subtopics=candidate.subtopics or [],
            decisions=candidate.decisions or [],
            context=candidate.context or "",
        )
        # JSONB list columns are passed as JSON strings so asyncpg binds them
        # to a single jsonb parameter (no per-row element expansion → no N+1).
        rows.append((
            rec.message_id, rec.subject, rec.from_address,
            rec.received_at, rec.topic_key, rec.run_id,
            rec.business_objective,
            json.dumps(rec.topics), json.dumps(rec.entities),
            json.dumps(rec.actions), json.dumps(rec.subtopics),
            json.dumps(rec.decisions), rec.context,
        ))

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Single batched upsert for the whole candidate set (executemany,
            # one statement) — never one INSERT per candidate.
            if rows:
                await conn.executemany(
                    """
                    INSERT INTO email_topics
                        (message_id, subject, from_address, received_at, topic_key, run_id,
                         business_objective, topics, entities, actions, subtopics, decisions, context)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                    ON CONFLICT (message_id) DO UPDATE SET
                        subject = EXCLUDED.subject,
                        from_address = EXCLUDED.from_address,
                        received_at = EXCLUDED.received_at,
                        topic_key = EXCLUDED.topic_key,
                        run_id = EXCLUDED.run_id,
                        business_objective = EXCLUDED.business_objective,
                        topics = EXCLUDED.topics,
                        entities = EXCLUDED.entities,
                        actions = EXCLUDED.actions,
                        subtopics = EXCLUDED.subtopics,
                        decisions = EXCLUDED.decisions,
                        context = EXCLUDED.context
                    """,
                    rows,
                )
    logger.info("[PG][TOPIC-WRITE] wrote %d topic row(s) at %s", len(rows), datetime.now(timezone.utc).isoformat())
    return len(rows)


async def read_email_semantics(
    message_ids: list[str],
    run_id: str = "",
    settings: Settings | None = None,
) -> dict[str, dict]:
    """Bulk-read the structured semantics persisted at the relevance stage.

    Returns a ``{message_id: semantics_dict}`` mapping where each value has
    the shape clustering consumes: ``business_objective`` (str), ``topics`` /
    ``entities`` / ``actions`` / ``subtopics`` / ``decisions`` (list[str]),
    ``context`` (str), and ``topic_key`` (str).

    This is a SINGLE query over ``message_id = ANY($1)`` (optionally scoped
    to ``run_id``) — never one query per message — so reusing the extracted
    semantics for a large candidate pool stays one round-trip, with no N+1.
    """
    out: dict[str, dict] = {}
    ids = [mid for mid in (message_ids or []) if mid]
    if not ids:
        return out

    pool = get_pool()
    if pool is None:
        raise RuntimeError("read_email_semantics: PostgreSQL pool unavailable")

    def _as_list(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                return []
        return [str(v) for v in value] if isinstance(value, list) else []

    if run_id:
        query = (
            "SELECT message_id, topic_key, business_objective, topics, entities, "
            "actions, subtopics, decisions, context "
            "FROM email_topics WHERE message_id = ANY($1::text[]) AND run_id = $2"
        )
        args: tuple = (ids, run_id)
    else:
        query = (
            "SELECT message_id, topic_key, business_objective, topics, entities, "
            "actions, subtopics, decisions, context "
            "FROM email_topics WHERE message_id = ANY($1::text[])"
        )
        args = (ids,)

    async with pool.acquire() as conn:
        records = await conn.fetch(query, *args)

    for row in records:
        mid = row["message_id"]
        if not mid:
            continue
        out[mid] = {
            "topic_key": row["topic_key"] or "",
            "business_objective": row["business_objective"] or "",
            "topics": _as_list(row["topics"]),
            "entities": _as_list(row["entities"]),
            "actions": _as_list(row["actions"]),
            "subtopics": _as_list(row["subtopics"]),
            "decisions": _as_list(row["decisions"]),
            "context": row["context"] or "",
        }
    logger.info(
        "[PG][SEMANTICS-READ] requested=%d found=%d run_id=%r",
        len(ids), len(out), run_id,
    )
    return out


# ---------------------------------------------------------------------------
# Meeting embeddings
# ---------------------------------------------------------------------------

async def read_meeting_embeddings(settings: Settings | None = None) -> dict[str, tuple[str, str]]:
    """Return meeting_id -> (subject, embedding_json)."""
    active = settings or get_settings()
    pool = get_pool()
    if pool is None:
        return {}

    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT meeting_id, subject, embedding FROM meeting_embeddings")

    out: dict[str, tuple[str, str]] = {}
    for r in rows:
        meeting_id = (r["meeting_id"] or "").strip()
        if not meeting_id:
            continue
        # Embedding read straight from the system-of-record JSONB column.
        out[meeting_id] = (r["subject"] or "", _jsonb_to_embedding_str(r["embedding"]))
    return out


async def read_meeting_departments(settings: Settings | None = None) -> dict[str, str]:
    """Return meeting_id -> identified department for the meeting_embeddings rows.

    Kept as a dedicated, narrow accessor (rather than widening the
    ``read_meeting_embeddings`` (subject, embedding) tuple) so the many
    existing tuple-unpack call sites and the Redis list serialization are
    left untouched. Empty string for an untagged / low-confidence row.
    """
    active = settings or get_settings()
    pool = get_pool()
    if pool is None:
        return {}

    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT meeting_id, department FROM meeting_embeddings")

    out: dict[str, str] = {}
    for r in rows:
        meeting_id = (r["meeting_id"] or "").strip()
        if not meeting_id:
            continue
        out[meeting_id] = (r["department"] or "").strip()
    return out


async def write_meeting_embeddings(
    records: list[dict[str, str]],
    settings: Settings | None = None,
) -> int:
    """Full-replace meeting_embeddings + their Weaviate vectors, transactionally."""
    active = settings or get_settings()
    pool = get_pool()
    if pool is None:
        raise RuntimeError("write_meeting_embeddings: PostgreSQL pool unavailable")

    emb_model = active.embedding_model
    emb_dim = int(active.embedding_dim)

    async with pool.acquire() as conn:
        existing_hashes = {
            r["meeting_id"]: (r["content_hash"] or "")
            for r in await conn.fetch("SELECT meeting_id, content_hash FROM meeting_embeddings")
        }

        vectors: list[tuple[str, np.ndarray]] = []
        prepared: list[tuple] = []
        skipped = 0
        for rec in records:
            meeting_id = (rec.get("meeting_id") or "").strip()
            if not meeting_id:
                continue
            subject = rec.get("subject") or ""
            department = rec.get("department") or ""
            new_hash = meeting_embedding_content_hash(
                meeting_id=meeting_id, subject=subject, department=department,
            )
            if existing_hashes.get(meeting_id) == new_hash:
                skipped += 1
                continue

            weaviate_id = None
            embedding_json = None
            vec = _json_to_vec(rec.get("embedding") or "")
            if vec is not None:
                weaviate_id = stable_object_id(f"meeting:{meeting_id}")
                embedding_json = json.dumps([float(x) for x in np.asarray(vec).ravel()])
                vectors.append((weaviate_id, vec))
            prepared.append((
                meeting_id, subject, department, weaviate_id,
                embedding_json, emb_model if embedding_json else "",
                emb_dim if embedding_json else 0, new_hash,
            ))

        async with conn.transaction():
            for (meeting_id, subject, department, weaviate_id, embedding_json,
                 e_model, e_dim, c_hash) in prepared:
                await conn.execute(
                    """
                    INSERT INTO meeting_embeddings
                        (meeting_id, subject, department, weaviate_id, embedding,
                         embedding_model, embedding_dim, content_hash, updated_at)
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, now())
                    ON CONFLICT (meeting_id) DO UPDATE SET
                        subject=EXCLUDED.subject, department=EXCLUDED.department,
                        weaviate_id=EXCLUDED.weaviate_id,
                        embedding=EXCLUDED.embedding, embedding_model=EXCLUDED.embedding_model,
                        embedding_dim=EXCLUDED.embedding_dim, content_hash=EXCLUDED.content_hash,
                        updated_at=now()
                    """,
                    meeting_id, subject, department, weaviate_id, embedding_json,
                    e_model, e_dim, c_hash,
                )

    if vectors:
        await _to_thread(wvs.upsert_vectors, NAMESPACE_MEETINGS, vectors, active)
    logger.info(
        "[PG][MEETING-EMBED-WRITE] wrote %d row(s), %d vector(s), skipped %d unchanged",
        len(prepared), len(vectors), skipped,
    )
    return len(prepared)


# ---------------------------------------------------------------------------
# Meeting history
# ---------------------------------------------------------------------------

async def read_meeting_history(settings: Settings | None = None) -> dict[str, dict[str, str]]:
    """Return persisted occurrence identity, business context, and embedding."""
    active = settings or get_settings()
    pool = get_pool()
    if pool is None:
        return {}

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT meeting_id, subject, start_datetime, end_datetime, business_context, "
            "department, embedding, updated_at "
            "FROM meeting_history"
        )

    out: dict[str, dict[str, str]] = {}
    for r in rows:
        meeting_id = (r["meeting_id"] or "").strip()
        if not meeting_id:
            continue
        out[meeting_id] = {
            "meeting_id": meeting_id,
            "subject": r["subject"] or "",
            "start_datetime": r["start_datetime"] or "",
            "end_datetime": r["end_datetime"] or "",
            "business_context": r["business_context"] or "",
            "department": r["department"] or "",
            "embedding": _jsonb_to_embedding_str(r["embedding"]),
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else "",
        }
    return out


async def write_meeting_history(
    records: list[dict[str, str]],
    settings: Settings | None = None,
) -> int:
    """Full-replace meeting history metadata and its Weaviate vectors."""
    active = settings or get_settings()
    pool = get_pool()
    if pool is None:
        raise RuntimeError("write_meeting_history: PostgreSQL pool unavailable")

    emb_model = active.embedding_model
    emb_dim = int(active.embedding_dim)

    async with pool.acquire() as conn:
        existing_hashes = {
            r["meeting_id"]: (r["content_hash"] or "")
            for r in await conn.fetch("SELECT meeting_id, content_hash FROM meeting_history")
        }

        vectors: list[tuple[str, np.ndarray]] = []
        prepared: list[tuple] = []
        skipped = 0
        for rec in records:
            meeting_id = (rec.get("meeting_id") or "").strip()
            if not meeting_id:
                continue
            subject = rec.get("subject") or ""
            start_dt = rec.get("start_datetime") or ""
            end_dt = rec.get("end_datetime") or ""
            business_context = rec.get("business_context") or ""
            department = rec.get("department") or ""
            new_hash = meeting_history_content_hash(
                meeting_id=meeting_id,
                subject=subject,
                start_datetime=start_dt,
                end_datetime=end_dt,
                business_context=business_context,
                department=department,
            )
            if existing_hashes.get(meeting_id) == new_hash:
                skipped += 1
                continue

            weaviate_id = None
            embedding_json = None
            vec = _json_to_vec(rec.get("embedding") or "")
            if vec is not None:
                weaviate_id = stable_object_id(f"history:{meeting_id}")
                embedding_json = json.dumps([float(x) for x in np.asarray(vec).ravel()])
                vectors.append((weaviate_id, vec))
            prepared.append((
                meeting_id, subject, start_dt, end_dt, business_context, department, weaviate_id,
                embedding_json, emb_model if embedding_json else "",
                emb_dim if embedding_json else 0, new_hash,
            ))

        async with conn.transaction():
            for (meeting_id, subject, start_dt, end_dt, business_context, department, weaviate_id,
                 embedding_json, e_model, e_dim, c_hash) in prepared:
                await conn.execute(
                    """
                    INSERT INTO meeting_history
                        (meeting_id, subject, start_datetime, end_datetime, business_context,
                         department, weaviate_id, embedding, embedding_model, embedding_dim,
                         content_hash, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, now())
                    ON CONFLICT (meeting_id) DO UPDATE SET
                        subject=EXCLUDED.subject, start_datetime=EXCLUDED.start_datetime,
                        end_datetime=EXCLUDED.end_datetime,
                        business_context=EXCLUDED.business_context,
                        department=EXCLUDED.department,
                        weaviate_id=EXCLUDED.weaviate_id, embedding=EXCLUDED.embedding,
                        embedding_model=EXCLUDED.embedding_model,
                        embedding_dim=EXCLUDED.embedding_dim,
                        content_hash=EXCLUDED.content_hash, updated_at=now()
                    """,
                    meeting_id, subject, start_dt, end_dt, business_context, department, weaviate_id,
                    embedding_json, e_model, e_dim, c_hash,
                )

    if vectors:
        await _to_thread(wvs.upsert_vectors, NAMESPACE_MEETINGS, vectors, active)
    logger.info(
        "[PG][MEETING-HISTORY-WRITE] wrote %d occurrence(s), %d vector(s), skipped %d unchanged",
        len(prepared), len(vectors), skipped,
    )
    return len(prepared)


async def delete_meeting_occurrences(
    meeting_ids: list[str],
    settings: Settings | None = None,
) -> int:
    """Hard-delete occurrences from BOTH meeting stores and their vectors.

    This is the cascading counterpart to the upsert-only
    ``write_meeting_history`` / ``write_meeting_embeddings`` writers, which
    never remove rows. It exists so a calendar occurrence that has been
    deleted/cancelled upstream can be fully retired from every layer that
    keys off its ``meeting_id`` — leaving no half-ghost (e.g. a history row
    whose embedding lingers, or a stale Weaviate vector) behind.

    Removes, for each id:
      * the ``meeting_history`` row,
      * the ``meeting_embeddings`` row,
      * the ``NAMESPACE_MEETINGS`` Weaviate vectors for both the
        ``history:<id>`` and ``meeting:<id>`` derived object ids.

    It never touches clusters or any other table. Returns the number of
    distinct meeting_ids processed (0 for an empty input — a safe no-op).
    """
    active = settings or get_settings()
    ids = [str(m).strip() for m in (meeting_ids or []) if str(m).strip()]
    ids = list(dict.fromkeys(ids))  # de-dup, preserve order
    if not ids:
        return 0

    pool = get_pool()
    if pool is None:
        raise RuntimeError("delete_meeting_occurrences: PostgreSQL pool unavailable")

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM meeting_history WHERE meeting_id = ANY($1::text[])", ids
            )
            await conn.execute(
                "DELETE FROM meeting_embeddings WHERE meeting_id = ANY($1::text[])", ids
            )

    # Best-effort vector cleanup: the Postgres rows are the system of record,
    # so a Weaviate hiccup here must not fail the prune. Any leftover vector
    # is also reconciled away by the scoped clear+restore on the next run.
    weaviate_ids = [stable_object_id(f"history:{m}") for m in ids]
    weaviate_ids += [stable_object_id(f"meeting:{m}") for m in ids]
    try:
        await _to_thread(wvs.remove_vectors, NAMESPACE_MEETINGS, weaviate_ids, active)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[PG][MEETING-DELETE] vector cleanup failed (rows already deleted): %s",
            exc,
        )

    logger.info("[PG][MEETING-DELETE] deleted %d occurrence(s) from history+embeddings", len(ids))
    return len(ids)


# ---------------------------------------------------------------------------
# Weaviate sync / self-healing restore (moved to agent_email.weaviate_sync)
# ---------------------------------------------------------------------------
#
# These are re-exported here so existing callers
# (``postgres_store.restore_weaviate_from_postgres`` /
# ``ensure_weaviate_consistent`` / ``heal_clusters_if_empty`` /
# ``clusters_row_count``) keep working unchanged. The implementations live
# in ``weaviate_sync`` to keep this module focused on the row read/write
# path and under the module size limit.
from agent_email.weaviate_sync import (  # noqa: E402
    clusters_row_count,
    ensure_weaviate_consistent,
    heal_clusters_if_empty,
    restore_weaviate_from_postgres,
)


# ---------------------------------------------------------------------------
# Meeting notes
# ---------------------------------------------------------------------------

def _coerce_action_items(value) -> list[dict]:
    """
    Normalize a stored action_items value into a list of dicts.

    The action_items column is JSONB. Depending on the asyncpg codec it may be
    returned either as a native list/dict or as a raw JSON string (e.g. '[]').
    This guarantees callers always receive a real list, which is required by
    the MeetingNote Pydantic model.
    """
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _coerce_tags(value) -> list[str]:
    """
    Normalize a stored ``tags`` value into a list of non-empty strings.

    The ``tags`` column is JSONB (a string array of team/department labels a
    CEO attached to a note). asyncpg may hand it back as a native list or as
    a raw JSON string depending on the codec, and legacy rows may have NULL.
    This guarantees callers always receive a clean ``list[str]``.
    """
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


async def read_meeting_notes(meeting_id: str, settings: Settings | None = None) -> list[dict]:
    """
    Read all notes for a specific meeting_id.
    Returns list of dicts with keys: id, meeting_id, note_text, action_items, tags, source, created_at, updated_at.
    """
    active = settings or get_settings()
    pool = get_pool()
    if pool is None:
        logger.warning("[PG][MEETING-NOTES] read_meeting_notes: pool unavailable, returning []")
        return []

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, meeting_id, note_text, action_items, tags, source, created_at, updated_at
            FROM meeting_notes
            WHERE meeting_id = $1
            ORDER BY created_at ASC
            """,
            meeting_id,
        )

    # Strip CR/LF from the user-controlled meeting_id before logging to
    # prevent log injection / forged log lines (CWE-117).
    safe_meeting_id = str(meeting_id).replace("\r\n", "").replace("\r", "").replace("\n", "")
    logger.info("[PG][MEETING-NOTES] read_meeting_notes: meeting_id=%s rows=%d", safe_meeting_id, len(rows))
    return [
        {
            "id": r["id"],
            "meeting_id": r["meeting_id"],
            "note_text": r["note_text"],
            "action_items": _coerce_action_items(r["action_items"]),
            "tags": _coerce_tags(r["tags"]),
            "source": r["source"] or "ceo_note",
            "created_at": r["created_at"].isoformat() if r["created_at"] else "",
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else "",
        }
        for r in rows
    ]


async def write_meeting_notes(
    records: list[dict[str, str | dict]],
    settings: Settings | None = None,
) -> int:
    """
    Insert or update meeting notes transactionally.
    Each record should have: meeting_id, note_text, action_items (dict/list),
    tags (list[str], optional), source (optional).
    """
    active = settings or get_settings()
    pool = get_pool()
    if pool is None:
        raise RuntimeError("write_meeting_notes: PostgreSQL pool unavailable")

    if not records:
        logger.info("[PG][MEETING-NOTES] write_meeting_notes: no records, returning 0")
        return 0

    prepared: list[tuple] = []
    for rec in records:
        meeting_id = (rec.get("meeting_id") or "").strip()
        if not meeting_id:
            logger.warning("[PG][MEETING-NOTES] skipping record with empty meeting_id")
            continue
        note_text = rec.get("note_text")
        if not note_text:
            logger.warning("[PG][MEETING-NOTES] skipping record with empty note_text")
            continue

        # Normalize action_items to JSON
        action_items = rec.get("action_items", [])
        if isinstance(action_items, list):
            action_items_json = json.dumps(action_items)
        elif isinstance(action_items, dict):
            action_items_json = json.dumps(action_items)
        else:
            action_items_json = "[]"

        source = (rec.get("source") or "ceo_note").strip() or "ceo_note"

        # Structured team/department tags (e.g. ["Risk", "BB&RL"]) the CEO
        # attached to the note via @mentions. Normalised to a JSON string
        # array of non-empty strings; defaults to [] when absent/malformed.
        raw_tags = rec.get("tags", [])
        if isinstance(raw_tags, list):
            tags_json = json.dumps([str(t).strip() for t in raw_tags if str(t).strip()])
        else:
            tags_json = "[]"

        prepared.append((meeting_id, note_text, action_items_json, tags_json, source))

    if not prepared:
        logger.warning("[PG][MEETING-NOTES] write_meeting_notes: no valid records after filtering")
        return 0

    async with pool.acquire() as conn:
        # All-or-nothing transaction
        async with conn.transaction():
            for meeting_id, note_text, action_items_json, tags_json, source in prepared:
                await conn.execute(
                    """
                    INSERT INTO meeting_notes (meeting_id, note_text, action_items, tags, source, created_at, updated_at)
                    VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, now(), now())
                    ON CONFLICT DO NOTHING
                    """,
                    meeting_id, note_text, action_items_json, tags_json, source,
                )

    logger.info("[PG][MEETING-NOTES] write_meeting_notes: wrote %d note(s)", len(prepared))
    return len(prepared)


# ---------------------------------------------------------------------------
# bu_emails — the chatbot's raw-message source table
# ---------------------------------------------------------------------------
# Unlike every other writer in this module, this one is NOT part of the
# cluster synthesis storage model and is NOT gated on
# EMAIL_STORAGE_BACKEND. It mirrors what `service.save_email_to_csv` writes
# to email_data.csv, but into the table the chatbot reads for RAG
# (`chatbot/postgres_store.fetch_email_rows`), so freshly fetched mail
# becomes answerable without a manual import step.
#
# Dedup: the table has a UNIQUE index on message_id and the write is an
# `ON CONFLICT (message_id) DO UPDATE` upsert. Microsoft Graph message ids
# are globally stable, so re-running the pipeline over an unchanged mailbox
# refreshes the same rows rather than appending duplicates.
#
# Failure policy: this function never raises. The chatbot's source table is
# strictly downstream of cluster synthesis, so a problem here must not
# abort the pipeline. It logs and returns 0 instead.

# Every column this writer knows how to populate, in a fixed order. The
# actual INSERT uses the intersection of this list with the live table's
# columns, so the same code works against either this project's schema or
# an externally-created table with extra/differently-named fields.
#
# These are hard-coded identifiers (never user input), which is what makes
# it safe to interpolate them into the statement - PostgreSQL cannot bind
# column names as parameters.
_BU_EMAILS_WRITE_COLUMNS = [
    "unique_id",           # PK on externally-created tables
    "message_id",
    "conversation_id",
    "subject",             # this project's naming
    "raw_content",
    "attachment_data",
    "email_subject",       # standalone chatbot project's naming
    "email_content",
    "attachment_content",
    "received_at",
    "from_address",
    "from_name",
    "to_address",
]


def build_bu_emails_insert(
    table: str,
    existing_columns: set[str] | list[str],
    conflict_column: str | None,
) -> tuple[str, list[str]]:
    """Build the INSERT statement for whatever columns the table has.

    Shared by the writer and by ``scripts/inspect_bu_emails.py`` so the
    diagnostic probes the exact statement the pipeline will run, instead of
    a drifting copy of it.

    Args:
        table: validated table name.
        existing_columns: columns actually present on the live table.
        conflict_column: column backed by a unique index to use as the
            ON CONFLICT target, or None for a plain INSERT.

    Returns:
        ``(sql, columns)`` where ``columns`` is the ordered column list the
        caller must bind values for, positionally.
    """
    existing = set(existing_columns)
    columns = [c for c in _BU_EMAILS_WRITE_COLUMNS if c in existing]
    column_list = ", ".join(columns)
    placeholders = ", ".join(f"${i}" for i in range(1, len(columns) + 1))

    sql = f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})"

    if conflict_column:
        assignments = [
            f"{c} = EXCLUDED.{c}" for c in columns if c != conflict_column
        ]
        # Only this project's schema has updated_at; bump it when present
        # so a refreshed row is distinguishable from an untouched one.
        if "updated_at" in existing:
            assignments.append("updated_at = now()")
        sql += (
            f" ON CONFLICT ({conflict_column}) DO UPDATE SET "
            + ", ".join(assignments)
        )

    return sql, columns


def _attachment_data_for_message(
    message_id: str,
    attachments: list,
) -> str:
    """Build the " || "-joined attachment blob for ONE message.

    Callers may hand us a thread-wide attachment list (see
    `service.get_recent_conversation`), so filter on
    ``ParsedAttachment.source_message_id`` rather than taking the list as-is
    — otherwise every row in a thread gets every sibling's attachment text.

    Format matches `service.save_email_to_csv` exactly ("<name>" or
    "<name>: <extracted_text>") so the CSV and the table never disagree.
    """
    parts: list[str] = []
    for att in attachments or []:
        if not getattr(att, "name", ""):
            continue
        # A None source_message_id means "unattributed"; keep it rather than
        # silently dropping evidence, matching the CSV writer's behaviour.
        source = getattr(att, "source_message_id", None)
        if source and source != message_id:
            continue
        text = getattr(att, "extracted_text", "") or ""
        parts.append(f"{att.name}: {text}" if text else f"{att.name}")
    return " || ".join(parts)


async def save_emails_to_bu_emails(
    pairs: list[tuple],
    settings: Settings | None = None,
) -> int:
    """Upsert one row per email message into the chatbot's source table.

    Args:
        pairs: list of ``(EmailMessage, list[ParsedAttachment])`` tuples.
            The attachment list may be thread-wide; it is filtered per
            message.
        settings: optional Settings override (table name comes from
            ``chatbot_pg_table``).

    Returns:
        Number of rows written (inserted or updated), or 0 if the write was
        skipped or failed.
    """
    active = settings or get_settings()
    pool = get_pool()
    if pool is None:
        # PostgreSQL not configured. The chatbot has no CSV fallback, so
        # there is simply nothing to do here.
        logger.debug("[PG][BU-EMAILS] pool unavailable; skipping write")
        return 0

    if not pairs:
        return 0

    try:
        table = validate_table_name(active.chatbot_pg_table)
    except ValueError as e:
        logger.error("[PG][BU-EMAILS] %s", e)
        return 0

    # Collapse to one row per message_id, keeping the LAST occurrence.
    # Two reasons this matters:
    #  1. A single statement cannot apply ON CONFLICT DO UPDATE to the same
    #     row twice — PostgreSQL raises "ON CONFLICT DO UPDATE command
    #     cannot affect row a second time".
    #  2. A thread fetch can legitimately contain the same message twice.
    deduped: dict[str, dict] = {}
    skipped = 0
    for message, attachments in pairs:
        message_id = (getattr(message, "id", "") or "").strip()
        conversation_id = (getattr(message, "conversation_id", "") or "").strip()
        # The chatbot's SELECT discards rows with either field blank, so
        # writing them would just be dead weight.
        if not message_id or not conversation_id:
            skipped += 1
            continue

        subject = getattr(message, "subject", "") or ""
        # Keep newlines: the CSV writer flattens them only to stay within
        # one CSV record. The chatbot's _clean_content normalises
        # whitespace on read anyway.
        body = getattr(message, "body_text", "") or ""
        attachment_data = _attachment_data_for_message(message_id, attachments)

        deduped[message_id] = {
            # Primary key on externally-created tables. Deriving it from the
            # Graph message id keeps it deterministic, so it doubles as the
            # dedup key: re-running the pipeline targets the same PK row.
            "unique_id": message_id,
            "message_id": message_id,
            "conversation_id": conversation_id,
            # Column names used by this project's schema...
            "subject": subject,
            "raw_content": body,
            "attachment_data": attachment_data,
            # ...and the equivalents used by the standalone chatbot
            # project's layout. Both sets are written when both exist so
            # the row is coherent for either reader.
            "email_subject": subject,
            "email_content": body,
            "attachment_content": attachment_data,
            # Native datetime -> TIMESTAMPTZ, so the chatbot's
            # "ORDER BY received_at" sorts chronologically.
            "received_at": getattr(message, "received_at", None),
            "from_address": getattr(message, "from_address", "") or "",
            "from_name": getattr(message, "from_name", "") or "",
            # The "to" recipients fetched via Graph (toRecipients), joined
            # into a single ", "-delimited string for the TEXT column.
            "to_address": ", ".join(getattr(message, "to_recipients", []) or []),
        }

    if skipped:
        logger.warning(
            "[PG][BU-EMAILS] skipped %d message(s) missing message_id/conversation_id",
            skipped,
        )

    if not deduped:
        return 0

    try:
        async with pool.acquire() as conn:
            # Introspect instead of assuming: the live table may be this
            # project's 10-column schema or an externally-created one with
            # extra NOT NULL columns and different field names.
            existing = await bu_emails_existing_columns(conn, table)

            # Intersect our candidate values with what the table actually
            # has, preserving a deterministic order for parameter binding.
            # Column names come from this fixed whitelist (never user
            # input), so interpolating them into the SQL is safe.
            columns = [c for c in _BU_EMAILS_WRITE_COLUMNS if c in existing]
            if "message_id" not in columns or "conversation_id" not in columns:
                logger.error(
                    "[PG][BU-EMAILS] table '%s' lacks message_id/conversation_id; "
                    "cannot write. Found columns: %s",
                    table, sorted(existing),
                )
                return 0

            # Pick a conflict target backed by a real unique index.
            # Prefer message_id (the natural key). Fall back to unique_id,
            # which is the primary key on externally-created tables and is
            # populated above with the same deterministic value.
            conflict_column = None
            for candidate in ("message_id", "unique_id"):
                if candidate in columns and await has_unique_single_column_index(
                    conn, table, candidate
                ):
                    conflict_column = candidate
                    break

            if not conflict_column:
                logger.warning(
                    "[PG][BU-EMAILS] table '%s' has no UNIQUE index on "
                    "message_id or unique_id; inserting without dedup. "
                    "Duplicate rows are possible until an index is added.",
                    table,
                )

            sql, columns = build_bu_emails_insert(
                table, existing, conflict_column
            )

            rows = [
                tuple(values[c] for c in columns) for values in deduped.values()
            ]

            async with conn.transaction():
                await conn.executemany(sql, rows)
    except Exception as e:
        logger.error(
            "[PG][BU-EMAILS] write to '%s' failed (%d row(s) not persisted): %r",
            table, len(deduped), e,
            exc_info=True,
        )
        return 0

    logger.info(
        "[PG][BU-EMAILS] upserted %d message row(s) into '%s' "
        "(conflict_target=%s, columns=%d) at %s",
        len(rows), table, conflict_column or "none", len(columns),
        datetime.now(timezone.utc).isoformat(),
    )
    return len(rows)