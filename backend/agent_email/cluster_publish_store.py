"""
agent_email/cluster_publish_store.py
====================================
PostgreSQL access for the incremental cluster publish flow.

Owns every SQL statement that touches ``cluster_publish_staging`` plus the
read-only lookups the flow needs against the raw-email table (``bu_emails``).
Business rules (matching, thresholds, retry policy) live in
``agent_email/cluster_publish.py``; this module is deliberately dumb.

Safety properties
-----------------
* Staging rows for a run are inserted in ONE transaction, so a crash can
  never leave a cluster with only some of its rows ("all rows available").
* Inserts are ``ON CONFLICT DO NOTHING`` on the unique key, so re-staging
  after a crash is a no-op and never resets ``retry_count``.
* ``delete_cluster_rows`` is the only delete, and it is scoped to exactly
  one ``(run_id, cluster_id)``. Nothing here deletes pending work in bulk.
* ``try_advisory_lock`` gives single-writer semantics across processes.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from core.config import Settings, get_settings
from core.db import get_pool
from core.logging_config import get_agent_logger, sanitize_log_value

from agent_email.db_schema import bu_emails_existing_columns, validate_table_name
from agent_email.schema import EmailMessage

logger = get_agent_logger("agent_email")

STAGING_TABLE = "cluster_publish_staging"
NEW_CLUSTER = "NEW_CLUSTER"
UPDATED_CLUSTER = "UPDATED_CLUSTER"

# Arbitrary constant: pg_try_advisory_lock key for "one publish run at a time".
_ADVISORY_LOCK_KEY = 0x434C5055  # "CLPU"


@dataclass(frozen=True)
class StagingRow:
    run_id: str
    cluster_id: str
    email_message_id: str
    email_subject: str | None
    cluster_action: str
    retry_count: int = 0


@dataclass(frozen=True)
class ClusterStaging:
    """Aggregate view of one ``(run_id, cluster_id)`` group of staging rows."""

    run_id: str
    cluster_id: str
    row_count: int
    retry_count: int          # MAX over the group's rows
    oldest_created_at: datetime


def new_run_id(now: datetime | None = None) -> str:
    """``run_{YYYYMMDDHHMMSS}_{uuid}`` (UTC)."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d%H%M%S")
    return f"run_{stamp}_{uuid.uuid4().hex}"


def _pool():
    pool = get_pool()
    if pool is None:
        raise RuntimeError("cluster_publish_store: PostgreSQL pool unavailable")
    return pool


@asynccontextmanager
async def advisory_lock():
    """Hold the publish-run advisory lock; yields False if another run has it.

    The lock lives on a dedicated connection for the whole run and is
    released explicitly (and implicitly if the connection dies), so a crashed
    process can never wedge future runs.
    """
    pool = _pool()
    async with pool.acquire() as conn:
        acquired = bool(await conn.fetchval("SELECT pg_try_advisory_lock($1)", _ADVISORY_LOCK_KEY))
        try:
            yield acquired
        finally:
            if acquired:
                try:
                    await conn.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_KEY)
                except Exception as exc:  # noqa: BLE001 - connection teardown also releases it
                    logger.warning("[CLUSTER-PUBLISH] advisory unlock failed: %s", sanitize_log_value(repr(exc)))


# ---------------------------------------------------------------------------
# Staging table
# ---------------------------------------------------------------------------

async def insert_staging_rows(rows: list[StagingRow]) -> int:
    """Insert all rows atomically; duplicates are ignored. Returns rows actually inserted."""
    if not rows:
        return 0
    pool = _pool()
    payload = [
        (r.run_id, r.cluster_id, r.email_message_id, r.email_subject, r.cluster_action)
        for r in rows
    ]
    async with pool.acquire() as conn:
        async with conn.transaction():
            before = await conn.fetchval(
                f"SELECT count(*) FROM {STAGING_TABLE} WHERE run_id = $1", rows[0].run_id,
            )
            await conn.executemany(
                f"""
                INSERT INTO {STAGING_TABLE}
                    (run_id, cluster_id, email_message_id, email_subject, cluster_action)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (run_id, cluster_id, email_message_id) DO NOTHING
                """,
                payload,
            )
            after = await conn.fetchval(
                f"SELECT count(*) FROM {STAGING_TABLE} WHERE run_id = $1", rows[0].run_id,
            )
    return int(after) - int(before)


async def list_pending_run_ids() -> list[str]:
    """Run ids that still have staging rows, oldest first."""
    pool = _pool()
    async with pool.acquire() as conn:
        recs = await conn.fetch(
            f"""
            SELECT run_id FROM {STAGING_TABLE}
            GROUP BY run_id
            ORDER BY min(created_at), run_id
            """
        )
    return [r["run_id"] for r in recs]


async def run_counts(run_id: str) -> tuple[int, int]:
    """(distinct cluster ids, distinct email ids) staged for ``run_id``."""
    pool = _pool()
    async with pool.acquire() as conn:
        rec = await conn.fetchrow(
            f"""
            SELECT count(DISTINCT cluster_id) AS clusters,
                   count(DISTINCT email_message_id) AS emails
            FROM {STAGING_TABLE} WHERE run_id = $1
            """,
            run_id,
        )
    return int(rec["clusters"]), int(rec["emails"])


async def list_cluster_groups(run_id: str) -> list[ClusterStaging]:
    """One entry per cluster staged under ``run_id``, oldest first."""
    pool = _pool()
    async with pool.acquire() as conn:
        recs = await conn.fetch(
            f"""
            SELECT cluster_id,
                   count(*) AS row_count,
                   max(retry_count) AS retry_count,
                   min(created_at) AS oldest
            FROM {STAGING_TABLE}
            WHERE run_id = $1
            GROUP BY cluster_id
            ORDER BY min(created_at), cluster_id
            """,
            run_id,
        )
    return [
        ClusterStaging(
            run_id=run_id,
            cluster_id=r["cluster_id"],
            row_count=int(r["row_count"]),
            retry_count=int(r["retry_count"]),
            oldest_created_at=r["oldest"],
        )
        for r in recs
    ]


async def rows_for_cluster(run_id: str, cluster_id: str) -> list[StagingRow]:
    pool = _pool()
    async with pool.acquire() as conn:
        recs = await conn.fetch(
            f"""
            SELECT run_id, cluster_id, email_message_id, email_subject,
                   cluster_action, retry_count
            FROM {STAGING_TABLE}
            WHERE run_id = $1 AND cluster_id = $2
            ORDER BY created_at, id
            """,
            run_id, cluster_id,
        )
    return [
        StagingRow(
            run_id=r["run_id"], cluster_id=r["cluster_id"],
            email_message_id=r["email_message_id"], email_subject=r["email_subject"],
            cluster_action=r["cluster_action"], retry_count=int(r["retry_count"]),
        )
        for r in recs
    ]


async def staged_members_by_cluster() -> dict[str, list[tuple[str, str | None]]]:
    """``{cluster_id: [(message_id, subject), ...]}`` across EVERY pending run.

    Used both to de-duplicate new emails (already staged => not new) and so
    that a new email can still match a cluster whose publish is deferred.
    """
    pool = _pool()
    async with pool.acquire() as conn:
        recs = await conn.fetch(
            f"SELECT cluster_id, email_message_id, email_subject FROM {STAGING_TABLE} "
            f"ORDER BY created_at, id"
        )
    out: dict[str, list[tuple[str, str | None]]] = {}
    for r in recs:
        out.setdefault(r["cluster_id"], []).append((r["email_message_id"], r["email_subject"]))
    return out


async def bump_retry(run_id: str, cluster_id: str) -> int:
    """Increment retry_count on every row of the group; return the new value."""
    pool = _pool()
    async with pool.acquire() as conn:
        value = await conn.fetchval(
            f"""
            WITH bumped AS (
                UPDATE {STAGING_TABLE}
                   SET retry_count = retry_count + 1, updated_at = now()
                 WHERE run_id = $1 AND cluster_id = $2
             RETURNING retry_count
            )
            SELECT max(retry_count) FROM bumped
            """,
            run_id, cluster_id,
        )
    return int(value or 0)


async def delete_cluster_rows(run_id: str, cluster_id: str) -> int:
    """Delete the staging rows of exactly one ``(run_id, cluster_id)``."""
    pool = _pool()
    async with pool.acquire() as conn:
        status = await conn.execute(
            f"DELETE FROM {STAGING_TABLE} WHERE run_id = $1 AND cluster_id = $2",
            run_id, cluster_id,
        )
    # asyncpg returns e.g. "DELETE 12"
    return int(status.rsplit(" ", 1)[-1])


async def remaining_rows(run_id: str | None = None) -> int:
    pool = _pool()
    async with pool.acquire() as conn:
        if run_id is None:
            return int(await conn.fetchval(f"SELECT count(*) FROM {STAGING_TABLE}"))
        return int(await conn.fetchval(
            f"SELECT count(*) FROM {STAGING_TABLE} WHERE run_id = $1", run_id,
        ))


# ---------------------------------------------------------------------------
# Raw email lookups (bu_emails)
# ---------------------------------------------------------------------------

_SUBJECT_COLS = ("subject", "email_subject")
_BODY_COLS = ("raw_content", "email_content")
_ATTACH_COLS = ("attachment_data", "attachment_content")


def _first_present(columns: set[str], options: tuple[str, ...]) -> str | None:
    return next((c for c in options if c in columns), None)


async def read_emails_by_ids(
    message_ids: list[str], settings: Settings | None = None,
) -> tuple[dict[str, EmailMessage], dict[str, str]]:
    """Load stored raw emails. Returns ``({id: EmailMessage}, {id: attachment_text})``.

    Missing ids are simply absent; the caller decides how to treat them.
    """
    ids = [m for m in dict.fromkeys(message_ids) if m]
    if not ids:
        return {}, {}
    active = settings or get_settings()
    table = validate_table_name(active.chatbot_pg_table)
    pool = _pool()
    async with pool.acquire() as conn:
        cols = await bu_emails_existing_columns(conn, table)
        subj = _first_present(cols, _SUBJECT_COLS)
        body = _first_present(cols, _BODY_COLS)
        attach = _first_present(cols, _ATTACH_COLS)
        wanted = ["message_id", "conversation_id"]
        wanted += [c for c in (subj, body, attach) if c]
        wanted += [c for c in ("received_at", "from_address", "from_name", "to_address") if c in cols]
        if "message_id" not in cols:
            return {}, {}
        # Column names come from fixed whitelists above, never user input.
        recs = await conn.fetch(
            f"SELECT {', '.join(wanted)} FROM {table} WHERE message_id = ANY($1::text[])",
            ids,
        )

    messages: dict[str, EmailMessage] = {}
    attachments: dict[str, str] = {}
    for r in recs:
        mid = r["message_id"]
        received = r["received_at"] if "received_at" in r.keys() else None
        if received is None:
            logger.warning("[CLUSTER-PUBLISH] event=email_missing_received_at email_message_id=%s", sanitize_log_value(mid))
            received = datetime.now(timezone.utc)
        to_addr = (r["to_address"] if "to_address" in r.keys() else "") or ""
        messages[mid] = EmailMessage(
            id=mid,
            conversation_id=(r["conversation_id"] or ""),
            subject=(r[subj] if subj else "") or "",
            from_name=(r["from_name"] if "from_name" in r.keys() else None),
            from_address=(r["from_address"] if "from_address" in r.keys() else None),
            to_recipients=[a.strip() for a in to_addr.split(",") if a.strip()],
            received_at=received,
            body_text=(r[body] if body else "") or "",
        )
        if attach and r[attach]:
            attachments[mid] = r[attach]
    return messages, attachments


async def message_ids_for_conversations(conversation_ids: list[str], settings: Settings | None = None) -> dict[str, str]:
    """``{message_id: conversation_id}`` for every stored mail in the given threads."""
    convs = [c for c in dict.fromkeys(conversation_ids) if c]
    if not convs:
        return {}
    active = settings or get_settings()
    table = validate_table_name(active.chatbot_pg_table)
    pool = _pool()
    async with pool.acquire() as conn:
        cols = await bu_emails_existing_columns(conn, table)
        if "message_id" not in cols or "conversation_id" not in cols:
            return {}
        recs = await conn.fetch(
            f"SELECT message_id, conversation_id FROM {table} WHERE conversation_id = ANY($1::text[])",
            convs,
        )
    return {r["message_id"]: r["conversation_id"] for r in recs}
