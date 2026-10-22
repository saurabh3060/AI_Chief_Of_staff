"""
documents/store.py
===================
Async PostgreSQL data layer for the documents bot, with a pluggable vector
backend (Weaviate or FAISS, controlled by ``CHATBOT_VECTOR_DB`` in ``.env``).

Row data lives in PostgreSQL.  Block and series embeddings are delegated
to whichever vector backend is active — see ``_vec()`` below.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Iterable

import asyncpg

from core.db import get_pool
from core.logging_config import get_agent_logger

logger = get_agent_logger("documents")


# ---------------------------------------------------------------------------
# Vector backend dispatch
# ---------------------------------------------------------------------------

def _vec():
    """Return the active vector-store module (faiss_store or weaviate_store).

    Resolved on every call so the toggle can be changed without restarting
    (useful for tests).  The import is cheap — both modules cache their state.
    """
    from core.config import get_settings
    if get_settings().use_faiss:
        from . import faiss_store
        return faiss_store
    from . import weaviate_store
    return weaviate_store


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

@asynccontextmanager
async def acquire():
    """Acquire a connection from the shared ``core.db`` pool.

    Usage::

        async with store.acquire() as conn:
            rows = await store.documents(conn)
    """
    pool = get_pool()
    if pool is None:
        raise RuntimeError(
            "PostgreSQL pool not initialised. Ensure POSTGRES_DSN is set "
            "and core.db.init_db() has run."
        )
    async with pool.acquire() as conn:
        yield conn


# ---------------------------------------------------------------------------
# Delete helpers
# ---------------------------------------------------------------------------

async def delete_document(conn: asyncpg.Connection, doc_id: str,
                          keep_row: bool = False) -> None:
    """Delete every derived row for ``doc_id``.

    Also removes the document's vectors from Weaviate (blocks + series).
    Corrections are deliberately kept, matching the SQLite behaviour.
    """
    # Collect block/series ids so we can tell Weaviate what to drop.
    block_ids = [r["block_id"] for r in await conn.fetch(
        "SELECT block_id FROM blocks WHERE doc_id = $1", doc_id)]
    series_ids = [r["series_id"] for r in await conn.fetch(
        "SELECT series_id FROM series WHERE doc_id = $1", doc_id)]

    for table in ("blocks", "series", "facts", "issues", "pages",
                  "calculated_metrics"):
        await conn.execute(f"DELETE FROM {table} WHERE doc_id = $1", doc_id)
    if not keep_row:
        await conn.execute("DELETE FROM documents WHERE doc_id = $1", doc_id)

    # Vector cleanup is best-effort — a backend hiccup must not fail the DB delete.
    try:
        import asyncio
        vec = _vec()
        if block_ids:
            await asyncio.to_thread(vec.delete_block_ids, block_ids)
        if series_ids:
            await asyncio.to_thread(vec.delete_series_ids, series_ids)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[DOC-STORE] delete_document: vector cleanup failed for doc_id=%s: %s",
            doc_id, exc,
        )


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------

async def add_block(
    conn: asyncpg.Connection,
    doc_id: str,
    page: int,
    kind: str,
    text: str,
    *,
    section: str = "",
    title: str = "",
    content=None,
    bbox=None,
    source: str = "",
    confidence: float = 1.0,
    section_path: list[str] | None = None,
    anchor_block_id: int | None = None,
) -> int:
    """Insert a block row.  Returns the new ``block_id``.

    The ``tsv`` column is populated automatically by the ``blocks_tsv_trigger``
    PostgreSQL trigger — no explicit FTS insert needed.
    """
    return int(await conn.fetchval(
        """
        INSERT INTO blocks
            (doc_id, page, kind, section, title, text, content, bbox,
             source, confidence, section_path, anchor_block_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING block_id
        """,
        doc_id, page, kind, section, title, text,
        json.dumps(content, ensure_ascii=False) if content is not None else None,
        json.dumps(bbox) if bbox is not None else None,
        source, float(confidence),
        json.dumps(section_path, ensure_ascii=False) if section_path else None,
        anchor_block_id,
    ))


async def add_series(
    conn: asyncpg.Connection,
    doc_id: str,
    page: int,
    block_id: int | None,
    label: str,
    name: str,
    metric_id: str,
    kind: str,
    unit: str,
    entity: str,
    source: str,
    confidence: float,
    path: list[str] | None = None,
    scale: float | None = None,
) -> int:
    """Insert a series row.  Returns the new ``series_id``.

    The ``tsv`` column (over ``label``) is populated automatically by the
    ``series_tsv_trigger`` PostgreSQL trigger.
    """
    return int(await conn.fetchval(
        """
        INSERT INTO series
            (doc_id, page, block_id, label, name, metric_id, kind, unit,
             entity, source, confidence, path, scale)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        RETURNING series_id
        """,
        doc_id, page, block_id, label, name, metric_id, kind, unit, entity,
        source, float(confidence),
        json.dumps(path, ensure_ascii=False) if path else None,
        scale,
    ))


UNDATED = "period not stated"


async def add_fact(
    conn: asyncpg.Connection,
    series_id: int,
    doc_id: str,
    page: int,
    period,
    value: float,
    as_printed: str,
    validated: bool,
    confidence: float,
    note: str = "",
) -> None:
    """Insert a fact row.

    ``period`` may be ``None``: a value for a category/text row with no
    period on the page.  Period columns are then empty strings — no
    period filter matches, matching the SQLite behaviour.
    """
    when = (
        (period.key, period.kind, period.start.isoformat(), period.end.isoformat(),
         period.label, int(period.partial))
        if period else ("", "", "", "", UNDATED, 0)
    )
    await conn.execute(
        """
        INSERT INTO facts
            (series_id, doc_id, page, period_key, period_kind,
             period_start, period_end, period_label, partial,
             value, as_printed, validated, confidence, note)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """,
        series_id, doc_id, page, *when,
        float(value) if value is not None else None,
        as_printed, int(validated), float(confidence), note,
    )


async def add_issue(
    conn: asyncpg.Connection,
    doc_id: str,
    page: int,
    kind: str,
    ref: str,
    message: str,
    suggestion: str = "",
) -> None:
    await conn.execute(
        """
        INSERT INTO issues (doc_id, page, kind, ref, message, suggestion)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        doc_id, page, kind, ref, message, suggestion,
    )


# ---------------------------------------------------------------------------
# Vectors (Weaviate)
# ---------------------------------------------------------------------------

async def set_embeddings(
    conn: asyncpg.Connection,  # kept in signature for symmetry; unused
    table: str,
    doc_id: str,
    rows: Iterable[tuple[int, list[float]]],
) -> int:
    """Push embeddings for ``table`` ('blocks' or 'series') to the active
    vector backend (Weaviate or FAISS, per ``CHATBOT_VECTOR_DB``).

    Returns the number of vectors written.
    """
    import asyncio
    items = [(int(row_id), str(doc_id), vec) for row_id, vec in rows]
    if not items:
        return 0
    vec = _vec()
    if table == "blocks":
        return await asyncio.to_thread(vec.upsert_block_embeddings, items)
    if table == "series":
        return await asyncio.to_thread(vec.upsert_series_embeddings, items)
    raise ValueError(f"Unknown embedding table: {table!r}")


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

async def documents(conn: asyncpg.Connection) -> list[asyncpg.Record]:
    """Every document, newest first by as-of key then ingested_at."""
    return await conn.fetch(
        "SELECT * FROM documents "
        "ORDER BY as_of_key DESC, ingested_at DESC"
    )
