"""
documents/sync_state.py
==========================
OneDrive -> document library sync state, stored in PostgreSQL.

Two tables (created by core.db.init_db):

- ``onedrive_sync_files``  — one row per tracked SharePoint file
- ``onedrive_sync_runs``   — one row per sync run (most recent = current status)

All public functions are async because they talk to PostgreSQL.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from core.db import get_pool
from core.logging_config import get_agent_logger

logger = get_agent_logger("documents")


# ---------------------------------------------------------------------------
# Load / query helpers
# ---------------------------------------------------------------------------

async def load_state() -> dict:
    """Load the full sync state from PostgreSQL.

    Returns a dict shaped like the old JSON file so callers don't change::

        {
            "last_sync_at": ...,
            "last_sync_status": ...,
            "last_sync_summary": {...},
            "files": { "<item_id>": { ... }, ... },
        }
    """
    pool = get_pool()
    if pool is None:
        logger.warning("PostgreSQL not available — returning empty sync state")
        return _empty_state()

    async with pool.acquire() as conn:
        # Latest run
        run = await conn.fetchrow(
            "SELECT status, summary, created_at "
            "FROM onedrive_sync_runs ORDER BY created_at DESC LIMIT 1"
        )
        # All tracked files
        rows = await conn.fetch(
            "SELECT item_id, name, relative_path, last_modified, "
            "       size, doc_id, synced_at "
            "FROM onedrive_sync_files"
        )

    state = _empty_state()
    if run:
        state["last_sync_at"] = run["created_at"].isoformat() if run["created_at"] else None
        state["last_sync_status"] = run["status"]
        state["last_sync_summary"] = run["summary"]  # already a dict (JSONB)

    # Collect all doc_ids referenced by sync files, then check which ones
    # are actually missing from the documents table.  This lets needs_sync()
    # flag files for re-ingestion when the document row was deleted.
    doc_ids = {r["doc_id"] for r in rows if r["doc_id"]}
    missing_doc_ids: set[str] = set()
    if doc_ids:
        async with pool.acquire() as conn2:
            existing = await conn2.fetch(
                "SELECT doc_id FROM documents WHERE doc_id = ANY($1::text[])",
                list(doc_ids),
            )
            existing_ids = {r["doc_id"] for r in existing}
            missing_doc_ids = doc_ids - existing_ids
            if missing_doc_ids:
                logger.info(
                    "[SYNC-STATE] %d tracked file(s) have missing document rows — "
                    "will re-sync: %s",
                    len(missing_doc_ids),
                    ", ".join(sorted(missing_doc_ids)),
                )

    for r in rows:
        record = {
            "name": r["name"],
            "relative_path": r["relative_path"],
            "last_modified": r["last_modified"],
            "size": r["size"],
            "doc_id": r["doc_id"],
            "synced_at": r["synced_at"].isoformat() if r["synced_at"] else "",
        }
        if r["doc_id"] in missing_doc_ids:
            record["_doc_missing"] = True
        state["files"][r["item_id"]] = record
    return state


async def save_state(state: dict) -> None:
    """Persist the full state to PostgreSQL.

    This is called after every file sync and at the end of a run.
    It upserts all files and the latest run result in one go.
    """
    pool = get_pool()
    if pool is None:
        logger.warning("PostgreSQL not available — cannot save sync state")
        return

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Upsert each tracked file
            for item_id, record in state.get("files", {}).items():
                await conn.execute(
                    """
                    INSERT INTO onedrive_sync_files
                        (item_id, name, relative_path, last_modified, size, doc_id, synced_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (item_id) DO UPDATE SET
                        name          = EXCLUDED.name,
                        relative_path = EXCLUDED.relative_path,
                        last_modified = EXCLUDED.last_modified,
                        size          = EXCLUDED.size,
                        doc_id        = EXCLUDED.doc_id,
                        synced_at     = EXCLUDED.synced_at
                    """,
                    item_id,
                    record.get("name", ""),
                    record.get("relative_path", ""),
                    record.get("last_modified", ""),
                    record.get("size", 0),
                    record.get("doc_id", ""),
                    _parse_ts(record.get("synced_at")),
                )

            # Insert a new run row if the caller recorded one
            if state.get("last_sync_status"):
                await conn.execute(
                    """
                    INSERT INTO onedrive_sync_runs (status, summary, created_at)
                    VALUES ($1, $2, $3)
                    """,
                    state["last_sync_status"],
                    json.dumps(state.get("last_sync_summary"), default=str)
                    if state.get("last_sync_summary") is not None else None,
                    _parse_ts(state.get("last_sync_at")) or datetime.now(timezone.utc),
                )


# ---------------------------------------------------------------------------
# State helpers (same API as before — callers don't change)
# ---------------------------------------------------------------------------

def _empty_state() -> dict:
    """Return a fresh, empty state dict."""
    return {
        "last_sync_at": None,
        "last_sync_status": None,
        "last_sync_summary": None,
        "files": {},
    }


def needs_sync(drive_item: dict, state: dict) -> bool:
    """Check whether a OneDrive drive item needs to be downloaded.

    Compares ``lastModifiedDateTime`` against the stored record.
    Also flags a file for re-sync if the associated ``doc_id`` is missing
    from the ``documents`` table (e.g. it was deleted or ingestion failed).
    """
    item_id = drive_item.get("id", "")
    if not item_id:
        return True

    record = state.get("files", {}).get(item_id)
    if record is None:
        return True

    # If the document row was deleted from PostgreSQL, re-sync even if
    # the SharePoint metadata hasn't changed.
    if record.get("_doc_missing"):
        return True

    remote_modified = drive_item.get("lastModifiedDateTime", "")
    stored_modified = record.get("last_modified", "")
    if remote_modified and remote_modified != stored_modified:
        return True

    return False


def record_sync(
    state: dict,
    item_id: str,
    drive_item: dict,
    doc_id: str,
) -> None:
    """Update the in-memory state for a successfully synced file.

    Caller must call ``await save_state(state)`` afterwards to persist.
    """
    state.setdefault("files", {})[item_id] = {
        "name": drive_item.get("name", ""),
        "relative_path": drive_item.get("_relative_path", ""),
        "last_modified": drive_item.get("lastModifiedDateTime", ""),
        "size": drive_item.get("size", 0),
        "doc_id": doc_id,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


def update_sync_result(
    state: dict,
    status: str,
    summary: dict | None = None,
) -> None:
    """Record the overall sync run result in the in-memory state.

    ``status`` is one of ``"success"``, ``"partial"``, ``"failed"``.
    Caller must call ``await save_state(state)`` afterwards to persist.
    """
    state["last_sync_at"] = datetime.now(timezone.utc).isoformat()
    state["last_sync_status"] = status
    state["last_sync_summary"] = summary


def get_tracked_files(state: dict) -> list[dict]:
    """Return a list of tracked file records for the status API."""
    files = []
    for item_id, record in state.get("files", {}).items():
        files.append({
            "item_id": item_id,
            **record,
        })
    return sorted(files, key=lambda f: f.get("name", ""))


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _parse_ts(value) -> datetime | None:
    """Best-effort parse an ISO timestamp string into a datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
