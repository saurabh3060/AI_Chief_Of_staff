"""
agent_open_tracker/deleted_store.py
========================================
Persistence layer for tracker items that have been permanently deleted
by the CEO via the trash icon on the Ledger / Launchpad / BU view.

Mirrors the completed_store pattern exactly:

Storage:
  - **PostgreSQL** (preferred): ``tracker_deleted_items`` table, used
    when ``POSTGRES_DSN`` is configured and the connection pool is live.
  - **JSON fallback**: ``deleted_tracker_items.json`` in the data
    directory, used when PostgreSQL is not configured (pool is None).
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from core.config import get_settings
from core.db import get_pool
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_open_tracker")

DELETED_FILENAME = "deleted_tracker_items.json"

_write_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Path helpers (JSON fallback only)
# ---------------------------------------------------------------------------

def _deleted_json_path() -> Path:
    settings = get_settings()
    return settings.resolved_bu_email_data_dir / DELETED_FILENAME


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

async def load_deleted_ids() -> set[str]:
    """Return the set of item IDs that have been deleted."""
    pool = get_pool()

    # -- PostgreSQL path --
    if pool is not None:
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT id FROM tracker_deleted_items")
        return {row["id"] for row in rows}

    # -- JSON fallback --
    path = _deleted_json_path()
    if not path.exists():
        return set()
    async with _write_lock:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {entry["id"] for entry in data if "id" in entry}
        except (json.JSONDecodeError, KeyError):
            logger.warning("deleted_tracker_items.json is malformed - treating as empty")
            return set()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

async def save_deleted_item(item_id: str) -> dict:
    """
    Persist a deleted item ID. Returns the saved dict.
    No-ops (without error) if the item ID is already present so
    double-clicks are harmless.
    """
    pool = get_pool()

    # -- PostgreSQL path --
    if pool is not None:
        deleted_at = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tracker_deleted_items (id, deleted_at)
                VALUES ($1, $2)
                ON CONFLICT (id) DO NOTHING
                """,
                item_id, deleted_at,
            )
        entry = {"id": item_id, "deleted_at": deleted_at.isoformat()}
        safe_item_id = str(item_id).replace("\r\n", "").replace("\r", "").replace("\n", "")
        logger.info("Deleted item %s (postgres)", safe_item_id)
        return entry

    # -- JSON fallback --
    path = _deleted_json_path()

    async with _write_lock:
        existing: list[dict] = []
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = []

        safe_item_id = str(item_id).replace("\r\n", "").replace("\r", "").replace("\n", "")
        if any(e.get("id") == item_id for e in existing):
            logger.info("Item %s already deleted - skipping", safe_item_id)
            return next(e for e in existing if e["id"] == item_id)

        deleted_at_str = datetime.now(timezone.utc).isoformat()
        entry = {"id": item_id, "deleted_at": deleted_at_str}
        existing.append(entry)
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Deleted item %s (json)", safe_item_id)
        return entry
