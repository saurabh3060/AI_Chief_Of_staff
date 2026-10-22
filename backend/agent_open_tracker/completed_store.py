"""
agent_open_tracker/completed_store.py
==========================================
Persistence layer for tracker items that have been marked complete by the
CEO via the Open Actions Status Tracker card on the Executive Launchpad.

Storage:
  - **PostgreSQL** (preferred): ``tracker_completed_items`` table, used
    when ``POSTGRES_DSN`` is configured and the connection pool is live.
  - **JSON fallback**: ``completed_tracker_items.json`` in the data
    directory, used when PostgreSQL is not configured (pool is None).

Item identity: each tracker item is identified by a deterministic SHA-256
hash of ``owner + "||" + text + "||" + business_unit`` (all lower-cased,
stripped).  This means the same action item will always get the same ID
whether it comes from the GET or the POST endpoint, with no database
required.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from core.config import get_settings
from core.db import get_pool
from core.logging_config import get_agent_logger
from core.utils import sanitize_log_input

logger = get_agent_logger("agent_open_tracker")

COMPLETED_FILENAME = "completed_tracker_items.json"

_write_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def make_item_id(owner: str, text: str, business_unit: str) -> str:
    """
    Return a stable, deterministic 16-char hex ID for a tracker action item.

    Built from a SHA-256 of the lowercased, stripped concatenation of
    ``owner``, ``text``, and ``business_unit`` so the same logical item
    always maps to the same ID regardless of when the request is made.
    """
    raw = f"{owner.strip().lower()}||{text.strip().lower()}||{business_unit.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Path helpers (JSON fallback only)
# ---------------------------------------------------------------------------

def _completed_json_path() -> Path:
    settings = get_settings()
    return settings.resolved_bu_email_data_dir / COMPLETED_FILENAME


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

async def load_completed_ids() -> set[str]:
    """Return the set of item IDs that have been marked complete."""
    pool = get_pool()

    # ── PostgreSQL path ──
    if pool is not None:
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT id FROM tracker_completed_items")
        return {row["id"] for row in rows}

    # ── JSON fallback ──
    path = _completed_json_path()
    if not path.exists():
        return set()
    async with _write_lock:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {entry["id"] for entry in data if "id" in entry}
        except (json.JSONDecodeError, KeyError):
            logger.warning("completed_tracker_items.json is malformed — treating as empty")
            return set()


async def load_completed_items() -> list[dict]:
    """Return the full list of completed item dicts."""
    pool = get_pool()

    # ── PostgreSQL path ──
    if pool is not None:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, text, owner, business_unit, due_date, source_topic, completed_at "
                "FROM tracker_completed_items ORDER BY completed_at DESC"
            )
        return [dict(row) for row in rows]

    # ── JSON fallback ──
    path = _completed_json_path()
    if not path.exists():
        return []
    async with _write_lock:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("completed_tracker_items.json is malformed — returning empty list")
            return []


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

async def save_completed_item(
    item_id: str,
    text: str,
    owner: str,
    business_unit: str,
    due_date: str | None,
    source_topic: str,
) -> dict:
    """
    Persist a newly completed item.  Returns the saved dict.
    No-ops (without error) if the item ID is already present so
    double-clicks are harmless.
    """
    pool = get_pool()

    # ── PostgreSQL path ──
    if pool is not None:
        completed_at = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            # Idempotent — ON CONFLICT DO NOTHING so double-clicks are safe.
            await conn.execute(
                """
                INSERT INTO tracker_completed_items (id, text, owner, business_unit, due_date, source_topic, completed_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO NOTHING
                """,
                item_id, text, owner, business_unit, due_date, source_topic, completed_at,
            )
        entry = {
            "id": item_id,
            "text": text,
            "owner": owner,
            "business_unit": business_unit,
            "due_date": due_date,
            "source_topic": source_topic,
            "completed_at": completed_at.isoformat(),
        }
        safe_item_id = str(item_id).replace("\r\n", "").replace("\r", "").replace("\n", "")
        logger.info("Marked item %s complete (postgres) (%s / %s)", safe_item_id, owner, business_unit)
        return entry

    # ── JSON fallback ──
    path = _completed_json_path()

    async with _write_lock:
        existing: list[dict] = []
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = []

        # Idempotency — don't add the same item twice.
        safe_item_id = str(item_id).replace("\r\n", "").replace("\r", "").replace("\n", "")
        if any(e.get("id") == item_id for e in existing):
            logger.info("Item %s already marked complete — skipping", safe_item_id)
            return next(e for e in existing if e["id"] == item_id)

        completed_at_str = datetime.now(timezone.utc).isoformat()
        entry = {
            "id": item_id,
            "text": text,
            "owner": owner,
            "business_unit": business_unit,
            "due_date": due_date,
            "source_topic": source_topic,
            "completed_at": completed_at_str,
        }
        existing.append(entry)
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Marked item %s complete (json) (%s / %s)", safe_item_id, owner, business_unit)
        return entry


async def remove_completed_item(item_id: str) -> bool:
    """
    Un-mark a previously completed item, removing it from the completed
    store so it reappears in future ``GET /agents/open-tracker`` responses
    (in either the ``items`` or ``pending_decisions`` bucket, depending on
    its owner).

    Returns True if the item was found and removed, False if it was not
    present in the completed store (no-op, not an error — makes repeated
    undo clicks harmless).
    """
    pool = get_pool()

    # ── PostgreSQL path ──
    if pool is not None:
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM tracker_completed_items WHERE id = $1",
                item_id,
            )
        # asyncpg returns a string like "DELETE 1" or "DELETE 0".
        deleted = result.endswith("1")
        if deleted:
            safe_item_id = str(item_id).replace("\r\n", "").replace("\r", "").replace("\n", "")
            logger.info("Un-marked item %s complete (postgres)", safe_item_id)
        return deleted

    # ── JSON fallback ──
    path = _completed_json_path()
    if not path.exists():
        return False

    async with _write_lock:
        try:
            existing: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False

        remaining = [e for e in existing if e.get("id") != item_id]
        if len(remaining) == len(existing):
            return False

        path.write_text(json.dumps(remaining, indent=2, ensure_ascii=False), encoding="utf-8")
        safe_item_id = str(item_id).replace("\r\n", "").replace("\r", "").replace("\n", "")
        logger.info("Un-marked item %s complete (json)", safe_item_id)
        return True
