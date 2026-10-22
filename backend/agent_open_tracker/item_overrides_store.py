"""
agent_open_tracker/item_overrides_store.py
==============================================
Persistence layer for manually edited action item fields.

Stores overrides for any combination of: text (description), owner,
business_unit (team), and due_date.  Each override is keyed by the
deterministic item_id (same SHA-256 hash used everywhere).

Storage:
  - **PostgreSQL** (preferred): ``tracker_item_overrides`` table, used
    when ``POSTGRES_DSN`` is configured and the connection pool is live.
  - **JSON fallback**: ``item_overrides.json`` in the data directory,
    used when PostgreSQL is not configured (pool is None).

Thread-safety: JSON writes guarded by an asyncio lock; PostgreSQL
handles concurrency natively via connection pooling.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from core.config import get_settings
from core.db import get_pool
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_open_tracker")

OVERRIDES_FILENAME = "item_overrides.json"

_write_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Path helpers (JSON fallback only)
# ---------------------------------------------------------------------------

def _overrides_path() -> Path:
    settings = get_settings()
    return settings.resolved_bu_email_data_dir / OVERRIDES_FILENAME


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

async def load_item_overrides() -> dict[str, dict]:
    """
    Return a mapping of ``item_id -> override_dict`` for every item that
    has been manually edited.  Each override dict may contain any subset
    of: ``text``, ``owner``, ``business_unit``, ``due_date``.
    """
    pool = get_pool()

    # ── PostgreSQL path ──
    if pool is not None:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, text, owner, business_unit, due_date FROM tracker_item_overrides"
            )
        result: dict[str, dict] = {}
        for row in rows:
            override = {}
            if row["text"] is not None:
                override["text"] = row["text"]
            if row["owner"] is not None:
                override["owner"] = row["owner"]
            if row["business_unit"] is not None:
                override["business_unit"] = row["business_unit"]
            if row["due_date"] is not None:
                override["due_date"] = row["due_date"]
            if override:
                result[row["id"]] = override
        return result

    # ── JSON fallback ──
    path = _overrides_path()
    if not path.exists():
        return {}
    async with _write_lock:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                entry["id"]: {
                    k: v for k, v in entry.items()
                    if k not in ("id", "updated_at")
                }
                for entry in data
                if "id" in entry
            }
        except (json.JSONDecodeError, KeyError):
            logger.warning("item_overrides.json is malformed — treating as empty")
            return {}


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

async def save_item_override(
    item_id: str,
    text: str | None = None,
    owner: str | None = None,
    business_unit: str | None = None,
    due_date: str | None = None,
) -> dict:
    """
    Save (or update) manual overrides for the given item.

    Only non-None fields are written / updated.  Returns the full
    saved entry dict.
    """
    pool = get_pool()

    # ── PostgreSQL path ──
    if pool is not None:
        updated_at = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            # Try to fetch existing row first.
            existing = await conn.fetchrow(
                "SELECT id, text, owner, business_unit, due_date FROM tracker_item_overrides WHERE id = $1",
                item_id,
            )

            if existing is None:
                # Insert new row.
                await conn.execute(
                    """
                    INSERT INTO tracker_item_overrides (id, text, owner, business_unit, due_date, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    item_id, text, owner, business_unit,
                    due_date if due_date and due_date.strip() else None,
                    updated_at,
                )
            else:
                # Merge: only update non-None fields.
                final_text = text if text is not None else existing["text"]
                final_owner = owner if owner is not None else existing["owner"]
                final_bu = business_unit if business_unit is not None else existing["business_unit"]
                if due_date is not None:
                    final_due = due_date if due_date.strip() else None
                else:
                    final_due = existing["due_date"]

                await conn.execute(
                    """
                    UPDATE tracker_item_overrides
                    SET text = $2, owner = $3, business_unit = $4, due_date = $5, updated_at = $6
                    WHERE id = $1
                    """,
                    item_id, final_text, final_owner, final_bu, final_due, updated_at,
                )

        entry = {"id": item_id, "updated_at": updated_at.isoformat()}
        if text is not None:
            entry["text"] = text
        if owner is not None:
            entry["owner"] = owner
        if business_unit is not None:
            entry["business_unit"] = business_unit
        if due_date is not None:
            entry["due_date"] = due_date

        safe_item_id = str(item_id).replace("\r\n", "").replace("\r", "").replace("\n", "")
        logger.info(
            "Item override saved (postgres): item=%s fields=%s",
            safe_item_id,
            [k for k in ("text", "owner", "business_unit", "due_date") if entry.get(k)],
        )
        return entry

    # ── JSON fallback ──
    path = _overrides_path()

    async with _write_lock:
        existing_list: list[dict] = []
        if path.exists():
            try:
                existing_list = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing_list = []

        updated_at_str = datetime.now(timezone.utc).isoformat()

        # Find existing entry for this item.
        entry_found: dict | None = None
        for e in existing_list:
            if e.get("id") == item_id:
                entry_found = e
                break

        if entry_found is None:
            entry_found = {"id": item_id}
            existing_list.append(entry_found)

        # Apply non-None fields.
        if text is not None:
            entry_found["text"] = text
        if owner is not None:
            entry_found["owner"] = owner
        if business_unit is not None:
            entry_found["business_unit"] = business_unit
        if due_date is not None:
            if due_date.strip():
                entry_found["due_date"] = due_date
            elif "due_date" in entry_found:
                del entry_found["due_date"]
        entry_found["updated_at"] = updated_at_str

        path.write_text(
            json.dumps(existing_list, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        safe_item_id = str(item_id).replace("\r\n", "").replace("\r", "").replace("\n", "")
        logger.info(
            "Item override saved (json): item=%s fields=%s",
            safe_item_id,
            [k for k in ("text", "owner", "business_unit", "due_date") if entry_found.get(k)],
        )
        return entry_found
