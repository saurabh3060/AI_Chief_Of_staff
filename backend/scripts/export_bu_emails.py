"""
scripts/export_bu_emails.py
============================
Exports every row from the bu_emails table (+ attachment_data) to a
single JSON file so you can inspect the full dataset or import it elsewhere.

Run from the backend/ directory:

    python scripts/export_bu_emails.py

Options:
    --table       Table name (default: value of CHATBOT_PG_TABLE in .env, usually "bu_emails")
    --out         Output file path (default: export_bu_emails.json next to this script)
    --active-only Only export rows with is_active = 1  (default: exports ALL rows)
    --indent      JSON indentation level (default: 2; use 0 for compact)

Examples:
    python scripts/export_bu_emails.py
    python scripts/export_bu_emails.py --active-only
    python scripts/export_bu_emails.py --out /tmp/bu_emails_full.json
    python scripts/export_bu_emails.py --table bu_emails --active-only --out data/export.json
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── bootstrap sys.path so relative imports from backend/ work ─────────────
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.config import get_settings           # noqa: E402
from core.db import init_db, close_db, get_pool  # noqa: E402
from agent_email.db_schema import validate_table_name  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _serialize(value):
    """Make any PostgreSQL value JSON-safe."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return value.decode("utf-8", errors="replace")
    return value


async def export_rows(table: str, active_only: bool) -> list[dict]:
    """Fetch every row from the table and return as a list of dicts."""
    pool = get_pool()
    if pool is None:
        raise RuntimeError(
            "Database pool is not initialised. "
            "Make sure POSTGRES_DSN is set in backend/.env."
        )

    where = "WHERE is_active = 1" if active_only else ""

    query = f"""
        SELECT
            id,
            conversation_id,
            message_id,
            subject,
            raw_content,
            attachment_data,
            received_at,
            from_address,
            is_active,
            created_at,
            updated_at
        FROM {table}
        {where}
        ORDER BY received_at ASC NULLS LAST, id ASC
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query)

    result = []
    for row in rows:
        result.append({k: _serialize(v) for k, v in dict(row).items()})

    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all bu_emails rows (with attachment_data) to JSON."
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Table name (default: CHATBOT_PG_TABLE from .env)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON file path (default: export_bu_emails.json in scripts/)",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        default=False,
        help="Only export rows with is_active = 1",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent level (default: 2; use 0 for compact single-line)",
    )
    args = parser.parse_args()

    settings = get_settings()

    # Resolve table name
    raw_table = args.table or settings.chatbot_pg_table
    try:
        table = validate_table_name(raw_table)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Resolve output path
    out_path = Path(args.out) if args.out else Path(__file__).parent / "export_bu_emails.json"

    print(f"  POSTGRES_DSN  : {'(set)' if settings.postgres_dsn else '(NOT SET — check .env)'}")
    print(f"  Table         : {table}")
    print(f"  Filter        : {'is_active = 1 only' if args.active_only else 'ALL rows'}")
    print(f"  Output        : {out_path}")
    print()

    if not settings.postgres_dsn:
        print("[ERROR] POSTGRES_DSN is empty in backend/.env — cannot connect.")
        sys.exit(1)

    # Connect
    await init_db()
    try:
        print("Fetching rows ...")
        rows = await export_rows(table, args.active_only)
    finally:
        await close_db()

    if not rows:
        print(f"[WARN] No rows found in '{table}'" +
              (" (with is_active = 1)" if args.active_only else "") + ".")
        sys.exit(0)

    # Build output document
    output = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "table": table,
        "filter": "active_only" if args.active_only else "all",
        "row_count": len(rows),
        "rows": rows,
    }

    indent = args.indent if args.indent > 0 else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=indent)

    print(f"[OK] Exported {len(rows):,} row(s) → {out_path}")

    # Quick summary
    has_attachments = sum(1 for r in rows if (r.get("attachment_data") or "").strip())
    print(f"     Rows with attachment_data : {has_attachments:,}")
    print(f"     Rows without              : {len(rows) - has_attachments:,}")
    print(f"     File size                 : {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    asyncio.run(main())
