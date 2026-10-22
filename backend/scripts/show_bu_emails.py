"""
show_bu_emails.py
=================
Quick diagnostic: prints every row in bu_emails with full content.

Run from the backend/ directory:
    python scripts/show_bu_emails.py

Options:
    --subject TEXT   Only show rows whose subject contains TEXT (case-insensitive).
    --full           Print the full body/attachment text, not a 500-char preview.

Examples:
    python scripts/show_bu_emails.py --subject "June" --full
    python scripts/show_bu_emails.py --subject "Lumi" --full
"""

import argparse
import asyncio
import os
import sys

# Ensure the backend package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import get_settings
from core.logging_config import setup_logging


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default=None, help="Filter: subject contains this text (case-insensitive).")
    parser.add_argument("--full", action="store_true", help="Print full body/attachment text, no truncation.")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    from core.db import init_db, close_db, get_pool

    await init_db()
    pool = get_pool()

    if pool is None:
        print("ERROR: PostgreSQL pool not available. Check POSTGRES_DSN in .env")
        return

    table = settings.chatbot_pg_table

    async with pool.acquire() as conn:
        total = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
        active = await conn.fetchval(f"SELECT COUNT(*) FROM {table} WHERE is_active = 1")

        print("=" * 70)
        print(f"  Table: {table}")
        print(f"  Total rows : {total}")
        print(f"  Active rows: {active}")
        if args.subject:
            print(f"  Filter     : subject contains '{args.subject}'")
        print("=" * 70)

        if args.subject:
            rows = await conn.fetch(
                f"SELECT message_id, conversation_id, subject, raw_content, "
                f"attachment_data, received_at, from_address, is_active "
                f"FROM {table} "
                f"WHERE subject ILIKE $1 "
                f"ORDER BY conversation_id ASC, received_at ASC NULLS LAST",
                f"%{args.subject}%",
            )
        else:
            rows = await conn.fetch(
                f"SELECT message_id, conversation_id, subject, raw_content, "
                f"attachment_data, received_at, from_address, is_active "
                f"FROM {table} "
                f"ORDER BY conversation_id ASC, received_at ASC NULLS LAST"
            )

    current_conv = None
    for i, r in enumerate(rows, 1):
        conv_id = r["conversation_id"] or "(none)"
        if conv_id != current_conv:
            current_conv = conv_id
            print(f"\n{'─' * 70}")
            print(f"  CONVERSATION: {conv_id}")
            print(f"{'─' * 70}")

        status = "ACTIVE" if r["is_active"] == 1 else "INACTIVE"
        body = (r["raw_content"] or "").strip()
        attachment = (r["attachment_data"] or "").strip()

        print(f"\n  [{i}] {status}")
        print(f"  Message ID : {r['message_id']}")
        print(f"  Subject    : {r['subject']}")
        print(f"  From       : {r['from_address']}")
        print(f"  Received   : {r['received_at']}")
        print(f"  Body length: {len(body)} chars")
        if body:
            if args.full:
                print(f"  Body       :\n{body}")
            else:
                preview = body[:500]
                if len(body) > 500:
                    preview += f"\n  ... ({len(body) - 500} more chars)"
                print(f"  Body       : {preview}")
        if attachment:
            print(f"  Attachment : {len(attachment)} chars")
            if args.full:
                print(f"               \n{attachment}")
            else:
                print(f"               {attachment[:200]}")

    print(f"\n{'=' * 70}")
    print(f"  Done. {len(rows)} row(s) printed.")
    print(f"{'=' * 70}")

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
