"""
scripts/check_bu_emails_conversations.py
=========================================
Quick diagnostic: how many distinct conversations exist in bu_emails
and how many messages belong to each.

Run from the backend/ directory:

    python scripts/check_bu_emails_conversations.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncpg
from core.config import get_settings


async def main() -> None:
    settings = get_settings()
    dsn = (settings.postgres_dsn or "").strip()
    table = settings.chatbot_pg_table

    if not dsn:
        print("POSTGRES_DSN is empty in .env — cannot connect.")
        return

    print(f"Table: {table}")
    print(f"Connecting...")

    conn = await asyncpg.connect(dsn=dsn, timeout=20)
    try:
        # 1. Distinct conversation count
        count = await conn.fetchval(
            f"SELECT COUNT(DISTINCT conversation_id) FROM {table} WHERE is_active = 1"
        )
        print(f"\nDistinct conversation_id (is_active=1): {count}")

        # 2. Messages per conversation
        rows = await conn.fetch(f"""
            SELECT conversation_id, COUNT(*) AS msg_count,
                   MIN(subject) AS sample_subject
            FROM {table}
            WHERE is_active = 1
            GROUP BY conversation_id
            ORDER BY msg_count DESC
        """)

        print(f"\n{'Conv ID (first 40 chars)':<45} {'Msgs':>5}   Sample Subject")
        print("-" * 120)
        for r in rows:
            cid = (r["conversation_id"] or "")[:40]
            print(f"{cid:<45} {r['msg_count']:>5}   {(r['sample_subject'] or '')[:60]}")

        # 3. Check for rows with empty conversation_id or message_id
        empty_conv = await conn.fetchval(
            f"SELECT COUNT(*) FROM {table} WHERE is_active = 1 "
            f"AND (conversation_id IS NULL OR BTRIM(conversation_id) = '')"
        )
        empty_msg = await conn.fetchval(
            f"SELECT COUNT(*) FROM {table} WHERE is_active = 1 "
            f"AND (message_id IS NULL OR BTRIM(message_id) = '')"
        )
        print(f"\nRows with empty conversation_id: {empty_conv}")
        print(f"Rows with empty message_id:      {empty_msg}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
