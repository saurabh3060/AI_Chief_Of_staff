"""
scripts/view_pages.py
========================
Dump the full content of the ``pages`` table in PostgreSQL to a text file.

Usage:
    cd backend
    python -m scripts.view_pages
"""

import asyncio
from pathlib import Path

from core.db import init_db, get_pool, close_db

OUTPUT_FILE = Path(__file__).resolve().parent.parent / "pages_dump.txt"


async def main():
    await init_db()
    pool = get_pool()
    if pool is None:
        print("[ERROR] PostgreSQL pool not available. Check POSTGRES_DSN in .env")
        return

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT doc_id, page, heading, text, image "
            "FROM pages ORDER BY doc_id, page"
        )

    if not rows:
        print("[INFO] The pages table is empty — no documents have been ingested yet.")
        await close_db()
        return

    lines = []
    current_doc = None
    for r in rows:
        if r["doc_id"] != current_doc:
            current_doc = r["doc_id"]
            lines.append("")
            lines.append("=" * 80)
            lines.append(f"  DOCUMENT: {current_doc}")
            lines.append("=" * 80)

        lines.append(f"\n--- Page {r['page']} ---")
        if r["heading"]:
            lines.append(f"  Heading: {r['heading']}")
        if r["image"]:
            lines.append(f"  Image:   {r['image']}")
        lines.append("  Text:")
        text = r["text"] or "(empty)"
        for line in text.splitlines():
            lines.append(f"    {line}")

    lines.append(f"\n\nTotal: {len(rows)} page(s)")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Written {len(rows)} page(s) to {OUTPUT_FILE}")
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
