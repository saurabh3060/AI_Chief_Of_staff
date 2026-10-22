"""
scripts/dump_doc_tables.py
============================
Dump a summary of all documents module PostgreSQL tables to see what's
actually stored. Prints documents, sample series/facts, and row counts.

Usage:
    cd backend
    python -m scripts.dump_doc_tables
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


async def main() -> None:
    from core.db import init_db, close_db, get_pool

    await init_db()
    try:
        pool = get_pool()
        async with pool.acquire() as conn:

            # ── 1. documents table ──
            docs = await conn.fetch(
                "SELECT doc_id, file_name, title, doc_type, entity, "
                "       as_of_key, as_of_label, pages, status "
                "FROM documents ORDER BY file_name"
            )
            print(f"\n{'═' * 100}")
            print(f"  TABLE: documents  ({len(docs)} rows)")
            print(f"{'═' * 100}")
            for d in docs:
                print(f"  {d['file_name']}")
                print(f"    doc_id={d['doc_id']}  type={d['doc_type']}  status={d['status']}  "
                      f"pages={d['pages']}  as_of={d['as_of_label'] or '-'}")
                print(f"    title: {(d['title'] or '')[:80]}")
                print()

            if not docs:
                print("  (no documents ingested yet)\n")
                # Still check other tables in case of orphans

            # ── 2. Row counts for all tables ──
            tables = [
                "documents", "pages", "blocks", "series", "facts",
                "issues", "corrections", "metric_rules", "calculated_metrics",
            ]
            print(f"{'═' * 100}")
            print(f"  ROW COUNTS")
            print(f"{'═' * 100}")
            for t in tables:
                try:
                    n = await conn.fetchval(f"SELECT count(*) FROM {t}")
                    print(f"  {t:30s} {n:>8,} rows")
                except Exception as e:
                    print(f"  {t:30s} (error: {e})")
            print()

            # ── 3. Sample series (top 20) ──
            series_rows = await conn.fetch(
                "SELECT s.series_id, s.doc_id, s.label, s.name, s.metric_id, "
                "       s.kind, s.unit, s.entity, d.file_name "
                "FROM series s LEFT JOIN documents d ON d.doc_id = s.doc_id "
                "ORDER BY s.series_id LIMIT 20"
            )
            print(f"{'═' * 100}")
            print(f"  TABLE: series  (first 20)")
            print(f"{'═' * 100}")
            if series_rows:
                for s in series_rows:
                    print(f"  [{s['series_id']}] {s['label'][:70]}")
                    print(f"       metric_id={s['metric_id'] or '-'}  unit={s['unit'] or '-'}  "
                          f"kind={s['kind'] or '-'}  file={s['file_name'] or '-'}")
            else:
                print("  (no series)")
            print()

            # ── 4. Sample facts (top 20) ──
            fact_rows = await conn.fetch(
                "SELECT f.fact_id, f.series_id, f.period_label, f.value, "
                "       f.as_printed, s.label "
                "FROM facts f LEFT JOIN series s ON s.series_id = f.series_id "
                "ORDER BY f.fact_id LIMIT 20"
            )
            print(f"{'═' * 100}")
            print(f"  TABLE: facts  (first 20)")
            print(f"{'═' * 100}")
            if fact_rows:
                for f in fact_rows:
                    val = f['value'] if f['value'] is not None else 'NULL'
                    print(f"  [{f['fact_id']}] {(f['label'] or '')[:50]} | "
                          f"{f['period_label'] or '-'} = {val}  (printed: {f['as_printed'] or '-'})")
            else:
                print("  (no facts)")
            print()

            # ── 5. Sample blocks (top 10) ──
            block_rows = await conn.fetch(
                "SELECT b.block_id, b.doc_id, b.page, b.kind, b.section, "
                "       b.title, LEFT(b.text, 100) AS text_preview, d.file_name "
                "FROM blocks b LEFT JOIN documents d ON d.doc_id = b.doc_id "
                "ORDER BY b.block_id LIMIT 10"
            )
            print(f"{'═' * 100}")
            print(f"  TABLE: blocks  (first 10)")
            print(f"{'═' * 100}")
            if block_rows:
                for b in block_rows:
                    print(f"  [{b['block_id']}] page {b['page']} | {b['kind']} | "
                          f"file={b['file_name'] or '-'}")
                    if b['section']:
                        print(f"       section: {b['section'][:60]}")
                    if b['title']:
                        print(f"       title: {b['title'][:60]}")
                    print(f"       text: {(b['text_preview'] or '').strip()[:80]}...")
            else:
                print("  (no blocks)")
            print()

    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
