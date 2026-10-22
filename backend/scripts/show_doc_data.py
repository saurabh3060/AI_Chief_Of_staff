"""
show_doc_data.py
================
Dump all structured data for a specific document from PostgreSQL.

Shows: document metadata, pages, blocks, series + facts, issues,
       calculated_metrics.

Usage (from backend/):
    python scripts/show_doc_data.py
    python scripts/show_doc_data.py --doc "Credit Cards Updated Dec-25"
    python scripts/show_doc_data.py --all-series
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import get_settings
from core.logging_config import setup_logging


async def main():
    parser = argparse.ArgumentParser(description="Show document data from PostgreSQL")
    parser.add_argument("--doc", default="Credit Cards Updated Dec-25",
                        help="Substring to match in file_name (case-insensitive)")
    parser.add_argument("--all-series", action="store_true",
                        help="Print every series+fact (can be long)")
    parser.add_argument("--page", type=int, default=None,
                        help="Filter to a specific page number")
    args = parser.parse_args()

    setup_logging()
    get_settings()

    from core.db import init_db, close_db, get_pool
    await init_db()
    pool = get_pool()

    if pool is None:
        print("ERROR: PostgreSQL pool not available. Check POSTGRES_DSN in .env")
        return

    async with pool.acquire() as conn:
        # ── 1. Find the document ──
        docs = await conn.fetch(
            "SELECT * FROM documents WHERE LOWER(file_name) LIKE $1 ORDER BY ingested_at DESC",
            f"%{args.doc.lower()}%",
        )
        if not docs:
            print(f"No document found matching '{args.doc}'")
            all_docs = await conn.fetch("SELECT doc_id, file_name, status FROM documents ORDER BY file_name")
            print("\nAvailable documents:")
            for d in all_docs:
                print(f"  {d['status']:8}  {d['file_name']}")
            await close_db()
            return

        doc = docs[0]
        doc_id = doc["doc_id"]

        print("=" * 80)
        print(f"  DOCUMENT: {doc['file_name']}")
        print(f"  doc_id:   {doc_id}")
        print(f"  title:    {doc['title']}")
        print(f"  type:     {doc['doc_type']}")
        print(f"  as_of:    {doc['as_of_label']}")
        print(f"  pages:    {doc['pages']}")
        print(f"  status:   {doc['status']}")
        print("=" * 80)

        # ── 2. Pages summary ──
        pages = await conn.fetch(
            "SELECT page, heading FROM pages WHERE doc_id=$1 ORDER BY page", doc_id
        )
        print(f"\n📄 PAGES ({len(pages)}):")
        for p in pages:
            print(f"  p.{p['page']:2d}  {p['heading'][:80]}")

        # ── 3. Blocks summary ──
        page_filter = "AND page=$2" if args.page else ""
        page_args = [doc_id, args.page] if args.page else [doc_id]

        blocks = await conn.fetch(
            f"SELECT block_id, page, kind, title, section, source, confidence, "
            f"LENGTH(text) as text_len FROM blocks WHERE doc_id=$1 {page_filter} "
            f"ORDER BY page, block_id",
            *page_args,
        )
        print(f"\n📦 BLOCKS ({len(blocks)}):")
        for b in blocks:
            title = b['title'][:50] if b['title'] else b['section'][:50]
            print(f"  #{b['block_id']:4d}  p.{b['page']:2d}  {b['kind']:10s}  "
                  f"{b['source'] or '-':8s}  conf={b['confidence']:.2f}  "
                  f"chars={b['text_len']:5d}  {title}")

        # ── 4. Series + Facts ──
        series = await conn.fetch(
            f"SELECT series_id, page, label, name, metric_id, kind, unit, source, "
            f"confidence, path, scale FROM series WHERE doc_id=$1 {page_filter} "
            f"ORDER BY page, series_id",
            *page_args,
        )
        print(f"\n📊 SERIES ({len(series)}):")

        display_limit = None if args.all_series else 30
        for i, s in enumerate(series):
            if display_limit and i >= display_limit:
                print(f"  ... and {len(series) - display_limit} more (use --all-series to see all)")
                break

            metric = f"[{s['metric_id']}]" if s['metric_id'] else ""
            print(f"\n  Series #{s['series_id']}  p.{s['page']}  {metric}")
            print(f"    label: {s['label'][:100]}")
            print(f"    name:  {s['name'][:60]}  kind={s['kind'] or '-'}  unit={s['unit'] or '-'}  "
                  f"source={s['source'] or '-'}  conf={s['confidence']:.2f}")

            facts = await conn.fetch(
                "SELECT period_label, period_key, period_kind, value, as_printed, "
                "validated, confidence, note FROM facts "
                "WHERE series_id=$1 ORDER BY period_start",
                s['series_id'],
            )
            if facts:
                for f in facts:
                    validated = "✓" if f['validated'] else " "
                    printed = f['as_printed'] or ""
                    val = f"{f['value']:,.2f}" if f['value'] is not None else "NULL"
                    print(f"    {validated} {f['period_label']:12s}  {val:>14s}  "
                          f"printed='{printed}'  {f['period_kind']}  conf={f['confidence']:.2f}"
                          f"{'  ' + f['note'] if f['note'] else ''}")
            else:
                print(f"    (no facts)")

        # ── 5. Issues ──
        issues = await conn.fetch(
            f"SELECT page, kind, ref, message, suggestion FROM issues "
            f"WHERE doc_id=$1 {page_filter} ORDER BY page, issue_id",
            *page_args,
        )
        print(f"\n⚠️  ISSUES ({len(issues)}):")
        for iss in issues[:20]:
            print(f"  p.{iss['page']:2d}  {iss['kind']:25s}  {iss['ref'][:30]:30s}  {iss['message'][:60]}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")

        # ── 6. Calculated metrics ──
        calc = await conn.fetch(
            f"SELECT metric_id, metric_name, metric_family, entity, period_label, "
            f"formatted_value, unit, formula, calculation_status "
            f"FROM calculated_metrics WHERE doc_id=$1 {page_filter} "
            f"ORDER BY metric_name, period_label",
            *page_args,
        )
        print(f"\n🔢 CALCULATED METRICS ({len(calc)}):")
        for c in calc[:30]:
            print(f"  {c['metric_name'][:30]:30s}  {c['entity'][:20]:20s}  "
                  f"{c['period_label']:12s}  {c['formatted_value']:>14s}  "
                  f"{c['unit'] or '-':5s}  [{c['calculation_status']}]")
        if len(calc) > 30:
            print(f"  ... and {len(calc) - 30} more")

        # ── 7. Metric rules ──
        rules = await conn.fetch(
            "SELECT rule_id, metric_name, operation, status, enabled "
            "FROM metric_rules ORDER BY rule_id LIMIT 20"
        )
        print(f"\n📐 METRIC RULES (first 20 of total):")
        for r in rules:
            enabled = "ON" if r['enabled'] else "OFF"
            print(f"  {r['rule_id']:25s}  {r['metric_name'][:35]:35s}  "
                  f"{r['operation']:15s}  {r['status']:12s}  {enabled}")
        total_rules = await conn.fetchval("SELECT COUNT(*) FROM metric_rules")
        print(f"  Total rules: {total_rules}")

    await close_db()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
