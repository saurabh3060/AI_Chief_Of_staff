"""
scripts/dump_cc_docs.py
========================
Dump all series, facts, and block summaries for Credit Cards
documents. Output goes to both console and cc_docs_dump.txt.

Usage:
    cd backend
    python -m scripts.dump_cc_docs
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# File name fragments to match
TARGETS = ["cc12", "credit cards"]

OUT_FILE = Path(__file__).resolve().parent.parent / "cc_docs_dump.txt"


async def main() -> None:
    from core.db import init_db, close_db, get_pool

    await init_db()
    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            # Find matching documents
            docs = await conn.fetch(
                "SELECT doc_id, file_name, title, doc_type, status, pages, as_of_label "
                "FROM documents ORDER BY file_name"
            )
            matched = [d for d in docs
                       if any(t in d['file_name'].lower() for t in TARGETS)]

            if not matched:
                out("No matching documents found. Available docs:")
                for d in docs:
                    out(f"  - {d['file_name']}")
                return

            for doc in matched:
                doc_id = doc['doc_id']
                out(f"\n{'█' * 100}")
                out(f"  FILE: {doc['file_name']}")
                out(f"  doc_id={doc_id}  type={doc['doc_type']}  "
                    f"pages={doc['pages']}  as_of={doc['as_of_label'] or '-'}")
                out(f"  title: {doc['title']}")
                out(f"{'█' * 100}")

                # ── Blocks summary ──
                blocks = await conn.fetch(
                    "SELECT block_id, page, kind, section, title, "
                    "       LEFT(text, 150) AS preview "
                    "FROM blocks WHERE doc_id = $1 ORDER BY page, block_id",
                    doc_id,
                )
                out(f"\n  ── BLOCKS ({len(blocks)}) ──")
                for b in blocks:
                    sec = b['section'][:50] if b['section'] else ''
                    ttl = b['title'][:50] if b['title'] else ''
                    preview = (b['preview'] or '').replace('\n', ' ').strip()[:100]
                    out(f"    p.{b['page']} [{b['kind']:10s}] {sec}")
                    if ttl:
                        out(f"         title: {ttl}")
                    out(f"         {preview}...")

                # ── Series + Facts ──
                series = await conn.fetch(
                    "SELECT series_id, page, label, name, metric_id, kind, unit, entity "
                    "FROM series WHERE doc_id = $1 ORDER BY page, series_id",
                    doc_id,
                )
                out(f"\n  ── SERIES ({len(series)}) with FACTS ──")
                for s in series:
                    facts = await conn.fetch(
                        "SELECT period_label, value, as_printed "
                        "FROM facts WHERE series_id = $1 ORDER BY period_start",
                        s['series_id'],
                    )
                    fact_str = ", ".join(
                        f"{f['period_label']}={f['value']}"
                        for f in facts if f['value'] is not None
                    )
                    if not fact_str:
                        fact_str = "(no numeric facts)"
                    out(f"\n    p.{s['page']} [{s['kind'] or '-':8s}] {s['label'][:80]}")
                    out(f"         unit={s['unit'] or '-'}  metric_id={s['metric_id'] or '-'}  "
                        f"entity={s['entity'] or '-'}")
                    out(f"         facts: {fact_str}")

                # ── Issues ──
                issues = await conn.fetch(
                    "SELECT page, kind, ref, message FROM issues "
                    "WHERE doc_id = $1 ORDER BY page LIMIT 10",
                    doc_id,
                )
                if issues:
                    out(f"\n  ── ISSUES (first 10 of ...) ──")
                    for iss in issues:
                        out(f"    p.{iss['page']} [{iss['kind']}] {iss['ref']}: "
                            f"{iss['message'][:80]}")

                # ── Calculated metrics ──
                calcs = await conn.fetch(
                    "SELECT metric_name, metric_family, period_label, "
                    "       formatted_value, unit, answer "
                    "FROM calculated_metrics WHERE doc_id = $1 "
                    "ORDER BY metric_family, metric_name, period_label LIMIT 30",
                    doc_id,
                )
                if calcs:
                    out(f"\n  ── CALCULATED METRICS (first 30) ──")
                    for c in calcs:
                        out(f"    {c['metric_family']} › {c['metric_name']} "
                            f"[{c['period_label']}] = {c['formatted_value']} {c['unit'] or ''}")
                        if c['answer']:
                            out(f"         answer: {c['answer'][:80]}")

                out()

    finally:
        # Write to file
        OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  ✓ Output saved to {OUT_FILE}")
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
