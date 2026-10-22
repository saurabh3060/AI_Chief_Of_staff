"""Dump every document's facts and derived (calculated) metrics to a text file.

Pulls from PostgreSQL: documents, series, facts, calculated_metrics.
Writes a human-readable report to documents/facts_dump.txt (or --out).

Usage:
  python -m documents.export_facts_dump
  python -m documents.export_facts_dump --out documents/facts_dump.txt
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from . import store


async def dump(out_path: Path) -> dict:
    lines: list[str] = []
    counts = {"documents": 0, "series": 0, "facts": 0, "calculated_metrics": 0}

    async with store.acquire() as conn:
        docs = await conn.fetch(
            "SELECT doc_id, file_name, title, doc_type, as_of_label, pages, status "
            "FROM documents ORDER BY file_name"
        )
        counts["documents"] = len(docs)

        for doc in docs:
            lines.append("=" * 100)
            lines.append(f"DOCUMENT: {doc['file_name']}")
            lines.append(f"  title     : {doc['title']}")
            lines.append(f"  doc_type  : {doc['doc_type']}")
            lines.append(f"  as_of     : {doc['as_of_label']}")
            lines.append(f"  pages     : {doc['pages']}")
            lines.append(f"  status    : {doc['status']}")
            lines.append(f"  doc_id    : {doc['doc_id']}")
            lines.append("")

            # --- raw extracted facts, grouped by series -----------------
            series_rows = await conn.fetch(
                "SELECT series_id, page, label, name, metric_id, kind, unit, entity "
                "FROM series WHERE doc_id = $1 ORDER BY page, label",
                doc["doc_id"],
            )
            counts["series"] += len(series_rows)
            if series_rows:
                lines.append("  --- FACTS (raw, extracted from source) ---")
            for s in series_rows:
                facts = await conn.fetch(
                    "SELECT period_label, value, as_printed, validated, confidence "
                    "FROM facts WHERE series_id = $1 ORDER BY period_start",
                    s["series_id"],
                )
                counts["facts"] += len(facts)
                if not facts:
                    continue
                label = s["label"] or s["name"] or f"series {s['series_id']}"
                unit = f" ({s['unit']})" if s["unit"] else ""
                lines.append(f"  [p.{s['page']}] {label}{unit}")
                for f in facts:
                    val = f["as_printed"] or (str(f["value"]) if f["value"] is not None else "")
                    lines.append(
                        f"      {f['period_label']}: {val}"
                        f"  [source: {doc['file_name']} p.{s['page']}, series_id={s['series_id']}]"
                    )
            lines.append("")

            # --- derived / calculated metrics ---------------------------
            calc_rows = await conn.fetch(
                "SELECT source_page, metric_name, metric_family, period_label, "
                "comparison_period_label, formatted_value, unit, formula, answer, "
                "calculation_status "
                "FROM calculated_metrics WHERE doc_id = $1 ORDER BY source_page, metric_name",
                doc["doc_id"],
            )
            counts["calculated_metrics"] += len(calc_rows)
            if calc_rows:
                lines.append("  --- DERIVED / CALCULATED METRICS ---")
            for c in calc_rows:
                cmp_part = f" vs {c['comparison_period_label']}" if c["comparison_period_label"] else ""
                lines.append(
                    f"  [p.{c['source_page']}] {c['metric_name']} ({c['metric_family']}) "
                    f"@ {c['period_label']}{cmp_part} = {c['formatted_value']} {c['unit']} "
                    f"[{c['calculation_status']}]"
                )
                if c["formula"]:
                    lines.append(f"      formula : {c['formula']}")
                if c["answer"]:
                    lines.append(f"      answer  : {c['answer']}")
                lines.append(
                    f"      [source: {doc['file_name']} p.{c['source_page']}]"
                )
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="documents/facts_dump.txt")
    args = ap.parse_args()

    from core.db import init_db, close_db

    async def run():
        await init_db()
        try:
            counts = await dump(Path(args.out))
            print(
                f"Wrote {args.out}: {counts['documents']} documents, "
                f"{counts['series']} series, {counts['facts']} facts, "
                f"{counts['calculated_metrics']} calculated metrics."
            )
        finally:
            await close_db()

    asyncio.run(run())


if __name__ == "__main__":
    main()
