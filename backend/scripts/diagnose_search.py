"""
scripts/diagnose_search.py
=============================
Diagnose why the document chatbot can't find data.
Checks Postgres tables, FAISS index, and FTS for a given search term.

Usage:
    cd backend
    python -m scripts.diagnose_search
"""

import asyncio
import json
from pathlib import Path

from core.db import init_db, get_pool, close_db


async def main():
    await init_db()
    pool = get_pool()
    if pool is None:
        print("[ERROR] PostgreSQL pool not available.")
        return

    async with pool.acquire() as conn:
        # 1. Check documents table
        print("=" * 60)
        print("  1. DOCUMENTS TABLE")
        print("=" * 60)
        docs = await conn.fetch("SELECT doc_id, file_name, title, status, pages FROM documents")
        for d in docs:
            print(f"  doc_id={d['doc_id']}  status={d['status']}  pages={d['pages']}  file={d['file_name']}")
        if not docs:
            print("  (empty — no documents ingested)")

        # 2. Check blocks table
        print("\n" + "=" * 60)
        print("  2. BLOCKS TABLE")
        print("=" * 60)
        block_count = await conn.fetchval("SELECT COUNT(*) FROM blocks")
        print(f"  Total blocks: {block_count}")
        block_ids = await conn.fetch(
            "SELECT MIN(block_id) as min_id, MAX(block_id) as max_id, doc_id, COUNT(*) as cnt "
            "FROM blocks GROUP BY doc_id"
        )
        for b in block_ids:
            print(f"  doc_id={b['doc_id']}  blocks={b['cnt']}  id_range=[{b['min_id']}..{b['max_id']}]")

        # 3. Check series table
        print("\n" + "=" * 60)
        print("  3. SERIES TABLE")
        print("=" * 60)
        series_count = await conn.fetchval("SELECT COUNT(*) FROM series")
        print(f"  Total series: {series_count}")
        series_ids = await conn.fetch(
            "SELECT MIN(series_id) as min_id, MAX(series_id) as max_id, doc_id, COUNT(*) as cnt "
            "FROM series GROUP BY doc_id"
        )
        for s in series_ids:
            print(f"  doc_id={s['doc_id']}  series={s['cnt']}  id_range=[{s['min_id']}..{s['max_id']}]")

        # 4. Check if NII exists in series labels
        print("\n" + "=" * 60)
        print("  4. SERIES CONTAINING 'NII'")
        print("=" * 60)
        nii = await conn.fetch("SELECT series_id, doc_id, page, label, unit FROM series WHERE label ILIKE '%NII%'")
        for s in nii:
            print(f"  series_id={s['series_id']}  page={s['page']}  label={s['label']}  unit={s['unit']}")
        if not nii:
            print("  (none found — NII not in any series label)")

        # 5. Check facts for NII series
        print("\n" + "=" * 60)
        print("  5. FACTS FOR NII SERIES")
        print("=" * 60)
        if nii:
            for s in nii:
                facts = await conn.fetch(
                    "SELECT period_label, value, as_printed FROM facts WHERE series_id = $1 ORDER BY period_start",
                    s['series_id']
                )
                print(f"  series_id={s['series_id']} label='{s['label']}':")
                for f in facts:
                    print(f"    {f['period_label']}: {f['value']}  (printed: {f['as_printed']})")
        else:
            # Show some sample series labels to understand what's there
            samples = await conn.fetch("SELECT DISTINCT label FROM series LIMIT 30")
            print("  Sample series labels in DB:")
            for s in samples:
                print(f"    - {s['label']}")

        # 6. FTS test
        print("\n" + "=" * 60)
        print("  6. FULL-TEXT SEARCH TEST")
        print("=" * 60)
        for term in ["NII", "revenue", "credit", "PBT", "spend"]:
            block_hits = await conn.fetchval(
                "SELECT COUNT(*) FROM blocks WHERE tsv @@ to_tsquery('english', $1)", term
            )
            series_hits = await conn.fetchval(
                "SELECT COUNT(*) FROM series WHERE tsv @@ to_tsquery('english', $1)", term
            )
            print(f"  '{term}': blocks={block_hits}  series={series_hits}")

        # 7. Check FAISS index vs Postgres IDs
        print("\n" + "=" * 60)
        print("  7. FAISS INDEX vs POSTGRES")
        print("=" * 60)
        faiss_dir = Path("data/faiss_indexes")
        for name, table, id_col in [
            ("doc_blocks_meta.json", "blocks", "block_id"),
            ("doc_series_meta.json", "series", "series_id"),
        ]:
            meta_path = faiss_dir / name
            if not meta_path.exists():
                print(f"  {name}: FILE NOT FOUND")
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            faiss_ids = {m["row_id"] for m in meta}
            faiss_docs = {m["doc_id"] for m in meta}
            pg_ids_rows = await conn.fetch(f"SELECT {id_col} FROM {table}")
            pg_ids = {r[id_col] for r in pg_ids_rows}

            in_both = faiss_ids & pg_ids
            in_faiss_only = faiss_ids - pg_ids
            in_pg_only = pg_ids - faiss_ids

            print(f"  {name}:")
            print(f"    FAISS vectors: {len(meta)}  (doc_ids: {faiss_docs})")
            print(f"    Postgres rows: {len(pg_ids)}")
            print(f"    Match (both):  {len(in_both)}")
            print(f"    FAISS only (stale): {len(in_faiss_only)}")
            print(f"    Postgres only (missing from FAISS): {len(in_pg_only)}")
            if in_pg_only and len(in_pg_only) <= 20:
                print(f"    Missing IDs: {sorted(in_pg_only)}")

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
