"""Check whether local PostgreSQL doc_id and fact_id values match the other developer's export.

Usage:
  python -m documents.check_lineage C:\\Users\\C59248\\Downloads\\metric_rules_and_calculated_metrics_postgres_20261014
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

from . import store


async def main_async(package_dir: Path) -> int:
    from core.db import init_db, close_db

    doc_manifest = package_dir / "document_manifest.csv"
    fact_manifest = package_dir / "source_fact_manifest.csv"
    calc_csv = package_dir / "calculated_metrics.csv"

    if not doc_manifest.is_file():
        print(f"ERROR: {doc_manifest} not found")
        return 1

    # --- Load manifest CSVs ---
    with open(doc_manifest, encoding="utf-8-sig") as f:
        expected_docs = list(csv.DictReader(f))
    print(f"Other developer's manifest: {len(expected_docs)} documents")

    expected_facts: list[dict] = []
    if fact_manifest.is_file():
        with open(fact_manifest, encoding="utf-8-sig") as f:
            expected_facts = list(csv.DictReader(f))
        print(f"Other developer's manifest: {len(expected_facts)} source fact rows")

    calc_doc_ids: set[str] = set()
    calc_fact_ids: set[tuple[str, int]] = set()  # (doc_id, fact_id)
    if calc_csv.is_file():
        with open(calc_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                calc_doc_ids.add(row["doc_id"])
                for fid in json.loads(row.get("source_fact_ids", "[]") or "[]"):
                    calc_fact_ids.add((row["doc_id"], int(fid)))
        print(f"Calculated_metrics.csv references {len(calc_doc_ids)} doc_ids, "
              f"{len(calc_fact_ids)} (doc_id, fact_id) pairs")

    await init_db()
    try:
        async with store.acquire() as conn:
            # --- Check documents ---
            print("\n=== DOCUMENT ID CHECK ===")
            your_docs = {row["doc_id"]: row["file_name"]
                         for row in await conn.fetch("SELECT doc_id, file_name FROM documents")}
            print(f"Your database: {len(your_docs)} documents")

            doc_match = 0
            doc_missing = 0
            doc_mismatch = 0
            for edoc in expected_docs:
                eid = edoc["doc_id"]
                ename = edoc["file_name"]
                if eid not in your_docs:
                    doc_missing += 1
                    print(f"  MISSING  doc_id={eid}  file={ename}")
                elif your_docs[eid] != ename:
                    doc_mismatch += 1
                    print(f"  MISMATCH doc_id={eid}  theirs={ename}  yours={your_docs[eid]}")
                else:
                    doc_match += 1

            print(f"\nResult: {doc_match} match, {doc_missing} missing, {doc_mismatch} name mismatch")

            # --- Check fact IDs (sample from calculated_metrics references) ---
            if calc_fact_ids:
                print("\n=== FACT ID CHECK (sampled from calculated_metrics.csv) ===")
                # Check a sample of fact IDs per doc
                sample_size = min(500, len(calc_fact_ids))
                sampled = sorted(calc_fact_ids)[:sample_size]

                fact_ok = 0
                fact_missing = 0
                fact_wrong_doc = 0

                for doc_id, fact_id in sampled:
                    row = await conn.fetchrow(
                        "SELECT doc_id FROM facts WHERE fact_id = $1", fact_id
                    )
                    if row is None:
                        fact_missing += 1
                        if fact_missing <= 5:
                            print(f"  MISSING  fact_id={fact_id} (expected doc_id={doc_id})")
                    elif row["doc_id"] != doc_id:
                        fact_wrong_doc += 1
                        if fact_wrong_doc <= 5:
                            print(f"  WRONG DOC  fact_id={fact_id} theirs={doc_id} yours={row['doc_id']}")
                    else:
                        fact_ok += 1

                print(f"\nResult ({sample_size} sampled): {fact_ok} match, "
                      f"{fact_missing} missing, {fact_wrong_doc} wrong document")
            elif expected_facts:
                print("\n=== FACT ID CHECK (from source_fact_manifest.csv) ===")
                sample = expected_facts[:500]
                fact_ok = 0
                fact_missing = 0
                for ef in sample:
                    fid = int(ef.get("fact_id", 0))
                    edoc = ef.get("doc_id", "")
                    row = await conn.fetchrow("SELECT doc_id FROM facts WHERE fact_id = $1", fid)
                    if row is None:
                        fact_missing += 1
                    elif row["doc_id"] == edoc:
                        fact_ok += 1
                    else:
                        fact_missing += 1
                print(f"\nResult ({len(sample)} sampled): {fact_ok} match, {fact_missing} missing/mismatch")

            # --- Verdict ---
            print("\n=== VERDICT ===")
            if doc_missing == 0 and doc_mismatch == 0 and (
                    not calc_fact_ids or (fact_missing == 0 and fact_wrong_doc == 0)):
                print("ALL MATCH — safe to run: import_rules_and_results.sql (Option A)")
                return 0
            elif doc_missing == 0 and doc_mismatch == 0:
                print("Documents match but some fact IDs differ.")
                print("Use: import_rules.sql (Option B) then re-materialize.")
                return 1
            else:
                print("Document IDs do NOT match.")
                print("Use: import_rules.sql (Option B) then re-materialize.")
                return 1
    finally:
        await close_db()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path,
                        help="Path to extracted metric_rules_and_calculated_metrics folder")
    args = parser.parse_args()
    if not args.package_dir.is_dir():
        parser.error(f"Not a directory: {args.package_dir}")
    return asyncio.run(main_async(args.package_dir))


if __name__ == "__main__":
    raise SystemExit(main())
