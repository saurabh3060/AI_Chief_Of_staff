"""Dump the full contents of metric_rules and calculated_metrics to a text file.

Usage:
  python -m documents.dump_metric_tables
  python -m documents.dump_metric_tables --out my_dump.txt
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from . import store


async def _dump_table(conn, table: str, out) -> None:
    rows = await conn.fetch(f"SELECT * FROM {table} ORDER BY 1")
    out.write(f"\n{'=' * 100}\n")
    out.write(f"TABLE: {table}  ({len(rows)} rows)\n")
    out.write(f"{'=' * 100}\n")
    if not rows:
        out.write("(no rows)\n")
        return
    columns = list(rows[0].keys())
    for i, row in enumerate(rows, 1):
        out.write(f"\n--- {table} row {i} ---\n")
        for col in columns:
            value = row[col]
            # Pretty-print JSON-looking text columns for readability.
            if isinstance(value, str) and value.strip().startswith(("{", "[")):
                try:
                    value = json.dumps(json.loads(value), indent=2)
                except (TypeError, ValueError):
                    pass
            out.write(f"{col}: {value}\n")


async def main_async(out_path: Path) -> None:
    from core.db import init_db, close_db

    await init_db()
    try:
        async with store.acquire() as conn:
            with out_path.open("w", encoding="utf-8") as out:
                await _dump_table(conn, "metric_rules", out)
                await _dump_table(conn, "calculated_metrics", out)
        print(f"Wrote table contents to {out_path.resolve()}")
    finally:
        await close_db()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("metric_tables_dump.txt"),
                        help="Output text file path (default: metric_tables_dump.txt)")
    args = parser.parse_args()
    asyncio.run(main_async(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
