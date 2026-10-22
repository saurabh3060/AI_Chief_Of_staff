"""
migrations/migrate_to_postgres.py
====================================
One-time migration script: creates the tracker tables in PostgreSQL
and copies existing data from the local JSON files.

Run once from the backend/ directory:

    python migrations/migrate_to_postgres.py

What it does:
  1. Reads POSTGRES_DSN from backend/.env
  2. Connects to the AWS PostgreSQL database
  3. Creates the two tables (IF NOT EXISTS — safe to run twice)
  4. Reads item_overrides.json and completed_tracker_items.json
  5. Inserts every row into PostgreSQL (ON CONFLICT — safe to run twice)
  6. Prints a summary of what was migrated

Prerequisites:
  - POSTGRES_DSN must be set in backend/.env
  - pip install asyncpg python-dotenv  (already in requirements.txt)
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the backend directory to sys.path so we can import core modules.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import asyncpg
from dotenv import dotenv_values


async def main() -> None:
    # Load DSN from .env
    env = dotenv_values(BACKEND_ROOT / ".env")
    dsn = env.get("POSTGRES_DSN", "").strip()

    if not dsn:
        print("ERROR: POSTGRES_DSN is empty in backend/.env")
        print("Set it to something like: postgresql://dbadmin:PASSWORD@host:6543/DBNAME")
        sys.exit(1)

    print(f"Connecting to PostgreSQL...")
    conn = await asyncpg.connect(dsn=dsn)
    print("Connected successfully!\n")

    # ── Step 1: Create tables ──
    print("Creating tables (if they don't exist)...")

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS tracker_item_overrides (
            id          VARCHAR(16) PRIMARY KEY,
            text        TEXT,
            owner       TEXT,
            business_unit TEXT,
            due_date    TEXT,
            updated_at  TIMESTAMPTZ DEFAULT now()
        );
    """)
    print("  tracker_item_overrides  OK")

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS tracker_completed_items (
            id          VARCHAR(16) PRIMARY KEY,
            text        TEXT NOT NULL,
            owner       TEXT NOT NULL,
            business_unit TEXT NOT NULL,
            due_date    TEXT,
            source_topic TEXT,
            completed_at TIMESTAMPTZ DEFAULT now()
        );
    """)
    print("  tracker_completed_items  OK\n")

    # ── Step 2: Migrate item_overrides.json ──
    data_dir = BACKEND_ROOT / env.get("BU_EMAIL_DATA_DIR", "data")
    overrides_path = data_dir / "item_overrides.json"
    overrides_count = 0

    if overrides_path.exists():
        print(f"Migrating {overrides_path.name}...")
        data = json.loads(overrides_path.read_text(encoding="utf-8"))

        for entry in data:
            item_id = entry.get("id")
            if not item_id:
                continue
            await conn.execute(
                """
                INSERT INTO tracker_item_overrides (id, text, owner, business_unit, due_date, updated_at)
                VALUES ($1, $2, $3, $4, $5, now())
                ON CONFLICT (id) DO NOTHING
                """,
                item_id,
                entry.get("text"),
                entry.get("owner"),
                entry.get("business_unit"),
                entry.get("due_date"),
            )
            overrides_count += 1
        print(f"  {overrides_count} item overrides migrated\n")
    else:
        print(f"  {overrides_path.name} not found — skipping\n")

    # ── Step 3: Migrate completed_tracker_items.json ──
    completed_path = data_dir / "completed_tracker_items.json"
    completed_count = 0

    if completed_path.exists():
        print(f"Migrating {completed_path.name}...")
        data = json.loads(completed_path.read_text(encoding="utf-8"))

        for entry in data:
            item_id = entry.get("id")
            if not item_id:
                continue
            await conn.execute(
                """
                INSERT INTO tracker_completed_items (id, text, owner, business_unit, due_date, source_topic, completed_at)
                VALUES ($1, $2, $3, $4, $5, $6, now())
                ON CONFLICT (id) DO NOTHING
                """,
                item_id,
                entry.get("text", ""),
                entry.get("owner", ""),
                entry.get("business_unit", ""),
                entry.get("due_date"),
                entry.get("source_topic", ""),
            )
            completed_count += 1
        print(f"  {completed_count} completed items migrated\n")
    else:
        print(f"  {completed_path.name} not found — skipping\n")

    await conn.close()

    # ── Summary ──
    print("=" * 50)
    print("MIGRATION COMPLETE")
    print(f"  Item overrides:   {overrides_count} rows")
    print(f"  Completed items:  {completed_count} rows")
    print()
    print("The JSON files can now be kept as backup or deleted.")
    print("The backend will read/write PostgreSQL going forward.")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
