"""
inspect_bu_emails.py
====================
Read-only diagnostic for the chatbot's source table (default "bu_emails").

Answers the questions that decide whether the email pipeline's write will
actually succeed against a table that already existed before this feature
was added:

  1. What columns does the live table really have, and of what type?
  2. Are there NOT NULL columns without defaults that the pipeline's
     7-column INSERT does not populate? (These would block every write.)
  3. Is `received_at` a timestamp, or TEXT left over from an older schema?
  4. Is there a UNIQUE index on `message_id`? (Required as the
     ON CONFLICT target that makes repeat runs dedupe instead of append.)
  5. How many rows are in there, and how many are actually usable by the
     chatbot's own filtered query?
  6. Would the pipeline's exact INSERT statement succeed? Verified by
     running it for real inside a transaction that is always ROLLED BACK.

Run from the backend/ directory:

    python scripts/inspect_bu_emails.py

Optional flags:
    --samples N     number of sample rows to print (default 5, 0 to skip)
    --full          print untruncated body/attachment text in samples
    --no-dry-run    skip the rollback-protected INSERT probe

SAFETY: this script never commits. The only write it attempts is inside an
explicitly rolled-back transaction, so the table is left exactly as found.
Email bodies are truncated by default; pass --full only if you are
comfortable printing message content to your terminal.
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import asyncpg  # noqa: E402

from core.config import get_settings  # noqa: E402
from agent_email.db_schema import (  # noqa: E402
    bu_emails_existing_columns,
    has_unique_single_column_index,
)
from agent_email.postgres_store import (  # noqa: E402
    _BU_EMAILS_WRITE_COLUMNS,
    build_bu_emails_insert,
)

# Every column the writer can populate. The real INSERT uses the
# intersection of this and the live table's columns.
PIPELINE_COLUMNS = list(_BU_EMAILS_WRITE_COLUMNS)

# Columns the chatbot's fetch_email_rows() SELECTs. Same set today, kept
# separate so a future divergence is obvious.
CHATBOT_REQUIRED_COLUMNS = list(PIPELINE_COLUMNS)

LINE = "=" * 78
SUB = "-" * 78


def header(title: str) -> None:
    print()
    print(LINE)
    print(f"  {title}")
    print(LINE)


def truncate(value, limit: int) -> str:
    if value is None:
        return "<NULL>"
    text = str(value).replace("\n", " ").replace("\r", " ")
    if limit and len(text) > limit:
        return text[:limit] + f"... [{len(text)} chars total]"
    return text


async def table_exists(conn, table: str) -> bool:
    return bool(
        await conn.fetchval(
            "SELECT 1 FROM information_schema.tables WHERE table_name = $1",
            table,
        )
    )


async def show_columns(conn, table: str) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT ordinal_position, column_name, data_type,
               is_nullable, column_default, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = $1
        ORDER BY ordinal_position
        """,
        table,
    )
    columns = [dict(r) for r in rows]

    header(f"1. COLUMNS ({len(columns)} total) in '{table}'")
    print(f"{'#':>3}  {'column':<28} {'type':<28} {'null?':<6} default")
    print(SUB)
    for c in columns:
        default = c["column_default"]
        default_text = truncate(default, 24) if default is not None else "-"
        marker = ""
        if c["column_name"] in PIPELINE_COLUMNS:
            marker = "  <-- written by pipeline"
        print(
            f"{c['ordinal_position']:>3}  {c['column_name']:<28} "
            f"{c['data_type']:<28} {c['is_nullable']:<6} {default_text}{marker}"
        )
    return columns


def check_contract(columns: list[dict]) -> bool:
    """Verify every column the chatbot reads and the pipeline writes exists."""
    names = {c["column_name"] for c in columns}

    header("2. COLUMN CONTRACT")
    ok = True

    missing_read = [c for c in CHATBOT_REQUIRED_COLUMNS if c not in names]
    if missing_read:
        ok = False
        print("  [FAIL] Columns the chatbot SELECTs but the table lacks:")
        for c in missing_read:
            print(f"         - {c}")
    else:
        print("  [OK]   All columns the chatbot reads are present.")

    received = next(
        (c for c in columns if c["column_name"] == "received_at"), None
    )
    if received is None:
        ok = False
        print("  [FAIL] received_at column is missing entirely.")
    elif "timestamp" in received["data_type"]:
        print(f"  [OK]   received_at is {received['data_type']}.")
    else:
        ok = False
        print(
            f"  [FAIL] received_at is '{received['data_type']}', expected a "
            f"timestamp type."
        )
        print(
            "         The pipeline passes a native datetime, which will not "
            "bind to this column."
        )
        print("         Fix without dropping the table:")
        print(
            "           ALTER TABLE bu_emails ALTER COLUMN received_at "
            "TYPE TIMESTAMPTZ"
        )
        print(
            "             USING NULLIF(BTRIM(received_at), '')::timestamptz;"
        )

    # The real blocker: NOT NULL, no default, and not populated by the
    # writer. PIPELINE_COLUMNS already includes unique_id and the legacy
    # email_subject/email_content/attachment_content aliases, so a table
    # using those names is not flagged.
    blockers = [
        c for c in columns
        if c["is_nullable"] == "NO"
        and c["column_default"] is None
        and c["column_name"] not in PIPELINE_COLUMNS
    ]
    if blockers:
        ok = False
        print()
        print("  [FAIL] NOT NULL columns with no default that the pipeline")
        print("         does NOT write. Every INSERT will be rejected:")
        for c in blockers:
            print(f"         - {c['column_name']} ({c['data_type']})")
        print()
        print("         Options, least destructive first:")
        print("           a) give them defaults:")
        print("              ALTER TABLE bu_emails ALTER COLUMN <col> "
              "SET DEFAULT '<value>';")
        print("           b) relax them:")
        print("              ALTER TABLE bu_emails ALTER COLUMN <col> "
              "DROP NOT NULL;")
        print("           c) have the pipeline populate them (code change).")
    else:
        print("  [OK]   No NOT NULL/no-default columns block the insert.")

    return ok


async def show_indexes(conn, table: str) -> str | None:
    header("3. INDEXES & CONSTRAINTS")

    rows = await conn.fetch(
        "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = $1 "
        "ORDER BY indexname",
        table,
    )
    if not rows:
        print("  (no indexes at all)")
    for r in rows:
        print(f"  {r['indexname']}")
        print(f"      {r['indexdef']}")

    # The writer needs a single-column UNIQUE index to use as its
    # ON CONFLICT target. It prefers message_id (the natural key) and falls
    # back to unique_id (the primary key on externally-created tables,
    # which the writer populates deterministically from the message id).
    # With neither, the write degrades to a plain INSERT and repeat runs
    # can duplicate.
    print()
    existing = await bu_emails_existing_columns(conn, table)
    dedup_column = None
    for candidate in ("message_id", "unique_id"):
        if candidate not in existing:
            continue
        if await has_unique_single_column_index(conn, table, candidate):
            dedup_column = candidate
            break

    if dedup_column:
        print(f"  [OK]   UNIQUE index on '{dedup_column}' will be the")
        print("         ON CONFLICT target. Repeat runs UPSERT, not duplicate.")
        if dedup_column == "unique_id":
            print("         (unique_id is set to the Graph message id, so this")
            print("          still gives exactly one row per message.)")
    else:
        print("  [WARN] No UNIQUE index on message_id or unique_id.")
        print("         The pipeline falls back to a plain INSERT, so repeat")
        print("         runs CAN create duplicate rows. To add one:")
        print("           CREATE UNIQUE INDEX uq_bu_emails_message_id")
        print("             ON bu_emails (message_id);")
        print("         If that errors, duplicates already exist - find them:")
        print("           SELECT message_id, COUNT(*) FROM bu_emails")
        print("             GROUP BY message_id HAVING COUNT(*) > 1;")
    return dedup_column


async def show_counts(conn, table: str) -> None:
    header("4. ROW COUNTS")

    total = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
    print(f"  Total rows:                      {total}")

    # The chatbot discards rows with either key blank, so the usable count
    # is what actually matters for retrieval.
    usable = await conn.fetchval(
        f"""
        SELECT COUNT(*) FROM {table}
        WHERE NULLIF(BTRIM(conversation_id), '') IS NOT NULL
          AND NULLIF(BTRIM(message_id), '') IS NOT NULL
        """
    )
    print(f"  Usable by the chatbot:           {usable}")
    if total and usable < total:
        print(f"  -> {total - usable} row(s) are invisible to the chatbot")
        print("     (blank conversation_id or message_id).")

    dupes = await conn.fetch(
        f"""
        SELECT message_id, COUNT(*) AS n FROM {table}
        GROUP BY message_id HAVING COUNT(*) > 1
        ORDER BY n DESC LIMIT 10
        """
    )
    if dupes:
        print(f"  [WARN] {len(dupes)} duplicated message_id(s), top 10:")
        for d in dupes:
            print(f"         {truncate(d['message_id'], 50)} x{d['n']}")
    else:
        print("  [OK]   No duplicate message_id values.")

    convs = await conn.fetchval(
        f"SELECT COUNT(DISTINCT conversation_id) FROM {table}"
    )
    print(f"  Distinct conversation_ids:       {convs}")

    if total:
        span = await conn.fetchrow(
            f"SELECT MIN(received_at) AS oldest, MAX(received_at) AS newest "
            f"FROM {table}"
        )
        print(f"  Oldest received_at:              {span['oldest']}")
        print(f"  Newest received_at:              {span['newest']}")


async def show_samples(conn, table: str, limit: int, full: bool) -> None:
    if limit <= 0:
        return

    header(f"5. SAMPLE ROWS (newest {limit})")
    body_limit = 0 if full else 200

    try:
        rows = await conn.fetch(
            f"SELECT * FROM {table} ORDER BY received_at DESC NULLS LAST "
            f"LIMIT {limit}"
        )
    except Exception as e:
        print(f"  Could not order by received_at ({e}); falling back.")
        rows = await conn.fetch(f"SELECT * FROM {table} LIMIT {limit}")

    if not rows:
        print("  (table is empty)")
        return

    for i, row in enumerate(rows, start=1):
        print()
        print(f"  --- row {i} " + "-" * 60)
        for key, value in dict(row).items():
            print(f"    {key:<24} {truncate(value, body_limit)}")


async def dry_run_insert(conn, table: str) -> bool:
    """Run the pipeline's real INSERT, then always roll it back.

    This is the definitive answer to "will the pipeline write succeed?".
    The statement is built by the pipeline's own
    ``build_bu_emails_insert`` against the live schema, so there is no
    chance of the diagnostic drifting from what the writer actually runs.
    """
    header("6. DRY-RUN INSERT PROBE (always rolled back)")

    existing = await bu_emails_existing_columns(conn, table)

    conflict_column = None
    for candidate in ("message_id", "unique_id"):
        if candidate in existing and await has_unique_single_column_index(
            conn, table, candidate
        ):
            conflict_column = candidate
            break

    sql, columns = build_bu_emails_insert(table, existing, conflict_column)

    if conflict_column:
        print(f"  Conflict target : ON CONFLICT ({conflict_column}) DO UPDATE")
    else:
        print("  Conflict target : none (plain INSERT, duplicates possible)")
    print(f"  Columns written : {len(columns)} -> {', '.join(columns)}")
    print()

    probe_values = {
        "unique_id": "__diagnostic_message__",
        "message_id": "__diagnostic_message__",
        "conversation_id": "__diagnostic_conversation__",
        "subject": "diagnostic probe subject",
        "raw_content": "diagnostic probe body",
        "attachment_data": "",
        "email_subject": "diagnostic probe subject",
        "email_content": "diagnostic probe body",
        "attachment_content": "",
        "received_at": datetime.now(timezone.utc),
        "from_address": "diagnostic@example.invalid",
        "from_name": "Diagnostic Probe",
    }
    probe = tuple(probe_values[c] for c in columns)

    try:
        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute(sql, *probe)
            print("  [OK]   The pipeline's INSERT statement SUCCEEDS.")
            print("         bu_emails will populate on the next run.")
            return True
        finally:
            # Unconditional rollback: the probe row is never persisted.
            await tx.rollback()
            print("         (probe row rolled back - table unchanged)")
    except Exception as e:
        print("  [FAIL] The pipeline's INSERT statement FAILS:")
        print(f"         {type(e).__name__}: {e}")
        print()
        print("         This is why bu_emails is not being populated.")
        print("         Section 2 above lists the likely blocking columns.")
        return False


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only diagnostic for the chatbot's bu_emails table."
    )
    parser.add_argument("--samples", type=int, default=5,
                        help="sample rows to print (0 to skip)")
    parser.add_argument("--full", action="store_true",
                        help="print untruncated body/attachment text")
    parser.add_argument("--no-dry-run", action="store_true",
                        help="skip the rollback-protected INSERT probe")
    args = parser.parse_args()

    settings = get_settings()
    dsn = (settings.postgres_dsn or "").strip()
    table = settings.chatbot_pg_table

    print(LINE)
    print("  bu_emails DIAGNOSTIC (read-only)")
    print(LINE)
    print(f"  CHATBOT_PG_TABLE    : {table}")
    print(f"  POSTGRES_DSN        : {'set' if dsn else 'NOT SET'}")

    if not dsn:
        print()
        print("  POSTGRES_DSN is empty in backend/.env.")
        print("  The pipeline skips the bu_emails write entirely without it.")
        return 1

    try:
        conn = await asyncpg.connect(dsn=dsn, timeout=20)
    except Exception as e:
        print()
        print(f"  Could not connect to PostgreSQL: {type(e).__name__}: {e}")
        return 1

    try:
        db = await conn.fetchval("SELECT current_database()")
        who = await conn.fetchval("SELECT current_user")
        print(f"  Connected to        : {db} as {who}")

        if not await table_exists(conn, table):
            print()
            print(f"  Table '{table}' does NOT exist.")
            print("  Restart the backend - startup creates it automatically.")
            return 1

        columns = await show_columns(conn, table)
        contract_ok = check_contract(columns)
        dedup_column = await show_indexes(conn, table)
        await show_counts(conn, table)
        await show_samples(conn, table, args.samples, args.full)

        probe_ok = None
        if not args.no_dry_run:
            probe_ok = await dry_run_insert(conn, table)

        header("SUMMARY")
        print(f"  Column contract satisfied : {'yes' if contract_ok else 'NO'}")
        print(f"  Dedup conflict target     : {dedup_column or 'NONE'}")
        if probe_ok is not None:
            print(f"  Pipeline INSERT works     : {'YES' if probe_ok else 'NO'}")
        print()
        if probe_ok:
            print("  Ready. Run the pipeline to populate the table:")
            print("    POST /agents/email/clusters/refresh")
        elif probe_ok is False:
            print("  Not ready - see section 6 for the blocking error.")
        return 0 if (probe_ok is not False) else 2
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
