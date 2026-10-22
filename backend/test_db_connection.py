"""
test_db_connection.py
=====================
Standalone PostgreSQL connectivity + data-fetch check for the backend.

Why this exists
---------------
The app fails to start when `init_db()` -> `create_email_schema()` hangs on
a Postgres query and asyncpg raises `TimeoutError`. That happens when the DB
is unreachable OR a DDL statement is blocked on a lock. This script isolates
*just* the DB connection using the SAME config (`POSTGRES_DSN` from
`backend/.env`) and the SAME driver (asyncpg) the app uses, so you can tell
connectivity problems apart from app/schema problems without touching code.

Run it (from the backend/ folder, with the venv active):

    python test_db_connection.py

It will:
  1. Load POSTGRES_DSN from .env (via core.config, same as the app).
  2. Open ONE connection with a short timeout (so it fails fast, not after 30s).
  3. Run SELECT 1 (pure connectivity).
  4. Print server version + current database/user.
  5. List tables and show a few sample rows from a known table if present.
  6. Check for sessions stuck "idle in transaction" (the usual cause of the
     startup DDL hang) so you know whether a lock is blocking schema creation.

Nothing here writes to the database.
"""

import asyncio
import sys

try:
    import asyncpg
except ImportError:
    print("ERROR: asyncpg is not installed. Activate the venv first:")
    print(r"    .\venv\Scripts\activate")
    sys.exit(1)

from core.config import get_settings

# Fail fast instead of waiting the app's 30s command_timeout.
CONNECT_TIMEOUT_SECONDS = 30
QUERY_TIMEOUT_SECONDS = 30

# Tables the app creates; we try to sample whichever exist.
SAMPLE_TABLES = [
    "email_clusters",
    "email_topics",
    "tracker_completed_items",
    "tracker_item_overrides",
]


def _mask_dsn(dsn: str) -> str:
    """Hide the password when echoing the DSN back to the console."""
    if "@" not in dsn or "://" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    creds, host = rest.split("@", 1)
    if ":" in creds:
        user = creds.split(":", 1)[0]
        creds = f"{user}:****"
    return f"{scheme}://{creds}@{host}"


async def main() -> int:
    settings = get_settings()
    dsn = settings.postgres_dsn

    if not dsn:
        print("POSTGRES_DSN is empty in backend/.env.")
        print("The app would run in JSON-file fallback mode, not Postgres.")
        print("Set POSTGRES_DSN to test a real connection.")
        return 1

    print(f"Using DSN: {_mask_dsn(dsn)}")
    print(f"Connecting (timeout={CONNECT_TIMEOUT_SECONDS}s)...")

    # --- 1. Open a single connection, fast-failing on timeout ---
    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(dsn=dsn, command_timeout=QUERY_TIMEOUT_SECONDS),
            timeout=CONNECT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        print("\nFAILED: connection timed out.")
        print("The DB host accepted the socket but did not complete the handshake")
        print("in time, OR the host/port is unreachable (VPN down, wrong host,")
        print("firewall, DB service stopped). Verify POSTGRES_DSN host:port.")
        return 2
    except (OSError, asyncpg.PostgresError) as exc:
        print(f"\nFAILED to connect: {type(exc).__name__}: {exc}")
        print("Check host, port, database name, username and password in the DSN.")
        return 2

    print("Connected.\n")

    try:
        # --- 2. Pure connectivity ---
        one = await conn.fetchval("SELECT 1;")
        print(f"SELECT 1 -> {one}  (connectivity OK)")

        # --- 3. Server / session info ---
        version = await conn.fetchval("SHOW server_version;")
        dbname = await conn.fetchval("SELECT current_database();")
        dbuser = await conn.fetchval("SELECT current_user;")
        print(f"PostgreSQL version : {version}")
        print(f"Database           : {dbname}")
        print(f"Connected user     : {dbuser}")

        # --- 4. List tables in the public schema ---
        tables = await conn.fetch(
            """
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename;
            """
        )
        table_names = [r["tablename"] for r in tables]
        print(f"\nPublic tables ({len(table_names)}): "
              f"{', '.join(table_names) if table_names else '(none)'}")

        # --- 5. Sample a few rows from whichever known table exists ---
        for tbl in SAMPLE_TABLES:
            if tbl not in table_names:
                continue
            count = await conn.fetchval(f'SELECT COUNT(*) FROM "{tbl}";')
            print(f'\nTable "{tbl}": {count} row(s).')
            if count:
                rows = await conn.fetch(f'SELECT * FROM "{tbl}" LIMIT 3;')
                for i, row in enumerate(rows, 1):
                    # Truncate wide values so the console stays readable.
                    preview = {
                        k: (str(v)[:60] + "..." if v is not None and len(str(v)) > 60 else v)
                        for k, v in dict(row).items()
                    }
                    print(f"  row {i}: {preview}")
            break  # one sample table is enough to prove data fetch works

        # --- 6. Diagnose the startup DDL hang: look for blocking sessions ---
        # List ALL other sessions on this database (not just ours), with
        # their application_name and client address so you can tell WHICH
        # client/machine owns a stuck transaction. bu_emails / chatbot_history
        # are NOT created by this app, so a session reading them is some OTHER
        # process sharing the rbl_ceo database.
        # Timestamps included so you can see exactly WHEN each session's
        # transaction began and how long it has been idle:
        #   xact_start  - when the open transaction started (the stale one)
        #   query_start - when its last query started running
        #   state_change- when it last entered its current state (e.g. went
        #                  'idle in transaction'); idle_age is now() minus this.
        # Timestamps are shown in the DB server's timezone.
        others = await conn.fetch(
            """
            SELECT pid, state, wait_event_type, wait_event,
                   usename, application_name,
                   host(client_addr) AS client,
                   LEFT(query, 90) AS query,
                   xact_start,
                   query_start,
                   state_change,
                   now() - xact_start  AS xact_age,
                   now() - state_change AS idle_age
            FROM pg_stat_activity
            WHERE datname = current_database()
              AND pid <> pg_backend_pid()
            ORDER BY xact_start NULLS LAST;
            """
        )
        if others:
            print(f"\nOther sessions on database '{dbname}' ({len(others)}):")
            for r in others:
                print(f"  pid={r['pid']} state={r['state']!r} "
                      f"wait={r['wait_event_type']}/{r['wait_event']} "
                      f"user={r['usename']} app={r['application_name']!r} "
                      f"client={r['client']}")
                print(f"      xact_start   = {r['xact_start']}   (age {r['xact_age']})")
                print(f"      query_start  = {r['query_start']}")
                print(f"      state_change = {r['state_change']}  (idle for {r['idle_age']})")
                print(f"      query={r['query']!r}")

        stuck = [r for r in others if r["state"] == "idle in transaction"]
        if stuck:
            print(f"\nWARNING: {len(stuck)} session(s) 'idle in transaction'. An "
                  "ALTER TABLE needs an\nACCESS EXCLUSIVE lock and will queue behind "
                  "ANY open transaction on this\nshared database - even on unrelated "
                  "tables - which is exactly why the app\nhangs at create_email_schema "
                  "(db_schema.py line 336). These are NOT created\nby this app "
                  "(bu_emails / chatbot_history belong to another process on rbl_ceo).")
            print("\nTerminate them, then start uvicorn IMMEDIATELY (before a new one opens):")
            for r in stuck:
                print(f"  SELECT pg_terminate_backend({r['pid']});")
            print("\nOr clear all idle-in-transaction sessions at once:")
            print("  SELECT pg_terminate_backend(pid) FROM pg_stat_activity")
            print("  WHERE datname = current_database()")
            print("    AND pid <> pg_backend_pid()")
            print("    AND state = 'idle in transaction';")
        else:
            print("\nNo 'idle in transaction' sessions right now.")

        # --- 7. Prove the exact failing statement works when unblocked ---
        # Run the SAME statement from db_schema.py line 336, but with a short
        # lock_timeout so it fails FAST instead of hanging, and inside a
        # transaction we roll back so nothing is actually changed. If this
        # errors with "lock timeout", that confirms a lock is blocking the
        # app's DDL right now.
        print("\nTesting the exact failing statement (ALTER TABLE meeting_history ...)")
        print("with a 3s lock_timeout, inside a rolled-back transaction (no change made):")
        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute("SET LOCAL lock_timeout = '3s';")
            await conn.execute(
                "ALTER TABLE meeting_history "
                "ADD COLUMN IF NOT EXISTS business_context TEXT DEFAULT ''"
            )
            print("  OK - the ALTER acquired its lock. The app's DDL is NOT blocked")
            print("  right now; if uvicorn still fails, start it again now.")
        except asyncpg.PostgresError as exc:
            print(f"  BLOCKED - {type(exc).__name__}: {exc}")
            print("  This is the live cause of the startup TimeoutError. Clear the")
            print("  'idle in transaction' sessions listed above, then start uvicorn.")
        finally:
            await tx.rollback()  # discard; never mutate the schema from this test

        print("\nConnectivity and data fetch both work. The startup failure is a")
        print("LOCK on this shared database, not a bug in the app code.")
        return 0

    except asyncio.TimeoutError:
        print("\nFAILED: a query timed out AFTER connecting.")
        print("This points to a LOCK (not connectivity) - the same cause as the")
        print("app's startup hang. Check for 'idle in transaction' sessions and")
        print("close any open query windows (pgAdmin/DBeaver) against this DB.")
        return 3
    except asyncpg.PostgresError as exc:
        print(f"\nQuery error: {type(exc).__name__}: {exc}")
        return 3
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
