"""
scripts/diagnostics_precheck.py
===============================
Read-only health precheck for the three storage layers this backend
depends on:

* PostgreSQL  — the system of record (asyncpg, DSN from ``POSTGRES_DSN``).
* Weaviate    — the rebuildable vector copy (v4 client, split REST/gRPC).
* Redis       — the disposable read cache (``redis.asyncio``).

It uses the SAME config (``core.config.get_settings`` reading
``backend/.env``) and the SAME clients the app uses, so a green run here
means the app's storage wiring should come up cleanly too.

Nothing here writes to any store. Every check fails fast and is reported
independently, so one dead layer never hides the status of the others.

Usage (run from the backend/ directory, venv active)
----------------------------------------------------
    python scripts/diagnostics_precheck.py

Exit code
---------
    0  every CONFIGURED layer is healthy.
    1  at least one configured layer failed its check.

A layer that is intentionally disabled in ``.env`` (empty ``POSTGRES_DSN``
or empty ``REDIS_HOST``) is reported as SKIPPED and does not affect the
exit code — only configured-but-broken layers fail the precheck.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from core.config import get_settings  # noqa: E402

# Fail fast rather than waiting on the app's long command_timeout.
CONNECT_TIMEOUT_SECONDS = 8
QUERY_TIMEOUT_SECONDS = 8

# Tables the app owns; the precheck reports a row count for whichever exist.
_KNOWN_TABLES = [
    "email_clusters",
    "email_topics",
    "meeting_embeddings",
    "meeting_history",
    "bu_emails",
    "chatbot_history",
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


def _section(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------
async def check_postgres(settings) -> str:
    """Return 'ok', 'fail', or 'skip'. Pure connectivity + read-only probes."""
    _section("PostgreSQL")
    dsn = settings.postgres_dsn
    if not dsn:
        print("SKIPPED: POSTGRES_DSN is empty — Postgres disabled "
              "(app would use JSON-file fallback).")
        return "skip"

    try:
        import asyncpg
    except ImportError:
        print("FAIL: asyncpg is not installed. Activate the venv first.")
        return "fail"

    print(f"DSN           : {_mask_dsn(dsn)}")
    print(f"Connecting    : timeout={CONNECT_TIMEOUT_SECONDS}s ...")

    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(dsn=dsn, command_timeout=QUERY_TIMEOUT_SECONDS),
            timeout=CONNECT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        print("FAIL: connection timed out (host unreachable, VPN down, "
              "wrong host/port, or DB service stopped).")
        return "fail"
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not connect: {type(exc).__name__}: {exc}")
        return "fail"

    try:
        one = await conn.fetchval("SELECT 1;")
        if one != 1:
            print(f"FAIL: SELECT 1 returned {one!r}.")
            return "fail"
        version = await conn.fetchval("SHOW server_version;")
        dbname = await conn.fetchval("SELECT current_database();")
        dbuser = await conn.fetchval("SELECT current_user;")
        print(f"SELECT 1      : OK")
        print(f"Version       : {version}")
        print(f"Database      : {dbname}")
        print(f"User          : {dbuser}")

        existing = {
            r["tablename"]
            for r in await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public';"
            )
        }
        print("Known tables  :")
        for tbl in _KNOWN_TABLES:
            if tbl in existing:
                cnt = await conn.fetchval(f'SELECT COUNT(*) FROM "{tbl}";')
                print(f"  {tbl:22s} {cnt} row(s)")
            else:
                print(f"  {tbl:22s} (not created yet)")

        # Surface the classic cause of a stuck startup DDL: a session left
        # 'idle in transaction' holds locks that block ALTER TABLE.
        stuck = await conn.fetchval(
            """
            SELECT count(*) FROM pg_stat_activity
            WHERE datname = current_database()
              AND pid <> pg_backend_pid()
              AND state = 'idle in transaction';
            """
        )
        if stuck:
            print(f"WARNING       : {stuck} session(s) 'idle in transaction' "
                  "— these can block schema DDL (ALTER TABLE) at app startup.")
        else:
            print("Locks         : no 'idle in transaction' sessions")
        print("RESULT        : OK")
        return "ok"
    except asyncio.TimeoutError:
        print("FAIL: a query timed out AFTER connecting — points to a LOCK, "
              "not connectivity.")
        return "fail"
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: query error: {type(exc).__name__}: {exc}")
        return "fail"
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Weaviate
# ---------------------------------------------------------------------------
async def check_weaviate(settings) -> str:
    """Return 'ok' or 'fail'. Connects, verifies readiness, reports counts."""
    _section("Weaviate")
    print(f"REST          : {settings.weaviate_http_host}:"
          f"{settings.weaviate_http_port} secure={settings.weaviate_http_secure}")
    print(f"gRPC          : {settings.weaviate_grpc_host}:"
          f"{settings.weaviate_grpc_port} secure={settings.weaviate_grpc_secure}")
    print(f"Collections   : prefix={settings.weaviate_email_collection_prefix!r}")

    from agent_email import weaviate_vector_store as wvs
    from core.weaviate.client import get_client, close_client

    try:
        # get_client() calls is_ready() and raises if the server is not up.
        client = await asyncio.to_thread(get_client, settings)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not connect / not ready: {type(exc).__name__}: {exc}")
        return "fail"

    try:
        ready = await asyncio.to_thread(client.is_ready)
        print(f"is_ready()    : {ready}")
        if not ready:
            print("FAIL: Weaviate reports not ready.")
            return "fail"

        # Node count exercises the gRPC/cluster meta path.
        try:
            nodes = await asyncio.to_thread(wvs.cluster_node_count, settings)
            print(f"Nodes         : {nodes}")
        except Exception as exc:  # noqa: BLE001
            print(f"Nodes         : (could not read: {exc})")

        print("Owned collections (object_count is read-only, -1 = missing):")
        all_ok = True
        for namespace, name in wvs.owned_collection_names(settings).items():
            try:
                cnt = await asyncio.to_thread(wvs.object_count, namespace, settings)
                state = "MISSING" if cnt < 0 else f"{cnt} objects"
                print(f"  {namespace:10s} -> {name:28s} [{state}]")
            except Exception as exc:  # noqa: BLE001
                all_ok = False
                print(f"  {namespace:10s} -> {name:28s} [ERROR: {exc}]")

        # Chatbot's own collection (separate from the email namespaces).
        try:
            exists = await asyncio.to_thread(
                client.collections.exists, settings.chatbot_weaviate_collection
            )
            print(f"  chatbot    -> {settings.chatbot_weaviate_collection:28s} "
                  f"[{'exists' if exists else 'MISSING'}]")
        except Exception as exc:  # noqa: BLE001
            print(f"  chatbot    -> (could not check: {exc})")

        print(f"RESULT        : {'OK' if all_ok else 'OK (with per-collection warnings)'}")
        return "ok" if all_ok else "ok"
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return "fail"
    finally:
        await asyncio.to_thread(close_client)


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
async def check_redis(settings) -> str:
    """Return 'ok', 'fail', or 'skip'. Pings and reports owned-key count."""
    _section("Redis")
    if not settings.redis_enabled:
        print("SKIPPED: REDIS_HOST is empty — Redis disabled "
              "(chat history / cache simply off).")
        return "skip"

    print(f"Host          : {settings.redis_host}:{settings.redis_port} "
          f"db={settings.redis_db} ssl={settings.redis_ssl}")
    print(f"Key prefix    : {settings.redis_key_prefix!r}")

    from core.redis.client import get_redis, ping as redis_ping, close_redis

    try:
        connected = await redis_ping(settings)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: ping raised: {type(exc).__name__}: {exc}")
        return "fail"

    if not connected:
        print("FAIL: PING did not return PONG (unreachable / auth / wrong port).")
        await close_redis()
        return "fail"

    print("PING          : PONG")
    try:
        client = await get_redis(settings)
        # Count only THIS app's keys — never flushdb-scan another project.
        owned = 0
        async for _ in client.scan_iter(match=f"{settings.redis_key_prefix}*"):
            owned += 1
        print(f"Owned keys    : {owned} (matching {settings.redis_key_prefix!r})")
        print("RESULT        : OK")
        return "ok"
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return "fail"
    finally:
        await close_redis()


async def main() -> int:
    settings = get_settings()
    print("Storage precheck — using backend/.env config, read-only.")

    results = {
        "PostgreSQL": await check_postgres(settings),
        "Weaviate": await check_weaviate(settings),
        "Redis": await check_redis(settings),
    }

    _section("SUMMARY")
    label = {"ok": "OK", "fail": "FAIL", "skip": "SKIPPED (disabled)"}
    for layer, status in results.items():
        print(f"  {layer:12s} : {label[status]}")

    failed = [layer for layer, status in results.items() if status == "fail"]
    if failed:
        print(f"\nPRECHECK FAILED: {', '.join(failed)} not healthy.")
        return 1
    print("\nPRECHECK PASSED: every configured storage layer is healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
