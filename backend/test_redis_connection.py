"""
test_redis_connection.py
========================
Standalone Redis connectivity + read check for the backend, modelled on
``test_db_connection.py``.

Why this exists
---------------
Redis is the backend's disposable read cache (see ``agent_email/redis_cache.py``
and ``core/redis/client.py``). Nothing is ever the master copy in Redis —
every cached value can be regenerated from Weaviate/PostgreSQL — so a Redis
outage only costs cache hits, never correctness. This script isolates *just*
the Redis connection using the SAME config (the ``REDIS_*`` settings from
``backend/.env``, via ``core.config``) and the SAME driver (``redis.asyncio``)
the app uses, so you can tell a Redis problem apart from an app problem
without touching code.

Run it (from the backend/ folder, with the venv active):

    python test_redis_connection.py

It will:
  1. Load the REDIS_* settings from .env (via core.config, same as the app).
  2. Build a redis.asyncio client with a short timeout (fails fast).
  3. PING the server (pure connectivity).
  4. Print server version, mode and basic INFO (uptime, memory, clients).
  5. Count keys under this app's REDIS_KEY_PREFIX and show a few sample keys
     (names + types only — values are NOT printed).

Nothing here writes to Redis.
"""

import asyncio
import sys

from core.config import get_settings

# Fail fast instead of hanging on an unreachable host.
CONNECT_TIMEOUT_SECONDS = 5
OP_TIMEOUT_SECONDS = 5

# How many sample keys to list (names + types only, never values).
SAMPLE_KEY_LIMIT = 10


async def main() -> int:
    settings = get_settings()

    host = (settings.redis_host or "").strip()
    if not host:
        print("REDIS_HOST is empty in backend/.env.")
        print("The app would run with the Redis cache DISABLED (reads fall")
        print("through to Weaviate/PostgreSQL). Set REDIS_HOST to test a")
        print("real connection.")
        return 1

    print(
        f"Using Redis: host={host} port={settings.redis_port} "
        f"db={settings.redis_db} ssl={settings.redis_ssl} "
        f"password={'set' if settings.redis_password else '(none)'}"
    )
    print(f"Key prefix : {settings.redis_key_prefix!r}")
    print(f"Connecting (timeout={CONNECT_TIMEOUT_SECONDS}s)...")

    try:
        import redis.asyncio as aioredis
    except ImportError:
        print("ERROR: the 'redis' package is not installed. Activate the venv first:")
        print(r"    .\venv\Scripts\activate")
        return 1

    # Same construction the app uses (core/redis/client.py _build_client),
    # with short timeouts so this test fails fast.
    client = aioredis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password or None,
        ssl=settings.redis_ssl,
        decode_responses=True,
        socket_connect_timeout=CONNECT_TIMEOUT_SECONDS,
        socket_timeout=OP_TIMEOUT_SECONDS,
    )

    try:
        # --- 1. Pure connectivity (PING) ---
        try:
            pong = await asyncio.wait_for(client.ping(), timeout=CONNECT_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            print("\nFAILED: PING timed out.")
            print("The host/port is unreachable (VPN down, wrong host, firewall,")
            print("Redis service stopped) OR SSL is misconfigured. Verify")
            print("REDIS_HOST/REDIS_PORT/REDIS_SSL in backend/.env.")
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"\nFAILED to connect: {type(exc).__name__}: {exc}")
            print("Check host, port, db, password and SSL in the REDIS_* settings.")
            return 2

        if not pong:
            print("\nFAILED: server did not answer PING with PONG.")
            return 2
        print("PING -> PONG  (connectivity OK)\n")

        # --- 2. Server info ---
        try:
            info = await asyncio.wait_for(client.info(), timeout=OP_TIMEOUT_SECONDS)
            print(f"Redis version   : {info.get('redis_version')}")
            print(f"Mode            : {info.get('redis_mode')}")
            print(f"Uptime (days)   : {info.get('uptime_in_days')}")
            print(f"Connected clients: {info.get('connected_clients')}")
            print(f"Used memory     : {info.get('used_memory_human')}")
        except Exception as exc:  # noqa: BLE001
            # INFO can be disabled on some managed Redis; not fatal.
            print(f"(could not read server INFO: {type(exc).__name__}: {exc})")

        # --- 3. Count + sample this app's keys (names/types only) ---
        prefix = settings.redis_key_prefix or ""
        pattern = f"{prefix}*" if prefix else "*"
        print(f"\nScanning keys matching {pattern!r} (names/types only; no values)...")

        count = 0
        samples: list[tuple[str, str]] = []
        try:
            async for key in client.scan_iter(match=pattern, count=500):
                count += 1
                if len(samples) < SAMPLE_KEY_LIMIT:
                    try:
                        ktype = await client.type(key)
                    except Exception:  # noqa: BLE001
                        ktype = "?"
                    samples.append((key, ktype))
        except Exception as exc:  # noqa: BLE001
            print(f"(key scan failed: {type(exc).__name__}: {exc})")

        print(f"Keys under prefix: {count}")
        for name, ktype in samples:
            print(f"  {name}  (type={ktype})")

        print("\nRedis connectivity and read both work.")
        return 0

    finally:
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
