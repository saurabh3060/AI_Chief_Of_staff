"""
core/redis/client.py
=====================
Redis connection management (shared core infrastructure), mirroring the
layout of ``core/weaviate/client.py`` — a ``core/redis`` package whose
``client.py`` owns the single process-wide connection.

Redis is used purely as a **disposable read cache** by the email agent
(see ``agent_email/redis_cache.py``). Nothing is ever the master copy in
Redis: every cached value can be regenerated from Weaviate/PostgreSQL, so
a Redis outage degrades to "slower reads", never "wrong reads".

Lifecycle
---------
The FastAPI process is long-running, so we keep a single process-wide
async client (``redis.asyncio``), lazily created and cached, and close it
once at shutdown. The client is created from the typed ``REDIS_*`` settings
in ``core/config.py`` — no ``os.getenv`` here.

Connecting is lazy and tolerant: if ``REDIS_HOST`` is empty the cache is
disabled and ``get_redis()`` returns ``None`` so callers transparently
skip the cache. A configured-but-unreachable Redis is reported by
``ping()`` returning False (logged at startup) rather than raising on
every call — the cache simply misses and reads fall through to the system
of record.

Note on the package name
------------------------
This package is named ``redis`` but lives under ``core`` (``core.redis``).
The ``import redis.asyncio`` below is an absolute import, so it resolves to
the installed ``redis`` library, never to this package — the two never
collide.
"""

from __future__ import annotations

import asyncio

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_email")

_lock = asyncio.Lock()
_client = None  # redis.asyncio.Redis | None
_initialized = False


def _build_client(settings: Settings):
    """
    Construct a ``redis.asyncio.Redis`` from settings, or return None when
    Redis is not configured. ``decode_responses=True`` so values come back
    as ``str`` (we store JSON text).
    """
    if not settings.redis_enabled:
        return None
    # Imported lazily so a deployment with Redis disabled never needs the
    # dependency import to succeed at module import time.
    import redis.asyncio as aioredis

    return aioredis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password or None,
        ssl=settings.redis_ssl,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


async def get_redis(settings: Settings | None = None):
    """
    Return the process-wide async Redis client, creating it on first use.
    Returns ``None`` when Redis is disabled (``REDIS_HOST`` empty), so
    callers can treat the cache as absent without special-casing config.

    Creating the client object does NOT open a socket; the first command
    (or an explicit ``ping``) does. A creation failure is logged and
    swallowed (returns None) so the cache can never break a request path.
    """
    global _client, _initialized
    active = settings or get_settings()
    async with _lock:
        if _initialized:
            return _client
        try:
            _client = _build_client(active)
        except Exception as exc:  # noqa: BLE001
            logger.error("[REDIS] client construction failed, cache disabled: %s", exc)
            _client = None
        _initialized = True
        return _client


async def ping(settings: Settings | None = None) -> bool:
    """
    Ping Redis. Returns True on PONG, False when Redis is disabled or
    unreachable. Never raises — intended for the startup log and the
    /health endpoint.
    """
    client = await get_redis(settings)
    if client is None:
        return False
    try:
        return bool(await client.ping())
    except Exception as exc:  # noqa: BLE001
        logger.warning("[REDIS] ping failed: %s", exc)
        return False


async def close_redis() -> None:
    """
    Close the process-wide Redis client, if one is open. Safe to call
    multiple times. Intended for FastAPI's shutdown/lifespan hook.
    """
    global _client, _initialized
    async with _lock:
        if _client is not None:
            try:
                await _client.aclose()
                logger.info("[REDIS] connection closed")
            except Exception:  # noqa: BLE001
                pass
            finally:
                _client = None
        _initialized = False


def reset_for_tests() -> None:
    """Drop the cached client/flag without closing (test teardown only)."""
    global _client, _initialized
    _client = None
    _initialized = False
