"""
core/storage/factory.py
=========================
Selects and caches the active storage backend based on the
`STORAGE_BACKEND` setting (`memory` or `postgres`).

Agents should always obtain storage through `get_storage()` rather than
constructing `InMemoryStorageBackend`/`PostgresStorageBackend` directly,
so the toggle in `.env` is the single source of truth for which backend
is active.
"""

from functools import lru_cache

from core.config import get_settings
from core.exceptions import StorageError
from core.storage.base import StorageBackend
from core.storage.memory_backend import InMemoryStorageBackend
from core.storage.postgres_backend import PostgresStorageBackend


@lru_cache
def get_storage() -> StorageBackend:
    """
    Return the process-wide storage backend instance, chosen by the
    `STORAGE_BACKEND` setting. Cached so every caller shares the same
    in-memory state within a process.
    """
    settings = get_settings()

    if settings.storage_backend == "memory":
        return InMemoryStorageBackend()

    if settings.storage_backend == "postgres":
        return PostgresStorageBackend(dsn=settings.postgres_dsn)

    raise StorageError(
        f"Unknown STORAGE_BACKEND '{settings.storage_backend}'. Expected 'memory' or 'postgres'."
    )


def reset_storage_for_tests() -> None:
    """Test-only helper: clears the cached storage backend so the next call to get_storage() builds a fresh one."""
    get_storage.cache_clear()
