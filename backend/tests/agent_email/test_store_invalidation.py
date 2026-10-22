"""
tests/agent_email/test_store_invalidation.py
===============================================
The derived ``meeting_clusters:*`` cache is cleared only when a write really
changed rows (the PostgreSQL writers skip unchanged rows by content hash and
return how many they wrote). No Redis, PostgreSQL or Weaviate is needed.
"""

import asyncio
from types import SimpleNamespace

import pytest

from agent_email import redis_cache, store


@pytest.fixture
def calls(monkeypatch):
    """Stub the PostgreSQL writers/readers and the Redis cache; record clear calls."""
    recorded = {"clears": 0, "written": 0}

    async def fake_write(records, settings=None):
        return recorded["written"]

    async def fake_read(settings=None):
        return {}

    async def fake_set(*args, **kwargs):
        return True

    async def fake_clear(settings=None):
        recorded["clears"] += 1
        return 0

    for name in ("write_meeting_embeddings", "write_meeting_history", "write_cluster_records"):
        monkeypatch.setattr(store.postgres_store, name, fake_write)
    for name in ("read_meeting_embeddings", "read_meeting_history", "read_all_cluster_records"):
        monkeypatch.setattr(store.postgres_store, name, fake_read)
    for name in ("set_meeting_embeddings", "set_meeting_history", "set_clusters"):
        monkeypatch.setattr(redis_cache, name, fake_set)
    monkeypatch.setattr(redis_cache, "clear_meeting_clusters", fake_clear)
    return recorded


_SETTINGS = SimpleNamespace(redis_enabled=True)


@pytest.mark.parametrize("write", [
    lambda: store.write_meeting_embeddings([], settings=_SETTINGS),
    lambda: store.write_meeting_history([], settings=_SETTINGS),
    lambda: store.write_cluster_records([], settings=_SETTINGS),
])
def test_cache_is_cleared_only_when_rows_changed(calls, write):
    calls["written"] = 0
    assert asyncio.run(write()) == 0
    assert calls["clears"] == 0          # nothing changed -> the derived cache stays

    calls["written"] = 3
    assert asyncio.run(write()) == 3
    assert calls["clears"] == 1          # rows changed -> invalidated exactly once
