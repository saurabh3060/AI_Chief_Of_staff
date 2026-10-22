"""
tests/core/test_storage.py
=============================
Verifies the storage abstraction: the in-memory backend correctly
implements add/get/list/update semantics for all three repositories,
and `factory.get_storage()` selects the right backend class based on
the `STORAGE_BACKEND` setting.
"""

import pytest


@pytest.fixture(autouse=True)
def reset_storage(required_env):
    """Ensure each test starts with a fresh cached storage instance and settings."""
    from core.config import get_settings
    from core.storage.factory import reset_storage_for_tests

    reset_storage_for_tests()
    yield
    reset_storage_for_tests()
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# ActionRepository (in-memory)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_action_repository_add_and_get():
    """Adding an action item should make it retrievable by its id."""
    from core.storage.memory_backend import InMemoryActionRepository

    repo = InMemoryActionRepository()
    await repo.add({"id": "a1", "title": "Follow up with Finance", "status": "In Progress"})

    fetched = await repo.get("a1")
    assert fetched is not None
    assert fetched["title"] == "Follow up with Finance"


@pytest.mark.asyncio
async def test_action_repository_get_missing_returns_none():
    """Fetching a non-existent action id should return None, not raise."""
    from core.storage.memory_backend import InMemoryActionRepository

    repo = InMemoryActionRepository()
    assert await repo.get("does-not-exist") is None


@pytest.mark.asyncio
async def test_action_repository_list_with_filters():
    """list() should only return items matching every filter key/value pair."""
    from core.storage.memory_backend import InMemoryActionRepository

    repo = InMemoryActionRepository()
    await repo.add({"id": "a1", "status": "Overdue"})
    await repo.add({"id": "a2", "status": "In Progress"})
    await repo.add({"id": "a3", "status": "Overdue"})

    overdue = await repo.list(filters={"status": "Overdue"})
    assert {item["id"] for item in overdue} == {"a1", "a3"}

    all_items = await repo.list()
    assert len(all_items) == 3


@pytest.mark.asyncio
async def test_action_repository_update_merges_patch():
    """update() should merge the patch into the existing record and return it."""
    from core.storage.memory_backend import InMemoryActionRepository

    repo = InMemoryActionRepository()
    await repo.add({"id": "a1", "status": "In Progress", "owner": "Priya"})

    updated = await repo.update("a1", {"status": "Overdue"})
    assert updated is not None
    assert updated["status"] == "Overdue"
    assert updated["owner"] == "Priya"  # untouched field preserved


@pytest.mark.asyncio
async def test_action_repository_update_missing_returns_none():
    """update() on a non-existent id should return None, not raise or create a new record."""
    from core.storage.memory_backend import InMemoryActionRepository

    repo = InMemoryActionRepository()
    assert await repo.update("missing", {"status": "Overdue"}) is None


# ---------------------------------------------------------------------------
# BriefRepository / SummaryRepository (in-memory)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_brief_repository_save_and_get_latest():
    """save_brief() should overwrite the stored brief; get_latest_brief() should return the most recent one."""
    from core.storage.memory_backend import InMemoryBriefRepository

    repo = InMemoryBriefRepository()
    await repo.save_brief("meeting-1", {"summary": "first draft"})
    await repo.save_brief("meeting-1", {"summary": "final draft"})

    latest = await repo.get_latest_brief("meeting-1")
    assert latest is not None
    assert latest["summary"] == "final draft"


@pytest.mark.asyncio
async def test_summary_repository_save_get_and_list():
    """save_summary()/get_latest_summary()/list_summaries() should behave consistently."""
    from core.storage.memory_backend import InMemorySummaryRepository

    repo = InMemorySummaryRepository()
    await repo.save_summary("thread-1", {"text": "summary one", "meeting_id": "m1"})
    await repo.save_summary("thread-2", {"text": "summary two", "meeting_id": "m2"})

    latest = await repo.get_latest_summary("thread-1")
    assert latest is not None
    assert latest["text"] == "summary one"

    filtered = await repo.list_summaries(filters={"meeting_id": "m2"})
    assert len(filtered) == 1
    assert filtered[0]["text"] == "summary two"


# ---------------------------------------------------------------------------
# Storage isolation: mutating a returned dict must not corrupt stored state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_repository_returns_copies_not_references():
    """Mutating a dict returned by get()/list() should not affect the stored record (defensive copying)."""
    from core.storage.memory_backend import InMemoryActionRepository

    repo = InMemoryActionRepository()
    await repo.add({"id": "a1", "status": "In Progress"})

    fetched = await repo.get("a1")
    fetched["status"] = "TAMPERED"

    fetched_again = await repo.get("a1")
    assert fetched_again["status"] == "In Progress"


# ---------------------------------------------------------------------------
# factory.get_storage() backend selection
# ---------------------------------------------------------------------------

def test_factory_returns_memory_backend_by_default(monkeypatch):
    """With STORAGE_BACKEND unset/"memory", get_storage() should return an InMemoryStorageBackend."""
    from core.config import get_settings
    from core.storage.factory import get_storage
    from core.storage.memory_backend import InMemoryStorageBackend

    monkeypatch.setenv("STORAGE_BACKEND", "memory")
    get_settings.cache_clear()

    storage = get_storage()
    assert isinstance(storage, InMemoryStorageBackend)


def test_factory_returns_postgres_backend_when_configured(monkeypatch):
    """With STORAGE_BACKEND=postgres and a DSN set, get_storage() should return a PostgresStorageBackend."""
    from core.config import get_settings
    from core.storage.factory import get_storage
    from core.storage.postgres_backend import PostgresStorageBackend

    monkeypatch.setenv("STORAGE_BACKEND", "postgres")
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://user:pass@localhost/db")
    get_settings.cache_clear()

    storage = get_storage()
    assert isinstance(storage, PostgresStorageBackend)


def test_factory_raises_on_unknown_backend(monkeypatch):
    """An unrecognized STORAGE_BACKEND value should raise a clear StorageError."""
    from core.config import get_settings
    from core.exceptions import StorageError
    from core.storage.factory import get_storage

    monkeypatch.setenv("STORAGE_BACKEND", "redis")
    get_settings.cache_clear()

    with pytest.raises(StorageError):
        get_storage()


@pytest.mark.asyncio
async def test_postgres_backend_methods_raise_not_implemented(monkeypatch):
    """Every PostgresStorageBackend repository method should raise NotImplementedError until built."""
    from core.storage.postgres_backend import PostgresStorageBackend

    backend = PostgresStorageBackend(dsn="postgresql://user:pass@localhost/db")

    with pytest.raises(NotImplementedError):
        await backend.actions.add({"id": "a1"})
