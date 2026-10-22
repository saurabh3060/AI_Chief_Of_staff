"""
core/storage/memory_backend.py
=================================
In-process, in-memory implementation of the storage abstraction.

This is the default storage backend (`STORAGE_BACKEND=memory`) - it
keeps everything in plain Python dicts for the lifetime of the process.
Data does not survive a restart, but that's an accepted tradeoff for the
current on-demand, no-database setup. Swapping to Postgres later only
requires implementing `postgres_backend.py` and does not touch any
agent code, since agents only depend on the abstract interfaces in
`base.py`.

An `asyncio.Lock` guards each repository's dict so concurrent agent
calls (e.g. multiple requests updating action items at once) don't
race on the underlying dict.
"""

import asyncio
from copy import deepcopy
from typing import Any, Optional

from core.storage.base import ActionRepository, BriefRepository, StorageBackend, SummaryRepository


class InMemoryActionRepository(ActionRepository):
    """Dict-backed implementation of `ActionRepository`, keyed by action id."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def add(self, action: dict[str, Any]) -> dict[str, Any]:
        """Store a copy of `action` keyed by its 'id', overwriting any existing entry with the same id."""
        async with self._lock:
            self._items[action["id"]] = deepcopy(action)
            return deepcopy(self._items[action["id"]])

    async def get(self, action_id: str) -> Optional[dict[str, Any]]:
        """Return a copy of the stored action item, or None if it doesn't exist."""
        async with self._lock:
            item = self._items.get(action_id)
            return deepcopy(item) if item is not None else None

    async def list(self, filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Return copies of all action items matching every key/value pair in `filters` (or all items if no filters given)."""
        async with self._lock:
            items = list(self._items.values())
        if not filters:
            return [deepcopy(item) for item in items]
        return [
            deepcopy(item)
            for item in items
            if all(item.get(key) == value for key, value in filters.items())
        ]

    async def update(self, action_id: str, patch: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Merge `patch` into the existing action item; returns None if the action id is unknown."""
        async with self._lock:
            existing = self._items.get(action_id)
            if existing is None:
                return None
            existing.update(patch)
            return deepcopy(existing)


class InMemoryBriefRepository(BriefRepository):
    """Dict-backed implementation of `BriefRepository`, keyed by meeting id -> latest brief."""

    def __init__(self) -> None:
        self._briefs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def save_brief(self, meeting_id: str, brief: dict[str, Any]) -> dict[str, Any]:
        """Overwrite the stored brief for `meeting_id` with `brief`."""
        async with self._lock:
            self._briefs[meeting_id] = deepcopy(brief)
            return deepcopy(self._briefs[meeting_id])

    async def get_latest_brief(self, meeting_id: str) -> Optional[dict[str, Any]]:
        """Return a copy of the latest stored brief for `meeting_id`, or None if none exists."""
        async with self._lock:
            brief = self._briefs.get(meeting_id)
            return deepcopy(brief) if brief is not None else None


class InMemorySummaryRepository(SummaryRepository):
    """Dict-backed implementation of `SummaryRepository`, keyed by thread/meeting id -> latest summary."""

    def __init__(self) -> None:
        self._summaries: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def save_summary(self, thread_id: str, summary: dict[str, Any]) -> dict[str, Any]:
        """Overwrite the stored summary for `thread_id` with `summary`."""
        async with self._lock:
            self._summaries[thread_id] = deepcopy(summary)
            return deepcopy(self._summaries[thread_id])

    async def get_latest_summary(self, thread_id: str) -> Optional[dict[str, Any]]:
        """Return a copy of the latest stored summary for `thread_id`, or None if none exists."""
        async with self._lock:
            summary = self._summaries.get(thread_id)
            return deepcopy(summary) if summary is not None else None

    async def list_summaries(self, filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Return copies of all stored summaries matching every key/value pair in `filters` (or all if no filters given)."""
        async with self._lock:
            items = list(self._summaries.values())
        if not filters:
            return [deepcopy(item) for item in items]
        return [
            deepcopy(item)
            for item in items
            if all(item.get(key) == value for key, value in filters.items())
        ]


class InMemoryStorageBackend(StorageBackend):
    """Bundles the three in-memory repositories behind the `StorageBackend` interface."""

    def __init__(self) -> None:
        self.actions = InMemoryActionRepository()
        self.briefs = InMemoryBriefRepository()
        self.summaries = InMemorySummaryRepository()
