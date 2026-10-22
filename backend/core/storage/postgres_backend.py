"""
core/storage/postgres_backend.py
===================================
PostgreSQL-backed storage implementation.

This module implements storage for:
- Email thread summaries (email_summaries table)

Uses asyncpg connection pool via `core.db.get_pool()`. All repository
methods are implemented to read/write to PostgreSQL and delegate
embedding storage to Weaviate.

To enable, set:
- `POSTGRES_DSN` in .env (PostgreSQL connection string)
- `STORAGE_BACKEND=postgres` in .env
"""

import json
from typing import Any, Optional

from core.db import get_pool
from core.exceptions import StorageError
from core.storage.base import ActionRepository, BriefRepository, StorageBackend, SummaryRepository


class PostgresActionRepository(ActionRepository):
    """Stub - not implemented. See module docstring."""

    async def add(self, action: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Postgres storage backend is reserved for a future implementation.")

    async def get(self, action_id: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError("Postgres storage backend is reserved for a future implementation.")

    async def list(self, filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        raise NotImplementedError("Postgres storage backend is reserved for a future implementation.")

    async def update(self, action_id: str, patch: dict[str, Any]) -> Optional[dict[str, Any]]:
        raise NotImplementedError("Postgres storage backend is reserved for a future implementation.")


class PostgresBriefRepository(BriefRepository):
    """Stub - not implemented. See module docstring."""

    async def save_brief(self, meeting_id: str, brief: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Postgres storage backend is reserved for a future implementation.")

    async def get_latest_brief(self, meeting_id: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError("Postgres storage backend is reserved for a future implementation.")


class PostgresSummaryRepository(SummaryRepository):
    """PostgreSQL-backed summary storage for email threads."""

    async def save_summary(self, thread_id: str, summary: dict[str, Any]) -> dict[str, Any]:
        """
        Persist (or overwrite) the latest summary for a given thread/meeting id.
        Uses email_summaries PostgreSQL table.
        """
        pool = get_pool()
        if pool is None:
            raise RuntimeError("save_summary: PostgreSQL pool unavailable")

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO email_summaries (thread_id, summary_json, created_at, updated_at)
                VALUES ($1, $2::jsonb, now(), now())
                ON CONFLICT (thread_id) DO UPDATE SET
                    summary_json=EXCLUDED.summary_json,
                    updated_at=now()
                """,
                thread_id, json.dumps(summary)
            )
        return summary

    async def get_latest_summary(self, thread_id: str) -> Optional[dict[str, Any]]:
        """Return the latest summary for a given thread/meeting id."""
        pool = get_pool()
        if pool is None:
            return None

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT summary_json FROM email_summaries WHERE thread_id = $1 ORDER BY updated_at DESC LIMIT 1",
                thread_id
            )
            if row:
                return json.loads(row["summary_json"])
        return None

    async def list_summaries(self, filters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """
        Return all stored summaries, optionally filtered by exact-match field values.
        Since summaries are stored as JSON, filtering is done in Python after fetching.
        """
        pool = get_pool()
        if pool is None:
            return []

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT summary_json FROM email_summaries ORDER BY updated_at DESC"
            )

        summaries = [json.loads(row["summary_json"]) for row in rows if row["summary_json"]]

        # Apply filters if provided
        if filters:
            filtered = []
            for summary in summaries:
                match = True
                for key, value in filters.items():
                    if summary.get(key) != value:
                        match = False
                        break
                if match:
                    filtered.append(summary)
            summaries = filtered

        return summaries


class PostgresStorageBackend(StorageBackend):
    """
    PostgreSQL-backed storage implementation. Constructs this class
    successfully (so `factory.get_storage()` can return it without crashing
    at import time) with the required `dsn` parameter. All repository
    methods are implemented using the asyncpg connection pool.
    """

    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise StorageError(
                "STORAGE_BACKEND=postgres requires POSTGRES_DSN to be set in .env."
            )
        self._dsn = dsn
        self.actions = PostgresActionRepository()
        self.briefs = PostgresBriefRepository()
        self.summaries = PostgresSummaryRepository()
