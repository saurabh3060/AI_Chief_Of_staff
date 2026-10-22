"""
tests/documents/conftest.py
============================
Shared fixtures for documents module tests.

Initialises the core.db PostgreSQL pool once per session so that async tests
can use ``store.acquire()`` and other async DB helpers.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _init_db_pool(request):
    """Start the core.db pool at the beginning of the test session, shut it down at the end."""
    import asyncio
    from core.db import init_db, close_db

    asyncio.get_event_loop_policy()  # ensure a loop policy exists
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(init_db())
    except Exception:
        pytest.skip("PostgreSQL pool could not be initialised (is POSTGRES_DSN set?)")
        return

    yield

    loop.run_until_complete(close_db())
    loop.close()
