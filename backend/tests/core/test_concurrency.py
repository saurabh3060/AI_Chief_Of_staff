"""
tests/core/test_concurrency.py
=================================
Verifies that the semaphore-based concurrency limiters in
`core.concurrency` actually cap the number of simultaneously running
tasks at the configured limit.
"""

import asyncio

import pytest


@pytest.fixture(autouse=True)
def reset_semaphores(required_env):
    """Ensure each test gets a fresh semaphore bound to its own event loop."""
    from core.concurrency import reset_semaphores_for_tests

    reset_semaphores_for_tests()
    yield
    reset_semaphores_for_tests()


@pytest.mark.asyncio
async def test_graph_semaphore_limits_concurrency(monkeypatch):
    """No more than MAX_CONCURRENT_GRAPH_CALLS tasks should run inside the semaphore at once."""
    monkeypatch.setenv("MAX_CONCURRENT_GRAPH_CALLS", "2")
    from core.config import get_settings

    get_settings.cache_clear()
    from core.concurrency import get_graph_semaphore

    current_in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def worker():
        nonlocal current_in_flight, max_in_flight
        async with get_graph_semaphore():
            async with lock:
                current_in_flight += 1
                max_in_flight = max(max_in_flight, current_in_flight)
            await asyncio.sleep(0.05)
            async with lock:
                current_in_flight -= 1

    await asyncio.gather(*(worker() for _ in range(6)))

    assert max_in_flight == 2
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_ollama_semaphore_limits_concurrency(monkeypatch):
    """No more than MAX_CONCURRENT_OLLAMA_CALLS tasks should run inside the semaphore at once."""
    monkeypatch.setenv("MAX_CONCURRENT_OLLAMA_CALLS", "1")
    from core.config import get_settings

    get_settings.cache_clear()
    from core.concurrency import get_ollama_semaphore

    current_in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def worker():
        nonlocal current_in_flight, max_in_flight
        async with get_ollama_semaphore():
            async with lock:
                current_in_flight += 1
                max_in_flight = max(max_in_flight, current_in_flight)
            await asyncio.sleep(0.05)
            async with lock:
                current_in_flight -= 1

    await asyncio.gather(*(worker() for _ in range(4)))

    assert max_in_flight == 1
    get_settings.cache_clear()
