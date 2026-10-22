"""
core/concurrency.py
=====================
In-process worker-pool concurrency limiting for outbound calls to
Microsoft Graph API and the local Ollama server.

Implemented as `asyncio.Semaphore` pools sized from `.env`
(`MAX_CONCURRENT_GRAPH_CALLS`, `MAX_CONCURRENT_OLLAMA_CALLS`), so we
never fire more than N concurrent requests at either service - this
avoids overwhelming Graph API rate limits or a locally-hosted Ollama
instance that can only handle a small number of concurrent model runs.

This is deliberately simple (no external broker/queue) so it works with
zero extra infrastructure. If the project later moves to a distributed
task queue (Celery/RQ), only this module needs to change - agents and
core clients just do `async with get_graph_semaphore(): ...` and don't
know or care how the limiting is implemented underneath.

Semaphores are created lazily (not at import time) because they must be
bound to the event loop that is actually running when they're first
used; creating them eagerly at module import time can bind them to the
wrong loop in test environments that spin up a fresh loop per test.
"""

import asyncio

from core.config import get_settings

_graph_semaphore: asyncio.Semaphore | None = None
_ollama_semaphore: asyncio.Semaphore | None = None


def get_graph_semaphore() -> asyncio.Semaphore:
    """
    Return the process-wide semaphore limiting concurrent Graph API
    calls. Created on first use, sized from `MAX_CONCURRENT_GRAPH_CALLS`.
    """
    global _graph_semaphore
    if _graph_semaphore is None:
        settings = get_settings()
        _graph_semaphore = asyncio.Semaphore(settings.max_concurrent_graph_calls)
    return _graph_semaphore


def get_ollama_semaphore() -> asyncio.Semaphore:
    """
    Return the process-wide semaphore limiting concurrent Ollama calls.
    Created on first use, sized from `MAX_CONCURRENT_OLLAMA_CALLS`.
    """
    global _ollama_semaphore
    if _ollama_semaphore is None:
        settings = get_settings()
        _ollama_semaphore = asyncio.Semaphore(settings.max_concurrent_ollama_calls)
    return _ollama_semaphore


def reset_semaphores_for_tests() -> None:
    """
    Test-only helper: clears the cached semaphores so a fresh one is
    created (bound to the current test's event loop, and reflecting any
    monkeypatched concurrency limits) on next use. Not used by
    application startup code.
    """
    global _graph_semaphore, _ollama_semaphore
    _graph_semaphore = None
    _ollama_semaphore = None
