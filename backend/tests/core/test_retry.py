"""
tests/core/test_retry.py
==========================
Verifies `core.retry.with_retry` behaviour for both synchronous and
asynchronous functions: retries the configured number of times with
backoff, eventually succeeds if the underlying call recovers, and
re-raises the last exception once attempts are exhausted.

`time.sleep` / `asyncio.sleep` are monkeypatched to no-ops so these
tests run instantly instead of actually waiting out the backoff delays.
"""

import asyncio

import pytest


def test_sync_function_retries_then_succeeds(monkeypatch):
    """A flaky sync function that fails twice then succeeds should return its result after 3 attempts."""
    from core import retry as retry_module

    monkeypatch.setattr(retry_module.time, "sleep", lambda _: None)

    calls = {"count": 0}

    @retry_module.with_retry(max_attempts=3, base_delay=0.01)
    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("simulated connection drop")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 3


def test_sync_function_raises_after_exhausting_attempts(monkeypatch):
    """If every attempt fails, the last exception should be re-raised, not swallowed."""
    from core import retry as retry_module

    monkeypatch.setattr(retry_module.time, "sleep", lambda _: None)

    calls = {"count": 0}

    @retry_module.with_retry(max_attempts=3, base_delay=0.01)
    def always_fails():
        calls["count"] += 1
        raise TimeoutError("simulated timeout")

    with pytest.raises(TimeoutError):
        always_fails()
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_async_function_retries_then_succeeds(monkeypatch):
    """A flaky async function that fails once then succeeds should return its result after 2 attempts."""
    from core import retry as retry_module

    async def instant_sleep(_):
        return None

    monkeypatch.setattr(retry_module.asyncio, "sleep", instant_sleep)

    calls = {"count": 0}

    @retry_module.with_retry(max_attempts=3, base_delay=0.01)
    async def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise ConnectionError("simulated connection drop")
        return "ok"

    result = await flaky()
    assert result == "ok"
    assert calls["count"] == 2


def test_retry_does_not_retry_exceptions_outside_configured_set(monkeypatch):
    """Exceptions not listed in retry_exceptions should propagate immediately, without retrying."""
    from core import retry as retry_module

    monkeypatch.setattr(retry_module.time, "sleep", lambda _: None)

    calls = {"count": 0}

    @retry_module.with_retry(max_attempts=3, base_delay=0.01, retry_exceptions=(ConnectionError,))
    def raises_value_error():
        calls["count"] += 1
        raise ValueError("not a retryable error")

    with pytest.raises(ValueError):
        raises_value_error()
    assert calls["count"] == 1
