"""
core/retry.py
==============
Generic retry/backoff decorator used to wrap outbound calls to
Microsoft Graph API and the local Ollama server, so transient network
issues (connection resets, timeouts) are retried automatically instead
of failing the whole agent run.

No third-party retry library (e.g. tenacity) is used, per project
requirements - this is a small, dependency-free implementation of
exponential backoff with jitter.

`with_retry` works on BOTH synchronous and asynchronous functions: it
inspects the wrapped function with `inspect.iscoroutinefunction` and
returns an appropriately shaped wrapper.

Behaviour:
- `ConnectionError` and `TimeoutError` (and subclasses, e.g.
  `requests.exceptions.ConnectionError`/`Timeout` which are registered as
  virtual subclasses via `connection_exceptions`/`timeout_exceptions`
  arguments) are retried with exponential backoff + jitter.
- Any other exception in `retry_exceptions` is retried using the same
  backoff schedule but logged distinctly, so operators can tell "the
  network is flaky" apart from "the call keeps failing for another
  reason".
- Once `max_attempts` is exhausted, the last exception is re-raised
  (never swallowed) so callers can decide how to handle a permanent
  failure.
"""

import asyncio
import functools
import inspect
import logging
import random
import time
from typing import Callable, Sequence, Type, TypeVar

F = TypeVar("F", bound=Callable)

_DEFAULT_RETRY_LOGGER = logging.getLogger("app")


def _compute_delay(base_delay: float, attempt: int) -> float:
    """
    Compute the exponential backoff delay (with jitter) for the given
    attempt number (1-indexed). Delay = base_delay * 2^(attempt-1),
    plus up to 25% random jitter to avoid synchronized retry storms
    when multiple calls fail at once.
    """
    exponential = base_delay * (2 ** (attempt - 1))
    jitter = exponential * random.uniform(0, 0.25)
    return exponential + jitter


def _resolve_delay(exc: BaseException, base_delay: float, attempt: int) -> float:
    """
    Determine how long to sleep before the next retry. If the exception
    carries its own server-provided delay hint (a `retry_after_seconds`
    attribute, e.g. `core.exceptions.GraphThrottledError` populated from
    Graph API's `Retry-After` header), honor that instead of the
    generic exponential schedule - the server is telling us exactly how
    long it wants us to wait, which is more accurate than guessing.
    """
    retry_after = getattr(exc, "retry_after_seconds", None)
    if retry_after is not None and retry_after >= 0:
        return retry_after
    return _compute_delay(base_delay, attempt)


def _classify_exception(exc: BaseException, connection_exceptions: Sequence[Type[BaseException]], timeout_exceptions: Sequence[Type[BaseException]]) -> str:
    """Return a short label describing what kind of failure this exception represents, for logging."""
    if getattr(exc, "retry_after_seconds", None) is not None:
        return "throttled"
    if isinstance(exc, tuple(connection_exceptions)):
        return "connection_error"
    if isinstance(exc, tuple(timeout_exceptions)):
        return "timeout"
    return "exception"


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retry_exceptions: Sequence[Type[BaseException]] = (Exception,),
    connection_exceptions: Sequence[Type[BaseException]] = (ConnectionError,),
    timeout_exceptions: Sequence[Type[BaseException]] = (TimeoutError,),
    logger: logging.Logger | None = None,
):
    """
    Decorator factory: returns a decorator that retries the wrapped
    function up to `max_attempts` times with exponential backoff +
    jitter whenever it raises one of `retry_exceptions`.

    Args:
        max_attempts: total number of attempts (including the first),
            e.g. 3 means "try once, then retry up to twice more".
        base_delay: base delay in seconds used to compute the
            exponential backoff schedule.
        retry_exceptions: exception types that should trigger a retry.
            Defaults to `(Exception,)` so any failure is retried unless
            the caller narrows it down.
        connection_exceptions: subset of `retry_exceptions` treated as
            "connection" failures for logging purposes.
        timeout_exceptions: subset of `retry_exceptions` treated as
            "timeout" failures for logging purposes.
        logger: logger to use for retry/backoff messages. Defaults to
            the shared "app" logger if not provided.
    """

    def decorator(func: F) -> F:
        active_logger = logger or _DEFAULT_RETRY_LOGGER

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                """Async retry wrapper: awaits the wrapped coroutine, retrying on failure."""
                last_exc: BaseException | None = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except tuple(retry_exceptions) as exc:
                        last_exc = exc
                        kind = _classify_exception(exc, connection_exceptions, timeout_exceptions)
                        if attempt >= max_attempts:
                            active_logger.warning(
                                "%s failed permanently after %d attempts (%s): %s",
                                func.__qualname__, attempt, kind, exc,
                            )
                            raise
                        delay = _resolve_delay(exc, base_delay, attempt)
                        active_logger.warning(
                            "%s failed on attempt %d/%d (%s): %s - retrying in %.2fs",
                            func.__qualname__, attempt, max_attempts, kind, exc, delay,
                        )
                        await asyncio.sleep(delay)
                if last_exc is not None:
                    raise last_exc

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            """Sync retry wrapper: calls the wrapped function, retrying on failure."""
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(retry_exceptions) as exc:
                    last_exc = exc
                    kind = _classify_exception(exc, connection_exceptions, timeout_exceptions)
                    if attempt >= max_attempts:
                        active_logger.warning(
                            "%s failed permanently after %d attempts (%s): %s",
                            func.__qualname__, attempt, kind, exc,
                        )
                        raise
                    delay = _resolve_delay(exc, base_delay, attempt)
                    active_logger.warning(
                        "%s failed on attempt %d/%d (%s): %s - retrying in %.2fs",
                        func.__qualname__, attempt, max_attempts, kind, exc, delay,
                    )
                    time.sleep(delay)
            if last_exc is not None:
                raise last_exc

        return sync_wrapper  # type: ignore[return-value]

    return decorator
