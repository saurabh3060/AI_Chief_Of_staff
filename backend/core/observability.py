"""
core/observability.py
========================
Langfuse observability, gated behind a single toggle: `LANGFUSE_ENABLED`.

Design goal (per project requirements): the app must work identically
whether Langfuse is enabled or not.
- `LANGFUSE_ENABLED=false` (default): `get_callback_handler()` returns
  `None`. Every LangGraph/LangChain `.invoke()`/`.ainvoke()` call in this
  codebase passes `callbacks=get_langfuse_callbacks()`, which is an empty
  list when disabled - so nothing about the call path changes.
- `LANGFUSE_ENABLED=true`: `get_callback_handler()` constructs a real
  Langfuse `CallbackHandler` using `LANGFUSE_PUBLIC_KEY`/
  `LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST`. Construction is wrapped in a
  try/except - if Langfuse is misconfigured (bad keys, unreachable
  host), the app logs a warning and falls back to `None` rather than
  crashing, so a Langfuse outage or misconfiguration can never take the
  whole backend down.

Callers should generally use `get_langfuse_callbacks()` (returns a list,
directly usable as `callbacks=...` in LangChain/LangGraph calls) rather
than `get_callback_handler()` directly.
"""

from functools import lru_cache

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("app")


@lru_cache
def get_callback_handler():
    """
    Return a Langfuse `CallbackHandler` if `LANGFUSE_ENABLED=true` and
    construction succeeds, otherwise return `None`. Cached so we only
    attempt to construct the handler once per process.
    """
    settings = get_settings()

    if not settings.langfuse_enabled:
        logger.info("Langfuse observability is disabled (LANGFUSE_ENABLED=false)")
        return None

    try:
        # Imported lazily so environments that never enable Langfuse
        # don't need the dependency chain (langfuse -> langchain) to be
        # importable for the rest of the app to function.
        from langfuse.callback import CallbackHandler

        handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse observability enabled (host=%s)", settings.langfuse_host)
        return handler
    except Exception as exc:  # noqa: BLE001 - any Langfuse init failure must degrade gracefully
        logger.warning("Failed to initialise Langfuse; continuing without observability: %s", exc)
        return None


def get_langfuse_callbacks() -> list:
    """
    Return the callbacks list to pass into LangChain/LangGraph
    `.invoke()`/`.ainvoke()` calls: `[handler]` if Langfuse is active,
    or `[]` if disabled/misconfigured. Using this helper means call
    sites never need an `if` check - they just do `callbacks=get_langfuse_callbacks()`.
    """
    handler = get_callback_handler()
    return [handler] if handler is not None else []


def reset_observability_for_tests() -> None:
    """Test-only helper: clears the cached callback handler so tests can exercise both toggle states."""
    get_callback_handler.cache_clear()
