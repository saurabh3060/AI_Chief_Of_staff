"""
core/logging_config.py
========================
Centralised logging setup for the entire backend.

Design goals (per project requirements):
- One rotating `.log` file per agent (and per core client) inside a single
  `logs/` folder, so operators can tail exactly the component they care
  about.
- The console is quiet: only WARNING and above are printed to stdout, so
  routine request handling never spams the terminal.
- Log files capture INFO and above, but callers are expected to avoid
  logging routine "200 OK" completions - only meaningful events (request
  context, decisions, retries, errors) should be logged. This module
  provides a `log_request_outcome()` helper that enforces that policy in
  one place instead of leaving it to each call site's discipline.

Usage:
    from core.logging_config import setup_logging, get_agent_logger

    setup_logging()  # call once, at process startup
    logger = get_agent_logger("agent_email")
    logger.info("Fetched %d messages", count)
"""

import logging
import re
import sys
from logging.handlers import RotatingFileHandler

from core.config import get_settings
from core.utils import sanitize_log_input

# Matches CR, LF, and other ASCII control characters (except tab) that a
# malicious value could use to forge fake log lines when interpolated into
# a log message. `sanitize_log_value` collapses them into a visible token.
_LOG_CONTROL_CHARS_RE = re.compile(r"[\r\n\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_log_value(value: object, max_length: int = 2000) -> str:
    """
    Make an untrusted value safe to interpolate into a log message.

    Strips carriage returns, line feeds, and other control characters so
    a user-supplied string cannot inject newlines and forge additional
    log entries (CWE-117 log injection). Also caps length to keep a
    single oversized value from flooding a log file.

    Use this on any user-controlled value (URLs, query text, filenames,
    subjects, request params) before passing it to a logger.
    """
    text = value if isinstance(value, str) else str(value)
    # Explicit CR/LF stripping first. This is the plain `str.replace`
    # chain that static analyzers (e.g. CodeQL's py/log-injection query)
    # recognize as a sanitizing barrier for newline/log-forging payloads.
    text = text.replace("\r\n", "").replace("\r", "").replace("\n", "")
    # Then collapse any remaining ASCII control characters into a space.
    text = _LOG_CONTROL_CHARS_RE.sub(" ", text)
    if len(text) > max_length:
        text = text[:max_length] + "…[truncated]"
    return text

# Names of every dedicated log file this backend writes. Keeping this as an
# explicit list (rather than discovering it dynamically) makes it obvious
# at a glance which components have their own log stream.
_AGENT_LOG_NAMES = [
    "app",
    "calendar_agent",
    "email_agent",
    "action_tracker",
    "mis_agent",
    "chatbot",
    "graph_client",
    "ollama_client",
]

# Maps the short logger name callers use (e.g. "agent_email") to the log
# file stem it should be written to (e.g. "email_agent"). Extend this map
# whenever a new component needs its own log file.
_LOGGER_TO_FILE = {
    "app": "app",
    "agent_calendar": "calendar_agent",
    "agent_email": "email_agent",
    "agent_action_tracker": "action_tracker",
    "agent_mis": "mis_agent",
    "chatbot": "chatbot",
    "graph_client": "graph_client",
    "ollama_client": "ollama_client",
    "orchestrator": "app",
    "auth": "app",
    "db": "app",
    "secrets": "app",
}

_configured = False


def setup_logging() -> None:
    """
    Configure every known logger with a rotating file handler (writing to
    `logs/<component>.log`) and attach a single shared console handler
    (WARNING+) to the root logger.

    Safe to call multiple times - configuration only happens once per
    process thanks to the `_configured` guard, so re-imports or repeated
    lifespan events do not duplicate handlers.
    """
    global _configured
    if _configured:
        return

    settings = get_settings()
    log_dir = settings.resolved_log_dir
    file_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Root logger: only a console handler, deliberately quiet (WARNING+)
    # so normal operation does not spam the terminal.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root_logger.addHandler(console_handler)

    # One rotating file handler per component. Each component's logger is
    # configured to NOT propagate to the root logger's console handler
    # twice by relying on the root's own level filtering (console stays
    # WARNING+ regardless of which child logger emitted the record).
    for component_name in _AGENT_LOG_NAMES:
        file_path = log_dir / f"{component_name}.log"
        handler = RotatingFileHandler(
            filename=str(file_path),
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        handler.setLevel(file_level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )

        # Attach this handler to every logger name that maps to this file
        # stem, so e.g. both "agent_email" writes to email_agent.log.
        for logger_name, file_stem in _LOGGER_TO_FILE.items():
            if file_stem == component_name:
                component_logger = logging.getLogger(logger_name)
                component_logger.setLevel(file_level)
                component_logger.addHandler(handler)
                component_logger.propagate = True  # still bubble up to console for WARNING+

    _configured = True


def reset_logging_for_tests() -> None:
    """
    Test-only helper: clears the `_configured` guard and removes all
    handlers from every known logger, so `setup_logging()` can be
    exercised repeatedly within a single test session without handlers
    piling up across test cases. Not used by application startup code.
    """
    global _configured
    _configured = False

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    for logger_name in _LOGGER_TO_FILE:
        component_logger = logging.getLogger(logger_name)
        for handler in list(component_logger.handlers):
            component_logger.removeHandler(handler)


def get_agent_logger(name: str) -> logging.Logger:
    """
    Return the logger for a given component name (e.g. "agent_email",
    "graph_client", "app"). Callers should use this instead of calling
    `logging.getLogger()` directly so it's obvious which log file a
    message will end up in.

    If `setup_logging()` has not been called yet, this still returns a
    usable logger (handlers just won't be attached until setup runs),
    which keeps module-level `logger = get_agent_logger(...)` calls safe.
    """
    return logging.getLogger(name)


def log_request_outcome(logger: logging.Logger, method: str, url: str, status_code: int, duration_ms: float) -> None:
    """
    Log the outcome of an outbound HTTP call (Graph API, Ollama, etc.)
    with a consistent policy: successful 2xx responses are logged at
    DEBUG (so they never spam INFO-level files by default), while any
    non-2xx status is logged at WARNING with full context so failures
    are always visible.
    """
    safe_url = sanitize_log_value(url)
    sanitized_url = sanitize_log_input(safe_url)
    if 200 <= status_code < 300:
        logger.debug("%s %s -> %s (%.1fms)", method, sanitized_url, status_code, duration_ms)
    else:
        logger.warning("%s %s -> %s (%.1fms)", method, sanitized_url, status_code, duration_ms)
