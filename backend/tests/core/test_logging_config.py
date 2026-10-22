"""
tests/core/test_logging_config.py
====================================
Verifies the centralized logging setup: per-component rotating log
files are created and written to correctly, unrelated components don't
leak into each other's files, and the "no 200 OK spam" policy enforced
by `log_request_outcome()` behaves as documented (successes at DEBUG,
failures at WARNING).
"""

import logging
import logging.handlers
import sys
from urllib.parse import urlsplit

import pytest


@pytest.fixture
def clean_logging(required_env, tmp_path, monkeypatch):
    """
    Point LOG_DIR at a temporary directory for this test and ensure
    logging state is fully reset before and after, so tests don't
    interfere with each other or with the developer's real logs/ folder.
    """
    from core.config import get_settings
    from core.logging_config import reset_logging_for_tests

    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    get_settings.cache_clear()
    reset_logging_for_tests()
    yield tmp_path
    reset_logging_for_tests()
    get_settings.cache_clear()


def test_setup_logging_creates_expected_log_files(clean_logging):
    """After setup_logging(), every documented per-component log file should exist."""
    from core.logging_config import setup_logging

    setup_logging()

    expected_files = {
        "app.log",
        "calendar_agent.log",
        "email_agent.log",
        "action_tracker.log",
        "mis_agent.log",
        "chatbot.log",
        "graph_client.log",
        "ollama_client.log",
    }
    actual_files = {p.name for p in clean_logging.iterdir()}
    assert expected_files.issubset(actual_files)


def test_agent_logger_writes_only_to_its_own_file(clean_logging):
    """A message logged by 'agent_email' should land in email_agent.log, not in others."""
    from core.logging_config import get_agent_logger, setup_logging

    setup_logging()
    logger = get_agent_logger("agent_email")
    logger.warning("unique marker: mailbox sync failed")

    email_log = (clean_logging / "email_agent.log").read_text(encoding="utf-8")
    calendar_log = (clean_logging / "calendar_agent.log").read_text(encoding="utf-8")

    assert "unique marker: mailbox sync failed" in email_log
    assert "unique marker: mailbox sync failed" not in calendar_log


def test_log_request_outcome_suppresses_success_at_info_level(clean_logging):
    """
    A successful (2xx) outcome should be logged at DEBUG, so with the
    default INFO file level it does not appear in the log file at all -
    enforcing the "no 200 OK spam" requirement.
    """
    from core.logging_config import get_agent_logger, log_request_outcome, setup_logging

    setup_logging()
    logger = get_agent_logger("graph_client")
    log_request_outcome(logger, "GET", "https://graph.microsoft.com/v1.0/me/messages", 200, 42.0)

    graph_log = (clean_logging / "graph_client.log").read_text(encoding="utf-8")
    assert "200" not in graph_log


def test_log_request_outcome_captures_failure_with_full_context(clean_logging):
    """A non-2xx outcome should be logged at WARNING with method/url/status/duration visible."""
    from core.logging_config import get_agent_logger, log_request_outcome, setup_logging

    setup_logging()
    logger = get_agent_logger("graph_client")
    log_request_outcome(logger, "GET", "https://graph.microsoft.com/v1.0/me/messages", 503, 1200.5)

    graph_log = (clean_logging / "graph_client.log").read_text(encoding="utf-8")
    assert "503" in graph_log
    assert "GET" in graph_log
    # Assert on the parsed HOSTNAME of the logged URL, compared by
    # equality, rather than a substring/`in` match against a raw URL or
    # netloc. `hostname` (unlike `netloc`) excludes any userinfo/port, so
    # this cannot be fooled by `graph.microsoft.com` appearing in a
    # `user@host`, `:port`, path, or query segment
    # (py/incomplete-url-substring-sanitization).
    logged_hostnames = {
        urlsplit(token).hostname
        for line in graph_log.splitlines()
        for token in line.split()
        if token.startswith("http")
    }
    assert any(host == "graph.microsoft.com" for host in logged_hostnames)


def test_console_handler_is_warning_level_only(clean_logging):
    """
    The root logger's console handler (the one this module attaches)
    should be set to WARNING, not INFO/DEBUG. We only assert on handlers
    writing to stdout, since pytest itself may attach its own log-capture
    handler to the root logger during test runs.
    """
    from core.logging_config import setup_logging

    setup_logging()
    root_logger = logging.getLogger()
    console_handlers = [
        h for h in root_logger.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.handlers.RotatingFileHandler)
        and getattr(h, "stream", None) is sys.stdout
    ]
    assert console_handlers, "expected a console StreamHandler writing to stdout on the root logger"
    assert all(h.level == logging.WARNING for h in console_handlers)
