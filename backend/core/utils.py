"""
core/utils.py
===============

Generic utility functions shared across the application.
"""

from typing import Any


def sanitize_log_input(value: str | None) -> str | None:
    """
    Remove line breaks and carriage returns from user-provided values
    to prevent log injection attacks.

    Args:
        value: User-provided string value (may be None)

    Returns:
        Sanitized string with \r, \n, and \r\n removed, or None if input is None
    """
    if value is None:
        return None
    return value.replace("\r", "").replace("\n", "").replace("\r\n", "")


def sanitize_log_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively sanitize all string values in a dictionary to prevent
    log injection attacks.

    Args:
        entry: Dictionary with potentially unsanitized string values

    Returns:
        Dictionary with all string values sanitized
    """
    sanitized = {}
    for key, value in entry.items():
        if isinstance(value, str):
            sanitized[key] = sanitize_log_input(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_entry(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_log_input(v) if isinstance(v, str) else v
                for v in value
            ]
        else:
            sanitized[key] = value
    return sanitized


def parse_ollama_keep_alive(value: str) -> "int | str":
    """Convert a keep_alive setting for Ollama: a plain integer (seconds, ``-1`` =
    forever) becomes an int, anything else (e.g. ``"30m"``) is passed through."""
    try:
        return int(value)
    except ValueError:
        return value
