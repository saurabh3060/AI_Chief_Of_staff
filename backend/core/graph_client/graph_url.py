"""
core/graph_client/graph_url.py
=================================
Backwards-compatibility SHIM over :mod:`core.graph_client.url_safety`.

Historically there were TWO parallel Graph URL-safety implementations in
this package - this module and ``url_safety`` - with duplicated structured
URL parsing, allow-lists, and segment validation. That duplication was a
maintenance and correctness hazard (CWE-918 fixes had to be made twice and
could drift apart).

The security logic now lives in ONE place: ``core.graph_client.url_safety``.
This module remains ONLY so existing imports and tests that reference the
``graph_url`` API (``GRAPH_BASE_URL``, ``APPROVED_HOSTNAME``,
``_validate_base_url``, ``safe_segment``, ``graph_url``,
``validate_graph_url``) keep working. Every function here is a thin wrapper
that delegates to ``url_safety`` and, where older callers/tests depend on
specific error wording, re-raises the single canonical
``UnsafeGraphUrlError`` with a message preserving that wording.

Do NOT add new validation logic here. Add it to ``url_safety`` and, if
needed, expose it through a wrapper below.
"""

from __future__ import annotations

from core.graph_client.url_safety import (
    ALLOWED_GRAPH_SCHEME,
    APPROVED_HOSTNAME,
    UnsafeGraphUrlError,
    safe_path_segment,
    validate_graph_base_url,
    validate_graph_request_url,
)

# ---------------------------------------------------------------------------
# Trusted base URL (kept as a module constant for backwards compatibility).
#
# The AUTHORITATIVE base URL at runtime is ``settings.graph_base_url`` (read
# from the ``GRAPH_BASE_URL`` env/config and validated by the ``Settings``
# model validator). This constant is the validated default and is used by
# the settings-less ``graph_url()`` helper below, which a handful of static
# endpoint builders still call.
# ---------------------------------------------------------------------------

_RAW_BASE = "https://graph.microsoft.com/v1.0"

# Validated via the SINGLE canonical base-URL validator. Any drift from the
# approved origin fails fast at import time.
GRAPH_BASE_URL: str = validate_graph_base_url(_RAW_BASE)

__all__ = [
    "GRAPH_BASE_URL",
    "APPROVED_HOSTNAME",
    "safe_segment",
    "graph_url",
    "validate_graph_url",
]


def _validate_base_url(raw: str) -> str:
    """Validate a Graph base URL. Delegates to the canonical validator.

    Kept for callers/tests that import ``graph_url._validate_base_url``.
    Error messages are normalised to the historical wording this shim's
    tests assert (``HTTPS``, ``hostname``, ``port``, ``credentials``,
    ``query``, ``fragment``) while the actual structured checks live in
    ``url_safety.validate_graph_base_url``.
    """
    try:
        return validate_graph_base_url(raw)
    except UnsafeGraphUrlError as exc:
        raise UnsafeGraphUrlError(_normalise_base_url_message(str(exc))) from exc


def safe_segment(value: str, name: str = "segment") -> str:
    """Validate + percent-encode a dynamic path segment.

    Thin wrapper over ``url_safety.safe_path_segment``. The canonical
    function raises specific messages (``must not be empty``, ``must not
    contain '/'`` …); older callers/tests here expect the words ``empty``
    and ``forbidden``, so non-empty rejections are re-raised with a
    ``forbidden``-flavoured message. The empty-check wording already
    contains ``empty`` and is passed through unchanged.

    This wrapper is intentionally STRICTER than the canonical function for
    one character: ``@``. ``url_safety.safe_path_segment`` treats ``@`` as a
    legal (encoded) id character, which is safe once percent-encoded. The
    historical ``graph_url`` contract (and its callers, e.g. the SharePoint
    share-id / drive-id builders) forbid ``@`` outright to avoid any chance
    of userinfo/credential confusion, so we reject it here before
    delegating. Rejecting is strictly safe (it can only refuse more).
    """
    if isinstance(value, str) and "@" in value:
        raise UnsafeGraphUrlError(
            f"Graph API path segment {name!r} contains a forbidden "
            f"character or sequence: '@'"
        )
    try:
        return safe_path_segment(value, name)
    except UnsafeGraphUrlError as exc:
        message = str(exc)
        if "empty" in message:
            # Already matches the historical ``empty`` wording.
            raise
        raise UnsafeGraphUrlError(
            f"Graph API path segment {name!r} contains a forbidden "
            f"character or sequence: {message}"
        ) from exc


def graph_url(path: str) -> str:
    """Build a full Graph API URL from an application-defined ``path``.

    ``path`` must already consist of static strings and/or segments that
    were encoded via ``safe_segment()``; this helper only joins the trusted
    base. It does NOT accept user-controlled input directly.

    Uses the default/base origin (``GRAPH_BASE_URL``). Builders that have a
    ``Settings`` object should prefer ``url_safety.validate_graph_base_url
    (settings.graph_base_url)`` so a non-default ``GRAPH_BASE_URL`` config
    is honoured; this settings-less form is retained for static callers.
    """
    if not path.startswith("/"):
        raise ValueError(f"Graph URL path must start with '/', got: {path!r}")
    return f"{GRAPH_BASE_URL}{path}"


def validate_graph_url(url: str) -> str:
    """Validate a fully-formed URL (e.g. an ``@odata.nextLink``).

    Delegates to the single canonical egress guard
    ``url_safety.validate_graph_request_url`` and normalises the error
    wording to what this shim's tests assert (``HTTPS``, ``hostname``,
    ``port``, ``credentials``, ``empty``).
    """
    try:
        return validate_graph_request_url(url)
    except UnsafeGraphUrlError as exc:
        raise UnsafeGraphUrlError(_normalise_request_url_message(str(exc))) from exc


# ---------------------------------------------------------------------------
# Error-message normalisation helpers.
#
# The canonical ``url_safety`` functions phrase their errors slightly
# differently from the historical ``graph_url`` API. These helpers map a
# canonical message onto the keyword the legacy tests/callers match on, so
# consolidating onto one implementation does not change any public contract.
# ---------------------------------------------------------------------------

def _normalise_request_url_message(message: str) -> str:
    lowered = message.lower()
    if "scheme" in lowered:
        return f"Graph API URL must use HTTPS: {message}"
    if "host" in lowered:
        return f"Graph API URL hostname is not approved: {message}"
    if "port" in lowered:
        return f"Graph API URL uses an unexpected port: {message}"
    if "credential" in lowered:
        return f"Graph API URL must not contain embedded credentials: {message}"
    if "non-empty" in lowered or "empty" in lowered:
        return f"Graph API URL must not be empty: {message}"
    return message


def _normalise_base_url_message(message: str) -> str:
    lowered = message.lower()
    if "scheme" in lowered:
        return f"Graph base URL must use HTTPS: {message}"
    if "host" in lowered:
        return f"Graph base URL hostname must be graph.microsoft.com: {message}"
    if "port" in lowered:
        return f"Graph base URL uses an unexpected port: {message}"
    if "credential" in lowered:
        return f"Graph base URL must not contain embedded credentials: {message}"
    if "query" in lowered:
        return f"Graph base URL must not contain a query string: {message}"
    if "fragment" in lowered:
        return f"Graph base URL must not contain a fragment: {message}"
    return message
