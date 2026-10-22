"""
core/graph_client/url_safety.py
=================================
Centralised, structured defences against SSRF (CWE-918) for every
Microsoft Graph URL this backend assembles.

The problem this module solves
------------------------------
Graph request URLs are built by interpolating dynamic values - message
ids, attachment ids, and (from trusted config) a mailbox address - into
a path template. At least one of those values (`message_id` on the
`/bu/summarizer-test` route) originates from an HTTP request body, i.e.
it is attacker-controllable. If such a value were allowed to contain
`/`, `?`, `#`, `@`, `\\`, `..`, encoded separators, or a full/relative
URL, it could change the host, scheme, origin, or path the request is
actually sent to - a partial Server-Side Request Forgery.

The two guarantees
------------------
1. `validate_graph_base_url(raw)` - the ORIGIN (scheme + host + port) a
   request can ever reach is pinned to a single trusted base URL that is
   read exclusively from application configuration and validated with a
   structured URL parser (`urlsplit`), never a substring/regex check. It
   must be HTTPS, target exactly `graph.microsoft.com`, use the expected
   port, and carry no embedded credentials, query string, or fragment.

2. `safe_path_segment(value, field_name)` - every dynamic value spliced
   into the path is treated as a single OPAQUE path segment. It is
   rejected if empty, or if it contains path separators, encoded
   separators, traversal sequences, malformed percent escapes, control
   characters, or other URL syntax, and is then percent-encoded with
   `quote(safe="")` so even legitimately id-internal reserved characters
   can never terminate the segment or introduce new structure.

3. `validate_graph_request_url(url)` - the single LAST-LINE egress guard.
   Every fully-assembled URL about to be sent to `httpx` (and every
   server-returned URL such as an `@odata.nextLink`) is re-parsed with
   `urlsplit` and refused unless its scheme/hostname/port resolve to an
   approved Graph origin. This is defence-in-depth, independent of how the
   URL was built.

Together these mean the resolved origin is config-pinned and no dynamic
segment can escape its path position, so user input cannot influence the
scheme, host, port, origin, or overall URL.

This module is the SINGLE source of truth for Graph URL safety. The
sibling `graph_url` module is a thin backwards-compatibility shim that
re-exports from here (see `graph_url.py`); do not duplicate logic there.
"""

from __future__ import annotations

import re
from urllib.parse import quote, urlsplit

# The single hostname this backend is ever permitted to talk to. Exposed
# both as a scalar (`APPROVED_HOSTNAME`) and a frozenset (`ALLOWED_GRAPH_HOSTS`,
# used for exact, unambiguous membership checks - no substring or suffix
# confusion like "graph.microsoft.com.evil.com"). The two are kept in sync
# here so there is a single definition of the approved host.
APPROVED_HOSTNAME = "graph.microsoft.com"
ALLOWED_GRAPH_HOSTS = frozenset({APPROVED_HOSTNAME})

# The only scheme permitted. Pinning this blocks file://, gopher://,
# http:// (downgrade), etc.
ALLOWED_GRAPH_SCHEME = "https"

# Microsoft Graph is served on the default HTTPS port only. `None` (port
# omitted, i.e. implicit 443) and an explicit 443 are both acceptable;
# anything else is rejected so a crafted base URL cannot redirect traffic
# to an unexpected port on the right host.
ALLOWED_GRAPH_PORTS = frozenset({None, 443})

# A conservative allow-list for a RAW (not-yet-encoded) dynamic path
# segment. Microsoft Graph ids (message ids, attachment ids) are URL-safe
# base64/base64url-ish strings; real-world ids also contain '=', '+', '/'
# BEFORE encoding, but '/' is a path separator we must forbid pre-encoding
# to keep the value a single opaque segment. We allow the characters that
# legitimately appear in Graph ids and resource key segments and reject
# everything else outright, THEN still percent-encode what remains. This
# is an allow-list, not a deny-list: anything not explicitly permitted is
# refused.
_ALLOWED_SEGMENT_CHARS = re.compile(r"^[A-Za-z0-9._~!$'()*+,;:@=\- ]+$")

# Explicitly forbidden substrings (checked on the RAW value, before any
# decoding/encoding) that signal an attempt to break out of the segment
# or smuggle structure. Backslash is included because some clients/servers
# normalise it to '/'.
_FORBIDDEN_SUBSTRINGS = ("/", "\\", "?", "#", "//", "..")

# A well-formed percent escape is '%' followed by exactly two hex digits.
# Any '%' not matching this is a malformed escape and is rejected (it would
# otherwise be ambiguous / could enable double-decoding attacks downstream).
_VALID_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_ANY_PERCENT = re.compile(r"%")


class UnsafeGraphUrlError(ValueError):
    """
    Raised when a configured Graph base URL fails structured validation,
    or when a dynamic path segment contains unsafe/URL-structural content.

    Subclasses ``ValueError`` so existing config-loading / request-handling
    code that already treats bad input as a ``ValueError`` keeps working,
    while callers that want to be specific can catch this type.
    """


def validate_graph_base_url(raw: str) -> str:
    """
    Validate a Microsoft Graph base URL coming from trusted application
    configuration and return it normalised (no trailing slash).

    Uses a STRUCTURED parse (`urlsplit`), not string prefix checks or a
    regex over the whole URL. Enforces, in order:

    - non-empty, `str`;
    - scheme is exactly ``https`` (blocks http downgrade, file://, etc.);
    - a hostname is present and is EXACTLY an approved host - compared
      against ``hostname`` (which excludes any ``user:pass@`` and ``:port``)
      so ``https://graph.microsoft.com@evil.com`` resolves to host
      ``evil.com`` and is rejected;
    - no embedded credentials (``username`` / ``password``);
    - the port is the expected one (omitted/implicit 443, or explicit 443);
    - no query string and no fragment;
    - a path that, once normalised, contains no traversal (``..``) segment.

    Raises ``UnsafeGraphUrlError`` on any violation. Intended to be called
    once at startup / before first use (see ``Settings`` validation and the
    cached accessor in ``endpoints.py``).
    """
    if not isinstance(raw, str) or not raw.strip():
        raise UnsafeGraphUrlError("Graph base URL must be a non-empty string.")

    candidate = raw.strip()
    parts = urlsplit(candidate)

    if parts.scheme != ALLOWED_GRAPH_SCHEME:
        raise UnsafeGraphUrlError(
            f"Graph base URL must use the '{ALLOWED_GRAPH_SCHEME}' scheme, got "
            f"'{parts.scheme or '(none)'}' in {candidate!r}."
        )

    # `hostname` is lower-cased by urlsplit and strips any userinfo/port, so
    # this is the safe value to allow-list against.
    host = parts.hostname
    if not host:
        raise UnsafeGraphUrlError(f"Graph base URL has no hostname: {candidate!r}.")
    if host not in ALLOWED_GRAPH_HOSTS:
        raise UnsafeGraphUrlError(
            f"Graph base URL host '{host}' is not an approved Graph host "
            f"({sorted(ALLOWED_GRAPH_HOSTS)}): {candidate!r}."
        )

    if parts.username or parts.password:
        raise UnsafeGraphUrlError(
            f"Graph base URL must not contain embedded credentials: {candidate!r}."
        )

    # Accessing `.port` raises ValueError for a malformed port; surface it as
    # our own error type rather than leaking the raw parse exception.
    try:
        port = parts.port
    except ValueError as exc:
        raise UnsafeGraphUrlError(
            f"Graph base URL has a malformed port: {candidate!r}."
        ) from exc
    if port not in ALLOWED_GRAPH_PORTS:
        raise UnsafeGraphUrlError(
            f"Graph base URL uses an unexpected port '{port}'; only the default "
            f"HTTPS port is allowed: {candidate!r}."
        )

    if parts.query:
        raise UnsafeGraphUrlError(
            f"Graph base URL must not contain a query string: {candidate!r}."
        )
    if parts.fragment:
        raise UnsafeGraphUrlError(
            f"Graph base URL must not contain a fragment: {candidate!r}."
        )

    # A base path is allowed (Graph uses '/v1.0'), but it must not contain a
    # traversal segment that could later resolve the request off the pinned
    # path root.
    for segment in parts.path.split("/"):
        if segment in ("..", "."):
            raise UnsafeGraphUrlError(
                f"Graph base URL path must not contain traversal segments: {candidate!r}."
            )

    # Normalise: drop any trailing slash so callers can join with '/<path>'
    # without producing a double slash.
    normalised_path = parts.path.rstrip("/")
    return f"{parts.scheme}://{parts.netloc}{normalised_path}"


def validate_graph_request_url(url: str) -> str:
    """
    Last-line egress guard: validate a FULLY-ASSEMBLED URL that is about to
    be handed to ``httpx`` and refuse it unless it points at an approved
    Microsoft Graph origin.

    This is the single, canonical check that every outbound-request code
    path in ``client.py`` funnels through, and it is ALSO used to validate
    server-returned URLs (e.g. ``@odata.nextLink`` pagination links) whose
    host/scheme/port are not otherwise under our control.

    Uses a STRUCTURED parse (``urlsplit``), never a substring/prefix/regex
    check, and validates against the parsed ``scheme``/``hostname``/``port``
    so that:

    - ``https://graph.microsoft.com@evil.com/`` → hostname ``evil.com`` → rejected
      (``hostname`` excludes any ``user:pass@`` userinfo and ``:port``);
    - ``//evil.com/path`` (protocol-relative) → empty scheme → rejected;
    - ``http://``/``file://``/``gopher://`` … → scheme mismatch → rejected;
    - an explicit non-443 port → rejected;
    - embedded credentials → rejected.

    Returns the URL unchanged when valid; raises ``UnsafeGraphUrlError``
    (a ``ValueError`` subclass) otherwise.
    """
    if not isinstance(url, str) or not url.strip():
        raise UnsafeGraphUrlError("Graph request URL must be a non-empty string.")

    parts = urlsplit(url.strip())

    if parts.scheme != ALLOWED_GRAPH_SCHEME:
        raise UnsafeGraphUrlError(
            f"Graph request URL must use the '{ALLOWED_GRAPH_SCHEME}' scheme: {url!r}."
        )

    host = parts.hostname
    if not host or host not in ALLOWED_GRAPH_HOSTS:
        raise UnsafeGraphUrlError(
            f"Graph request URL host '{host}' is not an approved Graph host "
            f"({sorted(ALLOWED_GRAPH_HOSTS)}): {url!r}."
        )

    if parts.username or parts.password:
        raise UnsafeGraphUrlError(
            f"Graph request URL must not contain embedded credentials: {url!r}."
        )

    try:
        port = parts.port
    except ValueError as exc:
        raise UnsafeGraphUrlError(
            f"Graph request URL has a malformed port: {url!r}."
        ) from exc
    if port not in ALLOWED_GRAPH_PORTS:
        raise UnsafeGraphUrlError(
            f"Graph request URL uses an unexpected port '{port}': {url!r}."
        )

    return url


def safe_path_segment(value: str, field_name: str = "path segment") -> str:
    """
    Validate a single dynamic value destined for a URL PATH position and
    return it percent-encoded as one opaque segment.

    This is the per-segment half of the SSRF defence. The value is treated
    as data, never structure: it must be a non-empty string containing only
    the conservatively allow-listed set of characters that legitimately
    appear in Microsoft Graph ids, with NO path separators, encoded
    separators, traversal sequences, malformed percent escapes, control
    characters, or other URL syntax. The validated value is then encoded
    with ``quote(safe="")`` so that even allow-listed reserved characters
    (``@``, ``:``, ``=`` …) are escaped and cannot terminate the segment or
    introduce new URL structure.

    Validation order (fail-closed):
      1. type/empty check (reject ``None``, non-``str``, empty, whitespace-only);
      2. control-character check (C0/C1, incl. NUL, CR, LF, TAB, DEL);
      3. explicit forbidden substrings (``/`` ``\\`` ``?`` ``#`` ``//`` ``..``);
      4. percent-escape well-formedness (every ``%`` starts a valid ``%HH``);
      5. structural check: parsing the raw value as a relative URL must not
         reveal a scheme, netloc, query, or fragment (catches ``http:``,
         ``//evil.com``, ``a?b``, ``a#b`` that slip other checks);
      6. character allow-list;
      7. ``quote(safe="")`` encoding.

    Because the final value is percent-encoded, downstream single-decoding
    yields back exactly the validated raw value (no separator reappears),
    and because we reject raw ``%`` that is not already a valid escape, a
    caller cannot pre-encode a separator (e.g. ``%2f`` → ``/``) to smuggle
    one past a later decode (double-encoding defence).

    Raises ``UnsafeGraphUrlError`` on any violation.
    """
    if not isinstance(value, str):
        raise UnsafeGraphUrlError(
            f"{field_name} must be a string, got {type(value).__name__}."
        )
    if not value or not value.strip():
        raise UnsafeGraphUrlError(f"{field_name} must not be empty.")

    # 2. Control characters: C0 range (0x00-0x1F), DEL (0x7F), and C1 range
    #    (0x80-0x9F). These never belong in a Graph id and can corrupt the
    #    request line or be stripped/normalised unpredictably downstream.
    for ch in value:
        code = ord(ch)
        if code < 0x20 or code == 0x7F or 0x80 <= code <= 0x9F:
            raise UnsafeGraphUrlError(
                f"{field_name} contains a control character (U+{code:04X})."
            )

    # 3. Explicit forbidden substrings (path separators / URL delimiters /
    #    traversal). Checked on the RAW value before any encoding.
    for bad in _FORBIDDEN_SUBSTRINGS:
        if bad in value:
            raise UnsafeGraphUrlError(
                f"{field_name} must not contain {bad!r}: {value!r}."
            )

    # 4. Percent escapes must be well-formed. Reject any '%' that is not the
    #    start of a valid two-hex-digit escape (malformed escape / an attempt
    #    to hide a separator as '%2f', '%2e%2e', etc.). We reject ALL raw '%'
    #    that is not already a valid escape; the valid ones are additionally
    #    caught by the allow-list below (which forbids '%'), so a caller
    #    cannot pre-encode anything - the segment is encoded exactly once by
    #    us. This is the double-encoding guard.
    if _ANY_PERCENT.search(value):
        raise UnsafeGraphUrlError(
            f"{field_name} must not contain percent-escapes or '%': {value!r}."
        )

    # 5. Structural parse: a legitimate id is a bare relative reference with
    #    no scheme/host/query/fragment. If urlsplit finds any of those, the
    #    value is trying to be a URL, not an id.
    probe = urlsplit(value)
    if probe.scheme or probe.netloc or probe.query or probe.fragment:
        raise UnsafeGraphUrlError(
            f"{field_name} must be a bare identifier, not a URL: {value!r}."
        )

    # 6. Character allow-list (fail-closed): only characters known to appear
    #    in Graph ids / resource keys are permitted.
    if not _ALLOWED_SEGMENT_CHARS.match(value):
        raise UnsafeGraphUrlError(
            f"{field_name} contains characters outside the allowed set: {value!r}."
        )

    # 7. Encode as a single opaque segment. safe="" escapes '/', '@', ':',
    #    etc. so nothing can terminate the segment or add structure.
    return quote(value, safe="")
