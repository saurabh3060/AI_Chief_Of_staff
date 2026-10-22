"""
tests/core/test_graph_url_safety.py
======================================
Focused SSRF (CWE-918) tests for the centralised Graph URL-safety layer:

- `core.graph_client.url_safety.safe_path_segment` - opaque path-segment
  validation + encoding for dynamic ids (message id, attachment id, ...).
- `core.graph_client.url_safety.validate_graph_base_url` - structured
  validation of the single trusted base URL.
- `core.graph_client.endpoints.*` - end-to-end URL construction, proving
  a crafted id can never change the host/scheme/origin or escape its path
  segment, and that the resolved host always stays graph.microsoft.com.

These exercise the exact attack payloads called out in the fix: absolute
URLs, protocol-relative URLs, untrusted hosts, alternate schemes,
embedded credentials, unexpected ports, path traversal, encoded and
double-encoded separators/traversal, malformed percent escapes, and ids
containing `/ \\ ? # @ % whitespace` or control characters.
"""

from urllib.parse import urlsplit

import pytest

from core.graph_client.url_safety import (
    UnsafeGraphUrlError,
    safe_path_segment,
    validate_graph_base_url,
)


# ===========================================================================
# safe_path_segment - VALID inputs (real Graph ids / resource keys)
# ===========================================================================

@pytest.mark.parametrize(
    "value",
    [
        "AAMkADk1",  # short base64url-ish id
        # A realistic long Graph message id (base64url with '-' and '_'... but
        # '_' is allowed; no '/' or '+' since those would be a real-world id's
        # encoded form). This mirrors the shape of actual message ids.
        "AAMkAGI2THVSAAA=",
        "AQMkADAwATM3ZmYAZS0yML",
        "att-001",
        "message_id-with.dots_and-dashes~tilde",
        "id==padding",  # trailing base64 '=' padding
        "a1b2c3",
        "user@contoso.onmicrosoft.com",  # mailbox-style value (allowed chars)
    ],
)
def test_safe_path_segment_accepts_valid_ids(value):
    """Legitimate Graph ids pass and come back percent-encoded (single pass)."""
    encoded = safe_path_segment(value, "message_id")
    # The encoded value must not contain any raw path/url separators.
    assert "/" not in encoded
    assert "?" not in encoded
    assert "#" not in encoded
    # Decoding exactly once must reproduce the original value (round-trips as a
    # single opaque segment).
    from urllib.parse import unquote

    assert unquote(encoded) == value
    # And it must sit in a single path segment when placed in a URL.
    probe = urlsplit(f"https://graph.microsoft.com/v1.0/messages/{encoded}")
    assert probe.netloc == "graph.microsoft.com"
    assert probe.path.count("/messages/") == 1
    assert "/" not in probe.path.split("/messages/")[1]


def test_safe_path_segment_encodes_reserved_chars():
    """Allow-listed reserved chars (@, :, =) are escaped so they can't add structure."""
    encoded = safe_path_segment("a@b:c=d", "message_id")
    assert "@" not in encoded and ":" not in encoded
    assert encoded == "a%40b%3Ac%3Dd"


# ===========================================================================
# safe_path_segment - MALICIOUS / malformed inputs
# ===========================================================================

@pytest.mark.parametrize(
    "value",
    [
        "",  # empty
        "   ",  # whitespace-only
    ],
)
def test_safe_path_segment_rejects_empty(value):
    with pytest.raises(UnsafeGraphUrlError):
        safe_path_segment(value, "message_id")


@pytest.mark.parametrize(
    "value",
    [
        "https://evil.com",  # absolute URL
        "http://evil.com/x",  # absolute URL, alt scheme
        "//evil.com",  # protocol-relative URL
        "//evil.com/path",
        "ftp://evil.com",  # alternate scheme
        "file:///etc/passwd",  # file scheme
        "gopher://evil.com",
    ],
)
def test_safe_path_segment_rejects_url_like_values(value):
    """Absolute, protocol-relative, and alt-scheme URLs must be refused."""
    with pytest.raises(UnsafeGraphUrlError):
        safe_path_segment(value, "message_id")


@pytest.mark.parametrize(
    "value",
    [
        "a/b",  # raw path separator
        "/absolute",
        "a\\b",  # backslash (normalises to '/' on some stacks)
        "..",  # traversal
        "../../etc/passwd",  # traversal
        "a/../b",
        "a?b=c",  # query injection
        "a#frag",  # fragment injection
    ],
)
def test_safe_path_segment_rejects_separators_and_traversal(value):
    with pytest.raises(UnsafeGraphUrlError):
        safe_path_segment(value, "message_id")


@pytest.mark.parametrize(
    "value",
    [
        "%2f",  # encoded '/'
        "%2F",
        "%2e%2e",  # encoded '..'
        "%2e%2e%2f",  # encoded '../'
        "%252f",  # double-encoded '/'
        "%252e%252e",  # double-encoded '..'
        "a%2fb",
    ],
)
def test_safe_path_segment_rejects_encoded_and_double_encoded_payloads(value):
    """Pre-encoded or double-encoded separators/traversal must be refused."""
    with pytest.raises(UnsafeGraphUrlError):
        safe_path_segment(value, "message_id")


@pytest.mark.parametrize(
    "value",
    [
        "%",  # lone percent
        "%zz",  # non-hex escape
        "%2",  # truncated escape
        "abc%g1",  # bad hex digit
        "trailing%",
    ],
)
def test_safe_path_segment_rejects_malformed_percent_escapes(value):
    with pytest.raises(UnsafeGraphUrlError):
        safe_path_segment(value, "message_id")


@pytest.mark.parametrize(
    "value",
    [
        "a b",  # space -> actually allowed char-wise? space IS in allow-list
    ],
)
def test_safe_path_segment_space_is_encoded_not_rejected(value):
    """A space is a legal (if unusual) id char; it must be encoded, not dropped."""
    encoded = safe_path_segment(value, "message_id")
    assert encoded == "a%20b"


@pytest.mark.parametrize(
    "value",
    [
        "a\tb",  # tab
        "a\nb",  # newline
        "a\rb",  # carriage return
        "a\x00b",  # NUL
        "a\x7fb",  # DEL
        "a\x1bb",  # ESC (C0)
        "a\x85b",  # NEL (C1)
    ],
)
def test_safe_path_segment_rejects_control_characters(value):
    with pytest.raises(UnsafeGraphUrlError):
        safe_path_segment(value, "message_id")


@pytest.mark.parametrize(
    "value",
    [
        "a<b",  # angle brackets
        "a>b",
        "a[b]",  # brackets
        'a"b',  # quote
        "a{b}",  # braces
        "a|b",  # pipe
        "a^b",  # caret
        "a`b",  # backtick
    ],
)
def test_safe_path_segment_rejects_disallowed_punctuation(value):
    with pytest.raises(UnsafeGraphUrlError):
        safe_path_segment(value, "message_id")


def test_safe_path_segment_rejects_non_string():
    with pytest.raises(UnsafeGraphUrlError):
        safe_path_segment(None, "message_id")  # type: ignore[arg-type]
    with pytest.raises(UnsafeGraphUrlError):
        safe_path_segment(123, "message_id")  # type: ignore[arg-type]


# ===========================================================================
# validate_graph_base_url - VALID
# ===========================================================================

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://graph.microsoft.com/v1.0", "https://graph.microsoft.com/v1.0"),
        ("https://graph.microsoft.com/v1.0/", "https://graph.microsoft.com/v1.0"),
        ("https://graph.microsoft.com/beta", "https://graph.microsoft.com/beta"),
        ("https://graph.microsoft.com:443/v1.0", "https://graph.microsoft.com:443/v1.0"),
        ("https://graph.microsoft.com", "https://graph.microsoft.com"),
        ("  https://graph.microsoft.com/v1.0  ", "https://graph.microsoft.com/v1.0"),
    ],
)
def test_validate_graph_base_url_accepts_trusted(raw, expected):
    assert validate_graph_base_url(raw) == expected


# ===========================================================================
# validate_graph_base_url - REJECTED
# ===========================================================================

@pytest.mark.parametrize(
    "raw",
    [
        "",  # empty
        "   ",  # whitespace
        "http://graph.microsoft.com/v1.0",  # http downgrade
        "ftp://graph.microsoft.com",  # alt scheme
        "file:///etc/passwd",  # file scheme
        "//graph.microsoft.com/v1.0",  # protocol-relative (no scheme)
        "https://evil.com/v1.0",  # wrong host
        "https://graph.microsoft.com.evil.com/v1.0",  # suffix confusion
        "https://evil.graph.microsoft.com/v1.0",  # subdomain not allowed
        "https://graph.microsoft.com@evil.com/v1.0",  # userinfo -> host is evil.com
        "https://user:pass@graph.microsoft.com/v1.0",  # embedded credentials
        "https://graph.microsoft.com:8443/v1.0",  # unexpected port
        "https://graph.microsoft.com:80/v1.0",  # unexpected port
        "https://graph.microsoft.com/v1.0?x=1",  # query string
        "https://graph.microsoft.com/v1.0#frag",  # fragment
        "https://graph.microsoft.com/../v1.0",  # traversal in base path
    ],
)
def test_validate_graph_base_url_rejects_untrusted(raw):
    with pytest.raises(UnsafeGraphUrlError):
        validate_graph_base_url(raw)


def test_validate_graph_base_url_rejects_non_string():
    with pytest.raises(UnsafeGraphUrlError):
        validate_graph_base_url(None)  # type: ignore[arg-type]


# ===========================================================================
# End-to-end: endpoints.* always resolve to the trusted host, and a crafted
# id can never escape its path segment.
# ===========================================================================

@pytest.fixture
def settings_shared(required_env, monkeypatch):
    from core.config import get_settings

    monkeypatch.setenv("GRAPH_AUTH_MODE", "app_only")
    monkeypatch.setenv("MAILBOX_EMAIL", "shared.mailbox@rblbank.com")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def test_message_detail_url_pins_host_for_valid_id(settings_shared):
    from core.graph_client import endpoints

    url = endpoints.message_detail_url(settings_shared, "AAMkAGI2THVSAAA=")
    parsed = urlsplit(url)
    assert parsed.scheme == "https"
    assert parsed.hostname == "graph.microsoft.com"
    assert "/users/shared.mailbox@rblbank.com/messages/" in url
    # The '=' in the id is encoded so it stays inside the segment.
    assert url.endswith("/messages/AAMkAGI2THVSAAA%3D")


@pytest.mark.parametrize(
    "evil_id",
    [
        "https://evil.com",
        "//evil.com",
        "../../../../etc/passwd",
        "x/../../y",
        "%2e%2e%2fadmin",
        "%252e%252e",
        "a?to=evil",
        "a#evil",
        "id\x00null",
        "id with space and /slash",
    ],
)
def test_message_detail_url_rejects_crafted_ids(settings_shared, evil_id):
    """A crafted message_id must raise before any URL is built."""
    from core.graph_client import endpoints

    with pytest.raises(UnsafeGraphUrlError):
        endpoints.message_detail_url(settings_shared, evil_id)


@pytest.mark.parametrize(
    "evil_id",
    [
        "https://evil.com",
        "//evil.com",
        "../../secret",
        "%2fetc%2fpasswd",
    ],
)
def test_attachment_detail_url_rejects_crafted_ids(settings_shared, evil_id):
    from core.graph_client import endpoints

    # Bad message_id.
    with pytest.raises(UnsafeGraphUrlError):
        endpoints.attachment_detail_url(settings_shared, evil_id, "att-ok")
    # Bad attachment_id.
    with pytest.raises(UnsafeGraphUrlError):
        endpoints.attachment_detail_url(settings_shared, "msg-ok", evil_id)


def test_all_builders_resolve_to_trusted_host(settings_shared):
    """Every endpoint builder's output resolves to the pinned Graph host."""
    from core.graph_client import endpoints

    urls = [
        endpoints.me_url(settings_shared),
        endpoints.messages_url(settings_shared),
        endpoints.message_detail_url(settings_shared, "msg-1"),
        endpoints.message_attachments_url(settings_shared, "msg-1"),
        endpoints.attachment_detail_url(settings_shared, "msg-1", "att-1"),
        endpoints.calendar_events_url(settings_shared),
        endpoints.calendar_view_url(settings_shared),
    ]
    for url in urls:
        parsed = urlsplit(url)
        assert parsed.scheme == "https", url
        assert parsed.hostname == "graph.microsoft.com", url


# ===========================================================================
# validate_graph_request_url - the single last-line egress guard
# ===========================================================================
#
# This is the canonical structured validator that EVERY outbound request in
# `GraphClient` funnels through (the main `_get`/`_request_once` path, the
# SharePoint `/content` fallback, and server-returned `@odata.nextLink`
# pagination URLs). It is also what the thin `graph_url.validate_graph_url`
# shim delegates to. These tests prove it accepts only fully-formed URLs
# resolving to the pinned Graph origin and rejects the full SSRF payload
# matrix.

from core.graph_client.url_safety import validate_graph_request_url  # noqa: E402


@pytest.mark.parametrize(
    "url",
    [
        "https://graph.microsoft.com/v1.0/me/messages",
        "https://graph.microsoft.com/v1.0/me/messages?$top=10&$skip=20",
        "https://graph.microsoft.com/v1.0/drives/abc/items/def/children?$skiptoken=xyz",
        "https://graph.microsoft.com:443/v1.0/me",  # explicit default port
        "https://graph.microsoft.com/beta/me",
    ],
)
def test_validate_graph_request_url_accepts_trusted(url):
    """A well-formed URL on the pinned origin is returned unchanged."""
    assert validate_graph_request_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        # -- empty / non-string --
        "",
        "   ",
        # -- untrusted hosts (incl. suffix/subdomain confusion) --
        "https://evil.com/v1.0/me",
        "https://graph.microsoft.com.evil.com/v1.0/me",
        "https://evil.graph.microsoft.com/v1.0/me",
        # -- userinfo trick: hostname resolves to evil.com --
        "https://graph.microsoft.com@evil.com/v1.0/me",
        # -- embedded credentials on the right host --
        "https://user:pass@graph.microsoft.com/v1.0/me",
        "https://admin@graph.microsoft.com/v1.0/me",
        # -- protocol-relative (no scheme) --
        "//evil.com/v1.0/me",
        "//graph.microsoft.com/v1.0/me",
        # -- alternate / downgraded schemes --
        "http://graph.microsoft.com/v1.0/me",
        "ftp://graph.microsoft.com/v1.0/me",
        "file:///etc/passwd",
        "gopher://evil.com",
        # -- unexpected ports on the right host --
        "https://graph.microsoft.com:8080/v1.0/me",
        "https://graph.microsoft.com:80/v1.0/me",
    ],
)
def test_validate_graph_request_url_rejects_untrusted(url):
    with pytest.raises(UnsafeGraphUrlError):
        validate_graph_request_url(url)


def test_validate_graph_request_url_rejects_non_string():
    with pytest.raises(UnsafeGraphUrlError):
        validate_graph_request_url(None)  # type: ignore[arg-type]


# ===========================================================================
# GraphClient egress guard - hostile URLs never leave the process
# ===========================================================================
#
# `_assert_safe_graph_url` is the client-level wrapper around
# `validate_graph_request_url` that translates a rejection into the client's
# public `GraphApiError`. These prove the guard fires at the actual egress
# method (`_get`) for the headline SSRF payloads, matching the "validation
# failed" contract the client exposes.

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hostile_url",
    [
        "https://evil.com/v1.0/me",
        "https://graph.microsoft.com.evil.com/v1.0/me",
        "http://graph.microsoft.com/v1.0/me",
        "//evil.com/v1.0/me",
        "https://graph.microsoft.com@evil.com/v1.0/me",
        "https://user:pass@graph.microsoft.com/v1.0/me",
        "https://graph.microsoft.com:8080/v1.0/me",
        "file:///etc/passwd",
    ],
)
async def test_client_get_rejects_hostile_urls(required_env, monkeypatch, hostile_url):
    from core.config import get_settings
    from core.exceptions import GraphApiError
    from core.graph_client.client import GraphClient

    monkeypatch.setenv("GRAPH_AUTH_MODE", "delegated")
    get_settings.cache_clear()
    client = GraphClient(settings=get_settings())
    with pytest.raises(GraphApiError, match="validation failed"):
        await client._get(hostile_url, "fake-token")
    get_settings.cache_clear()


# ===========================================================================
# graph_url shim stays behaviourally aligned with the canonical module
# ===========================================================================
#
# The `graph_url` module is now a thin shim over `url_safety`. These guard
# against the two modules drifting apart again (the redundancy this change
# removed).

def test_graph_url_shim_delegates_to_canonical():
    from core.graph_client import graph_url as shim
    from core.graph_client import url_safety as canonical

    # Same pinned host and validated base.
    assert shim.APPROVED_HOSTNAME in canonical.ALLOWED_GRAPH_HOSTS
    assert shim.GRAPH_BASE_URL == canonical.validate_graph_base_url(
        "https://graph.microsoft.com/v1.0"
    )

    # safe_segment delegates to safe_path_segment for a normal id.
    assert shim.safe_segment("AAMkAGI2THVSAAA=") == canonical.safe_path_segment(
        "AAMkAGI2THVSAAA=", "segment"
    )

    # validate_graph_url delegates to the egress guard for a trusted URL.
    url = "https://graph.microsoft.com/v1.0/me/messages"
    assert shim.validate_graph_url(url) == canonical.validate_graph_request_url(url)


def test_graph_url_shim_rejects_same_payloads_as_canonical():
    from core.graph_client import graph_url as shim

    # Hostile request URL rejected by the shim (delegating to egress guard).
    with pytest.raises(UnsafeGraphUrlError):
        shim.validate_graph_url("https://evil.com/v1.0/me")

    # '@' is rejected by the shim's (stricter) safe_segment contract.
    with pytest.raises(UnsafeGraphUrlError):
        shim.safe_segment("user@evil.com")

    # Traversal / separators rejected just like the canonical function.
    for payload in ("../../etc", "a/b", "%2e%2e%2f", "%252f", "foo%2"):
        with pytest.raises(UnsafeGraphUrlError):
            shim.safe_segment(payload)
