"""
tests/core/test_graph_url.py
================================
Focused unit tests for the centralized Graph URL validation module
(``core.graph_client.graph_url``), verifying that the SSRF mitigations
(CWE-918 / CodeQL py/partial-ssrf) work correctly.

Test categories:
1. Base URL validation
2. safe_segment — valid Graph IDs
3. safe_segment — rejection of dangerous inputs
4. graph_url — path construction
5. validate_graph_url — external URL validation
6. Integration: endpoints.py produces safe URLs
"""

import pytest


# =====================================================================
# 1. Base URL validation (_validate_base_url)
# =====================================================================

class TestBaseUrlValidation:
    """The base URL is validated at import time. These tests exercise
    the validator directly with alternate inputs."""

    def test_approved_base_url_passes(self):
        from core.graph_client.graph_url import _validate_base_url
        result = _validate_base_url("https://graph.microsoft.com/v1.0")
        assert result == "https://graph.microsoft.com/v1.0"

    def test_http_scheme_rejected(self):
        from core.graph_client.graph_url import _validate_base_url
        with pytest.raises(ValueError, match="HTTPS"):
            _validate_base_url("http://graph.microsoft.com/v1.0")

    def test_wrong_hostname_rejected(self):
        from core.graph_client.graph_url import _validate_base_url
        with pytest.raises(ValueError, match="graph.microsoft.com"):
            _validate_base_url("https://evil.com/v1.0")

    def test_embedded_credentials_rejected(self):
        from core.graph_client.graph_url import _validate_base_url
        with pytest.raises(ValueError, match="credentials"):
            _validate_base_url("https://user:pass@graph.microsoft.com/v1.0")

    def test_nonstandard_port_rejected(self):
        from core.graph_client.graph_url import _validate_base_url
        with pytest.raises(ValueError, match="port"):
            _validate_base_url("https://graph.microsoft.com:8443/v1.0")

    def test_query_string_rejected(self):
        from core.graph_client.graph_url import _validate_base_url
        with pytest.raises(ValueError, match="query"):
            _validate_base_url("https://graph.microsoft.com/v1.0?foo=bar")

    def test_fragment_rejected(self):
        from core.graph_client.graph_url import _validate_base_url
        with pytest.raises(ValueError, match="fragment"):
            _validate_base_url("https://graph.microsoft.com/v1.0#section")

    def test_default_port_443_accepted(self):
        from urllib.parse import urlparse
        from core.graph_client.graph_url import _validate_base_url
        result = _validate_base_url("https://graph.microsoft.com:443/v1.0")
        # Validate via the parsed hostname rather than a substring check,
        # which would also match e.g. "https://graph.microsoft.com.evil.com".
        assert urlparse(result).hostname == "graph.microsoft.com"

    def test_ftp_scheme_rejected(self):
        from core.graph_client.graph_url import _validate_base_url
        with pytest.raises(ValueError, match="HTTPS"):
            _validate_base_url("ftp://graph.microsoft.com/v1.0")

    def test_file_scheme_rejected(self):
        from core.graph_client.graph_url import _validate_base_url
        with pytest.raises(ValueError, match="HTTPS"):
            _validate_base_url("file:///etc/passwd")


# =====================================================================
# 2. safe_segment — valid Graph IDs
# =====================================================================

class TestSafeSegmentValid:
    """Typical Graph API IDs should pass validation and be encoded."""

    def test_simple_alphanumeric_id(self):
        from core.graph_client.graph_url import safe_segment
        assert safe_segment("AAMkAGI2TG93") == "AAMkAGI2TG93"

    def test_id_with_hyphens(self):
        from core.graph_client.graph_url import safe_segment
        assert safe_segment("550e8400-e29b-41d4-a716-446655440000") == "550e8400-e29b-41d4-a716-446655440000"

    def test_id_with_equals_sign(self):
        """Graph message IDs often end with ``=`` (base64 padding)."""
        from core.graph_client.graph_url import safe_segment
        result = safe_segment("AAMkAGI2TG93AAA=")
        assert "AAMkAGI2TG93AAA" in result

    def test_id_with_underscores(self):
        from core.graph_client.graph_url import safe_segment
        assert safe_segment("drive_item_123") == "drive_item_123"

    def test_id_with_periods(self):
        from core.graph_client.graph_url import safe_segment
        assert safe_segment("v1.0") == "v1.0"

    def test_email_address_as_mailbox(self):
        """Mailbox email addresses like ``user@rblbank.com`` are used in
        ``/users/{email}`` paths — the ``@`` is forbidden by safe_segment
        because it can be used for credential injection in URLs."""
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("shared.mailbox@rblbank.com", "mailbox_email")

    def test_sharing_url_token(self):
        """Base64url-encoded sharing tokens (``u!aHR0cHM6...``) should pass."""
        from core.graph_client.graph_url import safe_segment
        result = safe_segment("u!aHR0cHM6Ly9yYmxiYW5r")
        assert result == "u!aHR0cHM6Ly9yYmxiYW5r"


# =====================================================================
# 3. safe_segment — rejection of dangerous inputs
# =====================================================================

class TestSafeSegmentDangerous:
    """Inputs that could cause SSRF or path traversal must be rejected."""

    def test_empty_string_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="empty"):
            safe_segment("")

    def test_whitespace_only_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="empty"):
            safe_segment("   ")

    # -- Path traversal --

    def test_dot_dot_slash_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("../../etc/passwd")

    def test_dot_dot_no_slash_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("..")

    def test_dot_dot_backslash_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("..\\windows\\system32")

    # -- Path separators --

    def test_forward_slash_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("foo/bar")

    def test_backslash_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("foo\\bar")

    # -- Encoded separators --

    def test_encoded_forward_slash_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("foo%2Fbar")

    def test_encoded_forward_slash_uppercase_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("foo%2fbar")

    def test_encoded_backslash_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("foo%5Cbar")

    def test_encoded_backslash_lowercase_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("foo%5cbar")

    def test_encoded_dot_dot_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("%2E%2E")

    # -- Double encoding --

    def test_double_encoded_slash_rejected(self):
        """``%252F`` decodes to ``%2F`` which decodes to ``/``.
        The first layer (``%25`` → ``%``) produces ``%2F`` which is
        caught by the encoded-slash check."""
        from core.graph_client.graph_url import safe_segment
        # %252F contains %25 which is a valid percent-escape, but the
        # result %2F is caught
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("%252F")

    # -- Query / fragment injection --

    def test_question_mark_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("id?$expand=children")

    def test_hash_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("id#fragment")

    # -- Credential injection --

    def test_at_sign_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("user@evil.com")

    # -- Malformed percent-escapes --

    def test_bare_percent_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("100%done")

    def test_percent_one_hex_digit_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("foo%2")

    def test_percent_non_hex_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("foo%ZZ")

    # -- Control characters --

    def test_null_byte_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("foo\x00bar")

    def test_newline_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("foo\nbar")

    def test_tab_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("foo\tbar")

    def test_carriage_return_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("foo\rbar")

    def test_del_character_rejected(self):
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("foo\x7fbar")

    # -- Absolute URL as ID --

    def test_absolute_https_url_rejected(self):
        """An attacker might submit a full URL as a message_id."""
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("https://evil.com/steal")

    def test_protocol_relative_url_rejected(self):
        """``//evil.com`` is a protocol-relative URL."""
        from core.graph_client.graph_url import safe_segment
        with pytest.raises(ValueError, match="forbidden"):
            safe_segment("//evil.com/steal")

    # -- Whitespace in IDs --

    def test_space_in_id_encoded(self):
        """Spaces are not rejected (some IDs could have them) but are encoded."""
        from core.graph_client.graph_url import safe_segment
        # Space is not in the dangerous pattern, but quote() will encode it
        result = safe_segment("id with space")
        assert " " not in result
        assert "%20" in result


# =====================================================================
# 4. graph_url — path construction
# =====================================================================

class TestGraphUrl:
    """Verify that ``graph_url()`` correctly joins the trusted base."""

    def test_basic_path(self):
        from core.graph_client.graph_url import graph_url
        result = graph_url("/me/messages")
        assert result == "https://graph.microsoft.com/v1.0/me/messages"

    def test_path_must_start_with_slash(self):
        from core.graph_client.graph_url import graph_url
        with pytest.raises(ValueError, match="start with '/'"):
            graph_url("me/messages")

    def test_path_with_encoded_segment(self):
        from core.graph_client.graph_url import graph_url, safe_segment
        mid = safe_segment("AAMkAGI2TG93")
        result = graph_url(f"/me/messages/{mid}")
        assert result == "https://graph.microsoft.com/v1.0/me/messages/AAMkAGI2TG93"


# =====================================================================
# 5. validate_graph_url — external URL validation
# =====================================================================

class TestValidateGraphUrl:
    """Verify that ``validate_graph_url()`` accepts legitimate Graph
    URLs and rejects everything else (used for @odata.nextLink, etc.)."""

    def test_valid_graph_url(self):
        from core.graph_client.graph_url import validate_graph_url
        url = "https://graph.microsoft.com/v1.0/me/messages?$top=10&$skip=20"
        assert validate_graph_url(url) == url

    def test_valid_graph_url_with_odata_params(self):
        from core.graph_client.graph_url import validate_graph_url
        url = "https://graph.microsoft.com/v1.0/drives/abc/items/def/children?$skiptoken=xyz"
        assert validate_graph_url(url) == url

    # -- Hostile hosts --

    def test_evil_com_rejected(self):
        from core.graph_client.graph_url import validate_graph_url
        with pytest.raises(ValueError, match="hostname"):
            validate_graph_url("https://evil.com/v1.0/me/messages")

    def test_subdomain_of_graph_rejected(self):
        """``evil.graph.microsoft.com`` is NOT ``graph.microsoft.com``."""
        from core.graph_client.graph_url import validate_graph_url
        with pytest.raises(ValueError, match="hostname"):
            validate_graph_url("https://evil.graph.microsoft.com/v1.0/me")

    def test_graph_as_subdomain_of_evil_rejected(self):
        """``graph.microsoft.com.evil.com`` should fail."""
        from core.graph_client.graph_url import validate_graph_url
        with pytest.raises(ValueError, match="hostname"):
            validate_graph_url("https://graph.microsoft.com.evil.com/v1.0/me")

    # -- Protocol-relative URL --

    def test_protocol_relative_rejected(self):
        from core.graph_client.graph_url import validate_graph_url
        with pytest.raises(ValueError, match="HTTPS"):
            validate_graph_url("//evil.com/v1.0/me/messages")

    # -- Alternate schemes --

    def test_http_rejected(self):
        from core.graph_client.graph_url import validate_graph_url
        with pytest.raises(ValueError, match="HTTPS"):
            validate_graph_url("http://graph.microsoft.com/v1.0/me/messages")

    def test_ftp_rejected(self):
        from core.graph_client.graph_url import validate_graph_url
        with pytest.raises(ValueError, match="HTTPS"):
            validate_graph_url("ftp://graph.microsoft.com/v1.0/me")

    def test_file_rejected(self):
        from core.graph_client.graph_url import validate_graph_url
        with pytest.raises(ValueError, match="HTTPS"):
            validate_graph_url("file:///etc/passwd")

    def test_gopher_rejected(self):
        from core.graph_client.graph_url import validate_graph_url
        with pytest.raises(ValueError, match="HTTPS"):
            validate_graph_url("gopher://evil.com")

    # -- Embedded credentials --

    def test_credentials_in_url_rejected(self):
        from core.graph_client.graph_url import validate_graph_url
        with pytest.raises(ValueError, match="credentials"):
            validate_graph_url("https://admin:pass@graph.microsoft.com/v1.0/me")

    def test_user_only_credential_rejected(self):
        from core.graph_client.graph_url import validate_graph_url
        with pytest.raises(ValueError, match="credentials"):
            validate_graph_url("https://admin@graph.microsoft.com/v1.0/me")

    # -- Unexpected port --

    def test_nonstandard_port_rejected(self):
        from core.graph_client.graph_url import validate_graph_url
        with pytest.raises(ValueError, match="port"):
            validate_graph_url("https://graph.microsoft.com:8080/v1.0/me")

    def test_port_443_accepted(self):
        from core.graph_client.graph_url import validate_graph_url
        url = "https://graph.microsoft.com:443/v1.0/me/messages"
        assert validate_graph_url(url) == url

    # -- Empty / blank --

    def test_empty_url_rejected(self):
        from core.graph_client.graph_url import validate_graph_url
        with pytest.raises(ValueError, match="empty"):
            validate_graph_url("")

    def test_blank_url_rejected(self):
        from core.graph_client.graph_url import validate_graph_url
        with pytest.raises(ValueError, match="empty"):
            validate_graph_url("   ")


# =====================================================================
# 6. Integration: endpoints produce safe URLs
# =====================================================================

class TestEndpointsIntegration:
    """Verify that ``endpoints.py`` functions produce URLs that pass
    ``validate_graph_url()`` and that dangerous IDs are rejected."""

    @pytest.fixture
    def settings(self, required_env, monkeypatch):
        from core.config import get_settings
        monkeypatch.setenv("GRAPH_AUTH_MODE", "delegated")
        monkeypatch.setenv("MAILBOX_EMAIL", "")
        get_settings.cache_clear()
        yield get_settings()
        get_settings.cache_clear()

    def test_message_detail_url_valid_id(self, settings):
        from core.graph_client import endpoints
        from core.graph_client.graph_url import validate_graph_url
        url = endpoints.message_detail_url(settings, "AAMkAGI2TG93AAA=")
        validate_graph_url(url)  # should not raise
        assert "AAMkAGI2TG93AAA" in url

    def test_message_detail_url_rejects_traversal(self, settings):
        from core.graph_client import endpoints
        with pytest.raises(ValueError, match="forbidden"):
            endpoints.message_detail_url(settings, "../../admin")

    def test_message_detail_url_rejects_slash(self, settings):
        from core.graph_client import endpoints
        with pytest.raises(ValueError, match="forbidden"):
            endpoints.message_detail_url(settings, "id/../../admin")

    def test_attachment_detail_url_rejects_hostile_ids(self, settings):
        from core.graph_client import endpoints
        with pytest.raises(ValueError, match="forbidden"):
            endpoints.attachment_detail_url(settings, "valid-msg", "att%2F../../etc")

    def test_drive_item_children_url_valid(self):
        from core.graph_client import endpoints
        from core.graph_client.graph_url import validate_graph_url
        url = endpoints.drive_item_children_url("b!driveId123", "01ITEM456")
        validate_graph_url(url)
        assert "b!driveId123" in url
        assert "01ITEM456" in url

    def test_drive_item_children_url_rejects_traversal_drive_id(self):
        from core.graph_client import endpoints
        with pytest.raises(ValueError, match="forbidden"):
            endpoints.drive_item_children_url("../other", "01ITEM456")

    def test_drive_item_children_url_rejects_empty_item_id(self):
        from core.graph_client import endpoints
        with pytest.raises(ValueError, match="empty"):
            endpoints.drive_item_children_url("b!driveId123", "")

    def test_shares_driveitem_url_produces_valid_url(self):
        from core.graph_client import endpoints
        from core.graph_client.graph_url import validate_graph_url
        url = endpoints.shares_driveitem_url("https://rblbank.sharepoint.com/sites/CEO/docs/folder1")
        validate_graph_url(url)
        assert "/shares/" in url
        assert "/driveItem" in url

    def test_messages_url_valid(self, settings):
        from core.graph_client import endpoints
        from core.graph_client.graph_url import validate_graph_url
        url = endpoints.messages_url(settings)
        validate_graph_url(url)
        assert "/mailFolders/inbox/messages" in url

    def test_calendar_events_url_valid(self, settings):
        from core.graph_client import endpoints
        from core.graph_client.graph_url import validate_graph_url
        url = endpoints.calendar_events_url(settings)
        validate_graph_url(url)
        assert "/events" in url


# =====================================================================
# 7. Integration: client._get rejects hostile URLs
# =====================================================================

class TestClientSsrfGuard:
    """Verify that ``GraphClient._get()`` blocks requests to non-Graph hosts."""

    @pytest.mark.asyncio
    async def test_get_rejects_evil_host(self, required_env, monkeypatch):
        from core.config import get_settings
        from core.exceptions import GraphApiError
        from core.graph_client.client import GraphClient

        monkeypatch.setenv("GRAPH_AUTH_MODE", "delegated")
        get_settings.cache_clear()
        client = GraphClient(settings=get_settings())
        with pytest.raises(GraphApiError, match="validation failed"):
            await client._get("https://evil.com/v1.0/me", "fake-token")
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_get_rejects_http_scheme(self, required_env, monkeypatch):
        from core.config import get_settings
        from core.exceptions import GraphApiError
        from core.graph_client.client import GraphClient

        monkeypatch.setenv("GRAPH_AUTH_MODE", "delegated")
        get_settings.cache_clear()
        client = GraphClient(settings=get_settings())
        with pytest.raises(GraphApiError, match="validation failed"):
            await client._get("http://graph.microsoft.com/v1.0/me", "fake-token")
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_get_rejects_credentials_in_url(self, required_env, monkeypatch):
        from core.config import get_settings
        from core.exceptions import GraphApiError
        from core.graph_client.client import GraphClient

        monkeypatch.setenv("GRAPH_AUTH_MODE", "delegated")
        get_settings.cache_clear()
        client = GraphClient(settings=get_settings())
        with pytest.raises(GraphApiError, match="validation failed"):
            await client._get("https://admin:pass@graph.microsoft.com/v1.0/me", "fake-token")
        get_settings.cache_clear()
