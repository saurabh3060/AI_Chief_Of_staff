"""
tests/core/test_graph_client.py
==================================
Verifies `GraphClient` behaviour for both auth modes: delegated mode
requires a caller-supplied bearer token, app-only mode acquires its own
token via MSAL; URL building respects the configured shared mailbox;
and retry triggers on simulated connection/timeout errors.

Outbound HTTP calls are mocked via `httpx.MockTransport` so no real
network access is required.
"""

import httpx
import pytest


@pytest.fixture
def settings_delegated(required_env, monkeypatch):
    """Settings configured for delegated auth mode, with a shared mailbox set."""
    from core.config import get_settings

    monkeypatch.setenv("GRAPH_AUTH_MODE", "delegated")
    monkeypatch.setenv("MAILBOX_EMAIL", "shared.mailbox@rblbank.com")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def settings_app_only(required_env, monkeypatch):
    """Settings configured for app-only client-credentials auth mode."""
    from core.config import get_settings

    monkeypatch.setenv("GRAPH_AUTH_MODE", "app_only")
    monkeypatch.setenv("MAILBOX_EMAIL", "shared.mailbox@rblbank.com")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def _mock_transport(json_body: dict, status_code: int = 200) -> httpx.MockTransport:
    """Build an httpx.MockTransport that always returns the given JSON body/status."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=json_body)

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# URL building (mailbox-aware)
# ---------------------------------------------------------------------------

def test_messages_url_uses_shared_mailbox_when_configured(settings_delegated):
    from core.graph_client import endpoints

    url = endpoints.messages_url(settings_delegated)
    assert "/users/shared.mailbox@rblbank.com/mailFolders/inbox/messages" in url


def test_messages_url_uses_me_when_no_shared_mailbox(required_env, monkeypatch):
    from core.config import get_settings
    from core.graph_client import endpoints

    monkeypatch.setenv("MAILBOX_EMAIL", "")
    get_settings.cache_clear()
    settings = get_settings()

    url = endpoints.messages_url(settings)
    assert "/me/mailFolders/inbox/messages" in url
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Delegated mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delegated_mode_requires_bearer_token(settings_delegated):
    """Calling a GraphClient method in delegated mode without a bearer token should raise GraphApiError."""
    from core.exceptions import GraphApiError
    from core.graph_client.client import GraphClient

    client = GraphClient(settings=settings_delegated)
    with pytest.raises(GraphApiError):
        await client.get_messages(bearer_token=None)


@pytest.mark.asyncio
async def test_delegated_mode_get_messages_returns_parsed_json(settings_delegated, monkeypatch, mail_thread_fixture):
    """With a bearer token supplied, get_messages() should return the mocked Graph API response."""
    import httpx as httpx_module

    from core.graph_client.client import GraphClient

    transport = _mock_transport(mail_thread_fixture)

    class MockAsyncClient(httpx_module.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx_module, "AsyncClient", MockAsyncClient)

    client = GraphClient(settings=settings_delegated)
    result = await client.get_messages(bearer_token="fake-delegated-token")
    assert result == mail_thread_fixture


# ---------------------------------------------------------------------------
# App-only mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_app_only_mode_acquires_its_own_token(settings_app_only, monkeypatch, mail_thread_fixture):
    """In app-only mode, get_messages() should succeed without a bearer_token, using an internally acquired token."""
    import httpx as httpx_module

    from core.graph_client import client as client_module
    from core.graph_client.client import GraphClient

    # client.py imports acquire_app_only_token by name, so it must be
    # patched on the client module itself (not the auth module) to take
    # effect there.
    monkeypatch.setattr(client_module, "acquire_app_only_token", lambda settings: "fake-app-only-token")

    transport = _mock_transport(mail_thread_fixture)

    class MockAsyncClient(httpx_module.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx_module, "AsyncClient", MockAsyncClient)

    client = GraphClient(settings=settings_app_only)
    result = await client.get_messages(bearer_token=None)
    assert result == mail_thread_fixture


def test_acquire_app_only_token_raises_on_msal_error(settings_app_only, monkeypatch):
    """If MSAL fails to acquire an app-only token, a clear RuntimeError should be raised."""
    from core.graph_client import auth as auth_module

    class FakeMsalApp:
        def acquire_token_for_client(self, scopes):
            return {"error": "invalid_client", "error_description": "Invalid client secret provided."}

    monkeypatch.setattr(auth_module, "build_msal_app", lambda settings: FakeMsalApp())

    with pytest.raises(RuntimeError):
        auth_module.acquire_app_only_token(settings_app_only)


# ---------------------------------------------------------------------------
# Retry on transient failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_retries_on_connect_error_then_succeeds(settings_delegated, monkeypatch, mail_thread_fixture):
    """A transient httpx.ConnectError should be retried and eventually succeed."""
    import httpx as httpx_module

    from core import retry as retry_module
    from core.graph_client import client as client_module

    async def instant_sleep(_):
        return None

    monkeypatch.setattr(retry_module.asyncio, "sleep", instant_sleep)

    call_count = {"n": 0}

    def handler(request: httpx_module.Request) -> httpx_module.Response:
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise httpx_module.ConnectError("simulated connection drop", request=request)
        return httpx_module.Response(200, json=mail_thread_fixture)

    transport = httpx_module.MockTransport(handler)

    class MockAsyncClient(httpx_module.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx_module, "AsyncClient", MockAsyncClient)

    client = client_module.GraphClient(settings=settings_delegated)
    result = await client.get_messages(bearer_token="fake-token")
    assert result == mail_thread_fixture
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_local_mode_reads_eml_files_without_graph(tmp_path, required_env, monkeypatch):
    """GRAPH_AUTH_MODE=local should serve Graph-shaped messages from .eml files."""
    from core.config import get_settings
    from core.graph_client.client import GraphClient
    from core.graph_client import local_mailbox as local_mailbox_module

    eml_path = tmp_path / "sample.eml"
    eml_path.write_bytes(
        b"From: Wholesale Head <wholesale.head@rblbank.com>\r\n"
        b"To: DigitalOffice AICoS <digitaloffice.aicos@rblbank.com>\r\n"
        b"Subject: WB Monthly Review - August\r\n"
        b"Date: Wed, 16 Sep 2026 21:13:20 +0000\r\n"
        b"Message-ID: <local-test-001@example.com>\r\n"
        b"MIME-Version: 1.0\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n"
        b"\r\n"
        b"Action: Circulate the August WB review pack.\r\n"
    )

    empty_calendar = tmp_path / "empty_calendar.json"
    empty_calendar.write_text('{"meetings": []}', encoding="utf-8")

    monkeypatch.setenv("GRAPH_AUTH_MODE", "local")
    monkeypatch.setenv("EML_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_CALENDAR_PATH", str(empty_calendar))
    get_settings.cache_clear()
    local_mailbox_module._mailbox_cache = None

    client = GraphClient(settings=get_settings())
    result = await client.get_messages(bearer_token=None)
    messages = result["value"]
    assert len(messages) == 1
    assert messages[0]["subject"] == "WB Monthly Review - August"
    assert messages[0]["from"]["emailAddress"]["address"] == "wholesale.head@rblbank.com"

    calendar = await client.get_calendar_events(bearer_token=None)
    assert calendar == {"value": []}
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_local_mode_serves_dummy_calendar_events(tmp_path, required_env, monkeypatch):
    """GRAPH_AUTH_MODE=local should serve Graph-shaped dummy meetings from JSON."""
    from datetime import datetime, timedelta, timezone

    from core.config import get_settings
    from core.graph_client.client import GraphClient
    from core.graph_client import local_calendar as local_calendar_module

    calendar_path = tmp_path / "local_calendar.json"
    calendar_path.write_text(
        """
        {
          "meetings": [
            {
              "id": "local-evt-wb-mbr-sept",
              "subject": "Wholesale banking monthly business review Sept",
              "start_ist": "10:00",
              "end_ist": "11:00",
              "organizer_name": "Ashutosh Verma",
              "organizer_email": "Ashutosh.verma1@rbl.bank.in",
              "attendee_emails": ["Ashutosh.verma1@rbl.bank.in"]
            },
            {
              "id": "local-evt-corporate-credit-tat",
              "subject": "Corporate credit Tat Review",
              "start_ist": "12:30",
              "end_ist": "13:30",
              "organizer_name": "Ashutosh Verma",
              "organizer_email": "Ashutosh.verma1@rbl.bank.in",
              "attendee_emails": ["Ashutosh.verma1@rbl.bank.in"]
            },
            {
              "id": "local-evt-wb-liabilities-deposit-mix",
              "subject": "wholesale liabilities and deposit mix review",
              "start_ist": "15:00",
              "end_ist": "16:00",
              "organizer_name": "Ashutosh Verma",
              "organizer_email": "Ashutosh.verma1@rbl.bank.in",
              "attendee_emails": ["Ashutosh.verma1@rbl.bank.in"]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    monkeypatch.setenv("GRAPH_AUTH_MODE", "local")
    monkeypatch.setenv("LOCAL_CALENDAR_PATH", str(calendar_path))
    get_settings.cache_clear()
    local_calendar_module._calendar_cache = None

    client = GraphClient(settings=get_settings())
    calendar = await client.get_calendar_events(bearer_token=None)
    events = calendar["value"]
    assert [event["subject"] for event in events] == [
        "Wholesale banking monthly business review Sept",
        "Corporate credit Tat Review",
        "wholesale liabilities and deposit mix review",
    ]
    assert all(
        event["attendees"][0]["emailAddress"]["address"] == "Ashutosh.verma1@rbl.bank.in"
        for event in events
    )

    ist = timezone(timedelta(hours=5, minutes=30))
    day_start = datetime.now(ist).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    day_end = day_start + timedelta(days=1)
    view = await client.get_calendar_view(
        start_date_time=day_start.isoformat(),
        end_date_time=day_end.isoformat(),
        bearer_token=None,
    )
    assert len(view["value"]) == 3

    empty_view = await client.get_calendar_view(
        start_date_time=(day_start + timedelta(days=1)).isoformat(),
        end_date_time=(day_end + timedelta(days=1)).isoformat(),
        bearer_token=None,
    )
    assert empty_view == {"value": []}
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_get_raises_graph_api_error_on_http_error_status(settings_delegated, monkeypatch):
    """A non-2xx HTTP response should raise GraphApiError with the response body included."""
    import httpx as httpx_module

    from core.exceptions import GraphApiError
    from core.graph_client.client import GraphClient

    def handler(request: httpx_module.Request) -> httpx_module.Response:
        return httpx_module.Response(403, json={"error": {"message": "Access denied"}})

    transport = httpx_module.MockTransport(handler)

    class MockAsyncClient(httpx_module.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx_module, "AsyncClient", MockAsyncClient)

    client = GraphClient(settings=settings_delegated)
    with pytest.raises(GraphApiError):
        await client.get_messages(bearer_token="fake-token")


# ---------------------------------------------------------------------------
# In-flight 401 refresh + retry (TokenProvider)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_401_triggers_single_refresh_and_retry(settings_delegated, monkeypatch, mail_thread_fixture):
    """
    A 401 mid-run should trigger ONE token refresh (force_refresh=True) via the
    provider, then retry the request once with the new token and succeed.
    """
    import httpx as httpx_module

    from core.graph_client.client import GraphClient

    call_log = {"headers": [], "n": 0}

    def handler(request: httpx_module.Request) -> httpx_module.Response:
        call_log["n"] += 1
        call_log["headers"].append(request.headers.get("Authorization"))
        # First call (stale token) -> 401; second call (refreshed) -> 200.
        if call_log["n"] < 2:
            return httpx_module.Response(401, json={"error": {"message": "expired"}})
        return httpx_module.Response(200, json=mail_thread_fixture)

    transport = httpx_module.MockTransport(handler)

    class MockAsyncClient(httpx_module.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx_module, "AsyncClient", MockAsyncClient)

    provider_calls = {"force_refresh": []}

    async def provider(*, force_refresh: bool = False):
        provider_calls["force_refresh"].append(force_refresh)
        return "fresh-token"

    client = GraphClient(settings=settings_delegated, token_provider=provider)
    result = await client.get_messages(bearer_token="stale-token")

    assert result == mail_thread_fixture
    assert call_log["n"] == 2  # one failed call + one retry
    assert provider_calls["force_refresh"] == [True]  # refreshed exactly once, forced
    assert call_log["headers"][0] == "Bearer stale-token"
    assert call_log["headers"][1] == "Bearer fresh-token"


@pytest.mark.asyncio
async def test_401_without_provider_propagates(settings_delegated, monkeypatch):
    """With no token provider, a 401 should surface as GraphApiError (no retry)."""
    import httpx as httpx_module

    from core.exceptions import GraphApiError
    from core.graph_client.client import GraphClient

    calls = {"n": 0}

    def handler(request: httpx_module.Request) -> httpx_module.Response:
        calls["n"] += 1
        return httpx_module.Response(401, json={"error": {"message": "expired"}})

    transport = httpx_module.MockTransport(handler)

    class MockAsyncClient(httpx_module.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx_module, "AsyncClient", MockAsyncClient)

    client = GraphClient(settings=settings_delegated)  # no provider
    with pytest.raises(GraphApiError):
        await client.get_messages(bearer_token="stale-token")
    assert calls["n"] == 1  # no retry attempted


@pytest.mark.asyncio
async def test_401_retry_still_401_does_not_loop(settings_delegated, monkeypatch):
    """
    If the refreshed token is ALSO rejected (persistent 401), the client must
    retry only once and then surface the error - never loop refreshing.
    """
    import httpx as httpx_module

    from core.exceptions import GraphApiError
    from core.graph_client.client import GraphClient

    calls = {"n": 0}

    def handler(request: httpx_module.Request) -> httpx_module.Response:
        calls["n"] += 1
        return httpx_module.Response(401, json={"error": {"message": "expired"}})

    transport = httpx_module.MockTransport(handler)

    class MockAsyncClient(httpx_module.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx_module, "AsyncClient", MockAsyncClient)

    refresh_count = {"n": 0}

    async def provider(*, force_refresh: bool = False):
        refresh_count["n"] += 1
        return "still-bad-token"

    client = GraphClient(settings=settings_delegated, token_provider=provider)
    with pytest.raises(GraphApiError):
        await client.get_messages(bearer_token="stale-token")

    assert calls["n"] == 2  # original + exactly one retry
    assert refresh_count["n"] == 1  # refreshed exactly once, no loop
