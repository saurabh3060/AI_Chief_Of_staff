"""
tests/auth/test_auth_routes.py
=================================
Integration tests for the delegated OAuth routes (`/auth/login`,
`/auth/callback`), using FastAPI's TestClient. Verifies the redirect to
Microsoft's authorize endpoint carries the expected client_id/redirect_uri,
and that `/auth/callback` handles both the success and error paths
without ever contacting the real Microsoft token endpoint (mocked).
"""

from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(required_env, monkeypatch):
    """
    Build a TestClient against the real FastAPI app, with required env
    vars set so `core.config.get_settings()` (imported at module load
    time in several places) succeeds.
    """
    monkeypatch.setenv("APP_BASE_URL", "https://digitaloffice-test.rblbank.com")
    from core.config import get_settings

    get_settings.cache_clear()

    import importlib

    import main as main_module

    importlib.reload(main_module)

    with TestClient(main_module.app) as test_client:
        yield test_client

    get_settings.cache_clear()


def test_login_redirects_to_microsoft_authorize_endpoint(client, monkeypatch):
    """
    GET /auth/login should redirect (302) to whatever authorize URL
    `build_authorization_url` returns, with the redirect_uri it was
    called with reflecting our configured APP_BASE_URL. MSAL's real
    ConfidentialClientApplication makes a live network call to validate
    the tenant on construction, so the authorize-URL builder itself is
    mocked here - the URL-building responsibility is core.graph_client.auth's,
    already covered by tests/core/test_graph_client.py.
    """
    from auth import routes as routes_module

    captured = {}

    def fake_build_authorization_url(settings, redirect_uri, prompt=None):
        captured["redirect_uri"] = redirect_uri
        return f"https://login.microsoftonline.com/{settings.tenant_id}/oauth2/v2.0/authorize?client_id={settings.client_id}&redirect_uri={redirect_uri}"

    monkeypatch.setattr(routes_module, "build_authorization_url", fake_build_authorization_url)

    response = client.get("/auth/login", follow_redirects=False)

    assert response.status_code in (302, 307)
    location = response.headers["location"]
    # Assert on the parsed HOSTNAME rather than a loose substring or
    # `netloc` match — `hostname` excludes any userinfo/port, so
    # "login.microsoftonline.com" cannot be smuggled into a `user@host`,
    # `:port`, path, or query segment of an attacker-controlled URL
    # (py/incomplete-url-substring-sanitization).
    assert urlsplit(location).hostname == "login.microsoftonline.com"
    assert "test-client-id" in location
    assert captured["redirect_uri"] == "https://digitaloffice-test.rblbank.com/auth/callback"


def test_callback_without_code_returns_400(client):
    """GET /auth/callback with no ?code= param should return a 400 with an error message."""
    response = client.get("/auth/callback")

    assert response.status_code == 400
    assert "error" in response.json()


def test_callback_exchanges_code_for_token(client, monkeypatch):
    """GET /auth/callback?code=... should exchange the code for a token via the mocked MSAL flow."""
    from auth import routes as routes_module

    monkeypatch.setattr(
        routes_module,
        "exchange_code_for_token",
        lambda settings, code, redirect_uri: {
            "access_token": "fake-access-token",
            "id_token": "fake-id-token",
            "expires_in": 3600,
        },
    )

    response = client.get("/auth/callback", params={"code": "fake-auth-code"})

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"] == "fake-access-token"
    assert body["expires_in"] == 3600


def test_callback_returns_400_on_token_exchange_error(client, monkeypatch):
    """If MSAL's token exchange returns an error, /auth/callback should surface it as a 400."""
    from auth import routes as routes_module

    monkeypatch.setattr(
        routes_module,
        "exchange_code_for_token",
        lambda settings, code, redirect_uri: {
            "error": "invalid_grant",
            "error_description": "The authorization code has expired.",
        },
    )

    response = client.get("/auth/callback", params={"code": "expired-code"})

    assert response.status_code == 400
    assert "expired" in response.json()["error"]
