"""
tests/auth/test_session_enforcement.py
=========================================
Security regression tests for the shared `require_session` gate applied
at the router level in `main.py`.

These lock in the VAPT remediation: every agent/chatbot endpoint that was
reachable WITHOUT a valid `rbl_session` cookie must now return 401 when no
(or an invalid/expired) session cookie is presented, while a signed-in
caller with a valid session sees its normal 200/204 behaviour unchanged.

They also pin the handler-ordering fix: a fake `item_id` presented WITH a
valid session returns 404 (the lookup runs and finds nothing), but the
SAME fake id WITHOUT a session returns 401 (auth runs first, so the id is
never looked up) - i.e. no item_id is reachable at all without a session.

The app is reloaded in delegated mode (the deployed posture the VAPT
retest targeted). Service-layer calls are monkeypatched so the
authenticated path returns deterministically without needing Postgres,
Redis, or Weaviate - these tests are about the auth gate, not the agents'
business logic.
"""

import importlib

import pytest
from fastapi.testclient import TestClient

from agent_open_tracker.schema import OpenTrackerResponse
from chatbot.schema import ChatHistoryResponse


# ---------------------------------------------------------------------------
# App + session fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app_module(required_env, monkeypatch):
    """
    Reload the FastAPI app in delegated mode with the agent/chatbot
    service functions stubbed so the authenticated happy path is
    infra-free and deterministic.
    """
    monkeypatch.setenv("APP_BASE_URL", "https://digitaloffice-test.rblbank.com")
    monkeypatch.setenv("GRAPH_AUTH_MODE", "delegated")

    from core.config import get_settings
    get_settings.cache_clear()

    # --- stub open-tracker service ---
    import agent_open_tracker.routes as ot_routes

    async def fake_get_open_tracker_items():
        return OpenTrackerResponse()

    async def fake_complete_item(item_id):
        return None  # "not found" -> handler maps to 404

    async def fake_uncomplete_item(item_id):
        return None

    async def fake_delete_item(item_id):
        return None

    async def fake_update_due_date(item_id, due_date_str):
        return None

    async def fake_update_item(**kwargs):
        return None

    monkeypatch.setattr(ot_routes, "get_open_tracker_items", fake_get_open_tracker_items)
    monkeypatch.setattr(ot_routes, "complete_item", fake_complete_item)
    monkeypatch.setattr(ot_routes, "uncomplete_item", fake_uncomplete_item)
    monkeypatch.setattr(ot_routes, "delete_item", fake_delete_item)
    monkeypatch.setattr(ot_routes, "update_due_date", fake_update_due_date)
    monkeypatch.setattr(ot_routes, "update_item", fake_update_item)

    # --- stub email service ---
    import agent_email.routes as email_routes

    async def fake_get_email_cluster_records(department=None):
        return []

    async def fake_get_latest_summaries():
        return []

    async def fake_run_email_cluster_pipeline(bearer_token=None):
        return 0

    monkeypatch.setattr(email_routes, "get_email_cluster_records", fake_get_email_cluster_records)
    monkeypatch.setattr(email_routes, "get_latest_summaries", fake_get_latest_summaries)
    monkeypatch.setattr(email_routes, "run_email_cluster_pipeline", fake_run_email_cluster_pipeline)

    # --- stub action-tracker service ---
    import agent_action_tracker.routes as at_routes

    async def fake_classify_all_action_items():
        return None

    async def fake_list_action_items(status=None):
        return []

    monkeypatch.setattr(at_routes, "classify_all_action_items", fake_classify_all_action_items)
    monkeypatch.setattr(at_routes, "list_action_items", fake_list_action_items)

    # --- stub chatbot history (Redis-backed) ---
    import chatbot.routes as cb_routes

    monkeypatch.setattr(cb_routes, "get_redis_history", lambda limit=50: [])
    monkeypatch.setattr(cb_routes, "clear_redis_history", lambda: True)

    # Reload main so the router-level dependencies are re-evaluated under
    # the patched settings/services.
    import main as main_module
    importlib.reload(main_module)

    yield main_module

    get_settings.cache_clear()


@pytest.fixture
def client(app_module):
    with TestClient(app_module.app) as test_client:
        # TestClient keeps cookies across requests by default; make sure
        # each test starts cookie-free.
        test_client.cookies.clear()
        yield test_client


@pytest.fixture
def valid_cookie():
    """
    Create a real session in the SAME in-memory store `/callback` uses and
    return {cookie_name: session_id} ready to pass as `cookies=...`.
    """
    from auth.session import get_session_store
    from core.config import get_settings

    session_id = get_session_store().create(
        access_token="fake-access-token",
        id_token="fake-id-token",
        ttl_seconds=3600,
        data={"name": "CEO Office", "email": "ceo.office@rblbank.com"},
    )
    cookie_name = get_settings().session_cookie_name
    return {cookie_name: session_id}


# ---------------------------------------------------------------------------
# The VAPT finding list, as (method, path) pairs.
#
# `/api` is stripped by nginx upstream; backend-side paths have no prefix.
# Each entry is a route that the retest confirmed was reachable WITHOUT a
# valid session cookie.
# ---------------------------------------------------------------------------

READ_ENDPOINTS = [
    ("GET", "/agents/open-tracker"),
    ("GET", "/agents/email/clusters"),
    ("GET", "/agents/email/clusters?department=BBRL"),
    ("GET", "/agents/email/summaries"),
    ("GET", "/agents/actions"),
    ("GET", "/agents/actions?status=Open"),
    ("GET", "/chatbot/history"),
    ("GET", "/chatbot/history?limit=10"),
]

WRITE_ENDPOINTS = [
    ("POST", "/agents/open-tracker/fake-id/complete", None),
    ("DELETE", "/agents/open-tracker/fake-id/complete", None),
    ("PATCH", "/agents/open-tracker/fake-id/due-date", {"due_date": "2026-09-30"}),
    ("DELETE", "/agents/open-tracker/fake-id", None),
    ("PATCH", "/agents/open-tracker/fake-id/edit", {"text": "updated"}),
    ("DELETE", "/chatbot/history", None),
    ("POST", "/agents/email/clusters/refresh", None),
]


def _call(client, method, path, json_body=None, cookies=None):
    return client.request(method, path, json=json_body, cookies=cookies)


# ---------------------------------------------------------------------------
# 401 without a session cookie
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,path", READ_ENDPOINTS)
def test_read_endpoint_401_without_cookie(client, method, path):
    resp = _call(client, method, path)
    assert resp.status_code == 401, f"{method} {path} should be 401 without a session"


@pytest.mark.parametrize("method,path,body", WRITE_ENDPOINTS)
def test_write_endpoint_401_without_cookie(client, method, path, body):
    resp = _call(client, method, path, json_body=body)
    assert resp.status_code == 401, f"{method} {path} should be 401 without a session"


def test_401_with_invalid_cookie(client):
    """A present-but-bogus session id is treated exactly like no session."""
    resp = _call(
        client, "GET", "/agents/open-tracker",
        cookies={"rbl_session": "not-a-real-session-id"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Normal behaviour WITH a valid session (unchanged for signed-in users)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,path", READ_ENDPOINTS)
def test_read_endpoint_200_with_valid_cookie(client, valid_cookie, method, path):
    resp = _call(client, method, path, cookies=valid_cookie)
    assert resp.status_code == 200, f"{method} {path} should be 200 with a valid session"


def test_clusters_refresh_200_with_valid_cookie(client, valid_cookie):
    resp = _call(client, "POST", "/agents/email/clusters/refresh", cookies=valid_cookie)
    assert resp.status_code == 200
    assert resp.json() == {"clusters_written": 0}


def test_clear_chat_history_200_with_valid_cookie(client, valid_cookie):
    resp = _call(client, "DELETE", "/chatbot/history", cookies=valid_cookie)
    assert resp.status_code == 200
    assert resp.json() == {"cleared": True}


# ---------------------------------------------------------------------------
# Handler ordering: auth runs BEFORE the item_id lookup.
#
# - fake id WITHOUT a session  -> 401 (never looked up)
# - fake id WITH a valid session -> 404 (looked up, not found) — unchanged
# ---------------------------------------------------------------------------

FAKE_ID_WRITES = [
    ("POST", "/agents/open-tracker/fake-id/complete", None),
    ("DELETE", "/agents/open-tracker/fake-id/complete", None),
    ("PATCH", "/agents/open-tracker/fake-id/due-date", {"due_date": "2026-09-30"}),
    ("DELETE", "/agents/open-tracker/fake-id", None),
    ("PATCH", "/agents/open-tracker/fake-id/edit", {"text": "updated"}),
]


@pytest.mark.parametrize("method,path,body", FAKE_ID_WRITES)
def test_fake_item_id_without_session_is_401_not_404(client, method, path, body):
    """No item_id is reachable at all without a valid session."""
    resp = _call(client, method, path, json_body=body)
    assert resp.status_code == 401


@pytest.mark.parametrize("method,path,body", FAKE_ID_WRITES)
def test_fake_item_id_with_valid_session_is_404(client, valid_cookie, method, path, body):
    """With a valid session the lookup runs and a nonexistent id yields 404."""
    resp = _call(client, method, path, json_body=body, cookies=valid_cookie)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Public routes must stay reachable pre-login (sign-in flow not broken).
# ---------------------------------------------------------------------------

def test_health_is_public(client):
    assert client.get("/health").status_code == 200


def test_auth_me_is_public_and_reports_unauthenticated(client):
    """/auth/me must be reachable with no cookie so the SPA can decide."""
    resp = client.get("/auth/me")
    assert resp.status_code == 401
    assert resp.json()["authenticated"] is False
