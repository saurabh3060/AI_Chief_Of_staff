"""
tests/agent_email/test_deletion_foundation.py
==============================================
Offline tests for the deleted-email detection FOUNDATION (no automatic
reconciliation is implemented yet, and these tests make NO live Graph
calls and touch NO production data):

  * ``ClusterEmailRef`` gains an optional Graph ``message_id`` that is
    backward compatible with legacy refs that lack it and is never
    fabricated.
  * ``_email_refs_for_indices`` populates ``message_id`` from the source
    message id.
  * Graph HTTP status is exposed structurally: a verified 404 raises
    ``GraphNotFoundError`` (the ONLY deletion candidate); 403/429/5xx/
    timeouts/exhausted retries stay inconclusive.
  * The scoped Weaviate/Redis clear + rebuild orchestration restores only
    ACTIVE clusters into Weaviate and only the required mappings into
    Redis, and never deletes anything outside this app's scope.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest


# ---------------------------------------------------------------------------
# ClusterEmailRef — optional Graph message_id, backward compatible
# ---------------------------------------------------------------------------

def test_cluster_email_ref_defaults_message_id_blank():
    """A ref built without a message_id (legacy shape) deserializes cleanly."""
    from agent_email.schema import ClusterEmailRef

    legacy = {"subject": "Q1 Review", "from_address": "a@b.com", "date": "2026-01-01"}
    ref = ClusterEmailRef.model_validate(legacy)
    assert ref.message_id == ""  # filled with the default, never fabricated
    assert ref.subject == "Q1 Review"


def test_cluster_email_ref_roundtrips_message_id():
    from agent_email.schema import ClusterEmailRef

    ref = ClusterEmailRef(
        message_id="AAMk-graph-id", subject="S", from_address="a@b.com", date="2026-01-01"
    )
    dumped = ref.model_dump()
    assert dumped["message_id"] == "AAMk-graph-id"
    # Round-trip through JSON (how email_refs is actually stored).
    again = ClusterEmailRef.model_validate(json.loads(json.dumps(dumped)))
    assert again.message_id == "AAMk-graph-id"


def test_legacy_email_refs_json_parses_with_blank_ids():
    """A stored email_refs JSON string from before this field parses fine."""
    from agent_email.premeeting import _parse_email_refs

    legacy_json = json.dumps([
        {"subject": "Old A", "from_address": "a@b.com", "date": "2026-01-01"},
        {"subject": "Old B", "from_address": "c@d.com", "date": "2026-01-02"},
    ])
    refs = _parse_email_refs(legacy_json)
    assert len(refs) == 2
    assert all(r.message_id == "" for r in refs)


def test_email_refs_for_indices_populates_message_id():
    from agent_email.schema import EmailCandidate, EmailMessage
    from agent_email.service import _email_refs_for_indices

    candidates = [
        EmailCandidate(message=EmailMessage(
            id="graph-msg-1", subject="S1", from_address="a@b.com",
            received_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )),
        EmailCandidate(message=EmailMessage(
            id="graph-msg-2", subject="S2", from_address="c@d.com",
            received_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )),
    ]
    refs = _email_refs_for_indices(candidates, [0, 1])
    assert [r.message_id for r in refs] == ["graph-msg-1", "graph-msg-2"]


# ---------------------------------------------------------------------------
# Graph HTTP status exposed structurally
# ---------------------------------------------------------------------------

def _settings_app_only(monkeypatch):
    """Minimal settings for GraphClient in app_only mode (no bearer needed)."""
    from types import SimpleNamespace

    # Avoid a real token acquisition: patch the app-only token helper.
    monkeypatch.setattr(
        "core.graph_client.client.acquire_app_only_token", lambda settings: "fake-token"
    )
    # The Graph concurrency semaphore is created lazily from the global
    # settings and bound to the running loop; reset it so each test gets a
    # fresh semaphore on the current test loop. The endpoints builder also
    # reads `mailbox_email` off the settings object passed to GraphClient.
    from core.concurrency import reset_semaphores_for_tests

    reset_semaphores_for_tests()
    return SimpleNamespace(
        graph_auth_mode="app_only",
        graph_request_timeout_seconds=5,
        mailbox_email="shared@contoso.com",
        is_local_mail_mode=False,
    )


class _FakeResponse:
    def __init__(self, status_code, text="", headers=None, json_body=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self._json = json_body if json_body is not None else {}

    def json(self):
        return self._json


def _patch_http_get(monkeypatch, response):
    """Patch httpx.AsyncClient so GraphClient._get receives `response`."""
    class _FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None, params=None):
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr("core.graph_client.client.httpx.AsyncClient", _FakeAsyncClient)


@pytest.mark.asyncio
async def test_graph_404_raises_not_found_with_status(monkeypatch, required_env):
    from core.graph_client.client import GraphClient
    from core.exceptions import GraphNotFoundError, GraphApiError

    settings = _settings_app_only(monkeypatch)
    _patch_http_get(monkeypatch, _FakeResponse(404, text="not found"))
    client = GraphClient(settings=settings)

    with pytest.raises(GraphNotFoundError) as excinfo:
        await client.get_message_detail("deleted-id")
    assert excinfo.value.status_code == 404
    # It is a GraphApiError subclass so existing broad handlers still catch it.
    assert isinstance(excinfo.value, GraphApiError)


@pytest.mark.asyncio
async def test_graph_403_is_inconclusive_not_not_found(monkeypatch, required_env):
    from core.graph_client.client import GraphClient
    from core.exceptions import GraphApiError, GraphNotFoundError

    settings = _settings_app_only(monkeypatch)
    _patch_http_get(monkeypatch, _FakeResponse(403, text="forbidden"))
    client = GraphClient(settings=settings)

    with pytest.raises(GraphApiError) as excinfo:
        await client.get_message_detail("some-id")
    # 403 must NOT be a deletion candidate.
    assert not isinstance(excinfo.value, GraphNotFoundError)
    assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_graph_429_is_throttled_and_inconclusive(monkeypatch, required_env):
    from core.graph_client.client import GraphClient
    from core.exceptions import GraphThrottledError, GraphNotFoundError

    settings = _settings_app_only(monkeypatch)
    _patch_http_get(
        monkeypatch, _FakeResponse(429, text="throttled", headers={"Retry-After": "1"})
    )
    client = GraphClient(settings=settings)

    # with_retry will retry a GraphThrottledError up to max_attempts then
    # re-raise it; patch asyncio.sleep so the test does not actually wait.
    monkeypatch.setattr("core.retry.asyncio.sleep", _noop_async_sleep)

    with pytest.raises(GraphThrottledError) as excinfo:
        await client.get_message_detail("some-id")
    assert not isinstance(excinfo.value, GraphNotFoundError)
    assert excinfo.value.status_code == 429
    assert excinfo.value.retry_after_seconds == 1.0


@pytest.mark.asyncio
async def test_graph_timeout_is_inconclusive(monkeypatch, required_env):
    from core.graph_client.client import GraphClient
    from core.exceptions import GraphApiError, GraphNotFoundError

    settings = _settings_app_only(monkeypatch)
    _patch_http_get(monkeypatch, httpx.ReadTimeout("timed out"))
    client = GraphClient(settings=settings)
    monkeypatch.setattr("core.retry.asyncio.sleep", _noop_async_sleep)

    # Exhausted retries on a timeout surface the transport error, which is
    # neither a GraphApiError nor a GraphNotFoundError => inconclusive.
    with pytest.raises(httpx.ReadTimeout) as excinfo:
        await client.get_message_detail("some-id")
    assert not isinstance(excinfo.value, GraphNotFoundError)
    assert not isinstance(excinfo.value, GraphApiError)


async def _noop_async_sleep(_seconds):
    return None


# ---------------------------------------------------------------------------
# Scoped Weaviate clear + restore (ACTIVE clusters only)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clear_and_restore_weaviate_restores_active_only(monkeypatch):
    """
    The orchestration must: ensure schema, clear owned namespaces, and
    restore from PostgreSQL. We stub the Weaviate layer and the PG vector
    loaders so the test is fully offline, and assert active-only restore by
    driving restore through the real `restore_weaviate_from_postgres` with a
    fake connection whose cluster query returns only is_active=1 rows.
    """
    from agent_email import weaviate_sync
    import numpy as np

    calls = {"ensure_schema": 0, "cleared": [], "upserted": {}}

    def fake_ensure_schema(settings=None):
        calls["ensure_schema"] += 1
        return {}

    def fake_clear(namespace, settings=None):
        calls["cleared"].append(namespace)
        return 0

    def fake_upsert(namespace, items, settings=None):
        calls["upserted"][namespace] = list(items)
        return len(items)

    monkeypatch.setattr(weaviate_sync.wvs, "ensure_schema", fake_ensure_schema)
    monkeypatch.setattr(weaviate_sync.wvs, "clear_namespace_objects", fake_clear)
    monkeypatch.setattr(weaviate_sync.wvs, "upsert_vectors", fake_upsert)

    # Fake PG: only ACTIVE clusters come back from the active-only query the
    # loader runs; the loader itself (SELECT ... WHERE is_active = 1) is the
    # contract, so we return exactly what that query would.
    active_vec = json.dumps([1.0, 0.0])

    class _FakeConn:
        async def fetch(self, sql, *args):
            if "email_clusters" in sql:
                # Mimic the WHERE is_active = 1 filter: only active rows.
                return [{"cluster_id": "active-1", "embedding": active_vec}]
            # meeting_embeddings / meeting_history
            return []

    class _FakeAcquire:
        async def __aenter__(self):
            return _FakeConn()

        async def __aexit__(self, *a):
            return False

    class _FakePool:
        def acquire(self):
            return _FakeAcquire()

    monkeypatch.setattr(weaviate_sync, "get_pool", lambda: _FakePool())

    result = await weaviate_sync.clear_and_restore_weaviate_from_postgres()

    assert calls["ensure_schema"] == 1
    # Both owned namespaces were cleared.
    assert set(calls["cleared"]) == {
        weaviate_sync.NAMESPACE_CLUSTERS,
        weaviate_sync.NAMESPACE_MEETINGS,
    }
    # Exactly one active cluster vector was restored into the clusters ns.
    assert len(calls["upserted"].get(weaviate_sync.NAMESPACE_CLUSTERS, [])) == 1
    assert result[weaviate_sync.NAMESPACE_CLUSTERS]["restored"] == 1


# ---------------------------------------------------------------------------
# Scoped Redis clear + refill (required mappings only)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_redis_clear_all_scoped_is_prefix_bounded(monkeypatch):
    from agent_email import redis_cache
    from types import SimpleNamespace

    deleted_keys: list[str] = []
    store = {
        "rbl_email:clusters:all": "x",
        "rbl_email:meeting_clusters:m1": "y",
        "other_app:clusters:all": "z",  # must NOT be matched
    }

    class _FakeRedis:
        async def scan_iter(self, match=None):
            # Emulate Redis glob: only keys under the given prefix match.
            prefix = match[:-1] if match.endswith("*") else match
            for k in list(store.keys()):
                if k.startswith(prefix):
                    yield k

        async def delete(self, *keys):
            for k in keys:
                deleted_keys.append(k)
                store.pop(k, None)
            return len(keys)

    settings = SimpleNamespace(redis_key_prefix="rbl_email:")
    monkeypatch.setattr(redis_cache, "get_redis", _make_async_return(_FakeRedis()))

    n = await redis_cache.clear_all_scoped(settings=settings)
    assert n == 2
    assert set(deleted_keys) == {"rbl_email:clusters:all", "rbl_email:meeting_clusters:m1"}
    # The other app's key is untouched.
    assert "other_app:clusters:all" in store


@pytest.mark.asyncio
async def test_redis_clear_and_refill_loads_required_mappings(monkeypatch):
    """
    clear_and_refill_from_source must clear, then reload exactly the
    meeting-matching / pre-brief input mappings (clusters, meeting
    embeddings, meeting history) — and NOT pre-fill the derived
    meeting_clusters cache.
    """
    from agent_email import redis_cache
    from types import SimpleNamespace

    cleared = {"called": False}

    async def fake_clear(settings=None):
        cleared["called"] = True
        return 0

    refilled = {"called": False, "warm_derived": None}

    async def fake_refill(settings=None, warm_derived=True):
        refilled["called"] = True
        refilled["warm_derived"] = warm_derived
        return {"clusters": 3, "meeting_embeddings": 2, "meeting_history": 1}

    monkeypatch.setattr(redis_cache, "clear_all_scoped", fake_clear)
    monkeypatch.setattr(redis_cache, "refill_from_source", fake_refill)

    settings = SimpleNamespace(redis_key_prefix="rbl_email:")
    result = await redis_cache.clear_and_refill_from_source(settings=settings)

    assert cleared["called"] and refilled["called"]
    # The start-of-run rebuild must NOT warm the derived meeting_clusters
    # cache: the pipeline rewrites inputs and warms the briefs itself at the
    # end of the run, so pre-filling here would be stale throwaway work.
    assert refilled["warm_derived"] is False
    assert result == {"clusters": 3, "meeting_embeddings": 2, "meeting_history": 1}


def _make_async_return(value):
    async def _coro(*a, **k):
        return value
    return _coro
