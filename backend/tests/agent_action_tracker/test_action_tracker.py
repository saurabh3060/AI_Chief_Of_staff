"""
tests/agent_action_tracker/test_action_tracker.py
=====================================================
Unit tests for the classification logic (owner-guessing, due-date
parsing, overdue/on-track/in-progress determination), plus integration
tests proving Action Tracker derives its data from the Email agent's
persisted summaries (via `agent_email.service.get_latest_summaries()`)
rather than fetching mail itself.
"""

from datetime import date, timezone

import pytest


# ---------------------------------------------------------------------------
# guess_due_date() / classify_status() - pure logic
# ---------------------------------------------------------------------------

def test_guess_due_date_parses_explicit_iso_date():
    from agent_action_tracker.service import guess_due_date

    result = guess_due_date("Send the report by 2026-09-12", reference=date(2026, 9, 1))
    assert result == date(2026, 9, 12)


def test_guess_due_date_parses_explicit_text_date():
    from agent_action_tracker.service import guess_due_date

    result = guess_due_date("Send the report by 12 Sep 2026", reference=date(2026, 9, 1))
    assert result == date(2026, 9, 12)


def test_guess_due_date_parses_relative_weekday():
    from agent_action_tracker.service import guess_due_date

    # 2026-09-01 is a Tuesday; "by Friday" should resolve to 2026-09-04.
    result = guess_due_date("Send the breakup by Friday", reference=date(2026, 9, 1))
    assert result == date(2026, 9, 4)


def test_guess_due_date_returns_none_when_no_date_phrase():
    from agent_action_tracker.service import guess_due_date

    assert guess_due_date("Just review the numbers", reference=date(2026, 9, 1)) is None


def test_classify_status_overdue_when_due_date_in_past():
    from agent_action_tracker.schema import ActionStatus
    from agent_action_tracker.service import classify_status

    result = classify_status(date(2026, 8, 20), reference=date(2026, 9, 1))
    assert result == ActionStatus.OVERDUE


def test_classify_status_on_track_when_due_date_in_future():
    from agent_action_tracker.schema import ActionStatus
    from agent_action_tracker.service import classify_status

    result = classify_status(date(2026, 9, 10), reference=date(2026, 9, 1))
    assert result == ActionStatus.ON_TRACK


def test_classify_status_in_progress_when_no_due_date():
    from agent_action_tracker.schema import ActionStatus
    from agent_action_tracker.service import classify_status

    result = classify_status(None, reference=date(2026, 9, 1))
    assert result == ActionStatus.IN_PROGRESS


# ---------------------------------------------------------------------------
# guess_owner() - heuristic + LLM fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guess_owner_uses_heuristic_when_pattern_matches():
    from agent_action_tracker.service import guess_owner

    owner = await guess_owner("Finance team to send the reconciled branch-wise breakup by Friday")
    assert owner == "Finance team"


@pytest.mark.asyncio
async def test_guess_owner_falls_back_to_llm_when_no_heuristic_match(monkeypatch):
    from core.ollama_client.client import OllamaClient
    from agent_action_tracker.service import guess_owner

    async def fake_chat(self, prompt, system_prompt=None):
        return "Risk Team"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    owner = await guess_owner("Review the compliance findings before Monday", ollama_client=OllamaClient())
    assert owner == "Risk Team"


@pytest.mark.asyncio
async def test_guess_owner_returns_none_without_ollama_client_and_no_heuristic():
    from agent_action_tracker.service import guess_owner

    owner = await guess_owner("Review the compliance findings before Monday", ollama_client=None)
    assert owner is None


# ---------------------------------------------------------------------------
# classify_all_action_items() - chains on agent_email.service, never GraphClient
# ---------------------------------------------------------------------------

@pytest.fixture
def stored_summary_fixture(required_env):
    """
    Seed the storage layer with one persisted EmailSummary (as if the
    Email agent had already run), so Action Tracker tests can verify
    they read from storage/agent_email.service rather than Graph API.
    """
    import asyncio
    from datetime import datetime

    from agent_email.schema import ActionCandidate, EmailSummary
    from core.storage.factory import get_storage, reset_storage_for_tests

    reset_storage_for_tests()
    storage = get_storage()

    summary = EmailSummary(
        thread_id="thread-001",
        subject="Q1 Retail Performance Numbers",
        generated_at=datetime.now(timezone.utc),
        summary_text="Q1 retail numbers reviewed.",
        action_candidates=[
            ActionCandidate(text="Finance team to send the reconciled branch-wise breakup by Friday", source_message_id="msg-001"),
            ActionCandidate(text="Review the compliance findings before Monday", source_message_id="msg-002"),
        ],
    )

    asyncio.run(storage.summaries.save_summary(summary.thread_id, summary.model_dump(mode="json")))

    yield storage
    reset_storage_for_tests()


@pytest.mark.asyncio
async def test_classify_all_action_items_reads_from_email_agent_summaries(stored_summary_fixture, monkeypatch):
    """classify_all_action_items() should derive items from agent_email.service, not from GraphClient."""
    from core.graph_client.client import GraphClient
    from core.ollama_client.client import OllamaClient
    from agent_action_tracker.service import classify_all_action_items

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("Action Tracker must not call GraphClient directly")

    monkeypatch.setattr(GraphClient, "get_messages", fail_if_called)
    monkeypatch.setattr(GraphClient, "get_calendar_events", fail_if_called)

    async def fake_chat(self, prompt, system_prompt=None):
        return "Risk Team"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    items = await classify_all_action_items(reference=date(2026, 9, 1))

    assert len(items) == 2
    texts = {item.text for item in items}
    assert "Finance team to send the reconciled branch-wise breakup by Friday" in texts


@pytest.mark.asyncio
async def test_classify_all_action_items_persists_results(stored_summary_fixture, monkeypatch):
    from core.ollama_client.client import OllamaClient
    from agent_action_tracker.service import classify_all_action_items, list_action_items

    async def fake_chat(self, prompt, system_prompt=None):
        return "Risk Team"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    await classify_all_action_items(reference=date(2026, 9, 1))
    stored_items = await list_action_items()

    assert len(stored_items) == 2


# ---------------------------------------------------------------------------
# Routes - integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def action_tracker_test_client(stored_summary_fixture, monkeypatch):
    from core.ollama_client.client import OllamaClient

    async def fake_chat(self, prompt, system_prompt=None):
        return "Risk Team"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    import importlib

    import main as main_module

    importlib.reload(main_module)

    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as test_client:
        yield test_client


def test_get_action_items_route_returns_classified_items(action_tracker_test_client):
    response = action_tracker_test_client.get("/agents/actions")

    assert response.status_code == 200
    body = response.json()
    assert len(body["action_items"]) == 2


def test_get_action_items_route_filters_by_status(action_tracker_test_client):
    response = action_tracker_test_client.get("/agents/actions", params={"status": "In Progress"})

    assert response.status_code == 200
    body = response.json()
    for item in body["action_items"]:
        assert item["status"] == "In Progress"


def test_patch_action_item_updates_status(action_tracker_test_client):
    list_response = action_tracker_test_client.get("/agents/actions")
    action_id = list_response.json()["action_items"][0]["id"]

    patch_response = action_tracker_test_client.patch(f"/agents/actions/{action_id}", json={"status": "On Track"})

    assert patch_response.status_code == 200
    assert patch_response.json()["action_item"]["status"] == "On Track"


def test_patch_action_item_returns_404_for_unknown_id(action_tracker_test_client):
    response = action_tracker_test_client.patch("/agents/actions/does-not-exist", json={"status": "On Track"})
    assert response.status_code == 404


def test_patch_action_item_returns_400_when_no_fields_given(action_tracker_test_client):
    list_response = action_tracker_test_client.get("/agents/actions")
    action_id = list_response.json()["action_items"][0]["id"]

    response = action_tracker_test_client.patch(f"/agents/actions/{action_id}", json={})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# LangGraph subgraph
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_action_tracker_graph_runs_end_to_end(stored_summary_fixture, monkeypatch):
    from core.ollama_client.client import OllamaClient

    async def fake_chat(self, prompt, system_prompt=None):
        return "Risk Team"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    from agent_action_tracker.graph import action_tracker_graph

    result = await action_tracker_graph.ainvoke({"action_items": []})

    assert len(result["action_items"]) == 2
