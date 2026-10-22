"""
tests/smoke/test_e2e_flow.py
===============================
End-to-end smoke test exercising the full agent chain through the real
FastAPI app (via TestClient), simulating a realistic sequence of calls:

1. Fetch the current/next meeting from the Calendar agent.
2. Generate an email summary from the Email agent (which OCRs an
   attachment along the way).
3. Classify action items via the Action Tracker agent, chained on the
   Email agent's persisted summary (no re-fetching of mail).
4. Ask the Chatbot a question that should ground its answer in the
   Action Tracker's data.

Only the outermost dependencies (Microsoft Graph API and the local
Ollama server) are mocked - everything else runs through the real
FastAPI routes, service functions, LangGraph subgraphs, and in-memory
storage, exactly as it would in production with STORAGE_BACKEND=memory.

Run separately from the unit suite via:
    pytest tests/smoke/
"""

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def smoke_test_client(required_env, monkeypatch, mail_thread_fixture, pdf_attachment_fixture_factory, calendar_events_fixture_factory):
    """
    Build a TestClient against the real app, with only Graph API and
    Ollama calls mocked. Uses app-only auth mode so no bearer token is
    required on any route, keeping the smoke test focused on the agent
    chain rather than the OAuth flow (already covered by
    tests/auth/test_auth_routes.py).
    """
    from datetime import datetime, timezone

    from core.config import get_settings
    from core.graph_client.client import GraphClient
    from core.ollama_client.client import OllamaClient
    from core.storage.factory import reset_storage_for_tests

    monkeypatch.setenv("GRAPH_AUTH_MODE", "app_only")
    get_settings.cache_clear()
    reset_storage_for_tests()

    now = datetime.now(timezone.utc)
    calendar_fixture = calendar_events_fixture_factory(now)

    async def fake_get_calendar_events(self, bearer_token=None, top=50):
        return calendar_fixture

    async def fake_get_messages(self, bearer_token=None, top=10, skip=0, received_after=None):
        return mail_thread_fixture

    async def fake_get_attachments(self, message_id, bearer_token=None):
        return {"value": [{"id": "att-001", "name": "branch_audit_report.pdf"}]}

    async def fake_get_attachment_content(self, message_id, attachment_id, bearer_token=None):
        return pdf_attachment_fixture_factory(attachment_id=attachment_id)

    async def fake_vision_ocr(self, image_bytes, instruction=None):
        return "Branch Audit Report - all findings closed."

    # Route canned responses by recognizable substrings in the prompt
    # text, rather than call order - the smoke test exercises the same
    # agents through two different paths (HTTP routes and direct
    # orchestrator calls), so call count/order isn't consistent across
    # them, but each prompt's distinguishing phrase (from that agent's
    # prompts.py) always is.
    async def fake_chat(self, prompt, system_prompt=None):
        if "Draft a short pre-meeting brief" in prompt:
            return "- Discuss wholesale loan book growth\n- Review NPA trend"
        if "Summarize the following email conversation" in prompt:
            return "Summary: Q1 retail numbers reviewed, NPA improved."
        if "Who owns this action item?" in prompt:
            return "Risk Team"
        if "Which tool should be used to answer this question?" in prompt:
            return "actions"
        if "Answer the question using only the context above" in prompt:
            return "You have 2 tracked action items, including one overdue item."
        return "General response."

    monkeypatch.setattr(GraphClient, "get_calendar_events", fake_get_calendar_events)
    monkeypatch.setattr(GraphClient, "get_messages", fake_get_messages)
    monkeypatch.setattr(GraphClient, "get_attachments", fake_get_attachments)
    monkeypatch.setattr(GraphClient, "get_attachment_content", fake_get_attachment_content)
    monkeypatch.setattr(OllamaClient, "vision_ocr", fake_vision_ocr)
    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    import main as main_module

    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        yield client

    reset_storage_for_tests()
    get_settings.cache_clear()


def test_full_agent_chain_calendar_to_email_to_actions_to_chatbot(smoke_test_client):
    """
    Simulates: fetch calendar -> generate email summary -> get classified
    actions -> chatbot query referencing them, all through real routes.
    """
    # Step 1: Calendar agent - fetch next meeting.
    calendar_response = smoke_test_client.get("/agents/calendar/next")
    assert calendar_response.status_code == 200
    next_meeting = calendar_response.json()["next_meeting"]
    assert next_meeting is not None
    assert next_meeting["subject"] == "Wholesale Banking Review"

    # Step 2: Email agent - generate a summary (OCRs the fixture attachment).
    email_response = smoke_test_client.get("/agents/email/summary")
    assert email_response.status_code == 200
    summary = email_response.json()["summary"]
    assert summary is not None
    assert len(summary["action_candidates"]) >= 1

    # Step 3: Action Tracker - classify action items chained on the
    # Email agent's summary generated in step 2 (no re-fetch of mail).
    actions_response = smoke_test_client.get("/agents/actions")
    assert actions_response.status_code == 200
    action_items = actions_response.json()["action_items"]
    assert len(action_items) >= 1
    # Every classified item should trace back to the thread summarized in step 2.
    assert all(item["source_thread_id"] == summary["thread_id"] for item in action_items)

    # Step 4: Chatbot - ask a question grounded in the Action Tracker's data.
    chat_response = smoke_test_client.post(
        "/chatbot/query", json={"message": "What are my overdue action items?"}
    )
    assert chat_response.status_code == 200
    chat_body = chat_response.json()
    assert chat_body["tool_used"] == "actions"
    assert "action item" in chat_body["answer"].lower()


def test_full_agent_chain_via_orchestrator_directly(smoke_test_client):
    """
    Same chain as above, but driving the Calendar/Email/Action-Tracker
    steps through the orchestrator's run_agent() directly (bypassing
    HTTP), proving the orchestrator wiring itself produces a consistent
    result independent of the route layer.
    """
    import asyncio

    from core.orchestrator.graph import run_agent

    calendar_result = asyncio.run(run_agent("calendar"))
    assert calendar_result["calendar_result"]["meeting_window"].next_meeting is not None

    email_result = asyncio.run(run_agent("email"))
    assert email_result["email_result"]["summary"] is not None

    action_result = asyncio.run(run_agent("actions"))
    assert len(action_result["action_result"]["action_items"]) >= 1


def test_mis_agent_remains_a_clean_stub_during_full_chain(smoke_test_client):
    """Even after the other agents have run, the MIS stub route should still return a clean 501, not a crash."""
    smoke_test_client.get("/agents/calendar/next")
    smoke_test_client.get("/agents/email/summary")
    smoke_test_client.get("/agents/actions")

    mis_response = smoke_test_client.get("/agents/mis/insights")
    assert mis_response.status_code == 501
