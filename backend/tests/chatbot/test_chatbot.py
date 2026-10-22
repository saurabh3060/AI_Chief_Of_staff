"""
tests/chatbot/test_chatbot.py
================================
Unit tests for tool selection and context formatting, plus integration
tests for the full answer_question() flow (grounded and ungrounded)
and the `/chatbot/query` route, with the orchestrator and Ollama calls
mocked.
"""

import pytest


# ---------------------------------------------------------------------------
# select_tool() - unit tests with a mocked OllamaClient
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_select_tool_returns_calendar_for_meeting_question(monkeypatch):
    from core.ollama_client.client import OllamaClient
    from chatbot.service import select_tool

    async def fake_chat(self, prompt, system_prompt=None):
        return "calendar"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    tool = await select_tool("What's my next meeting?", OllamaClient())
    assert tool == "calendar"


@pytest.mark.asyncio
async def test_select_tool_returns_none_for_unrelated_question(monkeypatch):
    from core.ollama_client.client import OllamaClient
    from chatbot.service import select_tool

    async def fake_chat(self, prompt, system_prompt=None):
        return "none"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    tool = await select_tool("Hello, how are you?", OllamaClient())
    assert tool is None


@pytest.mark.asyncio
async def test_select_tool_treats_unrecognized_response_as_none(monkeypatch):
    """If the model returns something that isn't a known tool name, treat it as 'no tool' rather than erroring."""
    from core.ollama_client.client import OllamaClient
    from chatbot.service import select_tool

    async def fake_chat(self, prompt, system_prompt=None):
        return "some nonsense response"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    tool = await select_tool("What's my next meeting?", OllamaClient())
    assert tool is None


# ---------------------------------------------------------------------------
# format_tool_result_as_context()
# ---------------------------------------------------------------------------

def test_format_actions_context_lists_items():
    from datetime import datetime, timezone
    from agent_action_tracker.schema import ActionItem, ActionStatus
    from chatbot.service import format_tool_result_as_context

    items = [
        ActionItem(
            id="t1::0", text="Send the breakup", source_thread_id="t1", source_subject="Q1",
            owner="Finance", status=ActionStatus.OVERDUE, classified_at=datetime.now(timezone.utc),
        )
    ]
    context = format_tool_result_as_context("actions", {"action_items": items})
    assert "Overdue" in context
    assert "Send the breakup" in context
    assert "Finance" in context


def test_format_actions_context_handles_empty_list():
    from chatbot.service import format_tool_result_as_context

    context = format_tool_result_as_context("actions", {"action_items": []})
    assert "no tracked action items" in context.lower()


def test_format_email_context_handles_missing_summary():
    from chatbot.service import format_tool_result_as_context

    context = format_tool_result_as_context("email", {"summary": None})
    assert "no recent email" in context.lower()


def test_format_calendar_context_handles_missing_window():
    from chatbot.service import format_tool_result_as_context

    context = format_tool_result_as_context("calendar", {"meeting_window": None})
    assert "no calendar data" in context.lower()


# ---------------------------------------------------------------------------
# answer_question() - full flow integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_answer_question_without_tool_answers_directly(required_env, monkeypatch):
    from core.ollama_client.client import OllamaClient

    calls = []

    async def fake_chat(self, prompt, system_prompt=None):
        calls.append(prompt)
        if len(calls) == 1:
            return "none"
        return "Hello! I'm doing well, thank you."

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    from chatbot.service import answer_question

    response = await answer_question("Hello, how are you?")

    assert response.tool_used is None
    assert "doing well" in response.answer


@pytest.mark.asyncio
async def test_answer_question_with_actions_tool_grounds_answer(required_env, monkeypatch):
    from core.ollama_client.client import OllamaClient
    from core.storage.factory import reset_storage_for_tests

    reset_storage_for_tests()

    call_count = {"n": 0}

    async def fake_chat(self, prompt, system_prompt=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "actions"
        return "You have no overdue action items right now."

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    from chatbot.service import answer_question

    response = await answer_question("What are my overdue action items?")

    assert response.tool_used == "actions"
    assert "no overdue" in response.answer.lower()

    reset_storage_for_tests()


@pytest.mark.asyncio
async def test_answer_question_with_calendar_tool_grounds_answer(required_env, monkeypatch, calendar_events_fixture_factory):
    from datetime import datetime, timezone

    from core.graph_client.client import GraphClient
    from core.ollama_client.client import OllamaClient

    now = datetime.now(timezone.utc)
    fixture_response = calendar_events_fixture_factory(now)

    async def fake_get_calendar_events(self, bearer_token=None, top=50):
        return fixture_response

    monkeypatch.setattr(GraphClient, "get_calendar_events", fake_get_calendar_events)

    # Three sequential OllamaClient.chat() calls occur in this flow:
    # 1) chatbot's tool-selection call -> "calendar"
    # 2) Calendar agent's brief-generation call (inside flag_for_brief) -> brief text
    # 3) chatbot's final answer-composition call -> the grounded answer
    call_count = {"n": 0}
    responses = ["calendar", "mocked brief text", "Your next meeting is the Wholesale Banking Review."]

    async def fake_chat(self, prompt, system_prompt=None):
        response = responses[call_count["n"]]
        call_count["n"] += 1
        return response

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    from chatbot.service import answer_question

    response = await answer_question("What's my next meeting?")

    assert response.tool_used == "calendar"
    assert "Wholesale Banking Review" in response.answer


# ---------------------------------------------------------------------------
# Routes - integration test
# ---------------------------------------------------------------------------

def test_chatbot_query_route(required_env, monkeypatch):
    from core.config import get_settings
    from core.ollama_client.client import OllamaClient
    from core.storage.factory import reset_storage_for_tests

    reset_storage_for_tests()
    get_settings.cache_clear()

    call_count = {"n": 0}

    async def fake_chat(self, prompt, system_prompt=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "none"
        return "General answer."

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    import importlib

    import main as main_module

    importlib.reload(main_module)

    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as client:
        response = client.post("/chatbot/query", json={"message": "Hello there"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "General answer."
    assert body["tool_used"] is None

    reset_storage_for_tests()
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# LangGraph subgraph
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chatbot_graph_runs_end_to_end(required_env, monkeypatch):
    from core.ollama_client.client import OllamaClient
    from core.storage.factory import reset_storage_for_tests

    reset_storage_for_tests()

    async def fake_chat(self, prompt, system_prompt=None):
        return "none"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    from chatbot.graph import chatbot_graph

    result = await chatbot_graph.ainvoke({"question": "Hi there", "bearer_token": None})

    assert result["response"] is not None
    assert result["response"].tool_used is None

    reset_storage_for_tests()
