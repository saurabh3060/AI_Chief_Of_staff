"""
tests/core/test_orchestrator.py
==================================
Verifies the top-level LangGraph orchestrator: routing to the correct
agent node based on intent, correct state population, and that
Langfuse callbacks are attached (or omitted) based on the observability
toggle.

Only the outermost dependencies (Graph API, Ollama) are mocked; the
real compiled agent subgraphs run, so this proves the orchestrator
wiring dispatches to and collects results from each agent correctly.
"""

import pytest

from core.ollama_client.client import OllamaClient


@pytest.mark.asyncio
async def test_orchestrator_routes_to_calendar_agent(monkeypatch, calendar_events_fixture_factory):
    """intent='calendar' should invoke the Calendar agent subgraph and populate calendar_result."""
    from datetime import datetime, timezone

    from core.config import get_settings
    from core.graph_client.client import GraphClient
    from core.storage.factory import reset_storage_for_tests

    get_settings.cache_clear()
    reset_storage_for_tests()

    now = datetime.now(timezone.utc)
    calendar_fixture = calendar_events_fixture_factory(now)

    async def fake_get_calendar_events(self, bearer_token=None, top=50):
        return calendar_fixture

    async def fake_chat(self, prompt, system_prompt=None):
        return "- Discuss wholesale loan book growth\n- Review NPA trend"

    monkeypatch.setattr(GraphClient, "get_calendar_events", fake_get_calendar_events)
    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    from core.orchestrator.graph import run_agent

    result = await run_agent("calendar")

    assert result.get("calendar_result") is not None
    assert result["calendar_result"]["meeting_window"] is not None


@pytest.mark.asyncio
async def test_orchestrator_routes_to_email_agent(monkeypatch, mail_thread_fixture):
    """intent='email' should invoke the Email agent subgraph and populate email_result."""
    from core.graph_client.client import GraphClient
    from core.storage.factory import reset_storage_for_tests

    reset_storage_for_tests()

    async def fake_get_messages(self, bearer_token=None, top=10, skip=0, received_after=None):
        return mail_thread_fixture

    async def fake_chat(self, prompt, system_prompt=None):
        return "Summary: Q1 retail numbers reviewed, NPA improved."

    monkeypatch.setattr(GraphClient, "get_messages", fake_get_messages)
    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    from core.orchestrator.graph import run_agent

    result = await run_agent("email")

    assert result.get("email_result") is not None


@pytest.mark.asyncio
async def test_orchestrator_routes_to_action_tracker_agent():
    """intent='actions' should invoke the Action Tracker agent subgraph (with no summaries, an empty list)."""
    from core.storage.factory import reset_storage_for_tests

    reset_storage_for_tests()

    from core.orchestrator.graph import run_agent

    result = await run_agent("actions")

    assert result["action_result"] is not None


@pytest.mark.asyncio
async def test_orchestrator_ends_immediately_on_unrecognized_intent():
    """An unrecognized intent should route straight to END without populating any *_result field."""
    from core.orchestrator.graph import run_agent

    result = await run_agent("not_a_real_intent")  # type: ignore[arg-type]

    assert result.get("calendar_result") is None
    assert result.get("email_result") is None
    assert result.get("action_result") is None


@pytest.mark.asyncio
async def test_run_agent_passes_no_callbacks_when_langfuse_disabled(monkeypatch):
    """When Langfuse is disabled (default), run_agent() should invoke the graph without a callbacks config."""
    from core.orchestrator import graph as graph_module

    captured_config = {}

    async def fake_ainvoke(self, state, config=None):
        captured_config["config"] = config
        return {"action_items": []}

    monkeypatch.setattr(graph_module.orchestrator_graph.__class__, "ainvoke", fake_ainvoke)

    from core.orchestrator.graph import run_agent

    await run_agent("actions")

    assert captured_config["config"] is None
