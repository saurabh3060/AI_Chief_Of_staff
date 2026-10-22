"""
tests/agent_email/test_email_agent.py
========================================
Unit tests for the Email & Summary agent: subject normalization,
action-candidate extraction, the OCR pipeline (with a mocked vision
model), and prompt construction; plus integration tests for the
`/agents/email/*` routes with mocked Graph/Ollama clients.
"""

import pytest


# ---------------------------------------------------------------------------
# normalize_subject()
# ---------------------------------------------------------------------------

def test_normalize_subject_strips_re_and_fwd_prefixes():
    from agent_email.service import normalize_subject

    assert normalize_subject("RE: Q1 Retail Performance Numbers") == "Q1 Retail Performance Numbers"
    assert normalize_subject("Fwd: RE: Budget Review") == "Budget Review"
    assert normalize_subject("Plain Subject") == "Plain Subject"


# ---------------------------------------------------------------------------
# extract_action_candidates()
# ---------------------------------------------------------------------------

def test_extract_action_candidates_finds_action_prefixed_lines():
    from agent_email.schema import EmailMessage
    from agent_email.service import extract_action_candidates
    from datetime import datetime, timezone

    message = EmailMessage(
        id="msg-001",
        subject="Test",
        received_at=datetime.now(timezone.utc),
        body_text="Some context. Action: Finance team to send the reconciled branch-wise breakup by Friday.",
    )

    candidates = extract_action_candidates(message)

    assert len(candidates) == 1
    assert "Finance team" in candidates[0].text
    assert candidates[0].source_message_id == "msg-001"


def test_extract_action_candidates_returns_empty_when_no_action_line():
    from agent_email.schema import EmailMessage
    from agent_email.service import extract_action_candidates
    from datetime import datetime, timezone

    message = EmailMessage(
        id="msg-002",
        subject="Test",
        received_at=datetime.now(timezone.utc),
        body_text="Just a regular update, nothing actionable here.",
    )

    assert extract_action_candidates(message) == []


# ---------------------------------------------------------------------------
# OCR pipeline (agent_email.ocr.parse_attachment)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parse_attachment_runs_ocr_on_pdf(pdf_attachment_fixture_factory, monkeypatch):
    """A PDF attachment should be converted to page images and OCR'd via the mocked vision model."""
    from core.ollama_client.client import OllamaClient

    async def fake_vision_ocr(self, image_bytes, instruction=None):
        return "RBL Bank - Sample Scanned Document"

    monkeypatch.setattr(OllamaClient, "vision_ocr", fake_vision_ocr)

    from agent_email.ocr import parse_attachment

    raw_attachment = pdf_attachment_fixture_factory()
    parsed = await parse_attachment(raw_attachment)

    assert parsed.ocr_applied is True
    assert "RBL Bank" in parsed.extracted_text


@pytest.mark.asyncio
async def test_parse_attachment_skips_unsupported_file_types():
    """A non-PDF/image attachment (e.g. .xlsx) should not be OCR'd."""
    from agent_email.ocr import parse_attachment

    raw_attachment = {
        "id": "att-002",
        "name": "financials.xlsx",
        "contentType": "application/vnd.ms-excel",
        "contentBytes": "ZmFrZS1jb250ZW50",
    }
    parsed = await parse_attachment(raw_attachment)

    assert parsed.ocr_applied is False
    assert parsed.extracted_text == ""


@pytest.mark.asyncio
async def test_parse_attachment_handles_missing_content_bytes_gracefully():
    """An attachment with no contentBytes should be skipped, not raise."""
    from agent_email.ocr import parse_attachment

    raw_attachment = {"id": "att-003", "name": "report.pdf", "contentType": "application/pdf"}
    parsed = await parse_attachment(raw_attachment)

    assert parsed.ocr_applied is False


# ---------------------------------------------------------------------------
# get_recent_conversation() / summarize_thread() - service-level integration
# ---------------------------------------------------------------------------

@pytest.fixture
def mocked_email_clients(monkeypatch, mail_thread_fixture, pdf_attachment_fixture_factory):
    """
    Patch GraphClient.get_messages/get_attachments/get_attachment_content
    to return the fixture mail thread (which includes one message with
    hasAttachments=True), and OllamaClient.chat/vision_ocr to return
    canned responses, so service-level tests don't need real network/Ollama access.
    """
    from core.graph_client.client import GraphClient
    from core.ollama_client.client import OllamaClient

    async def fake_get_messages(self, bearer_token=None, top=10, skip=0, received_after=None):
        return mail_thread_fixture

    async def fake_get_attachments(self, message_id, bearer_token=None):
        return {"value": [{"id": "att-001", "name": "branch_audit_report.pdf"}]}

    async def fake_get_attachment_content(self, message_id, attachment_id, bearer_token=None):
        return pdf_attachment_fixture_factory(attachment_id=attachment_id)

    async def fake_vision_ocr(self, image_bytes, instruction=None):
        return "Branch Audit Report - all findings closed."

    async def fake_chat(self, prompt, system_prompt=None):
        return "Summary: Q1 retail numbers reviewed, NPA improved.\n- Finance team to send breakup by Friday."

    monkeypatch.setattr(GraphClient, "get_messages", fake_get_messages)
    monkeypatch.setattr(GraphClient, "get_attachments", fake_get_attachments)
    monkeypatch.setattr(GraphClient, "get_attachment_content", fake_get_attachment_content)
    monkeypatch.setattr(OllamaClient, "vision_ocr", fake_vision_ocr)
    monkeypatch.setattr(OllamaClient, "chat", fake_chat)


@pytest.mark.asyncio
async def test_get_recent_conversation_groups_replies_and_ocrs_attachments(required_env, mocked_email_clients):
    from agent_email.service import get_recent_conversation

    thread = await get_recent_conversation()

    assert thread is not None
    assert thread.subject == "Q1 Retail Performance Numbers"
    assert len(thread.messages) == 2  # both fixture messages share the normalized subject
    assert len(thread.attachments) == 1
    assert thread.attachments[0].ocr_applied is True


@pytest.mark.asyncio
async def test_summarize_thread_produces_summary_with_action_candidates(required_env, mocked_email_clients):
    from agent_email.service import get_recent_conversation, summarize_thread

    thread = await get_recent_conversation()
    summary = await summarize_thread(thread)

    assert "Q1 retail" in summary.summary_text
    assert len(summary.action_candidates) >= 1  # extracted from the "Action:" line in the fixture


# ---------------------------------------------------------------------------
# Routes - integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def email_test_client(required_env, monkeypatch, mail_thread_fixture, pdf_attachment_fixture_factory):
    """FastAPI TestClient with GraphClient/OllamaClient patched, similar to mocked_email_clients but wired through main.app."""
    from core.config import get_settings
    from core.graph_client.client import GraphClient
    from core.ollama_client.client import OllamaClient
    from core.storage.factory import reset_storage_for_tests

    monkeypatch.setenv("GRAPH_AUTH_MODE", "app_only")
    get_settings.cache_clear()
    reset_storage_for_tests()

    async def fake_get_messages(self, bearer_token=None, top=10, skip=0, received_after=None):
        return mail_thread_fixture

    async def fake_get_attachments(self, message_id, bearer_token=None):
        return {"value": [{"id": "att-001", "name": "branch_audit_report.pdf"}]}

    async def fake_get_attachment_content(self, message_id, attachment_id, bearer_token=None):
        return pdf_attachment_fixture_factory(attachment_id=attachment_id)

    async def fake_vision_ocr(self, image_bytes, instruction=None):
        return "Branch Audit Report - all findings closed."

    async def fake_chat(self, prompt, system_prompt=None):
        return "Summary: Q1 retail numbers reviewed.\n- Finance team to send breakup by Friday."

    monkeypatch.setattr(GraphClient, "get_messages", fake_get_messages)
    monkeypatch.setattr(GraphClient, "get_attachments", fake_get_attachments)
    monkeypatch.setattr(GraphClient, "get_attachment_content", fake_get_attachment_content)
    monkeypatch.setattr(OllamaClient, "vision_ocr", fake_vision_ocr)
    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    import importlib

    import main as main_module

    importlib.reload(main_module)

    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as test_client:
        yield test_client

    reset_storage_for_tests()
    get_settings.cache_clear()


def test_get_summary_route_is_read_only_and_returns_none_when_nothing_persisted(email_test_client):
    """
    /agents/email/summary no longer generates on demand - it only reads
    whatever has already been persisted. With nothing generated yet
    (no scheduler run in this test), it must return summary: None
    without making any Graph/Ollama call.
    """
    response = email_test_client.get("/agents/email/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] is None


def test_get_summary_route_returns_persisted_summary_without_generating(email_test_client):
    """Once a summary has been persisted (e.g. by the scheduler/graph), the route should surface it read-only."""
    import asyncio

    from agent_email.service import get_recent_conversation, get_or_generate_summary_for_thread

    async def _seed():
        thread = await get_recent_conversation()
        await get_or_generate_summary_for_thread(thread)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = email_test_client.get("/agents/email/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] is not None
    assert "Q1 retail" in body["summary"]["summary_text"]


def test_list_summaries_route_returns_persisted_summaries(email_test_client):
    import asyncio

    from agent_email.service import get_recent_conversation, get_or_generate_summary_for_thread

    async def _seed():
        thread = await get_recent_conversation()
        await get_or_generate_summary_for_thread(thread)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = email_test_client.get("/agents/email/summaries")

    assert response.status_code == 200
    body = response.json()
    assert len(body["summaries"]) == 1


def test_summary_route_requires_bearer_token_in_delegated_mode(required_env, monkeypatch):
    from core.config import get_settings

    monkeypatch.setenv("GRAPH_AUTH_MODE", "delegated")
    get_settings.cache_clear()

    import importlib

    import main as main_module

    importlib.reload(main_module)

    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as test_client:
        response = test_client.get("/agents/email/summary")

    assert response.status_code == 401
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# LangGraph subgraph
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_email_graph_runs_end_to_end(required_env, mocked_email_clients):
    from core.storage.factory import reset_storage_for_tests

    reset_storage_for_tests()

    from agent_email.graph import email_graph

    result = await email_graph.ainvoke({"bearer_token": None})

    assert result["thread"] is not None
    assert result["summary"] is not None
    assert "Q1 retail" in result["summary"].summary_text

    reset_storage_for_tests()

