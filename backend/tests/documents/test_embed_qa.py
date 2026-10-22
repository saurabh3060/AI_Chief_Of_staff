"""
tests/documents/test_embed_qa.py
=================================
Tests for the embedqa and retrieve QA pipelines.

Adapted from the SQLite reference to async PostgreSQL, matching the project's
production stack (asyncpg + store.acquire()).

Tests are marked ``@pytest.mark.integration`` because they require a live
PostgreSQL database with at least one ingested document (CC12.pdf) and its
associated series/facts/calculated_metrics.

Run:
    pytest tests/documents/test_embed_qa.py -v
    pytest tests/documents/test_embed_qa.py -k "not integration" -v   # unit only
"""
from __future__ import annotations

import json

import pytest

from documents import embedqa
from documents import retrieve
from documents import store


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

integration = pytest.mark.integration


# ---------------------------------------------------------------------------
# Async DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture
async def conn():
    """Acquire a pooled asyncpg connection for one test, then release it."""
    async with store.acquire() as c:
        yield c


# ---------------------------------------------------------------------------
# Helper: collect events from an async generator
# ---------------------------------------------------------------------------

async def _collect_events(async_gen):
    """Consume an async generator and return a list of (kind, value) tuples."""
    events = []
    async for event in async_gen:
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Unit tests — _plaintext_tool_request (pure logic, no DB)
# ---------------------------------------------------------------------------

def test_plaintext_calculate_request_is_recovered_without_inferring_formula():
    assert embedqa._plaintext_tool_request('calculate("278 / 21,168 * 100")') == (
        "calculate", {"expression": "278 / 21,168 * 100"}
    )


def test_plaintext_run_metric_json_is_recovered():
    content = '''run_metric for this row:\n```json\n{
    "label": "Write Off", "period_label": "Q3FY26", "page": "p.5"
    }\n```'''
    assert embedqa._plaintext_tool_request(content) == (
        "run_metric", {"label": "Write Off", "period": "Q3FY26"}
    )


def test_plaintext_run_metric_sentence_is_recovered():
    content = 'run_metric for label \u201cCIF Net Addition\u201d with period matching FY26 (period_label FY26).'
    assert embedqa._plaintext_tool_request(content) == (
        "run_metric", {"label": "CIF Net Addition", "period": "FY26"}
    )


def test_normal_answer_is_not_interpreted_as_a_tool_request():
    assert embedqa._plaintext_tool_request("Spend grew by 1.2% quarter over quarter.") is None


# ---------------------------------------------------------------------------
# Integration tests — _candidate_pages (soft router)
# ---------------------------------------------------------------------------

@integration
@pytest.mark.asyncio
async def test_soft_router_uses_document_metadata_without_a_supplied_page(conn):
    doc_ids = ["2a7323461ad9438b"]
    pages = await embedqa._candidate_pages(conn, "What share of ENR was Non-Term in Feb'26?", doc_ids)
    assert pages and pages[0] == 6
    pages = await embedqa._candidate_pages(conn, "How much was written off in Q3 FY26?", doc_ids)
    assert pages and pages[0] == 5
    pages = await embedqa._candidate_pages(conn, "How did Partner Sourcing volume change?", doc_ids)
    assert pages and pages[0] == 3


@integration
@pytest.mark.asyncio
async def test_soft_router_falls_back_for_a_deictic_question_with_no_metric_context(conn):
    result = await embedqa._candidate_pages(
        conn, "Is there a data-quality issue on this page?", ["2a7323461ad9438b"]
    )
    assert result == []


# ---------------------------------------------------------------------------
# Integration tests — _run_metric
# ---------------------------------------------------------------------------

@integration
@pytest.mark.asyncio
async def test_metric_lookup_uses_available_tagged_series_when_preferred_page_has_no_fact(conn):
    await embedqa._make_facts_view(conn, ["2a7323461ad9438b"], {})
    rows = json.loads(await embedqa._run_metric(conn, "cc_spends", "Q3FY26", [15]))
    assert {row["page"] for row in rows} == {5}
    assert rows[0]["value"] == 21446


# ---------------------------------------------------------------------------
# Integration tests — full pipeline short-circuits (no LLM needed)
# ---------------------------------------------------------------------------

@integration
@pytest.mark.asyncio
async def test_extremum_questions_short_circuit_the_model_in_both_pipelines(monkeypatch):
    def model_must_not_run(*args, **kwargs):
        raise AssertionError("model or embedding call should not run for a resolved extremum")

    monkeypatch.setattr(embedqa, "_candidate_pages", _async_stub([99]))
    monkeypatch.setattr(embedqa.llm, "embed", model_must_not_run)
    monkeypatch.setattr(embedqa.llm, "chat_message", model_must_not_run)
    monkeypatch.setattr(retrieve.llm, "chat_message", model_must_not_run)

    expected = "The lowest Month-end CIF was 4,541,171.00 in Aug-25. (p.4)"
    for answerer in (retrieve.answer_events, embedqa.answer_events):
        events = await _collect_events(answerer(
            "In which month was CIF lowest?", [],
            doc_ids=["2a7323461ad9438b"], reasoning=True,
        ))
        answer = next(value for kind, value in events if kind == "answer")
        assert answer == expected


@integration
@pytest.mark.asyncio
async def test_grouped_period_comparison_short_circuits_the_model_in_both_pipelines(monkeypatch):
    def model_must_not_run(*args, **kwargs):
        raise AssertionError("model or embedding call should not run for a resolved comparison")

    monkeypatch.setattr(embedqa, "_candidate_pages", _async_stub([99]))
    monkeypatch.setattr(embedqa.llm, "embed", model_must_not_run)
    monkeypatch.setattr(embedqa.llm, "chat_message", model_must_not_run)
    monkeypatch.setattr(retrieve.llm, "chat_message", model_must_not_run)

    question = "How does net addition compare between H1 and Oct\u2013Feb?"
    for answerer in (retrieve.answer_events, embedqa.answer_events):
        events = await _collect_events(answerer(
            question, [], doc_ids=["2a7323461ad9438b"], reasoning=True,
        ))
        answer = next(value for kind, value in events if kind == "answer")
        assert "-242,916 over H1" in answer
        assert "23,617 over Oct\u2013Feb" in answer


@integration
@pytest.mark.asyncio
async def test_direct_write_off_lookup_short_circuits_the_model_in_both_pipelines(monkeypatch):
    def model_must_not_run(*args, **kwargs):
        raise AssertionError("model or embedding call should not run for a resolved fact")

    monkeypatch.setattr(embedqa, "_candidate_pages", _async_stub([5]))
    monkeypatch.setattr(embedqa.llm, "embed", model_must_not_run)
    monkeypatch.setattr(embedqa.llm, "chat_message", model_must_not_run)
    monkeypatch.setattr(retrieve.llm, "chat_message", model_must_not_run)

    for answerer in (retrieve.answer_events, embedqa.answer_events):
        events = await _collect_events(answerer(
            "How much was written off in Q3 FY26?", [],
            doc_ids=["2a7323461ad9438b"], reasoning=True,
        ))
        answer = next(value for kind, value in events if kind == "answer")
        assert "Write Off was 663.00 Cr in Q3FY26" in answer
        assert "reported as -663.00 Cr in the source walk" in answer


@integration
@pytest.mark.asyncio
async def test_unscoped_source_facts_short_circuit_the_model_in_both_pipelines(monkeypatch):
    def model_must_not_run(*args, **kwargs):
        raise AssertionError("model or embedding call should not run for an exact source fact")

    monkeypatch.setattr(embedqa.llm, "embed", model_must_not_run)
    monkeypatch.setattr(embedqa.llm, "chat_message", model_must_not_run)
    monkeypatch.setattr(retrieve.llm, "chat_message", model_must_not_run)

    cases = [
        ("What is RFL's NPA recovery rate?", "RFL is 0.82%"),
        ("What were total advances in June 2026?", "Jun-26 is \u20b9 116,226"),
    ]
    for answerer in (retrieve.answer_events, embedqa.answer_events):
        for question, expected in cases:
            events = await _collect_events(answerer(question, [], reasoning=True))
            answer = next(value for kind, value in events if kind == "answer")
            assert expected in answer
            assert next(value for kind, value in events if kind == "passages") == []


@integration
@pytest.mark.asyncio
async def test_runtime_validated_growth_short_circuits_the_model_in_both_pipelines(monkeypatch):
    def model_must_not_run(*args, **kwargs):
        raise AssertionError("model or embedding call should not run for a runtime-validated calculation")

    monkeypatch.setattr(embedqa, "_candidate_pages", _async_stub([3]))
    monkeypatch.setattr(embedqa.llm, "embed", model_must_not_run)
    monkeypatch.setattr(embedqa.llm, "chat_message", model_must_not_run)
    monkeypatch.setattr(retrieve.llm, "chat_message", model_must_not_run)

    question = "What is the growth rate for proprietary sourcing from Jun-25 to Aug-25?"
    for answerer in (retrieve.answer_events, embedqa.answer_events):
        events = await _collect_events(answerer(
            question, [], doc_ids=["2a7323461ad9438b"], page=3,
        ))
        answer = next(value for kind, value in events if kind == "answer")
        assert "18.78%" in answer


@integration
@pytest.mark.asyncio
async def test_business_approved_premium_share_range_short_circuits_model(monkeypatch):
    def model_must_not_run(*args, **kwargs):
        raise AssertionError("model or embedding call should not run for a reviewed metric")

    monkeypatch.setattr(embedqa, "_candidate_pages", _async_stub([3]))
    monkeypatch.setattr(embedqa.llm, "embed", model_must_not_run)
    monkeypatch.setattr(embedqa.llm, "chat_message", model_must_not_run)
    monkeypatch.setattr(retrieve.llm, "chat_message", model_must_not_run)

    question = "What was the Premium share range?"
    for answerer in (retrieve.answer_events, embedqa.answer_events):
        events = await _collect_events(answerer(
            question, [], doc_ids=["2a7323461ad9438b"], page=3,
        ))
        answer = next(value for kind, value in events if kind == "answer")
        assert "Premium share range: 0.21% to 0.46%" in answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _async_stub(return_value):
    """Create an async function stub that always returns `return_value`.

    Used to monkeypatch async functions like _candidate_pages.
    """
    async def _stub(*args, **kwargs):
        return return_value
    return _stub
