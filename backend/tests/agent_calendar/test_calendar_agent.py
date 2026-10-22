"""
tests/agent_calendar/test_calendar_agent.py
==============================================
Unit tests for the meeting-detection logic in
`agent_calendar.service.detect_current_and_next_meeting` (pure function,
given fixture events), plus integration tests for the
`/agents/calendar/*` routes with a mocked `GraphClient`.
"""

from datetime import datetime, timedelta, timezone

import pytest


def _event(event_id: str, subject: str, start: datetime, end: datetime):
    from agent_calendar.schema import CalendarEvent

    return CalendarEvent(id=event_id, subject=subject, start=start, end=end, attendee_emails=[])


# ---------------------------------------------------------------------------
# detect_current_and_next_meeting() - pure logic
# ---------------------------------------------------------------------------

def test_detects_current_meeting_in_progress():
    """A meeting whose [start, end) window contains 'now' should be returned as current_meeting."""
    from agent_calendar.service import detect_current_and_next_meeting

    now = datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc)
    events = [
        _event("evt-1", "Ongoing Sync", now - timedelta(minutes=15), now + timedelta(minutes=45)),
    ]

    window = detect_current_and_next_meeting(events, as_of=now)

    assert window.current_meeting is not None
    assert window.current_meeting.id == "evt-1"
    assert window.next_meeting is None


def test_detects_next_meeting_when_none_in_progress():
    """When no meeting is currently happening, the earliest future meeting should be next_meeting."""
    from agent_calendar.service import detect_current_and_next_meeting

    now = datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc)
    events = [
        _event("evt-past", "Past Meeting", now - timedelta(hours=2), now - timedelta(hours=1)),
        _event("evt-far", "Far Future Meeting", now + timedelta(hours=5), now + timedelta(hours=6)),
        _event("evt-near", "Near Future Meeting", now + timedelta(hours=1), now + timedelta(hours=2)),
    ]

    window = detect_current_and_next_meeting(events, as_of=now)

    assert window.current_meeting is None
    assert window.next_meeting is not None
    assert window.next_meeting.id == "evt-near"


def test_no_meetings_returns_empty_window():
    """With no events at all, both current_meeting and next_meeting should be None."""
    from agent_calendar.service import detect_current_and_next_meeting

    window = detect_current_and_next_meeting([], as_of=datetime.now(timezone.utc))

    assert window.current_meeting is None
    assert window.next_meeting is None


def test_double_booked_picks_most_recently_started_as_current():
    """When two meetings overlap 'now', the one that started most recently should be treated as current."""
    from agent_calendar.service import detect_current_and_next_meeting

    now = datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc)
    events = [
        _event("evt-earlier", "Earlier Overlapping", now - timedelta(hours=1), now + timedelta(hours=1)),
        _event("evt-later", "Later Overlapping", now - timedelta(minutes=10), now + timedelta(minutes=50)),
    ]

    window = detect_current_and_next_meeting(events, as_of=now)

    assert window.current_meeting is not None
    assert window.current_meeting.id == "evt-later"


# ---------------------------------------------------------------------------
# filter_todays_events() - pure logic
# ---------------------------------------------------------------------------

def test_filter_todays_events_excludes_events_on_other_days():
    """Events starting on a different UTC calendar day than 'now' should be excluded."""
    from agent_calendar.service import filter_todays_events

    now = datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc)
    events = [
        _event("evt-yesterday", "Yesterday's Meeting", now - timedelta(days=1), now - timedelta(days=1) + timedelta(hours=1)),
        _event("evt-today-morning", "Today Morning", now.replace(hour=9), now.replace(hour=9, minute=30)),
        _event("evt-tomorrow", "Tomorrow's Meeting", now + timedelta(days=1), now + timedelta(days=1) + timedelta(hours=1)),
    ]

    today = filter_todays_events(events, as_of=now)

    assert [e.id for e in today.events] == ["evt-today-morning"]


def test_filter_todays_events_sorts_by_start_time():
    """Today's events should be returned sorted by start time, regardless of input order."""
    from agent_calendar.service import filter_todays_events

    now = datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc)
    events = [
        _event("evt-late", "Late Meeting", now.replace(hour=15), now.replace(hour=16)),
        _event("evt-early", "Early Meeting", now.replace(hour=8), now.replace(hour=9)),
    ]

    today = filter_todays_events(events, as_of=now)

    assert [e.id for e in today.events] == ["evt-early", "evt-late"]


def test_filter_todays_events_empty_when_no_events():
    """With no events at all, today's events list should be empty."""
    from agent_calendar.service import filter_todays_events

    today = filter_todays_events([], as_of=datetime.now(timezone.utc))

    assert today.events == []


# ---------------------------------------------------------------------------
# Routes - integration tests with a mocked GraphClient
# ---------------------------------------------------------------------------

@pytest.fixture
def calendar_test_client(required_env, monkeypatch, calendar_events_fixture_factory):
    """
    Build a FastAPI TestClient with GraphClient.get_calendar_events
    patched to return the dynamic fixture (relative to real "now"), so
    route tests don't depend on real Graph API/Ollama access.
    """
    from core.config import get_settings
    from core.storage.factory import reset_storage_for_tests

    monkeypatch.setenv("GRAPH_AUTH_MODE", "app_only")  # skip bearer-token requirement for route tests
    get_settings.cache_clear()
    reset_storage_for_tests()

    now = datetime.now(timezone.utc)
    fixture_response = calendar_events_fixture_factory(now)

    from core.graph_client.client import GraphClient

    async def fake_get_calendar_events(self, bearer_token=None, top=50):
        return fixture_response

    async def fake_get_calendar_view(self, start_date_time, end_date_time, bearer_token=None, top=50):
        return fixture_response

    monkeypatch.setattr(GraphClient, "get_calendar_events", fake_get_calendar_events)
    monkeypatch.setattr(GraphClient, "get_calendar_view", fake_get_calendar_view)

    from agent_email import service as email_service

    async def fake_precompute_meeting_embeddings(raw_events, settings=None):
        return 0

    monkeypatch.setattr(email_service, "precompute_meeting_embeddings", fake_precompute_meeting_embeddings)

    import importlib

    import main as main_module

    importlib.reload(main_module)

    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as test_client:
        yield test_client

    reset_storage_for_tests()
    get_settings.cache_clear()


def test_get_current_meeting_route_returns_in_progress_meeting(calendar_test_client):
    """GET /agents/calendar/current should return the fixture's currently in-progress meeting."""
    response = calendar_test_client.get("/agents/calendar/current")

    assert response.status_code == 200
    body = response.json()
    assert body["current_meeting"] is not None
    assert body["current_meeting"]["subject"] == "CEO x CRO Weekly Sync"


def test_get_next_meeting_route_returns_upcoming_meeting(calendar_test_client):
    """GET /agents/calendar/next should return the fixture's next upcoming meeting."""
    response = calendar_test_client.get("/agents/calendar/next")

    assert response.status_code == 200
    body = response.json()
    assert body["next_meeting"] is not None
    assert body["next_meeting"]["subject"] == "Wholesale Banking Review"


def test_get_today_events_route_returns_all_of_todays_meetings(calendar_test_client):
    """GET /agents/calendar/today should return every fixture event that falls on today, sorted by start time."""
    response = calendar_test_client.get("/agents/calendar/today")

    assert response.status_code == 200
    body = response.json()
    subjects = [e["subject"] for e in body["events"]]
    assert subjects == ["Q1 Retail Review", "CEO x CRO Weekly Sync", "Wholesale Banking Review"]


def test_get_brief_route_generates_and_returns_brief(calendar_test_client, monkeypatch):
    """GET /agents/calendar/brief should render a brief from a matched precomputed cluster."""
    from agent_email.schema import EmailClusterRecord
    from agent_email import service as email_service

    async def fake_match_meeting_to_cluster(meeting_title, meeting_body="", settings=None, **kwargs):
        return EmailClusterRecord(
            cluster_id="cluster_test",
            department="Wholesale Banking",
            topic="Loan book review",
            summary="Discuss wholesale loan book growth|Review NPA trend",
            top_questions="What is the NPA trend?",
            action_items="",
            email_refs="[]",
            email_count=1,
            cluster_embedding="[]",
            updated_at="2026-08-31T10:00:00+00:00",
        )

    monkeypatch.setattr(email_service, "match_meeting_to_cluster", fake_match_meeting_to_cluster)

    response = calendar_test_client.get("/agents/calendar/brief")

    assert response.status_code == 200
    body = response.json()
    assert body["brief"] is not None
    assert body["brief"]["meeting_subject"] == "Wholesale Banking Review"
    assert "wholesale loan book" in body["brief"]["brief_text"]


def test_get_brief_route_reuses_stored_brief_on_second_call(calendar_test_client, monkeypatch):
    """A second call to /agents/calendar/brief for the same meeting should reuse the stored brief."""
    from agent_email.schema import EmailClusterRecord
    from agent_email import service as email_service

    call_count = {"n": 0}

    async def fake_match_meeting_to_cluster(meeting_title, meeting_body="", settings=None, **kwargs):
        call_count["n"] += 1
        return EmailClusterRecord(
            cluster_id="cluster_test",
            department="Wholesale Banking",
            topic="Loan book review",
            summary="generated brief text",
            top_questions="",
            action_items="",
            email_refs="[]",
            email_count=1,
            cluster_embedding="[]",
            updated_at="2026-08-31T10:00:00+00:00",
        )

    monkeypatch.setattr(email_service, "match_meeting_to_cluster", fake_match_meeting_to_cluster)

    first = calendar_test_client.get("/agents/calendar/brief")
    second = calendar_test_client.get("/agents/calendar/brief")

    assert first.status_code == 200
    assert second.status_code == 200
    assert call_count["n"] == 1


def test_current_meeting_route_requires_bearer_token_in_delegated_mode(required_env, monkeypatch):
    """In delegated auth mode, /agents/calendar/current without an Authorization header should 401."""
    from core.config import get_settings

    monkeypatch.setenv("GRAPH_AUTH_MODE", "delegated")
    get_settings.cache_clear()

    import importlib

    import main as main_module

    importlib.reload(main_module)

    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as test_client:
        response = test_client.get("/agents/calendar/current")

    assert response.status_code == 401
    get_settings.cache_clear()



# ---------------------------------------------------------------------------
# LangGraph subgraph
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_calendar_graph_runs_end_to_end(required_env, monkeypatch, calendar_events_fixture_factory):
    """The compiled calendar_graph should run fetch_and_detect -> flag_for_brief and populate state."""
    from core.config import get_settings
    from core.storage.factory import reset_storage_for_tests

    monkeypatch.setenv("GRAPH_AUTH_MODE", "app_only")
    get_settings.cache_clear()
    reset_storage_for_tests()

    now = datetime.now(timezone.utc)
    fixture_response = calendar_events_fixture_factory(now)

    from agent_email.schema import EmailClusterRecord
    from agent_email import service as email_service
    from core.graph_client.client import GraphClient

    async def fake_get_calendar_events(self, bearer_token=None, top=50):
        return fixture_response

    async def fake_match_meeting_to_cluster(meeting_title, meeting_body="", settings=None, **kwargs):
        return EmailClusterRecord(
            cluster_id="cluster_graph",
            department="Wholesale Banking",
            topic="Review",
            summary="mocked brief text",
            top_questions="",
            action_items="",
            email_refs="[]",
            email_count=1,
            cluster_embedding="[]",
            updated_at="2026-08-31T10:00:00+00:00",
        )

    monkeypatch.setattr(GraphClient, "get_calendar_events", fake_get_calendar_events)
    monkeypatch.setattr(email_service, "match_meeting_to_cluster", fake_match_meeting_to_cluster)

    from agent_calendar.graph import calendar_graph

    result = await calendar_graph.ainvoke({"bearer_token": None})

    assert result["meeting_window"].next_meeting is not None
    assert result["brief"] is not None
    assert result["brief"].brief_text == "mocked brief text"

    reset_storage_for_tests()
    get_settings.cache_clear()


def test_generic_brief_questions_are_dropped_unless_they_cite_a_figure():
    from agent_calendar.service import _is_generic_brief_question

    assert _is_generic_brief_question("What is the current wholesale liability position?")
    assert _is_generic_brief_question("How do we manage corporate exposure?")
    assert _is_generic_brief_question("What is the purpose of this meeting?")
    assert not _is_generic_brief_question(
        "Why is standard-case TAT still 7.1 days against the 6-day target?"
    )
