"""
agent_calendar/service.py
============================
Business logic for the Calendar & Meeting agent.
"""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta, timezone

from core.config import IST, Settings, get_settings
from core.graph_client.client import GraphClient
from core.logging_config import get_agent_logger
from core.storage.base import StorageBackend
from core.storage.factory import get_storage
from core.text_match import html_to_text

from agent_calendar.schema import CalendarEvent, MeetingBrief, MeetingWindow, TodayEvents

logger = get_agent_logger("agent_calendar")


def _parse_event(raw_event: dict, settings: Settings | None = None) -> CalendarEvent:
    """Convert a raw Microsoft Graph API event JSON object into a normalized `CalendarEvent`."""
    organizer = raw_event.get("organizer", {}).get("emailAddress", {})
    attendees = raw_event.get("attendees", [])
    attendee_emails = [
        a.get("emailAddress", {}).get("address")
        for a in attendees
        if a.get("emailAddress", {}).get("address")
    ]
    body = raw_event.get("body", {})
    body_snippet = html_to_text(
        body.get("content", raw_event.get("bodyPreview", "")),
        body.get("contentType", ""),
    )
    return CalendarEvent(
        id=raw_event["id"],
        subject=raw_event.get("subject", "(no subject)"),
        start=_parse_graph_datetime(raw_event["start"], settings=settings),
        end=_parse_graph_datetime(raw_event["end"], settings=settings),
        organizer_name=organizer.get("name"),
        organizer_email=organizer.get("address"),
        attendee_emails=attendee_emails,
        body_snippet=body_snippet,
        matched_view=None,
    )


def _parse_graph_datetime(graph_datetime: dict, settings: Settings | None = None) -> datetime:
    active_settings = settings or get_settings()
    raw = graph_datetime["dateTime"]
    tz_name = graph_datetime.get("timeZone", "UTC")
    if "." in raw:
        date_part, frac_part = raw.split(".", 1)
        raw = f"{date_part}.{frac_part[:6]}"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is not None:
        return parsed

    if tz_name and tz_name.upper() == "UTC" and active_settings.calendar_source_is_mislabeled_utc:
        parsed = parsed.replace(tzinfo=IST)
        converted = parsed.astimezone(timezone.utc)
        logger.info(
            "Calendar time fix: raw dateTime='%s' (labeled UTC, treated as IST) -> true UTC='%s'",
            raw, converted.isoformat(),
        )
        return converted

    if tz_name and tz_name.upper() != "UTC":
        try:
            from zoneinfo import ZoneInfo

            parsed = parsed.replace(tzinfo=ZoneInfo(_windows_tz_to_iana(tz_name)))
            return parsed.astimezone(timezone.utc)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Unrecognized Graph timeZone '%s' for dateTime '%s'; treating as UTC.",
                tz_name, raw,
            )

    return parsed.replace(tzinfo=timezone.utc)


_WINDOWS_TO_IANA_TZ = {
    "India Standard Time": "Asia/Kolkata",
    "UTC": "UTC",
}


def _windows_tz_to_iana(windows_tz_name: str) -> str:
    return _WINDOWS_TO_IANA_TZ.get(windows_tz_name, windows_tz_name)


def detect_current_and_next_meeting(events: list[CalendarEvent], as_of: datetime | None = None) -> MeetingWindow:
    now = as_of or datetime.now(timezone.utc)
    current_meeting = None
    next_meeting = None
    earliest_next_start = None

    for event in events:
        if event.start <= now < event.end:
            if current_meeting is None or event.start > current_meeting.start:
                current_meeting = event
        elif event.start > now:
            if earliest_next_start is None or event.start < earliest_next_start:
                earliest_next_start = event.start
                next_meeting = event

    return MeetingWindow(as_of=now, current_meeting=current_meeting, next_meeting=next_meeting)


def filter_todays_events(events: list[CalendarEvent], as_of: datetime | None = None) -> TodayEvents:
    now = as_of or datetime.now(timezone.utc)
    today_ist = now.astimezone(IST).date()
    todays = sorted(
        (event for event in events if event.start.astimezone(IST).date() == today_ist),
        key=lambda event: event.start,
    )
    return TodayEvents(as_of=now, events=todays)


async def get_current_and_next_meeting(bearer_token: str | None = None, settings: Settings | None = None) -> MeetingWindow:
    active_settings = settings or get_settings()
    client = GraphClient(settings=active_settings)
    raw_response = await client.get_calendar_events(bearer_token=bearer_token)
    events = [_parse_event(raw_event, settings=active_settings) for raw_event in raw_response.get("value", [])]
    window = detect_current_and_next_meeting(events)
    logger.info(
        "Detected meeting window: current=%s next=%s",
        window.current_meeting.subject if window.current_meeting else None,
        window.next_meeting.subject if window.next_meeting else None,
    )
    return window


async def get_todays_events(bearer_token: str | None = None, settings: Settings | None = None) -> TodayEvents:
    active_settings = settings or get_settings()
    client = GraphClient(settings=active_settings)

    now = datetime.now(timezone.utc)
    now_ist = now.astimezone(IST)
    day_start_ist = datetime.combine(now_ist.date(), time.min, tzinfo=IST)
    day_end_ist = day_start_ist + timedelta(days=1)
    day_start = day_start_ist.astimezone(timezone.utc)
    day_end = day_end_ist.astimezone(timezone.utc)

    raw_response = await client.get_calendar_view(
        start_date_time=day_start.isoformat(),
        end_date_time=day_end.isoformat(),
        bearer_token=bearer_token,
        top=50,
    )
    raw_events = raw_response.get("value", [])
    from agent_email.service import precompute_meeting_embeddings

    try:
        await precompute_meeting_embeddings(raw_events, settings=active_settings)
    except Exception as exc:  # noqa: BLE001
        # Embedding precomputation is a non-fatal side-effect used only for
        # pre-meeting brief matching.  Never let it block the calendar response.
        logger.warning("precompute_meeting_embeddings failed (non-fatal): %s", exc)
    events = [_parse_event(raw_event, settings=active_settings) for raw_event in raw_events]
    today = filter_todays_events(events, as_of=now)
    await _tag_events_with_department_view(today.events, settings=active_settings)
    logger.info("Fetched %d event(s) for today", len(today.events))
    return today


async def _tag_events_with_department_view(events: list[CalendarEvent], settings: Settings | None = None) -> None:
    """
    Embeds calendar meetings once, then matches each title against
    precomputed cluster embeddings. Sets `matched_view` from the cluster
    department map, or leaves it None when nothing clears the confidence gate.
    """
    active_settings = settings or get_settings()
    view_map = active_settings.department_view_map

    from agent_email.service import match_meeting_to_cluster

    for event in events:
        try:
            cluster = await match_meeting_to_cluster(
                event.subject,
                event.body_snippet,
                settings=active_settings,
                meeting_id=event.id,
            )
            if cluster is None or not cluster.department:
                event.matched_view = None
                continue
            view = view_map.get(cluster.department.lower())
            event.matched_view = view
            logger.info(
                "[MEETING-MATCH] event id=%s subject=%r -> department=%r view=%r topic=%r",
                event.id, event.subject, cluster.department, view, cluster.topic,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to tag event id=%s with department view: %s", event.id, exc)


_GENERIC_MEETING_QUESTION_RE = re.compile(
    r"(?:"
    r"\b(?:subject|purpose|topic|expected\s+outcome)\s+of\s+(?:this|the)\s+meeting\b"
    r"|^what\s+(?:is|are)\s+the\s+(?:purpose|agenda|risks|timeline|next\s+steps|expected\s+outcomes?)\??$"
    r"|\bwho\s+(?:is\s+attending|organized)\b"
    r"|^what\s+is\s+the\s+current\s+.+\s+position\??$"
    r"|^how\s+do\s+we\s+manage\s+.+\??$"
    r"|^what\s+are\s+the\s+implications\s+(?:of\s+)?.+\??$"
    r")",
    re.IGNORECASE,
)
_GENERIC_FILLER_QUESTION_RE = re.compile(
    r"\b(?:current position|how do we|implications of|what is the current)\b",
    re.IGNORECASE,
)


def _is_generic_brief_question(question: str) -> bool:
    if _GENERIC_MEETING_QUESTION_RE.search(question):
        return True
    if re.search(r"\d", question):
        return False
    return bool(_GENERIC_FILLER_QUESTION_RE.search(question))


def _cluster_context_from_record(cluster) -> dict:
    summary = [s.strip() for s in (cluster.summary or "").split("|") if s.strip()]
    questions = [q.strip() for q in (cluster.top_questions or "").split("|") if q.strip()]
    action_items = [a.strip() for a in (cluster.action_items or "").split("|") if a.strip()]
    return {
        "department": cluster.department,
        "topics": [cluster.topic] if cluster.topic else [],
        "summary": summary,
        "questions": questions,
        "action_items": action_items,
    }


def _render_brief_text(cluster_context: dict | None) -> tuple[str, list[str]]:
    """Deterministically assemble brief text from precomputed cluster fields."""
    if not cluster_context:
        return "", []

    topics = [str(t).strip() for t in cluster_context.get("topics", []) if str(t).strip()]
    summary = [str(b).strip() for b in cluster_context.get("summary", []) if str(b).strip()]
    action_items = cluster_context.get("action_items", []) or []

    if not summary and not topics:
        return "", []

    lines: list[str] = []
    if topics:
        lines.append("Related topics: " + "; ".join(topics) + ".")
    lines.extend(f"- {bullet}" for bullet in summary)
    for ai in action_items:
        if isinstance(ai, dict):
            task = str(ai.get("task", "")).strip()
            if not task:
                continue
            owner = str(ai.get("owner") or "TBA").strip() or "TBA"
            deadline = str(ai.get("deadline") or "TBA").strip() or "TBA"
            lines.append(f"- Action ({owner}, by {deadline}): {task}")
        elif isinstance(ai, str) and ai.strip():
            lines.append(f"- Action: {ai.strip()}")
    brief_text = "\n".join(lines).strip()

    questions_raw = cluster_context.get("questions", []) or []
    questions: list[str] = []
    seen: set[str] = set()
    for value in questions_raw:
        if not isinstance(value, str):
            continue
        question = value.strip().rstrip(". ")
        if not question or _is_generic_brief_question(question):
            continue
        if not question.endswith("?"):
            question += "?"
        key = re.sub(r"\s+", " ", question).casefold()
        if key in seen:
            continue
        seen.add(key)
        questions.append(question)
        if len(questions) == 3:
            break

    return brief_text, questions


async def generate_brief(meeting: CalendarEvent, settings: Settings | None = None) -> MeetingBrief:
    """Generate a pre-meet brief from the matched precomputed cluster (no LLM)."""
    active_settings = settings or get_settings()

    from agent_email.service import match_meeting_to_cluster

    cluster = await match_meeting_to_cluster(
        meeting.subject,
        meeting.body_snippet,
        settings=active_settings,
        meeting_id=meeting.id,
    )

    cluster_context = _cluster_context_from_record(cluster) if cluster else None
    if cluster and cluster.department:
        view = active_settings.department_view_map.get(cluster.department.lower())
        meeting.matched_view = view

    brief_text, questions = _render_brief_text(cluster_context)

    brief = MeetingBrief(
        meeting_id=meeting.id,
        meeting_subject=meeting.subject,
        meeting_start=meeting.start,
        generated_at=datetime.now(timezone.utc),
        brief_text=brief_text,
        questions=questions,
        source_thread_ids=[],
    )
    logger.info(
        "Generated pre-meet brief for meeting_id=%s subject=%s matched=%s questions=%d",
        meeting.id, meeting.subject, cluster is not None, len(questions),
    )
    return brief


async def get_brief_for_meeting_id(
    meeting_id: str,
    bearer_token: str | None = None,
    storage: StorageBackend | None = None,
    settings: Settings | None = None,
) -> MeetingBrief | None:
    active_settings = settings or get_settings()
    today = await get_todays_events(bearer_token=bearer_token, settings=active_settings)
    meeting = next((event for event in today.events if event.id == meeting_id), None)
    if meeting is None:
        return None
    return await get_or_generate_brief(meeting, storage=storage, settings=active_settings)


async def get_or_generate_brief(
    meeting: CalendarEvent,
    storage: StorageBackend | None = None,
    settings: Settings | None = None,
) -> MeetingBrief:
    active_storage = storage or get_storage()
    stored = await active_storage.briefs.get_latest_brief(meeting.id)
    if stored is not None:
        stored_brief = MeetingBrief.model_validate(stored)
        if (stored_brief.brief_text or "").strip() or stored_brief.questions:
            return stored_brief

    brief = await generate_brief(meeting, settings=settings)
    await active_storage.briefs.save_brief(meeting.id, brief.model_dump(mode="json"))
    return brief
