"""
core/graph_client/local_calendar.py
======================================
Offline stand-in for Microsoft Graph calendar endpoints.

When `GRAPH_AUTH_MODE=local`, `GraphClient` serves calendar events from
`backend/data/local_calendar.json` instead of calling graph.microsoft.com.
Events are shaped like Graph JSON so `agent_calendar.service._parse_event`
and the meeting→email matcher stay unchanged.

Clock times in the JSON are IST wall-clock (`HH:MM`) and rematerialized
onto today's IST calendar day, so the dummy schedule never goes stale.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from core.config import IST, Settings, get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("local_calendar")

_calendar_cache: tuple[str, float, str, list[dict]] | None = None


def _parse_ist_clock(value: str) -> time:
    hour_text, minute_text = value.strip().split(":", 1)
    return time(int(hour_text), int(minute_text), tzinfo=IST)


def _graph_datetime(value: datetime) -> dict:
    utc = value.astimezone(timezone.utc)
    return {
        "dateTime": utc.strftime("%Y-%m-%dT%H:%M:%S.0000000"),
        "timeZone": "UTC",
    }


def _attendee_name(email: str, organizer_name: str, organizer_email: str) -> str:
    if email.lower() == organizer_email.lower() and organizer_name:
        return organizer_name
    local_part = email.split("@", 1)[0].replace(".", " ").replace("_", " ").strip()
    return local_part.title() or email


def _materialize_event(raw: dict, today_ist: datetime) -> dict | None:
    subject = (raw.get("subject") or "").strip()
    event_id = (raw.get("id") or "").strip()
    start_text = raw.get("start_ist")
    end_text = raw.get("end_ist")
    if not subject or not event_id or not start_text or not end_text:
        logger.warning("Skipping incomplete local calendar meeting: %s", raw)
        return None

    try:
        start = datetime.combine(today_ist.date(), _parse_ist_clock(start_text))
        end = datetime.combine(today_ist.date(), _parse_ist_clock(end_text))
    except (TypeError, ValueError):
        logger.warning("Skipping local calendar meeting with invalid IST times: %s", raw)
        return None

    if end <= start:
        end = start + timedelta(hours=1)

    organizer_name = (raw.get("organizer_name") or "").strip()
    organizer_email = (raw.get("organizer_email") or "").strip()
    attendee_emails = [
        email.strip()
        for email in raw.get("attendee_emails") or []
        if isinstance(email, str) and email.strip()
    ]

    return {
        "id": event_id,
        "subject": subject,
        "start": _graph_datetime(start),
        "end": _graph_datetime(end),
        "organizer": {
            "emailAddress": {
                "name": organizer_name,
                "address": organizer_email,
            }
        },
        "attendees": [
            {
                "type": "required",
                "emailAddress": {
                    "name": _attendee_name(email, organizer_name, organizer_email),
                    "address": email,
                },
            }
            for email in attendee_emails
        ],
    }


def _load_events(path: Path) -> list[dict]:
    if not path.is_file():
        logger.info("Local calendar file missing at %s; returning no events", path)
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read local calendar %s: %s", path, exc)
        return []

    meetings = payload.get("meetings") if isinstance(payload, dict) else payload
    if not isinstance(meetings, list):
        logger.warning("Local calendar %s has no meetings list", path)
        return []

    today_ist = datetime.now(IST)
    events = []
    for raw in meetings:
        if not isinstance(raw, dict):
            continue
        event = _materialize_event(raw, today_ist)
        if event is not None:
            events.append(event)
    events.sort(key=lambda item: item["start"]["dateTime"])
    return events


def ensure_local_calendar_exists(settings: Settings | None = None) -> Path:
    """
    Ensure the local calendar JSON exists for GRAPH_AUTH_MODE=local.

    Copies `backend/fixtures/local_calendar.json` into the configured
    LOCAL_CALENDAR_PATH when the target file is missing.
    """
    active = settings or get_settings()
    path = active.resolved_local_calendar_path
    if path.is_file():
        return path

    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "local_calendar.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if fixture_path.is_file():
        shutil.copyfile(fixture_path, path)
        logger.info("Seeded local calendar from %s -> %s", fixture_path, path)
    else:
        path.write_text('{"meetings": []}', encoding="utf-8")
        logger.info("Created empty local calendar at %s", path)
    return path


def load_local_calendar_events(settings: Settings | None = None) -> list[dict]:
    """
    Load Graph-shaped dummy calendar events for local mode.

    Cache key includes the file mtime and the IST calendar date so a
    process that stays up overnight still rematerializes onto the new day.
    """
    global _calendar_cache
    active = settings or get_settings()
    path = active.resolved_local_calendar_path
    cache_key = str(path.resolve()) if path.exists() else str(path)
    mtime = path.stat().st_mtime if path.is_file() else 0.0
    today_key = datetime.now(IST).date().isoformat()
    if (
        _calendar_cache
        and _calendar_cache[0] == cache_key
        and _calendar_cache[1] == mtime
        and _calendar_cache[2] == today_key
    ):
        return list(_calendar_cache[3])

    events = _load_events(path)
    _calendar_cache = (cache_key, mtime, today_key, events)
    logger.info("Loaded %d local calendar event(s) from %s", len(events), path)
    return list(events)


def _parse_bound(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _event_start(event: dict) -> datetime | None:
    raw = (event.get("start") or {}).get("dateTime")
    if not raw:
        return None
    if "." in raw:
        date_part, frac_part = raw.split(".", 1)
        raw = f"{date_part}.{frac_part[:6]}"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def list_local_calendar_events(top: int = 50, settings: Settings | None = None) -> dict:
    events = load_local_calendar_events(settings=settings)
    return {"value": events[: max(0, top)]}


def list_local_calendar_view(
    start_date_time: str,
    end_date_time: str,
    top: int = 50,
    settings: Settings | None = None,
) -> dict:
    events = load_local_calendar_events(settings=settings)
    window_start = _parse_bound(start_date_time)
    window_end = _parse_bound(end_date_time)
    if window_start is not None and window_end is not None:
        filtered = []
        for event in events:
            start = _event_start(event)
            if start is None:
                continue
            if window_start <= start < window_end:
                filtered.append(event)
        events = filtered
    return {"value": events[: max(0, top)]}
