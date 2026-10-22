"""
agent_calendar/routes.py
===========================
FastAPI routes for the Calendar & Meeting agent.

- `GET /agents/calendar/current` - the meeting happening right now, if any.
- `GET /agents/calendar/next` - the next upcoming meeting, if any.
- `GET /agents/calendar/today` - every meeting scheduled for today.
- `GET /agents/calendar/brief` - the pre-meet brief for the next upcoming
  meeting (generates one on first request, then reuses the stored copy).
- `GET /agents/calendar/brief/{meeting_id}` - the pre-meet brief for a
  specific meeting on today's calendar (by Graph event id), including
  up to 3 meeting-specific questions. Used when the user selects a
  specific row in the "Today's schedule" list.

Authentication is resolved by the shared `get_bearer_token` dependency
(see `auth/dependencies.py`): in delegated mode it comes from the
server-side session (via the HTTP-only session cookie set at
`/auth/callback`); in app-only mode it is `None` and `GraphClient`
manages its own token.
"""

from fastapi import APIRouter, Depends, HTTPException

from agent_calendar.service import (
    get_brief_for_meeting_id,
    get_current_and_next_meeting,
    get_or_generate_brief,
    get_todays_events,
)
from auth.dependencies import get_bearer_token
from core.config import get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_calendar")

router = APIRouter(prefix="/agents/calendar", tags=["calendar"])
notes_router = APIRouter(prefix="/agents/meeting-notes", tags=["meeting-notes"])


@router.get("/polling-config")
def get_calendar_polling_config():
    """
    Expose the calendar polling configuration to the SPA.

    The calendar agent has no server-side background loop - meetings are
    fetched on-demand per request - so "calendar polling" is the
    frontend re-hitting these endpoints on a timer. This returns both
    the master toggle (`calendar_polling_enabled`) and the interval
    (`calendar_polling_interval_seconds`) so the SPA can decide whether
    to poll at all and, if so, how often. When disabled, the frontend
    should stop its timer and fetch calendar data once/on demand only.
    """
    settings = get_settings()
    return {
        "calendar_polling_enabled": settings.calendar_polling_enabled,
        "calendar_polling_interval_seconds": settings.calendar_polling_interval_seconds,
    }


@router.get("/current")
async def get_current_meeting(bearer_token: str | None = Depends(get_bearer_token)):
    """Return the meeting currently in progress, or null if none is happening right now."""
    settings = get_settings()
    window = await get_current_and_next_meeting(bearer_token=bearer_token, settings=settings)
    return {"current_meeting": window.current_meeting, "as_of": window.as_of}


@router.get("/next")
async def get_next_meeting(bearer_token: str | None = Depends(get_bearer_token)):
    """Return the next upcoming meeting, or null if none is scheduled."""
    settings = get_settings()
    window = await get_current_and_next_meeting(bearer_token=bearer_token, settings=settings)
    return {"next_meeting": window.next_meeting, "as_of": window.as_of}


@router.get("/today")
async def get_today_events(bearer_token: str | None = Depends(get_bearer_token)):
    """Return every meeting scheduled for today, sorted by start time."""
    settings = get_settings()
    today = await get_todays_events(bearer_token=bearer_token, settings=settings)
    return {"events": today.events, "as_of": today.as_of}


@router.get("/brief")
async def get_next_meeting_brief(bearer_token: str | None = Depends(get_bearer_token)):
    """Return the pre-meet brief for the next upcoming meeting, generating one if none exists yet."""
    settings = get_settings()
    window = await get_current_and_next_meeting(bearer_token=bearer_token, settings=settings)
    if window.next_meeting is None:
        return {"brief": None, "message": "No upcoming meeting to brief."}

    brief = await get_or_generate_brief(window.next_meeting, settings=settings)
    return {"brief": brief}


@router.get("/brief/{meeting_id}")
async def get_meeting_brief(meeting_id: str, bearer_token: str | None = Depends(get_bearer_token)):
    """
    Return the pre-meet brief (including up to 3 meeting-specific
    questions) for a specific meeting on today's calendar, identified by
    its Graph event id. Generates one on first request, then reuses the
    stored copy. Returns 404 if `meeting_id` isn't one of today's events.
    """
    settings = get_settings()
    brief = await get_brief_for_meeting_id(meeting_id, bearer_token=bearer_token, settings=settings)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"No meeting with id '{meeting_id}' found on today's calendar.")
    return {"brief": brief}

from agent_email import postgres_store
from agent_email.schema import MeetingNoteCreate, MeetingNote, MeetingNoteResponse
from agent_email.service import extract_action_items_from_text


@notes_router.post("", response_model=MeetingNote)
async def create_meeting_note(
    note: MeetingNoteCreate,
    bearer_token: str | None = Depends(get_bearer_token),
):
    """
    Create a new meeting note. If action_items are not provided or empty,
    the LLM will extract them from note_text and update the record.
    """
    settings = get_settings()
    log = get_agent_logger("agent_calendar.notes")

    # Strip CR/LF from user-controlled values before logging to prevent
    # log injection / forged log lines (CWE-117).
    safe_meeting_id = str(note.meeting_id).replace("\r\n", "").replace("\r", "").replace("\n", "")
    safe_source = str(note.source).replace("\r\n", "").replace("\r", "").replace("\n", "")
    log.info(
        "[MEETING-NOTES] create_meeting_note: meeting_id=%s source=%s note_len=%d",
        safe_meeting_id,
        safe_source,
        len(note.note_text),
    )

    # Extract action items if not provided
    action_items = note.action_items
    if not action_items and note.note_text.strip():
        log.info("[MEETING-NOTES] extracting action items from note_text")
        try:
            action_items = await extract_action_items_from_text(note.note_text, settings)
            log.info(
                "[MEETING-NOTES] extracted %d action items from note",
                len(action_items),
            )
        except Exception as exc:
            log.warning(
                "[MEETING-NOTES] action item extraction failed: %s, proceeding with empty",
                exc,
            )

    # Write note to DB
    try:
        records_written = await postgres_store.write_meeting_notes(
            [
                {
                    "meeting_id": note.meeting_id,
                    "note_text": note.note_text,
                    "action_items": action_items,
                    "tags": note.tags,
                    "source": note.source,
                }
            ],
            settings=settings,
        )
        if records_written == 0:
            raise HTTPException(
                status_code=500,
                detail="Failed to write meeting note",
            )
    except RuntimeError as exc:
        logger.error("[MEETING-NOTES] DB error: %s", exc)
        raise HTTPException(status_code=500, detail="Database error") from exc
    except Exception as exc:
        logger.error("[MEETING-NOTES] unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Read back the created note to return with timestamps
    try:
        notes = await postgres_store.read_meeting_notes(note.meeting_id, settings=settings)
        created_note = next((n for n in notes if n["note_text"] == note.note_text), None)
        if created_note:
            return MeetingNote(
                id=created_note["id"],
                meeting_id=created_note["meeting_id"],
                note_text=created_note["note_text"],
                action_items=created_note["action_items"],
                tags=created_note.get("tags", []),
                source=created_note["source"],
                created_at=created_note["created_at"],
                updated_at=created_note.get("updated_at"),
            )
        raise HTTPException(status_code=500, detail="Note created but could not be retrieved")
    except Exception as exc:
        logger.error("[MEETING-NOTES] read_back error: %s", exc)
        raise HTTPException(status_code=500, detail="Note created but could not be verified") from exc


@notes_router.get("/{meeting_id}", response_model=MeetingNoteResponse)
async def get_meeting_notes(
    meeting_id: str,
    bearer_token: str | None = Depends(get_bearer_token),
):
    """Return all meeting notes for a specific meeting."""
    settings = get_settings()
    log = get_agent_logger("agent_calendar.notes")

    # Strip CR/LF from the user-controlled meeting_id before logging to
    # prevent log injection / forged log lines (CWE-117).
    safe_meeting_id = str(meeting_id).replace("\r\n", "").replace("\r", "").replace("\n", "")
    log.info("[MEETING-NOTES] get_meeting_notes: meeting_id=%s", safe_meeting_id)

    try:
        notes = await postgres_store.read_meeting_notes(meeting_id, settings=settings)
    except Exception as exc:
        logger.error("[MEETING-NOTES] DB error for meeting_id=%s: %s", safe_meeting_id, exc)
        raise HTTPException(status_code=500, detail="Database error") from exc

    return MeetingNoteResponse(
        notes=[
            MeetingNote(
                id=n["id"],
                meeting_id=n["meeting_id"],
                note_text=n["note_text"],
                action_items=n["action_items"],
                tags=n.get("tags", []),
                source=n["source"],
                created_at=n["created_at"],
                updated_at=n.get("updated_at"),
            )
            for n in notes
        ],
        count=len(notes),
    )
