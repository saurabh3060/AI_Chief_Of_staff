"""
agent_calendar/schema.py
==========================
Pydantic models for the Calendar & Meeting agent.

Stores the shapes used to represent:
- `CalendarEvent`: a normalized view of a single Microsoft Graph
  calendar event (parsed from the raw Graph API JSON).
- `MeetingWindow`: the result of detecting the "current" and/or "next"
  meeting from a list of events at a point in time.
- `MeetingBrief`: an auto-generated pre-meet brief for a flagged
  upcoming meeting.

These models are internal to this agent's routes/service - other
agents/packages must go through `agent_calendar.service` functions
rather than importing these models directly, to keep the "only talk
through service.py" boundary honest.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class CalendarEvent(BaseModel):
    """A normalized calendar event, parsed from a raw Microsoft Graph API event object."""

    id: str
    subject: str
    start: datetime
    end: datetime
    organizer_name: str | None = None
    organizer_email: str | None = None
    attendee_emails: list[str] = Field(default_factory=list)
    # Plain-text snippet of the Graph event body/description, used for
    # embedding-based meeting -> cluster matching.
    body_snippet: str = ""
    # Frontend sidebar view key (e.g. "Retail", "Wholesale") derived from
    # the matched cluster's department via `Settings.department_view_map`.
    # None when no cluster clears the meeting-match confidence gate.
    matched_view: str | None = None


class MeetingWindow(BaseModel):
    """
    Result of detecting the current/next meeting from a list of events
    at a given point in time (`as_of`).
    """

    as_of: datetime
    current_meeting: CalendarEvent | None = None
    next_meeting: CalendarEvent | None = None


class TodayEvents(BaseModel):
    """Every calendar event scheduled for today, as of a given point in time (`as_of`)."""

    as_of: datetime
    events: list[CalendarEvent] = Field(default_factory=list)


class MeetingBrief(BaseModel):
    """An auto-generated pre-meet brief for a flagged upcoming meeting."""

    meeting_id: str
    meeting_subject: str
    meeting_start: datetime
    generated_at: datetime
    brief_text: str
    # Up to 3 questions the CEO should walk in prepared to ask, specific
    # to this meeting. Empty list if the model's response couldn't be
    # parsed into structured questions (brief_text is always populated
    # regardless, so callers should treat this as a "nice to have").
    questions: list[str] = Field(default_factory=list)
    # Conversation ids (Graph `conversationId`, see
    # `agent_email.schema.BuEmailRecord.conversation_id`) of every BU
    # email thread that fed this brief's evidence, i.e. the
    # `related_records` retrieved by `find_related_emails_for_meeting`
    # and clustered into the picked cluster(s) used to render
    # `brief_text`. Purely for audit/traceability - lets you verify
    # exactly which threads a given brief was built from instead of
    # only trusting the short-TTL retrieval cache to agree with itself
    # each time. Empty when no related emails were found (metadata-only
    # brief) or for briefs generated before this field existed.
    source_thread_ids: list[str] = Field(default_factory=list)
