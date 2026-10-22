"""
agent_open_tracker/schema.py
================================
Pydantic models for the Open Actions Status Tracker card on the
Executive Launchpad.

Reads pre-extracted action items from the email cluster pipeline
(the ``email_clusters`` store) and classifies them into three display
buckets based on due date vs today:

- **Overdue**   — due date has passed → red card
- **Due Today** — due date is today   → amber card
- **Upcoming**  — due date in future, or no date → green card
"""

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class OpenTrackerStatus(str, Enum):
    """Display bucket for a single open action item."""

    OVERDUE = "Overdue"
    DUE_TODAY = "Due Today"
    UPCOMING = "Upcoming"


class OpenTrackerItem(BaseModel):
    """One action item shown on the Open Actions Status Tracker card."""

    id: str = ""
    """Deterministic 16-char hex ID (SHA-256 of owner+text+bu). Used to complete the item."""

    text: str
    """The action item text, e.g. 'Sandeep to lock funding cost assumption'."""

    owner: str = "Unassigned"
    """Who is responsible, extracted from the cluster action item string."""

    due_date: date | None = None
    """Parsed due date, if one was found in the action item string."""

    due_label: str = ""
    """Human-readable label: 'Overdue by X days', 'Due today', or 'Due on DD Mon YYYY'."""

    business_unit: str = ""
    """Business unit name exactly as stored in the email_clusters department column."""

    status: OpenTrackerStatus
    """Classified display status for colour-coding."""

    overdue_by_days: int | None = None
    """Number of days past due, only set when status == OVERDUE."""

    source_topic: str = ""
    """The cluster topic this action item came from, for context."""


class PendingDecisionItem(BaseModel):
    """An action item assigned to a decision-maker that should appear in
    'Awaiting Your Decision' on the Executive Launchpad instead of the
    Open Actions Status Tracker."""

    id: str = ""
    text: str
    owner: str = "Unassigned"
    due_date: date | None = None
    due_label: str = ""
    business_unit: str = ""
    source_topic: str = ""
    status: OpenTrackerStatus


class OpenTrackerResponse(BaseModel):
    """Full response payload for ``GET /agents/open-tracker``."""

    items: list[OpenTrackerItem] = Field(default_factory=list)
    pending_decisions: list[PendingDecisionItem] = Field(default_factory=list)
    overdue_count: int = 0
    due_today_count: int = 0
    upcoming_count: int = 0
    total_count: int = 0
    pending_overdue_count: int = 0
    pending_due_today_count: int = 0
    pending_upcoming_count: int = 0

    pipeline_fresh: bool = True
    """False when the email cluster pipeline has not yet completed a run
    in this backend process's lifetime - the data above (if any) may be
    leftover from before a restart and should be treated as not-yet-ready
    by the frontend rather than rendered as current."""

    pipeline_ready_at: str | None = None
    """ISO timestamp of when the pipeline first became fresh this
    session, or None if not yet fresh."""
