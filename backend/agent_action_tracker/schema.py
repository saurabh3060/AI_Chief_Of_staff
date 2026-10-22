"""
agent_action_tracker/schema.py
=================================
Pydantic models for the Action & Follow-up Tracker agent.

Stores the shapes used to represent:
- `ActionStatus`: the classification bucket for a tracked action item.
- `ActionItem`: a single tracked action item, derived from an
  `ActionCandidate` produced by the Email & Summary agent, enriched
  with an owner/due-date guess and a status classification.

Internal to this agent's routes/service - other packages should go
through `agent_action_tracker.service` functions rather than importing
these models directly.
"""

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel


class ActionStatus(str, Enum):
    """Classification bucket for a tracked action item."""

    OVERDUE = "Overdue"
    IN_PROGRESS = "In Progress"
    ON_TRACK = "On Track"
    # Set when the action item's own text indicates the work is already
    # done (e.g. "Done: sent the report", "Closed - approved by Finance"),
    # as determined by `agent_action_tracker.service.guess_is_completed`.
    # Takes priority over the due-date-derived buckets above regardless
    # of due date, since a completed item is no longer "overdue"/"on
    # track" even if its due date has passed.
    COMPLETED = "Completed"


class ActionItem(BaseModel):
    """
    A single tracked action item, classified from an email-derived
    action candidate.
    """

    id: str
    text: str
    source_thread_id: str
    source_subject: str
    owner: str | None = None
    due_date: date | None = None
    status: ActionStatus
    classified_at: datetime
