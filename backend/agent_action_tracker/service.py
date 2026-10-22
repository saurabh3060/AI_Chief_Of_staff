"""
agent_action_tracker/service.py
==================================
Business logic for the Action & Follow-up Tracker agent: pulling
action-item candidates from the Email & Summary agent's persisted
summaries (via `agent_email.service.get_latest_summaries()`), guessing
an owner and due date for each candidate, classifying the result into
Overdue / In Progress / On Track, and persisting the classified items
via the storage layer.

This module deliberately has NO Graph API or OCR logic - it must never
duplicate what `agent_email` already does. Its only external
dependencies are `agent_email.service` (for candidates) and
`core.ollama_client` (as a fallback for owner extraction when a simple
heuristic can't determine one).
"""

import re
from datetime import date, datetime, timezone

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger
from core.ollama_client.client import OllamaClient
from core.storage.base import StorageBackend
from core.storage.factory import get_storage

from agent_action_tracker.prompts import OWNER_SYSTEM_PROMPT, build_owner_extraction_prompt
from agent_action_tracker.schema import ActionItem, ActionStatus
from agent_email.schema import ActionCandidate
from agent_email.service import get_latest_summaries

logger = get_agent_logger("agent_action_tracker")

# Matches a leading "<Owner phrase> to <verb>" shape, e.g. "Finance team to
# send the reconciled breakup" -> captures "Finance team". This is the
# common phrasing used in the "Action: ..." lines the Email agent extracts.
_OWNER_HEURISTIC_RE = re.compile(r"^([A-Z][A-Za-z\s]{1,40}?)\s+to\s+\w")

# Weekday names (case-insensitive) that may appear as "by <weekday>" in an
# action item, e.g. "by Friday".
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAY_RE = re.compile(r"\bby\s+(" + "|".join(_WEEKDAYS) + r")\b", re.IGNORECASE)

# Matches an explicit date such as "by 12 Sep 2026" or "by 2026-09-12".
_EXPLICIT_DATE_RE = re.compile(
    r"\bby\s+(\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]+\s+\d{4})\b", re.IGNORECASE
)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Matches the "[done]" marker `agent_email.prompts._sanitize_and_normalize_action_items`
# appends when the SAME per-email LLM call that produced this action
# item's text/due_date also reported (per `build_bu_email_prompt` rule
# 3) that the source email itself says the work is already completed.
# Also matches a leading "Done:"/"Completed:"/"Closed:"/"Resolved:" as a
# fallback for any action text that predates the "[done]" tagging (or
# candidates sourced from `extract_action_candidates`'s "Action: ..."
# regex, which never goes through the LLM tagging path at all).
_COMPLETION_HEURISTIC_RE = re.compile(
    r"\[done\]|^(done|completed|closed|resolved)\b[:\-\s]", re.IGNORECASE
)


def _guess_owner_heuristic(text: str) -> str | None:
    """Try to extract an owner from the action text using the '<Owner> to <verb>' pattern."""
    match = _OWNER_HEURISTIC_RE.match(text)
    return match.group(1).strip() if match else None


async def guess_owner(text: str, ollama_client: OllamaClient | None = None) -> str | None:
    """
    Determine the owner of an action item: try the cheap regex heuristic
    first, and only fall back to an LLM call (via `OllamaClient.chat`)
    when the heuristic can't find one. Returns None if neither approach
    finds a clear owner ("Unassigned").
    """
    heuristic_owner = _guess_owner_heuristic(text)
    if heuristic_owner:
        return heuristic_owner

    if ollama_client is None:
        return None

    prompt = build_owner_extraction_prompt(text)
    response = await ollama_client.chat(prompt, system_prompt=OWNER_SYSTEM_PROMPT)
    cleaned = response.strip()
    if not cleaned or cleaned.lower() == "unassigned":
        return None
    return cleaned


def _next_weekday_on_or_after(reference: date, weekday_name: str) -> date:
    """
    Return the next date matching `weekday_name` (case-insensitive) that
    is on or after `reference`. If `reference` itself falls on that
    weekday, `reference` is returned (i.e. "by Friday" said on a Friday
    means today).
    """
    target_index = _WEEKDAYS.index(weekday_name.lower())
    days_ahead = (target_index - reference.weekday()) % 7
    from datetime import timedelta

    return reference + timedelta(days=days_ahead)


def _parse_explicit_date(raw: str) -> date | None:
    """Parse an explicit date string in 'YYYY-MM-DD' or 'D Month YYYY' format."""
    iso_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", raw)
    if iso_match:
        return date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))

    text_match = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$", raw)
    if text_match:
        day, month_name, year = text_match.groups()
        month = _MONTHS.get(month_name[:3].lower())
        if month:
            return date(int(year), month, int(day))
    return None


def guess_due_date(text: str, reference: date) -> date | None:
    """
    Determine a due date for an action item from its text: prefer an
    explicit date ("by 12 Sep 2026" / "by 2026-09-12") over a relative
    weekday reference ("by Friday"). Returns None if no date-shaped
    phrase is found.
    """
    explicit_match = _EXPLICIT_DATE_RE.search(text)
    if explicit_match:
        parsed = _parse_explicit_date(explicit_match.group(1))
        if parsed is not None:
            return parsed

    weekday_match = _WEEKDAY_RE.search(text)
    if weekday_match:
        return _next_weekday_on_or_after(reference, weekday_match.group(1))

    return None


def guess_is_completed(text: str) -> bool:
    """
    Determine whether an action item is already completed - reading
    the "[done]" tag `agent_email`'s per-email LLM call already wrote
    onto this action item's text (see `build_bu_email_prompt` rule 3 /
    `_sanitize_and_normalize_action_items`). No LLM call happens here:
    completion is tagged by the model at the SAME time it produces the
    action item's text/owner/due_date, since it is already reading the
    source email at that point - re-reading the same text a second
    time in this agent just to ask the model the same question again
    would be a wasted extra round-trip. The leading "Done:"/"Completed"/
    "Closed"/"Resolved" match is a fallback for text that predates the
    "[done]" tag (or came from `extract_action_candidates`'s "Action:
    ..." regex path, which bypasses the LLM tagging entirely).
    """
    return bool(_COMPLETION_HEURISTIC_RE.search(text.strip()))


def classify_status(due_date: date | None, reference: date, is_completed: bool = False) -> ActionStatus:
    """
    Classify an action item's status: COMPLETED takes priority (set by
    `guess_is_completed`, based on the item's own wording) regardless of
    due date. Otherwise, classify from the due date relative to
    `reference` (today): overdue if the due date has passed, on track
    if it's in the future, and in-progress if no due date could be
    determined at all (nothing to be overdue against yet).
    """
    if is_completed:
        return ActionStatus.COMPLETED
    if due_date is None:
        return ActionStatus.IN_PROGRESS
    if due_date < reference:
        return ActionStatus.OVERDUE
    return ActionStatus.ON_TRACK


async def classify_candidate(
    candidate: ActionCandidate,
    source_thread_id: str,
    source_subject: str,
    index: int,
    reference: date | None = None,
    ollama_client: OllamaClient | None = None,
) -> ActionItem:
    """Classify a single action-item candidate into a fully-formed `ActionItem`."""
    reference_date = reference or datetime.now(timezone.utc).date()
    owner = await guess_owner(candidate.text, ollama_client=ollama_client)
    due_date = guess_due_date(candidate.text, reference_date)
    is_completed = guess_is_completed(candidate.text)
    status = classify_status(due_date, reference_date, is_completed=is_completed)

    return ActionItem(
        id=f"{source_thread_id}::{index}",
        text=candidate.text,
        source_thread_id=source_thread_id,
        source_subject=source_subject,
        owner=owner,
        due_date=due_date,
        status=status,
        classified_at=datetime.now(timezone.utc),
    )


async def classify_all_action_items(
    reference: date | None = None,
    storage: StorageBackend | None = None,
    settings: Settings | None = None,
) -> list[ActionItem]:
    """
    Pull every persisted email summary (via `agent_email.service.get_latest_summaries()`),
    classify each summary's action candidates, persist the resulting
    `ActionItem`s via storage, and return the full classified list.

    Called on-demand (no background job) - each call re-classifies from
    the current set of summaries, so newly generated email summaries
    are picked up automatically on the next call.
    """
    active_storage = storage or get_storage()
    active_settings = settings or get_settings()
    ollama_client = OllamaClient(settings=active_settings)

    summaries = await get_latest_summaries(storage=active_storage)
    classified_items: list[ActionItem] = []

    for summary in summaries:
        for index, candidate in enumerate(summary.action_candidates):
            item = await classify_candidate(
                candidate,
                source_thread_id=summary.thread_id,
                source_subject=summary.subject,
                index=index,
                reference=reference,
                ollama_client=ollama_client,
            )
            await active_storage.actions.add(item.model_dump(mode="json"))
            classified_items.append(item)

    logger.info("Classified %d action item(s) from %d summary/summaries", len(classified_items), len(summaries))
    return classified_items


async def list_action_items(status: ActionStatus | None = None, storage: StorageBackend | None = None) -> list[ActionItem]:
    """Return currently stored action items, optionally filtered by status."""
    active_storage = storage or get_storage()
    filters = {"status": status.value} if status else None
    raw_items = await active_storage.actions.list(filters=filters)
    return [ActionItem.model_validate(raw) for raw in raw_items]


async def update_action_item(action_id: str, patch: dict, storage: StorageBackend | None = None) -> ActionItem | None:
    """Apply a manual update (e.g. status/owner override) to a tracked action item."""
    active_storage = storage or get_storage()
    updated = await active_storage.actions.update(action_id, patch)
    if updated is None:
        return None
    return ActionItem.model_validate(updated)
