"""
agent_open_tracker/service.py
===================================
Business logic for the Open Actions Status Tracker.

Reads the pre-synthesised ``email_clusters`` records produced by the Email
& Summary agent's cluster pipeline.  Each cluster row already contains:

- ``department``   — the business unit (BB&RL, Wholesale Banking, …)
- ``action_items`` — pipe-delimited strings in the form
                     ``"Owner: task text (by deadline)"``
- ``topic``        — the cluster topic, surfaced as context

This service parses those strings, resolves the due date, and
classifies each item into one of three display buckets:

    OVERDUE   → due date has already passed
    DUE_TODAY → due date is today
    UPCOMING  → due date is in the future (or unknown)

No LLM calls happen here — everything is derived from already-persisted
CSV data so the endpoint is fast and cheap.
"""

import json
import re
from datetime import date, datetime, timezone

from agent_email.store import read_all_cluster_records
from agent_open_tracker.completed_store import (
    load_completed_ids,
    make_item_id,
    remove_completed_item,
    save_completed_item,
)
from agent_open_tracker.deleted_store import load_deleted_ids, save_deleted_item
from agent_open_tracker.item_overrides_store import load_item_overrides, save_item_override
from agent_open_tracker.schema import OpenTrackerItem, OpenTrackerResponse, OpenTrackerStatus, PendingDecisionItem

# Owners whose action items should go to "Awaiting Your Decision" instead of
# the Open Actions Status Tracker. Stored lowercase and compared against a
# lowercased owner, so matching is case-insensitive.
_AWAITING_DECISION_OWNERS: set[str] = {o.lower() for o in {"r subramaniakumar"}}

# TEMP logic: if a cluster has any corresponding mail whose "to" recipient is
# one of these addresses, that cluster's action items are ALSO surfaced in
# "Awaiting Your Decision" (in addition to their normal routing). Matched on
# the bu_emails.to_address column (populated from Graph toRecipients). The CEO
# is addressed under two mailbox domains, so both must be matched.
_AWAITING_DECISION_TO_RECIPIENTS: set[str] = {
    "rskumar@rbl.bank.in",
    "rskumar@rblbank.com",
}
from core.logging_config import get_agent_logger
from core.utils import sanitize_log_input

logger = get_agent_logger("agent_open_tracker")


# ---------------------------------------------------------------------------
# Date-parsing helpers
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Matches the deadline suffix written by _format_action_item:
#   "(by 2026-09-30)"  or  "(by 30-Sep-2026)"  or  "(by 30 Sep 2026)"
#   or "(by 2026-09-30T21:49:48+00:00)" (full ISO datetime with time/offset,
#   e.g. produced when the LLM emits a timestamp instead of a plain date).
_DEADLINE_RE = re.compile(
    r"\(by\s+"
    r"(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?"  # ISO date or datetime
    r"|\d{1,2}-[A-Za-z]+-\d{4}"    # DMY:  30-Sep-2026
    r"|\d{1,2}\s+[A-Za-z]+\s+\d{4}"  # DMY: 30 Sep 2026
    r")"
    r"\s*\)",
    re.IGNORECASE,
)

# Matches the leading "Owner: " prefix.
_OWNER_PREFIX_RE = re.compile(r"^([^:]+):\s*(.+)$", re.DOTALL)

# Matches a bare trailing date suffix that some LLM outputs append instead of
# the canonical "(by …)" format, e.g.:
#   " - 2026-07-18"          (ISO)
#   " - 18-Apr-2026"         (DMY hyphen)
#   " - 18 Apr 2026"         (DMY space)
#   " - due 30 Jun"          (partial, no year — ignored / left as-is)
#   " - due 30 Jun 2026"     (full with "due" keyword)
#
# Only anchored at end-of-string ($) so a dash mid-sentence is never stripped.
# Captured group 1 is the raw date token, passed to _parse_date().
_TRAILING_DATE_RE = re.compile(
    r"\s*[-–]\s*(?:due\s+)?"
    r"(\d{4}-\d{2}-\d{2}"          # ISO:            2026-07-18
    r"|\d{1,2}-[A-Za-z]+-\d{4}"    # DMY hyphen:     18-Apr-2026
    r"|\d{1,2}\s+[A-Za-z]+\s+\d{4}"  # DMY space:   18 Apr 2026
    r")\s*$",
    re.IGNORECASE,
)


def _parse_date(raw: str) -> date | None:
    """Try to parse a date string in ISO or day-month-year format."""
    raw = raw.strip()

    # ISO datetime with time/offset: 2026-09-16T21:49:48+00:00 (also handles
    # a space instead of "T", and trailing "Z"). We only care about the date
    # portion for due-date classification, so the time is discarded.
    iso_dt = re.match(r"^(\d{4})-(\d{2})-(\d{2})[T ]\d{2}:\d{2}:\d{2}", raw)
    if iso_dt:
        try:
            return date(int(iso_dt.group(1)), int(iso_dt.group(2)), int(iso_dt.group(3)))
        except ValueError:
            return None

    # ISO: 2026-09-30
    iso = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", raw)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None

    # DMY with hyphen: 30-Sep-2026
    dmy_h = re.match(r"^(\d{1,2})-([A-Za-z]+)-(\d{4})$", raw)
    if dmy_h:
        month = _MONTHS.get(dmy_h.group(2)[:3].lower())
        if month:
            try:
                return date(int(dmy_h.group(3)), month, int(dmy_h.group(1)))
            except ValueError:
                return None

    # DMY with space: 30 Sep 2026
    dmy_s = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$", raw)
    if dmy_s:
        month = _MONTHS.get(dmy_s.group(2)[:3].lower())
        if month:
            try:
                return date(int(dmy_s.group(3)), month, int(dmy_s.group(1)))
            except ValueError:
                return None

    return None


def _parse_action_item_string(raw: str) -> tuple[str, str, date | None]:
    """
    Parse a cluster action item string of the form::

        "Owner: task text (by 2026-09-30)"

    Returns ``(owner, task_text, due_date)``.  Any field that cannot be
    parsed falls back gracefully (owner → "Unassigned", due_date → None).
    """
    raw = raw.strip()
    if not raw:
        return "Unassigned", "", None

    # Extract deadline first (so we can strip it from the task text).
    due_date: date | None = None
    deadline_match = _DEADLINE_RE.search(raw)
    if deadline_match:
        # Canonical "(by …)" format — extract date and strip the suffix.
        due_date = _parse_date(deadline_match.group(1))
        raw_no_deadline = raw[: deadline_match.start()].strip()
    else:
        # Fallback: look for a bare trailing " - YYYY-MM-DD" / " - DD-Mon-YYYY"
        # suffix produced by some LLM outputs.  Strip it and parse the date so
        # make_item_id always hashes the clean task text regardless of which
        # date format was used — preventing the same logical item from appearing
        # twice with different date-tail variants.
        trailing_match = _TRAILING_DATE_RE.search(raw)
        if trailing_match:
            due_date = _parse_date(trailing_match.group(1))
            raw_no_deadline = raw[: trailing_match.start()].strip()
        else:
            raw_no_deadline = raw

    # Strip "(by null)" — produced when the LLM couldn't determine a deadline.
    raw_no_deadline = re.sub(r"\s*\(by\s+null\)", "", raw_no_deadline, flags=re.IGNORECASE).strip()

    # Split "Owner: task text".
    owner_match = _OWNER_PREFIX_RE.match(raw_no_deadline)
    if owner_match:
        owner = owner_match.group(1).strip() or "Unassigned"
        task = owner_match.group(2).strip()
    else:
        owner = "Unassigned"
        task = raw_no_deadline.strip()

    # Fall back: if task is empty, use the full original string.
    if not task:
        task = raw.strip()

    return owner, task, due_date


def _classify(due_date: date | None, today: date) -> tuple[OpenTrackerStatus, int | None, str]:
    """
    Return ``(status, overdue_by_days, due_label)`` for a given due date.

    - OVERDUE   → due_date < today
    - DUE_TODAY → due_date == today
    - UPCOMING  → due_date > today, or no due_date
    """
    if due_date is None:
        return OpenTrackerStatus.UPCOMING, None, "No due date"

    if due_date < today:
        days = (today - due_date).days
        label = f"Overdue by {days} day{'s' if days != 1 else ''}"
        return OpenTrackerStatus.OVERDUE, days, label

    if due_date == today:
        return OpenTrackerStatus.DUE_TODAY, None, "Due today"

    # Future
    label = due_date.strftime("Due on %d %b %Y").replace(" 0", " ")
    return OpenTrackerStatus.UPCOMING, None, label


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _body_to_line_addresses_decision_recipient(body: str, recipients: list[str]) -> bool:
    """True if a quoted ``To:`` line inside a forwarded/replied body lists one
    of ``recipients``.

    Approval mails are frequently forwarded to the shared mailbox, so the
    CEO is NOT a Graph envelope recipient — the real ``To`` is the shared
    mailbox. The CEO only appears inside the quoted header block of the
    forwarded body, e.g.::

        From: OONAS Support <OONAS.Support@rbl.bank.in>
        Sent: 14 September 2026 20:31
        To: R S Kumar <RSKumar@rbl.bank.in>
        Cc: Jaideep Iyer <Jaideep.Iyer@rbl.bank.in>

    We only count the CEO as addressed when the address sits on a ``To:``
    line (not ``Cc:``/``From:`` and not loose prose), so a plain mention or
    a CC does not route the item to "Awaiting Your Decision".
    """
    if not body:
        return False
    lowered_recipients = [r for r in recipients if r]
    if not lowered_recipients:
        return False
    for raw_line in body.splitlines():
        line = raw_line.strip().lower()
        if not line:
            continue
        # Only consider quoted header "To:" lines. Starting with "to:" keeps
        # us off "Cc:"/"From:" lines and loose prose. Outlook forwards the
        # quoted header as its own line ("To: R S Kumar <rskumar@...>").
        if not line.startswith("to:"):
            continue
        # The address must appear AFTER the "to:" label on this same line.
        after_label = line[3:]
        for addr in lowered_recipients:
            if addr in after_label:
                return True
    return False


async def _message_ids_addressed_to_decision_recipient() -> set[str]:
    """Return the set of bu_emails message_ids addressed to the CEO, via
    EITHER the real Graph ``to_address`` OR a quoted ``To:`` line in the
    forwarded body (``raw_content`` / ``email_content``).

    TEMP helper for the "also surface in Awaiting Your Decision" rule. The
    to_address column is a ", "-joined list of Graph toRecipients, matched
    case-insensitively. Because approval mails are usually forwarded to the
    shared mailbox (so the CEO is only in the quoted body header, not the
    envelope), we ALSO scan the stored body for a quoted ``To:`` line naming
    the CEO. Returns an empty set (a safe no-op) when PostgreSQL is not
    configured or the query fails for any reason.
    """
    from core.config import get_settings
    from core.db import get_pool
    from agent_email.db_schema import validate_table_name

    try:
        pool = get_pool()
    except Exception:
        pool = None
    if pool is None:
        return set()

    # Normalise every address to lowercase so case never causes a miss. The
    # SQL side uses ILIKE (already case-insensitive), but lowercasing the
    # inputs too makes the case-insensitivity explicit and robust against any
    # uppercase values added to the set later.
    recipients = sorted({r.strip().lower() for r in _AWAITING_DECISION_TO_RECIPIENTS if r.strip()})
    if not recipients:
        return set()

    try:
        table = validate_table_name(get_settings().chatbot_pg_table)
        n = len(recipients)
        async with pool.acquire() as conn:
            # Discover which body column(s) this table actually has. Different
            # deployments use this project's `raw_content`, the chatbot
            # layout's `email_content`, or a `body_text` column — only query
            # the ones that exist so a missing column can't error the whole
            # lookup (which would silently drop every decision item).
            col_rows = await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = $1",
                table,
            )
            present = {r["column_name"] for r in col_rows}
            body_cols = [c for c in ("raw_content", "email_content", "body_text") if c in present]
            body_expr = (
                " || E'\\n' || ".join(f"coalesce({c},'')" for c in body_cols)
                if body_cols else "''"
            )

            # Envelope match: one case-insensitive clause per address on the
            # Graph to_address column.
            to_clauses = " OR ".join(
                f"lower(to_address) LIKE '%' || ${i + 1} || '%'" for i in range(n)
            )
            env_rows = await conn.fetch(
                f"SELECT message_id FROM {table} WHERE {to_clauses}",
                *recipients,
            )

            matched: set[str] = {r["message_id"] for r in env_rows if r["message_id"]}

            # Body candidate match: the address appears ANYWHERE in the stored
            # body. Cheap SQL pre-filter; the precise "on a To: line" check is
            # done in Python below to avoid false positives from CC / From /
            # prose mentions. Only run when a body column exists.
            if body_cols:
                body_clauses = " OR ".join(
                    f"lower({body_expr}) LIKE '%' || ${i + 1} || '%'" for i in range(n)
                )
                body_rows = await conn.fetch(
                    f"SELECT message_id, {body_expr} AS body "
                    f"FROM {table} WHERE {body_clauses}",
                    *recipients,
                )
                for r in body_rows:
                    mid = r["message_id"]
                    if not mid or mid in matched:
                        continue
                    if _body_to_line_addresses_decision_recipient(r["body"] or "", recipients):
                        matched.add(mid)

        return matched
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "could not load decision-recipient message_ids (continuing): %s", exc
        )
        return set()


def _cluster_addressed_to_decision_recipient(
    cluster, decision_message_ids: set[str]
) -> bool:
    """True if any of the cluster's email_refs message_ids is addressed to the
    decision recipient. Parses the cluster's email_refs JSON defensively."""
    if not decision_message_ids:
        return False
    raw_refs = getattr(cluster, "email_refs", "") or ""
    if not raw_refs.strip():
        return False
    try:
        refs = json.loads(raw_refs)
    except (ValueError, TypeError):
        return False
    if not isinstance(refs, list):
        return False
    for ref in refs:
        if isinstance(ref, dict) and (ref.get("message_id") or "") in decision_message_ids:
            return True
    return False


async def get_open_tracker_items() -> OpenTrackerResponse:
    """
    Read all clusters from the ``email_clusters`` store, parse their action
    items, classify each by due date, and return an ``OpenTrackerResponse``
    ready to be serialised by the route handler.

    Items that have already been marked complete (stored in
    ``completed_tracker_items.json``) are filtered out so they no longer
    appear on the Launchpad card.
    """
    today = datetime.now(timezone.utc).date()
    clusters = await read_all_cluster_records()

    # Load the set of already-completed IDs so we can filter them out.
    completed_ids = await load_completed_ids()

    # Load the set of deleted IDs so we can filter them out.
    deleted_ids = await load_deleted_ids()

    # Load full item overrides (text, owner, business_unit, due_date edits from edit modal).
    item_overrides = await load_item_overrides()

    # TEMP: message_ids whose mail was addressed to the decision recipient.
    # Clusters referencing any of these have their action items ALSO mirrored
    # into "Awaiting Your Decision".
    decision_message_ids = await _message_ids_addressed_to_decision_recipient()

    items: list[OpenTrackerItem] = []
    pending_decisions: list[PendingDecisionItem] = []

    for cluster in clusters:
        if not cluster.action_items or not cluster.action_items.strip():
            # Skip clusters with no action items
            continue

        cluster_to_decision = _cluster_addressed_to_decision_recipient(
            cluster, decision_message_ids
        )

        raw_items = [s.strip() for s in cluster.action_items.split("|") if s.strip()]
        for raw in raw_items:
            owner, task, due_date = _parse_action_item_string(raw)
            if not task:
                continue

            # Skip items the user has already marked complete or deleted.
            item_id = make_item_id(owner, task, cluster.department or "")
            if item_id in completed_ids or item_id in deleted_ids:
                continue

            business_unit = cluster.department or ""

            # Apply full item overrides (from edit modal) — these win over CSV values.
            if item_id in item_overrides:
                ov = item_overrides[item_id]
                if "text" in ov:
                    task = ov["text"]
                if "owner" in ov:
                    owner = ov["owner"]
                if "business_unit" in ov:
                    business_unit = ov["business_unit"]
                if "due_date" in ov and ov["due_date"]:  # Only if non-empty
                    parsed = _parse_date(ov["due_date"])
                    if parsed is not None:
                        due_date = parsed
                    # If override exists but is invalid, keep CSV date (don't set to None)

            status, overdue_by_days, due_label = _classify(due_date, today)

            if owner.strip().lower() in _AWAITING_DECISION_OWNERS:
                pending_decisions.append(
                    PendingDecisionItem(
                        id=item_id,
                        text=task,
                        owner=owner,
                        due_date=due_date,
                        due_label=due_label,
                        business_unit=business_unit,
                        source_topic=cluster.topic or "",
                        status=status,
                    )
                )
            else:
                items.append(
                    OpenTrackerItem(
                        id=item_id,
                        text=task,
                        owner=owner,
                        due_date=due_date,
                        due_label=due_label,
                        business_unit=business_unit,
                        status=status,
                        overdue_by_days=overdue_by_days,
                        source_topic=cluster.topic or "",
                    )
                )

                # TEMP: the cluster's mail was addressed to the decision
                # recipient, so ALSO surface this item in "Awaiting Your
                # Decision". It still appears in the Open tracker above; this
                # is an additive mirror, not a reroute. (Items already routed
                # to pending_decisions by owner are skipped to avoid dupes.)
                if cluster_to_decision:
                    pending_decisions.append(
                        PendingDecisionItem(
                            id=item_id,
                            text=task,
                            owner=owner,
                            due_date=due_date,
                            due_label=due_label,
                            business_unit=business_unit,
                            source_topic=cluster.topic or "",
                            status=status,
                        )
                    )

    # ---------------------------------------------------------------------------
    # Cross-cluster dedup: collapse items with the same (owner, task, bu)
    # identity that appear in multiple cluster rows (e.g. the same action was
    # captured across two meeting occurrences).  We keep the entry with the
    # latest due_date — if one variant has no date, prefer the dated one.
    # ---------------------------------------------------------------------------
    def _dedup_items(item_list: list) -> list:
        best: dict[str, object] = {}
        for item in item_list:
            key = f"{item.owner.strip().lower()}||{item.text.strip().lower()}||{item.business_unit.strip().lower()}"
            existing = best.get(key)
            if existing is None:
                best[key] = item
            else:
                # Prefer the item with a due_date; if both have one, keep later.
                existing_date = existing.due_date  # type: ignore[attr-defined]
                new_date = item.due_date
                if new_date is None:
                    pass  # existing is at least as good
                elif existing_date is None or new_date > existing_date:
                    best[key] = item
        return list(best.values())

    items = _dedup_items(items)
    pending_decisions = _dedup_items(pending_decisions)

    # Sort: Overdue first (most days overdue first), then Due Today, then Upcoming.
    _sort_order = {
        OpenTrackerStatus.OVERDUE: 0,
        OpenTrackerStatus.DUE_TODAY: 1,
        OpenTrackerStatus.UPCOMING: 2,
    }
    items.sort(key=lambda x: (
        _sort_order[x.status],
        -(x.overdue_by_days or 0),
    ))

    overdue_count = sum(1 for i in items if i.status == OpenTrackerStatus.OVERDUE)
    due_today_count = sum(1 for i in items if i.status == OpenTrackerStatus.DUE_TODAY)
    upcoming_count = sum(1 for i in items if i.status == OpenTrackerStatus.UPCOMING)

    logger.info(
        "Open tracker: %d overdue, %d due today, %d upcoming (total %d)",
        overdue_count, due_today_count, upcoming_count, len(items),
    )

    pending_overdue_count = sum(1 for p in pending_decisions if p.status == OpenTrackerStatus.OVERDUE)
    pending_due_today_count = sum(1 for p in pending_decisions if p.status == OpenTrackerStatus.DUE_TODAY)
    pending_upcoming_count = sum(1 for p in pending_decisions if p.status == OpenTrackerStatus.UPCOMING)

    return OpenTrackerResponse(
        items=items,
        pending_decisions=pending_decisions,
        overdue_count=overdue_count,
        due_today_count=due_today_count,
        upcoming_count=upcoming_count,
        total_count=len(items),
        pending_overdue_count=pending_overdue_count,
        pending_due_today_count=pending_due_today_count,
        pending_upcoming_count=pending_upcoming_count,
    )


async def complete_item(item_id: str) -> dict | None:
    """
    Mark a tracker action item as complete.

    Looks the item up in the current cluster data to fetch its full
    details, writes it to ``completed_tracker_items.json``, and returns
    the saved entry (or None if the item_id is not found in the clusters).
    """
    today = datetime.now(timezone.utc).date()
    clusters = await read_all_cluster_records()

    for cluster in clusters:
        if not cluster.action_items:
            continue
        for raw in [s.strip() for s in cluster.action_items.split("|") if s.strip()]:
            owner, task, due_date = _parse_action_item_string(raw)
            if not task:
                continue
            candidate_id = make_item_id(owner, task, cluster.department or "")
            if candidate_id != item_id:
                continue

            # Found — persist as completed.
            due_date_str = due_date.isoformat() if due_date else None
            entry = await save_completed_item(
                item_id=item_id,
                text=task,
                owner=owner,
                business_unit=cluster.department or "",
                due_date=due_date_str,
                source_topic=cluster.topic or "",
            )
            return entry

    sanitized_item_id = sanitize_log_input(item_id)
    logger.warning("complete_item: item_id %s not found in current clusters", sanitized_item_id)
    return None


async def uncomplete_item(item_id: str) -> dict | None:
    """
    Undo a previously completed tracker action item.

    Removes the item from ``completed_tracker_items.json`` (or the
    Postgres ``tracker_completed_items`` table) so it reappears on the
    next ``GET /agents/open-tracker`` call — routed back into either the
    ``items`` or ``pending_decisions`` bucket based on its owner, exactly
    as if it had never been completed.

    Looks the item up in the current cluster data to return its full
    details (owner, business_unit, source_topic, etc.) so the frontend
    can immediately restore it without waiting for the next poll.

    Returns None if the item_id is not found in the current clusters
    (e.g. the cluster pipeline has re-run and the item no longer exists),
    OR if it was not found in the completed store (nothing to undo).
    """
    # Strip CR/LF from the user-controlled item_id before logging to
    # prevent log injection / forged log lines (CWE-117).
    safe_item_id = str(item_id).replace("\r\n", "").replace("\r", "").replace("\n", "")

    removed = await remove_completed_item(item_id)
    if not removed:
        logger.warning("uncomplete_item: item_id %s was not marked complete", safe_item_id)
        return None

    today = datetime.now(timezone.utc).date()
    clusters = await read_all_cluster_records()

    for cluster in clusters:
        if not cluster.action_items:
            continue
        for raw in [s.strip() for s in cluster.action_items.split("|") if s.strip()]:
            owner, task, due_date = _parse_action_item_string(raw)
            if not task:
                continue
            candidate_id = make_item_id(owner, task, cluster.department or "")
            if candidate_id != item_id:
                continue

            status, overdue_by_days, due_label = _classify(due_date, today)
            return {
                "id": item_id,
                "text": task,
                "owner": owner,
                "business_unit": cluster.department or "",
                "due_date": due_date.isoformat() if due_date else None,
                "due_label": due_label,
                "status": status.value,
                "overdue_by_days": overdue_by_days,
                "source_topic": cluster.topic or "",
                "is_pending_decision": owner.strip().lower() in _AWAITING_DECISION_OWNERS,
            }

    # The item was successfully removed from the completed store, but the
    # cluster it originally belonged to no longer exists (CSV re-generated).
    # Still report success — the frontend just won't get full item details.
    logger.info(
        "uncomplete_item: item_id %s removed from completed store but no longer "
        "present in current clusters",
        safe_item_id,
    )
    return {"id": item_id}


async def update_due_date(item_id: str, due_date_str: str) -> dict | None:
    """
    Set a manual due-date override for the given tracker item.

    Validates that the item actually exists in the current clusters,
    persists the override to ``item_overrides`` via save_item_override,
    and returns the updated item details.  Returns None if the item_id is not found.
    """
    clusters = await read_all_cluster_records()

    for cluster in clusters:
        if not cluster.action_items:
            continue
        for raw in [s.strip() for s in cluster.action_items.split("|") if s.strip()]:
            owner, task, _csv_due = _parse_action_item_string(raw)
            if not task:
                continue
            candidate_id = make_item_id(owner, task, cluster.department or "")
            if candidate_id != item_id:
                continue

            # Found — persist via item_overrides (single source of truth).
            entry = await save_item_override(item_id, due_date=due_date_str)

            # Re-classify with the new date so the response reflects
            # the updated status immediately.
            today = datetime.now(timezone.utc).date()
            new_due = _parse_date(due_date_str)
            status, overdue_by_days, due_label = _classify(new_due, today)

            return {
                "id": item_id,
                "text": task,
                "owner": owner,
                "business_unit": cluster.department or "",
                "due_date": due_date_str,
                "due_label": due_label,
                "status": status.value,
                "overdue_by_days": overdue_by_days,
                "source_topic": cluster.topic or "",
                "updated_at": entry.get("updated_at"),
            }

    # Strip CR/LF from the user-controlled item_id before logging (CWE-117).
    safe_item_id = str(item_id).replace("\r\n", "").replace("\r", "").replace("\n", "")
    logger.warning("update_due_date: item_id %s not found in current clusters", safe_item_id)
    return None


async def update_item(
    item_id: str,
    text: str | None = None,
    owner: str | None = None,
    business_unit: str | None = None,
    due_date_str: str | None = None,
) -> dict | None:
    """
    Save manual overrides for any combination of an item's fields
    (text/description, owner, business_unit/team, due_date).

    Validates that the item exists in the current clusters, persists
    the overrides to ``item_overrides.json``, and returns the updated
    item with re-classified status.  Returns None if the item_id is
    not found.
    """
    clusters = await read_all_cluster_records()

    for cluster in clusters:
        if not cluster.action_items:
            continue
        for raw in [s.strip() for s in cluster.action_items.split("|") if s.strip()]:
            csv_owner, csv_task, csv_due = _parse_action_item_string(raw)
            if not csv_task:
                continue
            candidate_id = make_item_id(csv_owner, csv_task, cluster.department or "")
            if candidate_id != item_id:
                continue

            # Found — persist the overrides.
            entry = await save_item_override(
                item_id=item_id,
                text=text,
                owner=owner,
                business_unit=business_unit,
                due_date=due_date_str,
            )

            # Build the response using overridden values where provided,
            # falling back to CSV values.
            final_text = text if text is not None else csv_task
            final_owner = owner if owner is not None else csv_owner
            final_bu = business_unit if business_unit is not None else (cluster.department or "")

            today = datetime.now(timezone.utc).date()
            if due_date_str:
                final_due = _parse_date(due_date_str)
            else:
                final_due = csv_due

            status, overdue_by_days, due_label = _classify(final_due, today)

            return {
                "id": item_id,
                "text": final_text,
                "owner": final_owner,
                "business_unit": final_bu,
                "due_date": due_date_str or (final_due.isoformat() if final_due else None),
                "due_label": due_label,
                "status": status.value,
                "overdue_by_days": overdue_by_days,
                "source_topic": cluster.topic or "",
                "updated_at": entry.get("updated_at"),
            }

    # Strip CR/LF from the user-controlled item_id before logging (CWE-117).
    safe_item_id = str(item_id).replace("\r\n", "").replace("\r", "").replace("\n", "")
    logger.warning("update_item: item_id %s not found in current clusters", safe_item_id)
    return None


async def delete_item(item_id: str) -> dict | None:
    """
    Permanently delete a tracker action item.

    Looks the item up in the current cluster data to validate it exists,
    writes its ID to the deleted store (Postgres or JSON), and returns
    a confirmation dict (or None if the item_id is not found).
    """
    clusters = await read_all_cluster_records()

    for cluster in clusters:
        if not cluster.action_items:
            continue
        for raw in [s.strip() for s in cluster.action_items.split("|") if s.strip()]:
            owner, task, due_date = _parse_action_item_string(raw)
            if not task:
                continue
            candidate_id = make_item_id(owner, task, cluster.department or "")
            if candidate_id != item_id:
                continue

            # Found - persist as deleted.
            entry = await save_deleted_item(item_id)
            entry.update({
                "text": task,
                "owner": owner,
                "business_unit": cluster.department or "",
            })
            return entry

    sanitized_item_id = sanitize_log_input(item_id)
    logger.warning("delete_item: item_id %s not found in current clusters", sanitized_item_id)
    return None
