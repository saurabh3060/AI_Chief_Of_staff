"""
agent_open_tracker/routes.py
=================================
FastAPI routes for the Open Actions Status Tracker agent.

Exposes a read endpoint plus write endpoints (complete/uncomplete/
delete/due-date/edit).

Authentication: a valid server-side session is enforced at the router
level in ``main.py`` via ``Depends(require_session)``. That gate runs
BEFORE any handler body here, so every endpoint below - read and write -
rejects a missing/invalid/expired ``rbl_session`` cookie with 401 before
it ever looks up an ``item_id`` (the fix for "a fake itemId returned 404
instead of 401"). A signed-in caller still sees identical behaviour,
including a real 404 for an ``item_id`` that is not in the current
clusters.

Authorization (IDOR) note: these items have NO per-user ownership. An
``item_id`` is a deterministic SHA-256 hash of the cluster action-item's
(owner, text, business_unit), where ``owner`` is the business owner named
in the mail (e.g. "Sandeep"), not the authenticated app user. The tracker
is shared single-tenant data for the one CEO's office - there is no
user/mailbox column to scope a query to, and every authenticated session
is by definition authorized to act on that shared data. The IDOR exposure
is therefore closed by requiring authentication (an unauthenticated caller
can no longer iterate ids at all); there is no finer-grained per-user
boundary to enforce within the authenticated set without changing the data
model and the response contract. See ``require_session`` for where the
gate is applied.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent_open_tracker.service import (
    complete_item,
    delete_item,
    get_open_tracker_items,
    uncomplete_item,
    update_due_date,
    update_item,
)
from agent_open_tracker.schema import OpenTrackerResponse
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_open_tracker")

router = APIRouter(prefix="/agents/open-tracker", tags=["open-tracker"])


class DueDatePatch(BaseModel):
    """Request body for updating an item's due date."""
    due_date: str


class ItemEditPatch(BaseModel):
    """Request body for editing action item fields."""
    text: str | None = None
    owner: str | None = None
    business_unit: str | None = None
    due_date: str | None = None


@router.get("", response_model=OpenTrackerResponse)
async def get_open_tracker() -> OpenTrackerResponse:
    """
    Return all open action items from the email cluster pipeline,
    classified by due date and tagged with their business unit.

    This endpoint reads the ``email_clusters`` store on every call — it is
    fast (cache/DB read, no LLM) and safe to poll every 60 seconds from
    the frontend.  Items already marked complete are filtered out.
    """
    return await get_open_tracker_items()


@router.post("/{item_id}/complete")
async def complete_tracker_item(item_id: str) -> dict:
    """
    Mark a tracker action item as complete.

    Writes the item to ``completed_tracker_items.json`` so it is
    excluded from future ``GET /agents/open-tracker`` responses.
    Returns the completed item's details so the frontend can inject it
    into the Action & Accountability Ledger's Completed tab.

    Returns 404 if the item_id is not found in the current clusters
    (e.g. the cluster pipeline has re-run and the item no longer exists).
    """
    entry = await complete_item(item_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Action item '{item_id}' not found in current clusters.",
        )
    return {"completed": True, "item": entry}


@router.delete("/{item_id}/complete")
async def uncomplete_tracker_item(item_id: str) -> dict:
    """
    Undo a previous completion of a tracker action item.

    Removes the item from ``completed_tracker_items.json`` so it
    reappears in future ``GET /agents/open-tracker`` responses — back in
    the Open Actions Status Tracker or the "Awaiting Your Decision" queue,
    whichever bucket its owner routes it to.

    Returns 404 if the item was not previously marked complete (nothing
    to undo).
    """
    entry = await uncomplete_item(item_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Action item '{item_id}' was not marked complete.",
        )
    return {"uncompleted": True, "item": entry}


@router.delete("/{item_id}")
async def delete_tracker_item(item_id: str) -> dict:
    """
    Permanently delete a tracker action item.

    Writes the item ID to the ``tracker_deleted_items`` table (or
    ``deleted_tracker_items.json`` fallback) so it is excluded from
    future ``GET /agents/open-tracker`` responses.

    Returns 404 if the item_id is not found in the current clusters.
    """
    entry = await delete_item(item_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Action item '{item_id}' not found in current clusters.",
        )
    return {"deleted": True, "item": entry}


@router.patch("/{item_id}/due-date")
async def update_tracker_due_date(item_id: str, body: DueDatePatch) -> dict:
    """
    Set or update the due date for a tracker action item.

    Expects JSON body: ``{ "due_date": "2026-09-30" }`` (ISO date string).

    Persists the override to ``due_date_overrides.json`` so it survives
    page refreshes and CSV regenerations.  Returns the updated item with
    re-classified status/label so the frontend can reflect the change
    immediately.

    Returns 400 if due_date is missing or invalid.
    Returns 404 if the item_id is not found in the current clusters.
    """
    due_date_str = body.due_date.strip()
    if not due_date_str:
        raise HTTPException(status_code=400, detail="due_date is required (ISO date string, e.g. '2026-09-30').")

    # Basic format validation
    import re
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", due_date_str):
        raise HTTPException(status_code=400, detail="due_date must be in YYYY-MM-DD format.")

    entry = await update_due_date(item_id, due_date_str)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Action item '{item_id}' not found in current clusters.",
        )
    return {"updated": True, "item": entry}


@router.patch("/{item_id}/edit")
async def edit_tracker_item(item_id: str, body: ItemEditPatch) -> dict:
    """
    Edit one or more fields of a tracker action item.

    Expects JSON body with any subset of::

        {
            "text": "new description",
            "owner": "New Owner",
            "business_unit": "Credit Cards",
            "due_date": "2026-10-15"
        }

    Persists overrides to ``item_overrides.json``.  Returns the updated
    item with re-classified status so the frontend can reflect changes
    immediately.

    Returns 400 if no editable fields are provided.
    Returns 400 if due_date is present but not in YYYY-MM-DD format.
    Returns 404 if the item_id is not found in the current clusters.
    """
    text = body.text
    owner = body.owner
    business_unit = body.business_unit
    due_date = body.due_date

    # At least one field must be provided.
    if not any([text, owner, business_unit, due_date]):
        raise HTTPException(
            status_code=400,
            detail="At least one of text, owner, business_unit, or due_date is required.",
        )

    # Validate due_date format if provided.
    if due_date:
        import re as _re
        if not _re.match(r"^\d{4}-\d{2}-\d{2}$", due_date):
            raise HTTPException(status_code=400, detail="due_date must be in YYYY-MM-DD format.")

    entry = await update_item(
        item_id=item_id,
        text=text,
        owner=owner,
        business_unit=business_unit,
        due_date_str=due_date,
    )
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Action item '{item_id}' not found in current clusters.",
        )
    return {"updated": True, "item": entry}
