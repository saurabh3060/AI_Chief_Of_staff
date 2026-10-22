"""
agent_action_tracker/routes.py
=================================
FastAPI routes for the Action & Follow-up Tracker agent.

- `GET /agents/actions` - classify (or re-classify) action items from
  the Email agent's persisted summaries and return the full list,
  optionally filtered by `status` query param.
- `PATCH /agents/actions/{action_id}` - manually update a tracked
  action item (e.g. override its status or owner).

This agent never talks to Graph API directly, so no bearer token is
needed for its business logic. A valid server-side session is still
required, enforced at the router level in ``main.py`` via
``Depends(require_session)`` (delegated mode), so these endpoints are not
reachable without authentication - its data (the Email agent's persisted
summaries) is not public.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agent_action_tracker.schema import ActionStatus
from agent_action_tracker.service import classify_all_action_items, list_action_items, update_action_item
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_action_tracker")

router = APIRouter(prefix="/agents/actions", tags=["actions"])


class ActionItemPatch(BaseModel):
    """Request body for manually patching a tracked action item."""

    status: ActionStatus | None = None
    owner: str | None = None


@router.get("")
async def get_action_items(status: ActionStatus | None = Query(default=None)):
    """
    Re-classify action items from the Email agent's currently persisted
    summaries and return them, optionally filtered by `status`.
    """
    await classify_all_action_items()
    items = await list_action_items(status=status)
    return {"action_items": items}


@router.patch("/{action_id}")
async def patch_action_item(action_id: str, patch: ActionItemPatch):
    """Apply a manual override (status and/or owner) to a tracked action item."""
    patch_dict = patch.model_dump(mode="json", exclude_none=True)
    if not patch_dict:
        raise HTTPException(status_code=400, detail="No fields provided to update.")

    updated = await update_action_item(action_id, patch_dict)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Action item '{action_id}' not found.")
    return {"action_item": updated}
