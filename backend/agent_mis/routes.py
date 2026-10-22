"""
agent_mis/routes.py
======================
FastAPI routes for the MIS Insight agent - STUB.

- `GET /agents/mis/insights` - always returns HTTP 501 Not Implemented
  with a clear message, rather than a 404 (the route exists and is
  reachable, it's just not built yet) or an unhandled crash.
"""

from fastapi import APIRouter, HTTPException

from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_mis")

router = APIRouter(prefix="/agents/mis", tags=["mis"])


@router.get("/insights")
async def get_insights():
    """Stub endpoint: MIS Insight agent is not implemented yet - returns 501 with a clear message."""
    logger.info("GET /agents/mis/insights called but agent is not implemented")
    raise HTTPException(status_code=501, detail="MIS Insight agent is not implemented yet.")
