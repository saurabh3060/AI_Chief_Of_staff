"""
agent_mis/service.py
=======================
Stub business logic for the MIS Insight agent.

Every function raises `AgentNotImplementedError` - this agent is
scaffolded for folder-structure consistency only, per the
implementation plan ("MIS Insight Agent stub"). Real logic (extracting
insights from MIS data sources) is deferred to a future implementation
pass.
"""

from core.exceptions import AgentNotImplementedError
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_mis")


async def get_metric_insights() -> list:
    """Stub: would eventually extract and return MIS-derived metric insights. Not implemented yet."""
    logger.info("get_metric_insights() called but MIS agent is not yet implemented")
    raise AgentNotImplementedError("MIS Insight agent is not implemented yet.")
