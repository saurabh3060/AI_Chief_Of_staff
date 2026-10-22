"""
agent_mis/schema.py
======================
Placeholder pydantic model for the MIS Insight agent's future output.

Kept minimal on purpose - this agent is a stub (see package docstring).
`MetricInsight` sketches the eventual shape (a named metric with a
value and short commentary) but is not produced by any real logic yet.
"""

from pydantic import BaseModel


class MetricInsight(BaseModel):
    """Placeholder shape for a single MIS-derived metric insight (not yet populated by real logic)."""

    metric_name: str
    value: float | None = None
    commentary: str | None = None
