"""
agent_mis/graph.py
=====================
Stub LangGraph subgraph for the MIS Insight agent.

Compiles a minimal graph with a single node that raises
`AgentNotImplementedError` via `agent_mis.service.get_metric_insights()`.
This keeps the module importable and structurally consistent with the
other agents' `graph.py` files, without pretending any real
orchestration logic exists yet.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from agent_mis.service import get_metric_insights
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_mis")


class MisAgentState(TypedDict, total=False):
    """Placeholder state shape for the MIS agent stub."""

    insights: list


async def not_implemented_node(state: MisAgentState) -> MisAgentState:
    """Node: calls the stub service function, which raises AgentNotImplementedError."""
    await get_metric_insights()
    return {"insights": []}  # unreachable - get_metric_insights() always raises


def build_mis_graph():
    """Build and compile the MIS Insight agent's stub LangGraph subgraph."""
    graph = StateGraph(MisAgentState)
    graph.add_node("not_implemented_node", not_implemented_node)
    graph.set_entry_point("not_implemented_node")
    graph.add_edge("not_implemented_node", END)
    return graph.compile()


mis_graph = build_mis_graph()
