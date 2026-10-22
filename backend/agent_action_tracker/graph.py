"""
agent_action_tracker/graph.py
================================
LangGraph subgraph for the Action & Follow-up Tracker agent.

Flow (single node - this agent's work is simple enough not to need
multiple stages):
1. `classify_actions` - pull action candidates from the Email &
   Summary agent's persisted summaries and classify them, via
   `agent_action_tracker.service`.

This subgraph is compiled once (`build_action_tracker_graph()` returns
the compiled graph) and is what `core.orchestrator.graph` imports as a
node - the orchestrator never reaches into
`agent_action_tracker.service` directly, it only invokes this compiled
subgraph.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from agent_action_tracker.schema import ActionItem
from agent_action_tracker.service import classify_all_action_items
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_action_tracker")


class ActionTrackerState(TypedDict, total=False):
    """State passed through the Action Tracker subgraph: just the classified output list."""

    action_items: list[ActionItem]


async def classify_actions(state: ActionTrackerState) -> ActionTrackerState:
    """Node: classify all action items derived from the Email agent's persisted summaries."""
    items = await classify_all_action_items()
    return {"action_items": items}


def build_action_tracker_graph():
    """Build and compile the Action & Follow-up Tracker agent's LangGraph subgraph."""
    graph = StateGraph(ActionTrackerState)
    graph.add_node("classify_actions", classify_actions)
    graph.set_entry_point("classify_actions")
    graph.add_edge("classify_actions", END)
    return graph.compile()


# Compiled once at import time - agents/orchestrator reuse this instance
# rather than recompiling the graph on every request.
action_tracker_graph = build_action_tracker_graph()
