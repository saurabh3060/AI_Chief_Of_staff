"""
agent_calendar/graph.py
==========================
LangGraph subgraph for the Calendar & Meeting agent.

Flow:
1. `fetch_and_detect` - fetch calendar events via GraphClient and detect
   the current/next meeting (delegates to `agent_calendar.service`).
2. `flag_for_brief` - if a next meeting was found, generate (or reuse a
   previously stored) pre-meet brief for it.

This subgraph is compiled once (`build_calendar_graph()` returns the
compiled graph) and is what `core.orchestrator.graph` imports as a node
- the orchestrator never reaches into `agent_calendar.service` directly,
it only invokes this compiled subgraph.
"""

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent_calendar.schema import MeetingBrief, MeetingWindow
from agent_calendar.service import get_current_and_next_meeting, get_or_generate_brief
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_calendar")


class CalendarAgentState(TypedDict, total=False):
    """
    State passed between nodes in the Calendar agent subgraph.

    - `bearer_token`: delegated-mode auth token, if applicable (input).
    - `meeting_window`: result of detecting current/next meeting.
    - `brief`: generated pre-meet brief for the next meeting, if any.
    """

    bearer_token: Optional[str]
    meeting_window: Optional[MeetingWindow]
    brief: Optional[MeetingBrief]


async def fetch_and_detect(state: CalendarAgentState) -> CalendarAgentState:
    """Node: fetch calendar events and detect the current/next meeting."""
    window = await get_current_and_next_meeting(bearer_token=state.get("bearer_token"))
    return {"meeting_window": window}


async def flag_for_brief(state: CalendarAgentState) -> CalendarAgentState:
    """
    Node: if a next meeting was detected, flag it for a pre-meet brief
    by generating (or reusing) one via `agent_calendar.service`.
    """
    window = state.get("meeting_window")
    if window is None or window.next_meeting is None:
        logger.info("No upcoming meeting detected; skipping brief generation")
        return {"brief": None}

    brief = await get_or_generate_brief(window.next_meeting)
    return {"brief": brief}


def build_calendar_graph():
    """Build and compile the Calendar & Meeting agent's LangGraph subgraph."""
    graph = StateGraph(CalendarAgentState)
    graph.add_node("fetch_and_detect", fetch_and_detect)
    graph.add_node("flag_for_brief", flag_for_brief)
    graph.set_entry_point("fetch_and_detect")
    graph.add_edge("fetch_and_detect", "flag_for_brief")
    graph.add_edge("flag_for_brief", END)
    return graph.compile()


# Compiled once at import time - agents/orchestrator reuse this instance
# rather than recompiling the graph on every request.
calendar_graph = build_calendar_graph()
