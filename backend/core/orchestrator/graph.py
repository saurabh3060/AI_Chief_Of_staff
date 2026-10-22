"""
core/orchestrator/graph.py
=============================
The top-level LangGraph orchestrator: a single `StateGraph` with one
node per wired agent (Calendar, Email, Action Tracker), dispatched by
`intent` via a conditional edge from a router node.

Each agent node invokes that agent's own compiled subgraph (imported as
an opaque unit - see package docstring) and copies its result into the
shared `OrchestratorState`. This is the structural mechanism that
enforces "no intermixed agent logic": the orchestrator can only ever
call an agent through its public compiled graph, never through
internals.

`run_agent()` is the function other code should call - it wraps
`orchestrator_graph.ainvoke()` and attaches Langfuse callbacks (if
enabled) automatically, so callers don't need to think about
observability wiring.
"""

from langgraph.graph import END, StateGraph

from agent_action_tracker.graph import action_tracker_graph
from agent_calendar.graph import calendar_graph
from agent_email.graph import email_graph
from core.logging_config import get_agent_logger
from core.observability import get_langfuse_callbacks
from core.orchestrator.state import AgentIntent, OrchestratorState

logger = get_agent_logger("orchestrator")


async def route(state: OrchestratorState) -> OrchestratorState:
    """
    Router node (entry point): does no work itself. It exists so the
    graph has a single entry point from which the conditional edge
    (`route_by_intent`) can dispatch to exactly one agent node - keeping
    dispatch data-driven off `intent` rather than baked into the caller.
    """
    return {}


def route_by_intent(state: OrchestratorState) -> str:
    """
    Conditional-edge function: returns the name of the node to run next
    based on `intent`. An unrecognized intent routes straight to END so
    the graph terminates cleanly instead of raising.
    """
    intent = state.get("intent")
    if intent in ("calendar", "email", "actions"):
        return intent
    logger.warning("Unrecognized orchestrator intent: %s", intent)
    return END


async def run_calendar_agent(state: OrchestratorState) -> OrchestratorState:
    """Node: invoke the Calendar & Meeting agent's compiled subgraph."""
    result = await calendar_graph.ainvoke({"bearer_token": state.get("bearer_token")})
    return {"calendar_result": result}


async def run_email_agent(state: OrchestratorState) -> OrchestratorState:
    """Node: invoke the Email & Summary agent's compiled subgraph."""
    result = await email_graph.ainvoke({"bearer_token": state.get("bearer_token")})
    return {"email_result": result}


async def run_action_tracker_agent(state: OrchestratorState) -> OrchestratorState:
    """Node: invoke the Action & Follow-up Tracker agent's compiled subgraph."""
    result = await action_tracker_graph.ainvoke({"action_items": []})
    return {"action_result": result}


def build_orchestrator_graph():
    """
    Build and compile the top-level orchestrator graph: a router entry
    point that conditionally dispatches to exactly one agent node based
    on `intent`, then ends.
    """
    graph = StateGraph(OrchestratorState)

    graph.add_node("route", route)
    graph.add_node("calendar", run_calendar_agent)
    graph.add_node("email", run_email_agent)
    graph.add_node("actions", run_action_tracker_agent)

    graph.set_entry_point("route")

    # The router node does no work; it only decides which agent node to
    # dispatch to (or END for an unrecognized intent) via the conditional
    # edge below, so dispatch is data-driven off `intent`.
    graph.add_conditional_edges(
        "route",
        route_by_intent,
        {
            "calendar": "calendar",
            "email": "email",
            "actions": "actions",
            END: END,
        },
    )

    # Each agent node runs exactly once, then the graph ends.
    graph.add_edge("calendar", END)
    graph.add_edge("email", END)
    graph.add_edge("actions", END)

    return graph.compile()


# Compiled once at import time - callers reuse this instance rather than
# recompiling the graph on every request.
orchestrator_graph = build_orchestrator_graph()


async def run_agent(intent: AgentIntent, bearer_token: str | None = None) -> OrchestratorState:
    """
    Run the orchestrator for a single agent `intent`, returning the
    final `OrchestratorState` (with the dispatched agent's result
    populated in its `*_result` slot).

    Attaches Langfuse callbacks automatically when observability is
    enabled (via `core.observability.get_langfuse_callbacks`), and
    passes no callbacks config at all when it is disabled - so callers
    never have to wire observability themselves.
    """
    initial_state: OrchestratorState = {"intent": intent, "bearer_token": bearer_token}

    callbacks = get_langfuse_callbacks()
    config = {"callbacks": callbacks} if callbacks else None

    logger.info("Orchestrator dispatching intent=%s (langfuse=%s)", intent, bool(callbacks))
    return await orchestrator_graph.ainvoke(initial_state, config=config)
