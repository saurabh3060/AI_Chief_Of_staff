"""
core/orchestrator/state.py
=============================
Shared state passed between the orchestrator's router node and the
per-agent nodes.

`OrchestratorState` is the single `TypedDict` that flows through the
top-level LangGraph. The router reads `intent` to decide which agent
node to run; each agent node writes its result into its own dedicated
`*_result` slot so results never collide and a caller can tell which
agent actually ran by which slot is populated.

Kept in its own module (rather than inline in `graph.py`) so the state
shape can be imported by tests and by any future node without importing
the whole compiled graph.
"""

from typing import Literal, TypedDict

# The set of agent intents the orchestrator knows how to route. Anything
# outside this set routes straight to END without running an agent node.
AgentIntent = Literal["calendar", "email", "actions"]


class OrchestratorState(TypedDict, total=False):
    """
    State threaded through the top-level orchestrator graph.

    - `intent`: which agent to dispatch to (set by the caller of
      `run_agent`); the router node reads this.
    - `bearer_token`: optional delegated Graph access token, forwarded
      to the agent subgraphs that need it (Calendar, Email). None in
      app-only mode.
    - `calendar_result` / `email_result` / `action_result`: the output
      of whichever agent node ran, copied verbatim from that agent's
      compiled subgraph. Only the slot for the dispatched intent is
      populated; the rest stay unset.
    """

    intent: AgentIntent
    bearer_token: str | None
    calendar_result: dict
    email_result: dict
    action_result: dict
