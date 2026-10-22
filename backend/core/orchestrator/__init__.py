"""
core/orchestrator package
==========================
The top-level multi-agent orchestrator.

This package owns the single LangGraph `StateGraph` that dispatches a
request to exactly one agent (Calendar, Email, or Action Tracker) based
on an `intent`, and copies that agent's result into a shared state.

Design rule enforced here: each agent is invoked ONLY through its own
public compiled subgraph (`calendar_graph`, `email_graph`,
`action_tracker_graph`), never through that agent's `service.py`
internals. The orchestrator treats each agent as an opaque unit, so no
agent's logic leaks into another's.

Public surface: `run_agent(intent, bearer_token=None)` in
`core.orchestrator.graph` - the single entry point other code should
call. It wraps the compiled graph's `ainvoke` and attaches Langfuse
callbacks automatically when observability is enabled.
"""
