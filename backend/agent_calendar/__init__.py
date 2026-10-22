"""
agent_calendar package
========================
Calendar & Meeting Agent.

Responsibility: detect the current and next scheduled meeting in
real time (on-demand, per request - no background polling) from the
configured mailbox's calendar, and flag the next upcoming meeting for
an auto-generated pre-meet brief.

Everything this agent needs (schema, LangGraph subgraph, business
logic, prompts, routes) lives in this folder. Other agents/packages
should only ever call into `agent_calendar.service` functions (never
reach into `graph.py`/internals directly) if they need calendar data -
per the feature-based, no-intermixed-logic architecture requirement.
"""
