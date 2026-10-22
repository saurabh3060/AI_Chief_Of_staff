"""
agent_email/graph.py
=======================
LangGraph subgraph for the Email & Summary agent.

Flow:
1. `fetch_conversation` - fetch the most recent conversation thread
   (including OCR'd attachments) via `agent_email.service`.
2. `summarize` - generate (or reuse a previously stored) summary for
   that thread, including extracted action-item candidates.

This subgraph is compiled once (`build_email_graph()` returns the
compiled graph) and is what `core.orchestrator.graph` imports as a
node - the orchestrator never reaches into `agent_email.service`
directly, it only invokes this compiled subgraph.
"""

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent_email.schema import ConversationThread, EmailSummary
from agent_email.service import get_or_generate_summary_for_thread, get_recent_conversation
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_email")


class EmailAgentState(TypedDict, total=False):
    """
    State passed between nodes in the Email agent subgraph.

    - `bearer_token`: delegated-mode auth token, if applicable (input).
    - `thread`: the fetched conversation thread.
    - `summary`: the generated (or reused) summary for that thread.
    """

    bearer_token: Optional[str]
    thread: Optional[ConversationThread]
    summary: Optional[EmailSummary]


async def fetch_conversation(state: EmailAgentState) -> EmailAgentState:
    """Node: fetch the most recent conversation thread, including OCR'd attachments."""
    thread = await get_recent_conversation(bearer_token=state.get("bearer_token"))
    return {"thread": thread}


async def summarize(state: EmailAgentState) -> EmailAgentState:
    """Node: generate (or reuse) a summary for the fetched thread, if one was found."""
    thread = state.get("thread")
    if thread is None:
        logger.info("No conversation thread found; skipping summarization")
        return {"summary": None}

    summary = await get_or_generate_summary_for_thread(thread)
    return {"summary": summary}


def build_email_graph():
    """Build and compile the Email & Summary agent's LangGraph subgraph."""
    graph = StateGraph(EmailAgentState)
    graph.add_node("fetch_conversation", fetch_conversation)
    graph.add_node("summarize", summarize)
    graph.set_entry_point("fetch_conversation")
    graph.add_edge("fetch_conversation", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


# Compiled once at import time - agents/orchestrator reuse this instance
# rather than recompiling the graph on every request.
email_graph = build_email_graph()
