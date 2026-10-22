"""
chatbot/graph.py
===================
LangGraph subgraph for the Chatbot agent.

Wraps the RAG-based `answer_question` in a single-node StateGraph.
The chatbot uses Weaviate hybrid vector search over indexed email messages to
retrieve context and compose grounded answers via Ollama.

This subgraph is compiled once at import time. Unlike other agents,
the Chatbot is not wired into `core.orchestrator.graph` as a node -
it operates independently with its own RAG pipeline.
"""

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from chatbot.schema import ChatResponse
from chatbot.service import answer_question
from core.logging_config import get_agent_logger

logger = get_agent_logger("chatbot")


class ChatbotState(TypedDict, total=False):
    """
    State passed through the Chatbot subgraph.

    - `question`: the user's message (input).
    - `bearer_token`: kept for API compatibility (input).
    - `bm25_index`: the local BM25 index for re-ranking (input).
    - `wv_client`: the connected Weaviate client (input).
    - `response`: the composed `ChatResponse` (output).
    """

    question: str
    bearer_token: Optional[str]
    bm25_index: object  # BM25Index instance
    wv_client: object   # weaviate.WeaviateClient instance
    response: Optional[ChatResponse]


async def answer(state: ChatbotState) -> ChatbotState:
    """Node: run the RAG retrieval + answer-composition flow."""
    response = await answer_question(
        state["question"],
        bearer_token=state.get("bearer_token"),
        bm25_index=state.get("bm25_index"),
        wv_client=state.get("wv_client"),
    )
    return {"response": response}


def build_chatbot_graph():
    """Build and compile the Chatbot agent's LangGraph subgraph."""
    graph = StateGraph(ChatbotState)
    graph.add_node("answer", answer)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)
    return graph.compile()


# Compiled once at import time - reused across requests.
chatbot_graph = build_chatbot_graph()
