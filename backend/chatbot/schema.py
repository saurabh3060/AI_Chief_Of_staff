"""
chatbot/schema.py
====================
Pydantic models for the Chatbot agent.

- `ChatRequest`: the incoming user question (plus an optional bearer
  token, for delegated Graph auth mode when the chatbot needs to
  consult the Calendar or Email agents).
- `ChatResponse`: the composed answer with sources, timings, and
  follow-up questions — matching the standalone chatbot's /chat contract.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """
    Incoming chatbot query.

    `bearer_token` is retained for backward compatibility but is now
    ignored: the Graph token is resolved server-side from the session
    cookie (see `auth/dependencies.get_bearer_token`), not sent by the
    client.

    `mode` selects the answering pipeline:
      - "email" (default): RAG over indexed email content (existing pipeline).
      - "document": RAG over ingested MIS/PDF documents (documents module).
    """

    message: str
    bearer_token: str | None = None
    mode: str = "email"  # "email" | "document"


class SourceInfo(BaseModel):
    """One source citation returned alongside the chatbot answer."""

    conversation_id: str | None = None
    message_count: int | None = None
    first_received_at: str | None = None
    last_received_at: str | None = None
    participants: list[str] | None = None
    subject: str = "Unknown"
    filename: str = ""
    score: float = 0.0
    chunk_index: int | None = None
    source_type: str | None = None
    context_sent: str | None = None
    matched_message_id: str | None = None
    matched_message_subject: str | None = None
    matched_message_received_at: str | None = None
    matched_message_content: str | None = None


class TimingInfo(BaseModel):
    """Timing breakdown for a chatbot response."""

    retrieval: float | None = None
    context_format: float | None = None
    llm_generation: float | None = None
    first_token: float | None = None
    total: float = 0.0
    tokens: int | None = None
    tokens_per_sec: float | None = None


class ChatResponse(BaseModel):
    """Chatbot's composed answer with sources, timings, and follow-up questions.

    Matches the standalone chatbot backend's /chat response contract so
    the frontend can consume either backend interchangeably.
    """

    answer: str
    sources: list[SourceInfo] = []
    is_greeting: bool = False
    latency_ms: float = 0.0
    timings: TimingInfo = TimingInfo()
    follow_up_questions: list[str] = []
    tool_used: str | None = None


class ChatHistoryEntry(BaseModel):
    """A single persisted Q&A pair, as returned by GET /chatbot/history."""

    question: str
    answer: str
    timestamp: float


class ChatHistoryResponse(BaseModel):
    """Response body for GET /chatbot/history."""

    history: list[ChatHistoryEntry]


class HealthResponse(BaseModel):
    """Response body for GET /chatbot/health."""

    status: str
    index_ready: bool
    total_chunks: int = 0
    total_vectors: int = 0
    requests_served: int = 0
