"""
chatbot/service.py
======================
LLM-powered chatbot that answers questions using retrieved email context.

Ported from the standalone chatbot_final project's chatbot.py, adapted to:
  - read tuning knobs (LLM model, max tokens, context window) from
    core.config.get_settings() instead of a module-local config.py
  - use this project's centralised logger
  - expose `answer_question()` / `generate_response_stream()` with the
    signatures expected by chatbot/routes.py (bearer_token kept for
    backward compatibility, wv_client/bm25_index passed through from
    the module-level chatbot state)
"""

import asyncio
import json
import queue
import re
import threading
import time
from typing import Generator

import ollama

from chatbot.postgres_store import save_chat_message
from chatbot.redis_history import save_message
from chatbot.redis_qa_cache import get_cached_answer, set_cached_answer
from chatbot.retriever import format_context, retrieve_weaviate
from chatbot.schema import ChatResponse, SourceInfo, TimingInfo
from core.config import get_settings
from core.db import get_main_loop
from core.logging_config import get_agent_logger
from core.utils import sanitize_log_input

logger = get_agent_logger("chatbot")


SYSTEM_PROMPT = """You are a concise email and meeting assistant. You are given real email threads as context covering all business units — including Wholesale Banking, Retail, Credit Cards, Liabilities, and any other topics present in the emails.

CRITICAL ANTI-HALLUCINATION RULE (HIGHEST PRIORITY):
- You MUST answer ONLY from the email context provided below. If the context does not contain information about what the user is asking, respond with EXACTLY: "Not found in the provided emails."
- NEVER describe your own capabilities, features, or what you can help with.
- NEVER say things like "I can provide insights on", "I'm here to help you", "I can help you navigate", "What would you like to focus on", or anything similar.
- NEVER invent, assume, or hallucinate any information, names, numbers, or facts that are not explicitly written in the provided email context.
- If you are unsure whether the context answers the question, say "Not found in the provided emails." — do NOT guess.

STRICT RULES — follow every one:
1. Answer in 8-9 lines maximum. No exceptions.
2. Always include exact figures, names, dates, and percentages as they appear in the emails — never round or omit them.
3. Answer from what is written in the emails. You may paraphrase and connect related points — but never invent facts, people, or numbers not present in the context. If genuinely nothing relevant exists in the provided emails, say "Not found in the provided emails."
4. Do not repeat the question, do not add preamble like "According to Email 1", do not add conclusions or summaries beyond what was asked.
5. If the question asks for action items, list them as bullet points with owner and due date — still within 7 lines total.
6. Do not explain what the email is about or who sent it unless that is what was asked.
7. If the question specifies a number of items (e.g. "two areas", "three reasons"), list exactly that many as separately numbered points. Each point must be distinct — do not split one idea into two, and do not merge two different ideas into one.
8. If the emails show a question being raised but never answered — no one named accounts, no plan was given, no response was provided — say explicitly: "The emails raise this question but no answer was provided." Do not fill the silence with related numbers or context from the email.
9. Read ALL provided emails before answering. If multiple emails contain different figures or answers for the same metric, use the one from the email whose content is most specific and directly answers the question — do not default to the first email.
10. When the question asks for a specific number, duration, quantity, or percentage (e.g. "how much", "how many", "how long", "what percentage") — state that exact figure first, before any explanation. Do not lead with qualitative context when a specific number exists in the emails.
11. PERIOD-SPECIFIC FIGURES — CRITICAL: Every number in the emails belongs only to the exact month, quarter, or period it is explicitly labeled with. A figure labeled "March" is ONLY the answer for a question about March. It is NEVER the answer for April, February, June, or any other period. If the question asks about April sanctions and the emails only show March sanctions, say: "April sanctions are not found in the provided emails." Do not substitute, reuse, or carry over any figure from one period to answer a question about a different period.
12. If the question asks about a specific time period (month, quarter, year) and no figure is explicitly labeled with that exact period in the emails, say: "Not found in the provided emails for that period." Never infer, estimate, or borrow a figure from an adjacent period.

FINAL REMINDER: If the provided email context does not discuss the topic the user asked about, your ONLY allowed response is: "Not found in the provided emails." Do NOT describe yourself or your capabilities. Do NOT make up an answer."""

GREETING_RESPONSE = (
    "Hi! I'm your Meeting, Email & Document Assistant. I can dig through your meeting summaries, "
    "email discussions, and uploaded documents.\n\n"
    "Ask me anything about your meetings, emails, or documents!"
)

NOT_FOUND_RESPONSE = "Not found in the provided emails."

# Hard ceiling on the assembled prompt (system + context + question). If the
# built prompt exceeds this many characters we shrink the context budget and
# rebuild ONCE before calling the LLM, so we never rely on Ollama truncating
# the input (which drops whole threads and corrupts answers).
PROMPT_CHAR_CEILING = 13000
# Fraction to shrink the context budget by on the single rebuild.
PROMPT_SHRINK_FACTOR = 0.6
# Warn when Ollama reports it evaluated more prompt tokens than this.
PROMPT_EVAL_WARN_TOKENS = 3500


def _build_user_message(context: str, query: str) -> str:
    """Assemble the user turn sent to the LLM from context + question."""
    return (
        f"Context from meeting summaries and emails:\n{context}\n\n"
        f"User question: {query}\n\n"
        f"Answer only from the context above."
    )


def _build_prompt_with_guard(retrieved: list[dict], query: str) -> tuple[str, str]:
    """Return (context, user_message), rebuilding once with a smaller budget
    if the assembled prompt exceeds PROMPT_CHAR_CEILING.

    Never depends on Ollama truncation: if the first build is too large we
    rebuild the context with a reduced ``budget_chars`` a single time.

    When CHATBOT_DEBUG_PROMPT is on, logs which threads were retrieved
    (subject + rank) and how many sentences build_context kept per thread.
    """
    from chatbot.retriever import build_context

    stats: dict = {}
    context = build_context(retrieved, query, stats=stats)
    user_message = _build_user_message(context, query)

    if len(user_message) > PROMPT_CHAR_CEILING:
        settings = get_settings()
        shrunk = max(1000, int(settings.chatbot_context_max_chars * PROMPT_SHRINK_FACTOR))
        logger.warning(
            "Built prompt %d chars > ceiling %d - rebuilding context once with budget_chars=%d",
            len(user_message), PROMPT_CHAR_CEILING, shrunk,
        )
        stats = {}
        context = build_context(retrieved, query, budget_chars=shrunk, stats=stats)
        user_message = _build_user_message(context, query)

    _debug_log_retrieval(retrieved, stats)

    return context, user_message


def _debug_enabled() -> bool:
    """True when CHATBOT_DEBUG_PROMPT is set."""
    return get_settings().chatbot_debug_prompt


def _debug_log_retrieval(retrieved: list[dict], stats: dict) -> None:
    """Log retrieved threads (subject + rank) and per-thread kept counts.

    No-op unless CHATBOT_DEBUG_PROMPT is on.
    """
    if not _debug_enabled():
        return

    for rank, chunk in enumerate(retrieved, 1):
        subject = chunk.get("subject") or chunk.get("thread_subject") or "(no subject)"
        score = chunk.get("combined_score", chunk.get("score", 0))
        logger.info("[CHATBOT-DEBUG] retrieved rank=%d score=%.3f subject=%r", rank, score, subject)

    for rank, subject, kept in stats.get("kept_per_thread", []):
        logger.info(
            "[CHATBOT-DEBUG] build_context kept rank=%d sentences=%d subject=%r",
            rank, kept, subject,
        )


def _debug_dump_prompt(user_message: str, raw_answer: str) -> None:
    """Write the final prompt (system + context + question) and the RAW LLM
    answer (before the hallucination regex / NOT_FOUND replacement) to the
    debug file, overwriting it each request.

    No-op unless CHATBOT_DEBUG_PROMPT is on. Never raises: a debug-dump
    failure must not break the chat response.
    """
    if not _debug_enabled():
        return

    settings = get_settings()
    path = settings.chatbot_debug_prompt_path
    try:
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        contents = (
            "===== SYSTEM PROMPT =====\n"
            f"{SYSTEM_PROMPT}\n\n"
            "===== USER MESSAGE (context + question) =====\n"
            f"{user_message}\n\n"
            "===== RAW LLM ANSWER (pre hallucination/NOT_FOUND) =====\n"
            f"{raw_answer}\n"
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(contents)
        logger.info("[CHATBOT-DEBUG] wrote prompt + raw answer to %s", path)
    except Exception as e:
        logger.warning("[CHATBOT-DEBUG] could not write debug prompt file %s: %s", path, e)


def _ns_to_s(value) -> float:
    """Ollama reports *_duration in nanoseconds; convert to seconds."""
    try:
        return round(float(value) / 1e9, 2)
    except (TypeError, ValueError):
        return 0.0


def _log_llm_metrics(
    response_obj,
    *,
    prompt_chars: int,
    retrieval_ms: float,
) -> None:
    """Log one metrics line per request from the Ollama response object.

    Logs prompt_chars, prompt_eval_count, prompt_eval_duration (s),
    eval_count, eval_duration (s), tokens/s and retrieval time. Emits a
    WARNING if prompt_eval_count exceeds PROMPT_EVAL_WARN_TOKENS.
    """
    def _get(key, default=0):
        if isinstance(response_obj, dict):
            return response_obj.get(key, default)
        return getattr(response_obj, key, default)

    prompt_eval_count = _get("prompt_eval_count", 0) or 0
    eval_count = _get("eval_count", 0) or 0
    prompt_eval_s = _ns_to_s(_get("prompt_eval_duration", 0))
    eval_s = _ns_to_s(_get("eval_duration", 0))
    tokens_per_sec = round(eval_count / eval_s, 1) if eval_s > 0 else 0.0

    logger.info(
        "LLM metrics: prompt_chars=%d prompt_eval_count=%d prompt_eval_duration=%.2fs "
        "eval_count=%d eval_duration=%.2fs tokens_per_sec=%.1f retrieval=%.1fms",
        prompt_chars, prompt_eval_count, prompt_eval_s,
        eval_count, eval_s, tokens_per_sec, retrieval_ms,
    )
    if prompt_eval_count > PROMPT_EVAL_WARN_TOKENS:
        logger.warning(
            "prompt_eval_count=%d exceeds %d - prompt is still too large; "
            "reduce CHATBOT_CONTEXT_MAX_CHARS",
            prompt_eval_count, PROMPT_EVAL_WARN_TOKENS,
        )


# ── Pre-LLM relevance gate ────────────────────────────────────


def _is_context_relevant(retrieved: list[dict]) -> bool:
    """Return True if the retrieved chunks likely contain useful context.

    Checks whether at least one chunk had a BM25 keyword match. If BM25
    found zero results, the retrieved emails have no keyword overlap with
    the query at all — semantic similarity alone is unreliable (Weaviate
    normalises the top score to 1.0 regardless of actual relevance).

    When BM25 returns nothing, we short-circuit with 'Not found' instead
    of letting the LLM hallucinate from irrelevant context.
    """
    if not retrieved:
        return False

    # Check if ANY chunk had a BM25 keyword match
    has_bm25_hit = any(
        chunk.get("bm25_score", 0) > 0 or chunk.get("bm25_score_norm", 0) > 0
        for chunk in retrieved
    )

    if not has_bm25_hit:
        logger.info(
            "Pre-LLM relevance check FAILED: no BM25 keyword hits across %d chunks — skipping LLM",
            len(retrieved),
        )
        return False

    return True


# ── Post-LLM hallucination detection ─────────────────────────
# Phrases that indicate the LLM is describing its capabilities instead
# of answering from email context. All matched case-insensitively.
_HALLUCINATION_PATTERNS = [
    "i can provide insights",
    "i can help you",
    "i'm here to help",
    "i am here to help",
    "what would you like",
    "how can i assist",
    "i can assist you",
    "executive dashboard",
    "performance metrics and key",
    "today's meetings and preparation",
    "priority levels",
]


def _is_hallucinated_response(text: str) -> bool:
    """Return True if the LLM response looks like a hallucinated capability
    description rather than an answer grounded in email context.

    Checks the response against known hallucination phrases
    (case-insensitive).
    """
    lower = text.lower()
    return any(pattern in lower for pattern in _HALLUCINATION_PATTERNS)


def _persist_message(query: str, answer: str) -> None:
    """Save a Q&A pair to both Redis (recent-window cache) and the
    permanent Postgres `chatbot_history` table.

    `generate_response_stream` is a sync generator (iterated by
    FastAPI's StreamingResponse in a worker thread - no running event
    loop in that thread), so the Postgres write can't simply be
    `await`-ed. It also can't use `asyncio.run()`: that spins up a
    *brand new* event loop, but the shared asyncpg pool's connections
    are bound to the main loop created in `core.db.init_db()` - using
    them from a different loop raises
    "cannot perform operation: another operation is in progress".
    Instead, schedule the coroutine onto that same main loop with
    `run_coroutine_threadsafe` and wait for it here. Never raises: a
    Postgres hiccup must not break the chat response, same contract as
    `save_message`.
    """
    save_message(query, answer)
    try:
        loop = get_main_loop()
        if loop is None:
            logger.warning("Could not persist chat message to Postgres: no main event loop available")
            return
        future = asyncio.run_coroutine_threadsafe(save_chat_message(query, answer), loop)
        future.result(timeout=10)
    except Exception as e:
        logger.warning("Could not persist chat message to Postgres: %s", e)


GREETINGS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "howdy", "greetings", "what's up", "whats up", "sup", "yo",
    "hi there", "hello there", "hey there",
    "hii", "hiii", "hiiii", "helloo", "hellooo", "heyyy", "heyy",
    "namaste", "namaskar", "good day",
    "how are you", "how r u", "wassup", "wazzup",
}


def is_greeting(query: str) -> bool:
    """Check if the user query is a simple greeting.

    Handles variations like 'hii', 'hiii', 'helloo', 'heyyy', 'hiiii there', etc.
    """
    cleaned = query.strip().lower().rstrip("!.,?;:")

    if cleaned in GREETINGS:
        return True

    # Normalize repeated characters: "hiii" -> "hi", "heyyy" -> "hey", "helloo" -> "hello"
    normalized = re.sub(r"(.)\1{1,}", r"\1", cleaned)
    if normalized in GREETINGS:
        return True

    # Handle "hii", "hiii" specifically (collapse only trailing repeats)
    normalized2 = re.sub(r"(.)\1+$", r"\1", cleaned)
    if normalized2 in GREETINGS:
        return True

    # Handle greetings with extra words: "hi there!", "hello sir", "hey buddy"
    first_word = cleaned.split()[0] if cleaned.split() else ""
    first_normalized = re.sub(r"(.)\1{1,}", r"\1", first_word)
    greeting_words = {"hi", "hello", "hey", "howdy", "greetings", "yo", "sup"}
    if first_normalized in greeting_words and len(cleaned.split()) <= 3:
        return True

    return False


def _source_key(chunk: dict) -> str:
    """Match source records by durable conversation (thread) identity."""
    conversation_id = str(chunk.get("conversation_id") or "").strip()
    if conversation_id:
        return f"conversation:{conversation_id}"
    meeting_id = str(chunk.get("meeting_id") or "").strip()
    if meeting_id:
        return f"meeting:{meeting_id}"
    return f"legacy:{chunk.get('email_id', '')}:{chunk.get('subject', '')}"


def _extract_matched_message_content(chunk: dict) -> str:
    """Find the matched message's rendered content from the thread's messages_json."""
    matched_id = chunk.get("matched_message_id", "")
    if not matched_id:
        return ""

    raw_json = chunk.get("messages_json", "[]")
    try:
        messages = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    except (json.JSONDecodeError, TypeError):
        return ""

    for msg in messages:
        if str(msg.get("message_id", "")) == str(matched_id):
            parts = []
            if msg.get("from_address"):
                parts.append(f"From:    {msg['from_address']}")
            if msg.get("received_at"):
                parts.append(f"Date:    {msg['received_at']}")
            if msg.get("subject"):
                parts.append(f"Subject: {msg['subject']}")
            parts.append("")
            body = (msg.get("body") or "").strip()
            attachment = (msg.get("attachment_data") or "").strip()
            if body:
                parts.append(body)
            if attachment:
                if body:
                    parts.append("")
                parts.append(f"[Attachment]\n{attachment}")
            return "\n".join(parts)

    return ""


def _get_sources(retrieved: list[dict]) -> list[dict]:
    """Extract unique source info from retrieved chunks, including the content sent to the LLM."""
    sources = []
    seen = set()
    for i, chunk in enumerate(retrieved):
        source_type = chunk.get("source_type", "email")
        score = chunk.get("combined_score", chunk.get("score", 0))

        if source_type == "email_conversation":
            participants = chunk.get("participants", [])
            if isinstance(participants, str):
                participants = [participants] if participants else []
            chunk_context = "\n".join([
                f"--- Conversation {i + 1}: {chunk.get('subject', 'No Subject')} (relevance: {score:.3f}) ---",
                f"Conversation ID: {chunk.get('conversation_id', 'N/A')}",
                f"Messages: {chunk.get('message_count', 0)}",
                f"Date range: {chunk.get('first_received_at', 'N/A')} to {chunk.get('last_received_at', 'N/A')}",
                f"Participants: {', '.join(participants) if participants else 'N/A'}",
                "",
                chunk.get("raw_content", ""),
            ])
        elif source_type == "email":
            chunk_context = "\n".join([
                f"--- Email {i + 1}: {chunk.get('subject', 'No Subject')} (relevance: {score:.3f}) ---",
                f"From:     {chunk.get('from_address', 'N/A')}",
                f"Received: {chunk.get('received_at', 'N/A')}",
                "",
                chunk.get("raw_content", ""),
            ])
        elif source_type == "meeting":
            chunk_context = (
                f"--- Meeting {i + 1} (relevance: {score:.3f}) ---\n"
                f"Subject: {chunk.get('subject', 'N/A')}"
            )
        else:
            chunk_context = chunk.get("raw_content", "")

        if source_type in ("email", "email_conversation"):
            key = _source_key(chunk)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "conversation_id": chunk.get("conversation_id", ""),
                    "message_count": chunk.get("message_count", 1),
                    "first_received_at": chunk.get("first_received_at", chunk.get("received_at", "")),
                    "last_received_at": chunk.get("last_received_at", chunk.get("received_at", "")),
                    "participants": chunk.get("participants", []),
                    "subject": chunk.get("subject", "Unknown"),
                    "filename": chunk.get("from_address", "email"),
                    "score": score,
                    "chunk_index": i + 1,
                    "source_type": source_type,
                    "context_sent": chunk_context,
                    "matched_message_id": chunk.get("matched_message_id", ""),
                    "matched_message_subject": chunk.get("matched_message_subject", ""),
                    "matched_message_received_at": chunk.get("matched_message_received_at", ""),
                    "matched_message_content": _extract_matched_message_content(chunk),
                })

        elif source_type == "meeting":
            key = chunk.get("meeting_id", chunk.get("subject", ""))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "subject": chunk.get("subject", "Unknown Meeting"),
                    "filename": "meeting",
                    "score": score,
                    "chunk_index": i + 1,
                    "source_type": source_type,
                    "context_sent": chunk_context,
                })
    return sources


def _retrieve_for_query(query: str, bm25_index, wv_client) -> tuple[list[dict], str]:
    """Run Weaviate retrieval and return (retrieved_chunks, retriever_label)."""
    return retrieve_weaviate(query, wv_client, bm25_index=bm25_index), "weaviate"


async def answer_question(
    message: str,
    bearer_token: str | None = None,
    bm25_index=None,
    wv_client=None,
) -> ChatResponse:
    """Generate a chatbot response (non-streaming). Used by POST /chatbot/query.

    Returns a full ChatResponse with sources, timings, follow-up questions,
    and is_greeting — matching the standalone chatbot's /chat contract.

    `bearer_token` is accepted for backward compatibility with the
    existing request contract but is not used - the Graph token (if ever
    needed here) is resolved server-side from the session, not from the
    client, mirroring the rest of this backend's auth model.
    """
    settings = get_settings()
    query = message.strip()

    timings: dict = {}
    t_total = time.perf_counter()

    if is_greeting(query):
        total_ms = (time.perf_counter() - t_total) * 1000
        timings["total"] = round(total_ms, 1)
        logger.info("Greeting detected: '%s' | %.1fms (no LLM call)", sanitize_log_input(query), timings["total"])
        save_message(query, GREETING_RESPONSE)
        await save_chat_message(query, GREETING_RESPONSE)
        return ChatResponse(
            answer=GREETING_RESPONSE,
            sources=[],
            is_greeting=True,
            latency_ms=round(total_ms, 1),
            timings=TimingInfo(total=round(total_ms, 1)),
            follow_up_questions=[],
            tool_used="weaviate",
        )

    # ── Cache-first: same question answered before → return from Redis ──
    cached = get_cached_answer(query, mode="email")
    if cached:
        total_ms = (time.perf_counter() - t_total) * 1000
        logger.info("QA cache HIT (email): '%s' | %.1fms", sanitize_log_input(query), total_ms)
        save_message(query, cached["answer"])
        await save_chat_message(query, cached["answer"])
        return ChatResponse(
            answer=cached["answer"],
            sources=[SourceInfo(**s) for s in cached.get("sources", [])],
            is_greeting=False,
            latency_ms=round(total_ms, 1),
            timings=TimingInfo(total=round(total_ms, 1)),
            follow_up_questions=[],
            tool_used=cached.get("tool_used", "weaviate"),
        )

    logger.info("Processing query: '%s'", sanitize_log_input(query))

    t_retrieve = time.perf_counter()
    retrieved, retriever_label = _retrieve_for_query(query, bm25_index, wv_client)
    timings["retrieval"] = round((time.perf_counter() - t_retrieve) * 1000, 1)

    t_context = time.perf_counter()
    context = format_context(retrieved, query)
    timings["context_format"] = round((time.perf_counter() - t_context) * 1000, 1)

    # ── Pre-LLM relevance gate ────────────────────────────────
    # If retrieved context is clearly irrelevant, skip the LLM entirely
    # and return "Not found" — saves ~80s of LLM time and prevents hallucination.
    if not _is_context_relevant(retrieved):
        total_ms = (time.perf_counter() - t_total) * 1000
        timings["total"] = round(total_ms, 1)
        logger.info(
            "Query short-circuited (irrelevant context): retriever=%s retrieval=%.0fms total=%.0fms",
            retriever_label, timings["retrieval"], timings["total"],
        )
        save_message(query, NOT_FOUND_RESPONSE)
        await save_chat_message(query, NOT_FOUND_RESPONSE)
        sources = _get_sources(retrieved)
        return ChatResponse(
            answer=NOT_FOUND_RESPONSE,
            sources=[SourceInfo(**s) for s in sources],
            is_greeting=False,
            latency_ms=round(total_ms, 1),
            timings=TimingInfo(
                retrieval=timings.get("retrieval"),
                context_format=timings.get("context_format"),
                total=timings["total"],
            ),
            follow_up_questions=[],
            tool_used=retriever_label,
        )

    # Rebuild the prompt through the size guard (shrinks + rebuilds once if
    # it exceeds the char ceiling) so we never depend on Ollama truncation.
    context, user_message = _build_prompt_with_guard(retrieved, query)

    client = ollama.Client(host=settings.ollama_host)
    t_llm = time.perf_counter()
    response = client.chat(
        model=settings.chatbot_llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={
            "num_predict": settings.chatbot_max_tokens,
            "temperature": 0.1,
            "num_ctx": settings.chatbot_llm_num_ctx,
            "num_thread": 8,
            "repeat_penalty": 1.1,
        },
        keep_alive=-1,
    )
    timings["llm_generation"] = round((time.perf_counter() - t_llm) * 1000, 1)

    _log_llm_metrics(
        response,
        prompt_chars=len(user_message),
        retrieval_ms=timings.get("retrieval", 0.0),
    )

    answer_text = response["message"]["content"]

    # Debug dump of the final prompt + RAW answer (before any post-processing).
    _debug_dump_prompt(user_message, answer_text)

    # ── Post-LLM hallucination check ─────────────────────────
    # If the LLM ignored the system prompt and hallucinated a capability
    # description, replace it with the safe "Not found" response.
    if _is_hallucinated_response(answer_text):
        answer_text = NOT_FOUND_RESPONSE

    total_ms = (time.perf_counter() - t_total) * 1000
    timings["total"] = round(total_ms, 1)

    logger.info(
        "Query answered: retriever=%s retrieval=%.0fms llm=%.0fms total=%.0fms",
        retriever_label, timings["retrieval"], timings["llm_generation"], timings["total"],
    )

    save_message(query, answer_text)
    await save_chat_message(query, answer_text)

    sources = _get_sources(retrieved)

    set_cached_answer(query, {
        "answer": answer_text,
        "sources": sources,
        "follow_up_questions": [],
        "tool_used": retriever_label,
    }, mode="email")

    return ChatResponse(
        answer=answer_text,
        sources=[SourceInfo(**s) for s in sources],
        is_greeting=False,
        latency_ms=round(total_ms, 1),
        timings=TimingInfo(
            retrieval=timings.get("retrieval"),
            context_format=timings.get("context_format"),
            llm_generation=timings.get("llm_generation"),
            total=timings["total"],
        ),
        follow_up_questions=[],
        tool_used=retriever_label,
    )


def generate_response_stream(
    question: str,
    bm25_index=None,
    wv_client=None,
) -> Generator[str, None, None]:
    """Stream chatbot response as Server-Sent Events (SSE).

    Event types: step, sources, token, greeting, done, error — matching
    the existing frontend contract (see chatbot/routes.py docstring).
    """
    settings = get_settings()
    query = question.strip()

    timings: dict = {}
    t_total = time.perf_counter()

    # ── Handle greetings (instant, no LLM) ───────────────────
    if is_greeting(query):
        total_ms = (time.perf_counter() - t_total) * 1000
        logger.info("Greeting detected: '%s' | %.1fms (no LLM call)", sanitize_log_input(query), total_ms)
        _persist_message(query, GREETING_RESPONSE)
        yield f"data: {json.dumps({'type': 'greeting', 'content': GREETING_RESPONSE})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'timings': {'total': round(total_ms, 1)}, 'follow_up_questions': []})}\n\n"
        return

    # ── Cache-first: same question answered before → stream from Redis ──
    cached = get_cached_answer(query, mode="email")
    if cached:
        total_ms = (time.perf_counter() - t_total) * 1000
        logger.info("QA cache HIT (email, stream): '%s' | %.1fms", sanitize_log_input(query), total_ms)
        _persist_message(query, cached["answer"])
        yield f"data: {json.dumps({'type': 'step', 'step': 'cache', 'message': 'Returning cached answer...'})}\n\n"
        yield f"data: {json.dumps({'type': 'sources', 'sources': cached.get('sources', []), 'all_candidates': [], 'retriever': cached.get('tool_used', 'weaviate'), 'timings': {}})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'content': cached['answer']})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'timings': {'total': round(total_ms, 1)}, 'follow_up_questions': []})}\n\n"
        return

    logger.info("Processing query (streaming): '%s'", sanitize_log_input(query))

    yield f"data: {json.dumps({'type': 'step', 'step': 'analyzing', 'message': 'Analyzing your query...'})}\n\n"

    # ── Step 1: Retrieve ──────────────────────────────────────
    t_retrieve = time.perf_counter()
    retrieved, retriever_label = _retrieve_for_query(query, bm25_index, wv_client)
    retrieve_ms = (time.perf_counter() - t_retrieve) * 1000
    timings["retrieval"] = round(retrieve_ms, 1)

    # ── Step 2: Format context ───────────────────────────────
    yield f"data: {json.dumps({'type': 'step', 'step': 'searching', 'message': f'Searching relevant email conversations... ({retrieve_ms:.0f}ms)'})}\n\n"

    t_context = time.perf_counter()
    context = format_context(retrieved, query)
    context_ms = (time.perf_counter() - t_context) * 1000
    timings["context_format"] = round(context_ms, 1)

    # ── Send sources + retrieval timings immediately ─────────
    sources = _get_sources(retrieved)

    all_candidates = []
    if retrieved and "_all_candidates" in retrieved[0]:
        for i, c in enumerate(retrieved[0]["_all_candidates"]):
            sent_to_llm = any(_source_key(c) == _source_key(r) for r in retrieved)
            all_candidates.append({
                "rank": i + 1,
                "conversation_id": c.get("conversation_id", ""),
                "message_id": c.get("message_id", ""),
                "subject": c.get("thread_subject", c.get("subject", "Unknown")),
                "matched_message_subject": c.get("subject", ""),
                "message_count": c.get("message_count", 0),
                "first_received_at": c.get("first_received_at", ""),
                "last_received_at": c.get("last_received_at", ""),
                "matched_message_received_at": c.get("received_at", ""),
                "participants": c.get("participants", []),
                "score": round(c.get("combined_score", c.get("score", 0)), 4),
                "semantic_score": round(c.get("semantic_score_norm", c.get("score", 0)), 4),
                "bm25_score": round(c.get("bm25_score_norm", c.get("bm25_score", 0)), 4),
                "sent_to_llm": sent_to_llm,
                "source_type": c.get("source_type", "email_message"),
            })

    yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'all_candidates': all_candidates, 'retriever': retriever_label, 'timings': timings})}\n\n"

    # ── Pre-LLM relevance gate ────────────────────────────────
    # If retrieved context is clearly irrelevant, skip the LLM entirely
    # and return "Not found" — saves ~80s of LLM time and prevents hallucination.
    if not _is_context_relevant(retrieved):
        total_ms = (time.perf_counter() - t_total) * 1000
        timings["total"] = round(total_ms, 1)
        logger.info(
            "Streaming query short-circuited (irrelevant context): retrieval=%.0fms total=%.0fms",
            timings["retrieval"], timings["total"],
        )
        _persist_message(query, NOT_FOUND_RESPONSE)
        yield f"data: {json.dumps({'type': 'token', 'content': NOT_FOUND_RESPONSE})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'timings': timings, 'follow_up_questions': []})}\n\n"
        return

    # ── Step 3: Stream LLM generation ────────────────────────
    model = settings.chatbot_llm_model
    yield f"data: {json.dumps({'type': 'step', 'step': 'generating', 'message': f'Sending to LLM ({model})...'})}\n\n"

    # Rebuild the prompt through the size guard (shrinks + rebuilds once if it
    # exceeds the char ceiling) so we never depend on Ollama truncation.
    context, user_message = _build_prompt_with_guard(retrieved, query)

    prompt_tokens = len(user_message) // 4
    logger.info(
        "LLM streaming start: model=%s, ~%d prompt tokens, max_output=%d",
        model, prompt_tokens, settings.chatbot_max_tokens,
    )

    client = ollama.Client(host=settings.ollama_host)
    t_llm = time.perf_counter()
    t_first_token = None
    token_count = 0
    full_response = ""

    # ── Heartbeat-aware LLM streaming ────────────────────────
    # Ollama can take 60-120s+ to produce the first token (especially
    # on CPU). During that silence, reverse proxies / load balancers
    # kill the idle SSE connection. We run the Ollama stream in a
    # background thread and yield heartbeat events every 10s while
    # waiting, so the connection always has data flowing through it.
    HEARTBEAT_INTERVAL = 10  # seconds between keepalive events
    _SENTINEL = object()     # marks end of the Ollama stream

    token_queue: queue.Queue = queue.Queue()
    llm_error: list = []     # captured exception from the thread
    llm_final: list = []     # captured final chunk (carries eval counts)

    def _ollama_worker():
        """Run blocking Ollama streaming in a background thread,
        pushing each token (or error/sentinel) into token_queue."""
        try:
            stream = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                options={
                    "num_predict": settings.chatbot_max_tokens,
                    "temperature": 0.1,
                    "num_ctx": settings.chatbot_llm_num_ctx,
                    "num_thread": 8,
                    "repeat_penalty": 1.1,
                },
                keep_alive=-1,
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    token_queue.put(token)
                # The terminal chunk carries prompt_eval_count/eval_count/
                # *_duration once Ollama finishes; capture it for metrics.
                is_done = chunk.get("done") if isinstance(chunk, dict) else getattr(chunk, "done", False)
                if is_done:
                    llm_final.append(chunk)
        except Exception as e:
            llm_error.append(e)
        finally:
            token_queue.put(_SENTINEL)

    worker = threading.Thread(target=_ollama_worker, daemon=True)
    worker.start()

    try:
        while True:
            try:
                item = token_queue.get(timeout=HEARTBEAT_INTERVAL)
            except queue.Empty:
                # No token yet — send a heartbeat to keep the SSE
                # connection alive through proxies / load balancers.
                elapsed = time.perf_counter() - t_llm
                yield f"data: {json.dumps({'type': 'heartbeat', 'elapsed_s': round(elapsed, 1)})}\n\n"
                continue

            if item is _SENTINEL:
                break

            # Real token from Ollama
            if t_first_token is None:
                t_first_token = time.perf_counter()
                first_token_ms = (t_first_token - t_llm) * 1000
                logger.info("First token arrived: %.0fms", first_token_ms)
                yield f"data: {json.dumps({'type': 'step', 'step': 'writing', 'message': f'Generating response... (first token in {first_token_ms:.0f}ms)'})}\n\n"

            token_count += 1
            full_response += item
            yield f"data: {json.dumps({'type': 'token', 'content': item})}\n\n"

    except Exception as e:
        logger.error("LLM streaming error: %s", e)
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        return

    # Surface any error from the background thread.
    if llm_error:
        logger.error("LLM streaming error: %s", llm_error[0])
        yield f"data: {json.dumps({'type': 'error', 'content': str(llm_error[0])})}\n\n"
        return

    # ── Final timing summary ─────────────────────────────────
    llm_ms = (time.perf_counter() - t_llm) * 1000
    total_ms = (time.perf_counter() - t_total) * 1000
    tokens_per_sec = (token_count / (llm_ms / 1000)) if llm_ms > 0 else 0

    timings["llm_generation"] = round(llm_ms, 1)
    timings["first_token"] = round((t_first_token - t_llm) * 1000, 1) if t_first_token else 0
    timings["total"] = round(total_ms, 1)
    timings["tokens"] = token_count
    timings["tokens_per_sec"] = round(tokens_per_sec, 1)

    # Debug dump of the final prompt + RAW streamed answer (before the
    # hallucination regex / NOT_FOUND replacement below).
    _debug_dump_prompt(user_message, full_response)

    # ── Post-LLM hallucination check ─────────────────────────
    # If the LLM streamed a hallucinated response, send a correction
    # event so the frontend can replace the displayed text.
    if _is_hallucinated_response(full_response):
        full_response = NOT_FOUND_RESPONSE
        yield f"data: {json.dumps({'type': 'replace', 'content': NOT_FOUND_RESPONSE})}\n\n"

    logger.info(
        "Streaming query timing: retrieval=%.1fms context=%.1fms first_token=%.1fms "
        "llm_total=%.1fms (%d tokens, %.0f tok/s) total=%.1fms sources=%s",
        timings["retrieval"], timings["context_format"], timings["first_token"],
        timings["llm_generation"], token_count, tokens_per_sec, timings["total"],
        ", ".join(s["subject"][:30] for s in sources),
    )

    # ── Per-request LLM metrics (from Ollama's terminal chunk) ──
    if llm_final:
        _log_llm_metrics(
            llm_final[0],
            prompt_chars=len(user_message),
            retrieval_ms=timings.get("retrieval", 0.0),
        )
    else:
        # No terminal chunk captured — fall back to our own token count so the
        # metrics line is still emitted (prompt tokens estimated as chars/4).
        logger.info(
            "LLM metrics: prompt_chars=%d prompt_eval_count=%d (est) eval_count=%d "
            "eval_duration=%.2fs tokens_per_sec=%.1f retrieval=%.1fms",
            len(user_message), len(user_message) // 4, token_count,
            llm_ms / 1000, tokens_per_sec, timings.get("retrieval", 0.0),
        )
        if (len(user_message) // 4) > PROMPT_EVAL_WARN_TOKENS:
            logger.warning(
                "estimated prompt tokens %d exceed %d - reduce CHATBOT_CONTEXT_MAX_CHARS",
                len(user_message) // 4, PROMPT_EVAL_WARN_TOKENS,
            )

    _persist_message(query, full_response)

    set_cached_answer(query, {
        "answer": full_response,
        "sources": sources,
        "follow_up_questions": [],
        "tool_used": retriever_label,
    }, mode="email")

    yield f"data: {json.dumps({'type': 'done', 'timings': timings, 'follow_up_questions': []})}\n\n"
