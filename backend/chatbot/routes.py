"""
chatbot/routes.py
====================
FastAPI routes for the RAG-based Chatbot agent.

- `POST /chatbot/query` - answer a natural-language question using
  RAG over indexed email content (preserves existing frontend API contract).
- `POST /chatbot/query/stream` - same, but streamed as Server-Sent Events.
- `POST /chatbot/index` - rebuild the vector index from Postgres.

Retrieval backend: Weaviate hybrid vector + BM25 search via the shared
Weaviate deployment (core.weaviate.client), using the chatbot's own
dedicated collection (chatbot.weaviate_store). A local BM25 index is
built from the same message set for re-ranking.

Raw email rows are sourced from the shared PostgreSQL pool
(chatbot.postgres_store) - there is no CSV/.eml fallback for the
chatbot's data source.
"""

import json
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from auth.dependencies import get_bearer_token
from chatbot.bm25_index import build_bm25
from chatbot.postgres_store import fetch_chat_history
from chatbot.schema import (
    ChatHistoryEntry,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SourceInfo,
    TimingInfo,
)
from chatbot.redis_history import (
    clear_history as clear_redis_history,
    get_history as get_redis_history,
    save_message,
)
from chatbot.redis_qa_cache import clear_qa_cache, get_cached_answer, set_cached_answer
from chatbot.service import GREETING_RESPONSE, answer_question, generate_response_stream, is_greeting
from core.logging_config import get_agent_logger, sanitize_log_value
from core.utils import sanitize_log_input

logger = get_agent_logger("chatbot")

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

# The only error text a document stream ever sends to the client (CWE-209):
# whatever the pipeline reports, the details stay in the server log.
_DOC_STREAM_ERROR_MESSAGE = "Document search failed. Please try again later."

# ── Global chatbot state ──────────────────────────────────────
# Weaviate mode uses "wv_client"; "bm25" is the local keyword index
# built from the same message set for re-ranking.
_chatbot_state = {
    "bm25": None,
    "wv_client": None,
    "ready": False,
    "request_count": 0,
    "documents_available": False,
}


def get_chatbot_state() -> dict:
    """Return the module-level chatbot state dict."""
    return _chatbot_state


async def initialize_index():
    """Initialise the chatbot's retrieval backend at startup.

    Never raises - a failure here leaves the chatbot not-ready
    (POST /chatbot/index can build it manually later) rather than
    blocking the whole app's startup.
    """
    from core.config import get_settings
    if get_settings().use_faiss:
        await _initialize_faiss()
    else:
        await _initialize_weaviate()
    _initialize_documents()


async def _initialize_faiss():
    """Build the FAISS index + local BM25 from PostgreSQL emails."""
    from chatbot._vec_backend import vec_store
    from chatbot.indexer import build_weaviate_index, weaviate_collection_count
    from chatbot.retriever import fetch_all_chunks_weaviate

    logger.info("Initializing chatbot retrieval backend: FAISS")
    try:
        vs = vec_store()
        vs.ensure_collection()

        count = vs.collection_count()
        if count == 0:
            logger.info("Chatbot FAISS index is empty - ingesting emails...")
            t0 = time.perf_counter()
            inserted = await build_weaviate_index(None)  # client=None for FAISS
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("Ingestion complete: %d objects in %.0fms", inserted, elapsed)
        else:
            logger.info("Chatbot FAISS index ready - %d objects found.", count)

        count = vs.collection_count()
        logger.info("Building local BM25 index from FAISS chunks...")
        chunks = fetch_all_chunks_weaviate(None)
        _chatbot_state["bm25"] = build_bm25(chunks)
        logger.info("BM25 index ready - %d messages indexed.", len(chunks))

        _chatbot_state["ready"] = True
    except Exception as e:
        logger.error("Chatbot FAISS startup failed: %s", e)
        logger.info("Chatbot is NOT ready. Use POST /chatbot/index to retry manually.")


async def _initialize_weaviate():
    """Connect to Weaviate, ingest if the collection is empty, build BM25."""
    from chatbot.indexer import build_weaviate_index, weaviate_collection_count
    from chatbot.retriever import fetch_all_chunks_weaviate
    from core.weaviate.client import get_client

    logger.info("Initializing chatbot retrieval backend: Weaviate")
    try:
        client = get_client()
        _chatbot_state["wv_client"] = client

        count = weaviate_collection_count(client)
        if count == 0:
            logger.info("Chatbot Weaviate collection is empty - ingesting emails...")
            t0 = time.perf_counter()
            inserted = await build_weaviate_index(client)
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("Ingestion complete: %d objects in %.0fms", inserted, elapsed)
        else:
            logger.info("Chatbot Weaviate collection ready - %d objects found.", count)

        # Self-heal: sync Weaviate with Postgres (add missing, remove stale).
        try:
            from chatbot.weaviate_sync import ensure_weaviate_consistent
            sync_result = await ensure_weaviate_consistent(client)
            logger.info(
                "Chatbot Weaviate sync: added=%d removed=%d already_ok=%d",
                sync_result.get("added", 0),
                sync_result.get("removed", 0),
                sync_result.get("already_ok", 0),
            )
        except Exception as e:
            logger.warning("Chatbot Weaviate sync failed (non-fatal): %s", e)

        count = weaviate_collection_count(client)
        logger.info("Building local BM25 index from Weaviate messages...")
        wv_chunks = fetch_all_chunks_weaviate(client)
        if not wv_chunks and count > 0:
            raise RuntimeError(
                f"Weaviate message validation failed: {count} stored but 0 usable chunks fetched"
            )
        if len(wv_chunks) != count:
            logger.warning(
                "Weaviate aggregate count (%d) differs from fetched chunks (%d) "
                "— likely eventual consistency lag after ingestion; proceeding with fetched data.",
                count, len(wv_chunks),
            )
        _chatbot_state["bm25"] = build_bm25(wv_chunks)
        logger.info("BM25 index ready - %d messages indexed.", len(wv_chunks))

        _chatbot_state["ready"] = True
    except Exception as e:
        logger.error("Chatbot Weaviate startup failed: %s", e)
        logger.info("Chatbot is NOT ready. Use POST /chatbot/index to retry manually.")


def _initialize_documents():
    """Check if the MIS document Q&A module is functional (library.db exists and readable).

    Never raises - a failure here leaves document mode unavailable but
    email mode still works normally.
    """
    # The real availability check (PostgreSQL pool reachable) runs on first query, since
    # documents.service.is_available() is now async and this function is called from the
    # sync startup path. Flip to True optimistically; a failing query returns 503 cleanly.
    _chatbot_state["documents_available"] = True
    logger.info("Document Q&A module marked available (verified lazily on first query).")


@router.post("/query", response_model=None)
async def query(
    request: ChatRequest,
    bearer_token: str | None = Depends(get_bearer_token),
):
    """
    Answer a user's question using RAG.

    When `mode` is "email" (default), uses the Weaviate email pipeline.
    When `mode` is "document", uses the MIS document Q&A pipeline and
    returns a plain JSON ChatResponse (use POST /chatbot/query/stream
    instead for an SSE response).

    Simple greetings ("hi", "hello", etc.) are answered instantly with
    a canned response, regardless of mode - no LLM call needed.
    """
    message = request.message.strip()
    if is_greeting(message):
        _chatbot_state["request_count"] += 1
        return ChatResponse(
            answer=GREETING_RESPONSE,
            sources=[],
            is_greeting=True,
            latency_ms=0.0,
            timings=TimingInfo(total=0.0),
            follow_up_questions=[],
            tool_used=None,
        )

    if request.mode == "document":
        if not _chatbot_state["documents_available"]:
            raise HTTPException(
                status_code=503,
                detail="Document Q&A module is not available.",
            )
        _chatbot_state["request_count"] += 1
        from documents.service import answer_once
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # Cache-first: same question answered before → return from Redis.
        t_cache = time.perf_counter()
        cached = get_cached_answer(message, mode="document")
        if cached:
            cache_ms = round((time.perf_counter() - t_cache) * 1000, 1)
            logger.info("QA cache HIT (document): %.1fms", cache_ms)
            save_message(message, cached["answer"])
            return ChatResponse(
                answer=cached["answer"],
                sources=[SourceInfo(**s) for s in cached.get("sources", [])],
                is_greeting=False,
                latency_ms=cache_ms,
                timings=TimingInfo(total=cache_ms),
                follow_up_questions=[],
                tool_used="documents",
            )

        result = await answer_once(query=message)
        save_message(message, result["answer"])
        set_cached_answer(
            message,
            {"answer": result["answer"], "sources": result["sources"], "tool_used": "documents"},
            mode="document",
        )
        return ChatResponse(
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result["sources"]],
            is_greeting=False,
            latency_ms=result["timings"]["total"],
            timings=TimingInfo(total=result["timings"]["total"]),
            follow_up_questions=[],
            tool_used="documents",
        )

    # Default: email mode
    if not _chatbot_state["ready"]:
        raise HTTPException(
            status_code=503,
            detail="Index not ready. Please call POST /chatbot/index first.",
        )

    _chatbot_state["request_count"] += 1

    return await answer_question(
        request.message,
        bearer_token=bearer_token,
        bm25_index=_chatbot_state["bm25"],
        wv_client=_chatbot_state["wv_client"],
    )


@router.post("/query/stream")
async def query_stream(
    request: ChatRequest,
    bearer_token: str | None = Depends(get_bearer_token),
):
    """
    Stream a chatbot response as Server-Sent Events (SSE).

    Same input as /query but returns an event stream with types:
      - step:     loading/progress indicators
      - sources:  email source citations (or document source citations)
      - token:    individual response tokens
      - greeting: instant greeting response
      - done:     final timing summary
      - error:    error message

    When `mode` is "document", streams answers from the MIS document
    Q&A pipeline instead of the email pipeline.
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    _chatbot_state["request_count"] += 1

    # Simple greetings ("hi", "hello", etc.) get an instant canned
    # response regardless of mode - no LLM call needed.
    if is_greeting(message):
        import json as _json

        def _greeting_stream():
            yield f"data: {_json.dumps({'type': 'greeting', 'content': GREETING_RESPONSE})}\n\n"
            yield f"data: {_json.dumps({'type': 'done', 'timings': {'total': 0.0}, 'follow_up_questions': []})}\n\n"

        return StreamingResponse(
            _greeting_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if request.mode == "document":
        if not _chatbot_state["documents_available"]:
            raise HTTPException(
                status_code=503,
                detail="Document Q&A module is not available.",
            )

        item_message = sanitize_log_value(message)
        sanitized_item_id = sanitize_log_input(item_message)
        logger.info("New document streaming query: '%s'", sanitized_item_id)

        from documents.service import stream_answer

        async def _doc_stream_with_history():
            """Serve from the QA cache if possible; otherwise wrap stream_answer
            to capture the final answer, persist history, and fill the cache."""
            # Cache-first: same question answered before → stream from Redis.
            t_cache = time.perf_counter()
            cached = get_cached_answer(message, mode="document")
            if cached:
                cache_ms = round((time.perf_counter() - t_cache) * 1000, 1)
                logger.info("QA cache HIT (document, stream): %.1fms", cache_ms)
                save_message(message, cached["answer"])
                yield f"data: {json.dumps({'type': 'step', 'step': 'cache', 'message': 'Returning cached answer...'})}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'sources': cached.get('sources', []), 'source_mode': 'documents'})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': cached['answer']})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'timings': {'total': cache_ms}, 'follow_up_questions': [], 'source_mode': 'documents'})}\n\n"
                return

            answer_text = ""
            doc_sources: list = []
            had_error = False
            async for chunk in stream_answer(query=message):
                # Capture answer tokens + sources for history / cache persistence.
                try:
                    if chunk.startswith("data: "):
                        payload = json.loads(chunk[6:].strip())
                        ptype = payload.get("type")
                        if ptype == "token":
                            answer_text += payload.get("content", "")
                        elif ptype == "sources":
                            doc_sources = payload.get("sources", [])
                        elif ptype == "error":
                            had_error = True
                            # Never forward upstream error text: re-emit a fixed message.
                            chunk = (
                                "data: "
                                + json.dumps({"type": "error", "content": _DOC_STREAM_ERROR_MESSAGE})
                                + "\n\n"
                            )
                except (json.JSONDecodeError, ValueError):
                    pass
                yield chunk
            # Stream finished — persist the Q&A pair to Redis.
            if answer_text:
                save_message(message, answer_text)
                if not had_error:
                    set_cached_answer(
                        message,
                        {"answer": answer_text, "sources": doc_sources, "tool_used": "documents"},
                        mode="document",
                    )

        # stream_answer is an async generator; StreamingResponse handles those natively.
        return StreamingResponse(
            _doc_stream_with_history(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Default: email mode
    if not _chatbot_state["ready"]:
        raise HTTPException(
            status_code=503,
            detail="Index not ready. Please call POST /chatbot/index first.",
        )

    item_message = sanitize_log_value(message)
    sanitized_item_id = sanitize_log_input(item_message)
    logger.info("New streaming query: '%s'", sanitized_item_id)

    def event_generator():
        yield from generate_response_stream(
            question=message,
            bm25_index=_chatbot_state["bm25"],
            wv_client=_chatbot_state["wv_client"],
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history", response_model=ChatHistoryResponse)
async def history(
    limit: int = Query(default=50, ge=1, le=100),
) -> ChatHistoryResponse:
    """
    Return the most recent persisted Q&A pairs from Redis, oldest first.

    Backed by the Redis `chatbot:history` list (chatbot/redis_history.py).
    Redis is cleared on each pipeline run (run_pipeline.py), so history
    reflects only the current cycle. Returns an empty list - never an
    error - if Redis is unreachable or unconfigured.
    """
    entries = get_redis_history(limit=limit)
    return ChatHistoryResponse(
        history=[ChatHistoryEntry(**e) for e in entries]
    )


@router.delete("/history")
async def clear_chat_history():
    """
    Delete the persisted chat history (Redis `chatbot:history`).

    Used by the chat panel's refresh button. Only the history is removed;
    the QA answer cache is intentionally left alone. Returns 503 if Redis
    is configured but the delete failed, so the UI can keep the chat.
    """
    if not clear_redis_history():
        raise HTTPException(status_code=503, detail="Could not clear chat history")
    return {"cleared": True}


@router.post("/index")
async def rebuild_index():
    """Rebuild the chatbot's Weaviate retrieval index from Postgres."""
    logger.info("Index rebuild requested...")
    t0 = time.perf_counter()

    try:
        from chatbot.indexer import build_weaviate_index, weaviate_collection_count
        from chatbot.retriever import fetch_all_chunks_weaviate
        from core.weaviate.client import get_client

        client = _chatbot_state["wv_client"] or get_client()
        _chatbot_state["wv_client"] = client
        _chatbot_state["ready"] = False

        inserted = await build_weaviate_index(client, recreate=True)
        clear_qa_cache()  # index changed → previously cached answers may be stale

        wv_chunks = fetch_all_chunks_weaviate(client)
        if len(wv_chunks) != inserted:
            raise RuntimeError(
                f"Message rebuild validation failed: {inserted} inserted, {len(wv_chunks)} usable"
            )
        _chatbot_state["bm25"] = build_bm25(wv_chunks)
        _chatbot_state["ready"] = True

        total = weaviate_collection_count(client)
        build_ms = (time.perf_counter() - t0) * 1000
        logger.info("Weaviate message index rebuilt in %.0fms: %d objects", build_ms, total)
        return {
            "message": "Weaviate chatbot index rebuilt successfully from Postgres.",
            "total_chunks": total,
            "total_vectors": total,
            "build_time_ms": round(build_ms, 1),
        }
    except Exception as e:
        _chatbot_state["ready"] = False
        # Log the full error server-side; return a generic message to the
        # client so internal details / stack traces are not exposed (CWE-209).
        logger.error("Index rebuild failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to build index.")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check if the chatbot backend and index are ready.

    Reports Weaviate collection object count and total requests served,
    matching the standalone chatbot backend's /health contract.
    """
    from chatbot.indexer import weaviate_collection_count

    client = _chatbot_state["wv_client"]
    count = weaviate_collection_count(client) if client else 0

    return HealthResponse(
        status="ok" if _chatbot_state["ready"] else "not_ready",
        index_ready=_chatbot_state["ready"],
        total_chunks=count,
        total_vectors=count,
        requests_served=_chatbot_state["request_count"],
    )


# ── Document Q&A endpoints ────────────────────────────────────


@router.get("/documents")
async def list_documents():
    """Return the list of ingested MIS documents available for Q&A."""
    if not _chatbot_state["documents_available"]:
        raise HTTPException(
            status_code=503,
            detail="Document Q&A module is not available.",
        )
    from documents.service import get_document_names
    return {"documents": await get_document_names()}


@router.get("/documents/health")
async def documents_health():
    """Check if the document Q&A module is available."""
    return {
        "available": _chatbot_state["documents_available"],
        "mode": "document",
    }


@router.get("/modes")
async def available_modes():
    """Return which chatbot modes are currently available."""
    return {
        "modes": {
            "email": _chatbot_state["ready"],
            "document": _chatbot_state["documents_available"],
        }
    }
