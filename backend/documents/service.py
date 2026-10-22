"""Adapter: wraps the PDF bot's async ``embedqa.answer_events()`` or
``retrieve.answer_events()`` and translates its output into the SSE event
format the email chatbot frontend already understands.

Pipeline selection
------------------
Set ``DOC_PIPELINE`` in ``.env``:

* ``full``   (default) — ``embedqa.py``.  Extracts metadata filters (company,
  year, quarter), builds a scoped facts_view, hybrid search (vector + keyword),
  tool-calling loop with run_metric + calculate, number verification.
  More accurate, slower (~30-60 s on CPU).

* ``simple`` — ``retrieve.py``.  Embedding + keyword hybrid search only, no
  metadata filters, no facts_view, no run_metric.  Just top-K passages fed
  straight to the chat model with a calculator tool.  Faster (~10-20 s on CPU),
  good enough for narrative / "what does the report say" questions.

The PDF bot now runs on asyncpg + Weaviate (not SQLite); every function here is async.
"""
import json
import os
import time
import traceback

from . import embedqa
from . import retrieve as retrieve_pipeline
from . import store
from . import config
from core.logging_config import get_agent_logger
from core.utils import sanitize_log_input

logger = get_agent_logger("chatbot")


def _use_simple() -> bool:
    """True when the operator picked the simple (retrieve-only) pipeline."""
    return os.getenv("DOC_PIPELINE", "full").strip().lower() == "simple"


def _passage_to_source(p: dict, index: int) -> dict:
    """Reshape a PDF passage dict into the source-card shape the frontend renders."""
    return {
        "subject": f"{p.get('file_name', 'Document')} — p.{p.get('page', '?')}",
        "filename": p.get("file_name", ""),
        "score": 1.0 - (index * 0.05),
        "chunk_index": index + 1,
        "source_type": "document",
        "context_sent": embedqa.passage_text(p)[:2000] if p else "",
        "conversation_id": None,
        "message_count": None,
        "first_received_at": None,
        "last_received_at": None,
        "participants": None,
        "matched_message_id": None,
        "matched_message_subject": f"Page {p.get('page', '?')}",
        "matched_message_received_at": None,
    }


async def stream_answer(query: str, history: list[dict] | None = None,
                        doc_ids: list[str] | None = None):
    """Async generator yielding SSE-formatted strings.

    FastAPI's ``StreamingResponse`` handles async generators natively — callers simply pass
    the result of this function to ``StreamingResponse(..., media_type='text/event-stream')``.
    """
    t0 = time.perf_counter()
    sources: list[dict] = []
    tool_trace: list[dict] = []
    event_count = 0

    logger.info("[DOC-SVC] stream_answer() START  query='%s'  history=%d  doc_ids=%s",
                sanitize_log_input(query[:120]), len(history or []), doc_ids)

    try:
        simple = _use_simple()
        pipeline_mod = retrieve_pipeline if simple else embedqa
        pipeline_label = "retrieve (simple)" if simple else "embedqa (full)"

        logger.info("[DOC-SVC] Calling %s.answer_events()  model=%s  reasoning=False",
                    pipeline_label, config.CHAT_MODEL)

        events_kwargs: dict = dict(
            question=query,
            history=history,
            doc_ids=doc_ids,
            model=config.CHAT_MODEL,
            reasoning=False,
        )
        if not simple:
            events_kwargs["style"] = "full"

        async for kind, val in pipeline_mod.answer_events(**events_kwargs):
            event_count += 1
            elapsed = (time.perf_counter() - t0) * 1000

            if kind == "stage":
                logger.info("[DOC-SVC] event #%d [stage] %.0fms: %s", event_count, elapsed, sanitize_log_input(str(val)))
                yield f"data: {json.dumps({'type': 'step', 'step': 'searching', 'message': val})}\n\n"

            elif kind == "passages":
                sources = [_passage_to_source(p, i) for i, p in enumerate(val)]
                logger.info("[DOC-SVC] event #%d [passages] %.0fms: %d passages, %d sources",
                            event_count, elapsed, len(val), len(sources))
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'source_mode': 'documents'})}\n\n"

            elif kind == "series":
                logger.info("[DOC-SVC] event #%d [series] %.0fms: %d series", event_count, elapsed,
                            len(val) if val else 0)

            elif kind == "tools":
                tool_trace = val
                logger.info("[DOC-SVC] event #%d [tools] %.0fms: %d tool calls in trace",
                            event_count, elapsed, len(val) if val else 0)

            elif kind == "warning":
                logger.warning("[DOC-SVC] event #%d [warning] %.0fms: %s", event_count, elapsed, sanitize_log_input(str(val)))
                yield f"data: {json.dumps({'type': 'step', 'step': 'warning', 'message': val})}\n\n"

            elif kind == "answer":
                logger.info("[DOC-SVC] event #%d [answer] %.0fms: %d chars",
                            event_count, elapsed, len(val) if val else 0)
                yield f"data: {json.dumps({'type': 'token', 'content': val})}\n\n"

    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error("[DOC-SVC] stream_answer() EXCEPTION after %.0fms and %d events: %s\n%s",
                     elapsed, event_count, e, traceback.format_exc())
        # Return a generic message to the client; the exception detail /
        # stack trace is logged server-side only (CWE-209).
        yield f"data: {json.dumps({'type': 'error', 'content': 'Document search failed. Please try again later.'})}\n\n"
        return

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("[DOC-SVC] stream_answer() DONE  %.0fms  events=%d  sources=%d  tools=%d",
                elapsed_ms, event_count, len(sources), len(tool_trace))
    yield f"data: {json.dumps({'type': 'done', 'timings': {'total': round(elapsed_ms, 1)}, 'follow_up_questions': [], 'source_mode': 'documents'})}\n\n"


async def answer_once(query: str, history: list[dict] | None = None,
                      doc_ids: list[str] | None = None) -> dict:
    """Non-streaming counterpart to ``stream_answer`` for ``POST /chatbot/query``."""
    t0 = time.perf_counter()
    sources: list[dict] = []
    answer_text = ""
    event_count = 0

    logger.info("[DOC-SVC] answer_once() START  query='%s'  history=%d  doc_ids=%s",
                sanitize_log_input(query[:120]), len(history or []), doc_ids)

    try:
        simple = _use_simple()
        pipeline_mod = retrieve_pipeline if simple else embedqa

        events_kwargs: dict = dict(
            question=query,
            history=history,
            doc_ids=doc_ids,
            model=config.CHAT_MODEL,
            reasoning=False,
        )
        if not simple:
            events_kwargs["style"] = "full"

        async for kind, val in pipeline_mod.answer_events(**events_kwargs):
            event_count += 1
            if kind == "passages":
                sources = [_passage_to_source(p, i) for i, p in enumerate(val)]
            elif kind == "answer":
                answer_text = val or ""
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error("[DOC-SVC] answer_once() EXCEPTION after %.0fms and %d events: %s\n%s",
                     elapsed, event_count, e, traceback.format_exc())
        # Return a generic message to the client; the exception detail /
        # stack trace is logged server-side only (CWE-209).
        return {
            "answer": "Document search failed. Please try again later.",
            "sources": [],
            "timings": {"total": round(elapsed, 1)},
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("[DOC-SVC] answer_once() DONE  %.0fms  events=%d  sources=%d  answer_len=%d",
                elapsed_ms, event_count, len(sources), len(answer_text))
    return {
        "answer": answer_text,
        "sources": sources,
        "timings": {"total": round(elapsed_ms, 1)},
    }


async def get_document_names() -> list[str]:
    """Return file names of all ready documents in the PDF library."""
    try:
        async with store.acquire() as conn:
            docs = await store.documents(conn)
        names = [d["file_name"] for d in docs if d["status"] == "ready"]
        logger.info("[DOC-SVC] get_document_names(): %d ready documents", len(names))
        return names
    except Exception as e:
        logger.error("[DOC-SVC] get_document_names() failed: %s", e)
        return []


async def is_available() -> bool:
    """Check if the document module is functional (PostgreSQL pool reachable)."""
    try:
        async with store.acquire() as conn:
            docs = await store.documents(conn)
        logger.info("[DOC-SVC] is_available(): True (%d documents in library)", len(docs))
        return True
    except Exception as e:
        logger.warning("[DOC-SVC] is_available(): False — %s", e)
        return False
