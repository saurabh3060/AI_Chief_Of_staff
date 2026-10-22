
"""
agent_email/service.py
=========================
Email agent: conversation summarization widget + multi-stage cluster
synthesis pipeline + real-time meeting->cluster matching.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from core.config import Settings, get_settings
from core.graph_client.client import GraphClient
from core.logging_config import get_agent_logger
from core.ollama_client.client import OllamaClient
from core.storage.base import StorageBackend
from core.storage.factory import get_storage
from core.text_match import html_to_text, meaningful_tokens, strip_email_noise
from core.workers import run_in_worker_pool

from agent_email import postgres_store, store, test_log_store
from agent_email import weaviate_vector_store as wvs
from agent_email.weaviate_vector_store import NAMESPACE_CLUSTERS
from agent_email.storage_ids import stable_object_id
from agent_email.clustering import (
    cluster_candidates,
    embed_candidates,
    enrich_candidates_with_attachments,
    extract_mom_attachment_text,
    format_emails_for_synthesis,
    mean_embedding,
    normalize_subject,
    series_occurrence_key,
    series_period_token,
    series_subject_tokens,
    split_for_synthesis,
    tag_candidates_with_topics,
)
from agent_email.pipeline_metrics import PipelineMetricsCollector
from agent_email.ocr import parse_attachment
from agent_email.prompts import (
    BU_EMAIL_SYSTEM_PROMPT,
    CLUSTER_SYNTHESIS_SYSTEM_PROMPT,
    MEETING_CLUSTER_RELEVANCE_SYSTEM_PROMPT,
    MEETING_CONTEXT_SYSTEM_PROMPT,
    build_bu_email_prompt,
    build_cluster_synthesis_prompt,
    build_relevance_context_prompt,
    build_semantic_context_text,
    evaluate_bu_response_quality,
    parse_bu_email_response,
    parse_cluster_synthesis_response,
    parse_relevance_context,
    render_relevance_context,
)
from agent_email.dept_centroids import build_department_index, pick_department
from agent_email.relevance import compute_adaptive_cutoff, score_candidate_pool
from agent_email.schema import (
    ActionCandidate,
    ClusterEmailRef,
    ConversationThread,
    EmailCandidate,
    EmailClusterRecord,
    EmailItemSummary,
    EmailMessage,
    EmailSummarizerParsedResult,
    EmailSummarizerQualityReport,
    EmailSummarizerTestRequest,
    EmailSummarizerTestResponse,
    EmailSummarizerTestResult,
    EmailSummary,
    ParsedAttachment,
)
from agent_email.semantic_match import cosine_similarity, embed_text

logger = get_agent_logger("agent_email")

_SUBJECT_PREFIX_RE = re.compile(r"^(re|fw|fwd)\s*:\s*", re.IGNORECASE)
_ACTION_LINE_RE = re.compile(r"Action:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def _cluster_match_text(candidates: list[EmailCandidate], indices: list[int]) -> str:
    """Subject + stripped body of every mail in the cluster, for dept token tie-break."""
    parts: list[str] = []
    for i in indices:
        message = candidates[i].message
        parts.append(message.subject or "")
        parts.append(strip_email_noise(message.body_text or ""))
    return " ".join(p for p in parts if p)


def _parse_message(raw_message: dict) -> EmailMessage:
    from_info = raw_message.get("from", {}).get("emailAddress", {})
    body = raw_message.get("body", {})
    to_recipients = [
        recipient.get("emailAddress", {}).get("address", "")
        for recipient in raw_message.get("toRecipients", [])
        if recipient.get("emailAddress", {}).get("address")
    ]
    cc_recipients = [
        recipient.get("emailAddress", {}).get("address", "")
        for recipient in raw_message.get("ccRecipients", [])
        if recipient.get("emailAddress", {}).get("address")
    ]
    return EmailMessage(
        id=raw_message["id"],
        conversation_id=raw_message.get("conversationId", "") or "",
        subject=raw_message.get("subject", "(no subject)"),
        from_name=from_info.get("name"),
        from_address=from_info.get("address"),
        to_recipients=to_recipients,
        cc_recipients=cc_recipients,
        received_at=datetime.fromisoformat(raw_message["receivedDateTime"].replace("Z", "+00:00")),
        has_attachments=raw_message.get("hasAttachments", False),
        body_text=html_to_text(body.get("content", raw_message.get("bodyPreview", "")), body.get("contentType", "")),
    )


def extract_action_candidates(message: EmailMessage) -> list[ActionCandidate]:
    return [
        ActionCandidate(text=match.group(1).strip(), source_message_id=message.id)
        for match in _ACTION_LINE_RE.finditer(message.body_text)
    ]


async def get_raw_messages(bearer_token: str | None = None, top: int = 20, settings: Settings | None = None) -> list[dict]:
    active_settings = settings or get_settings()
    client = GraphClient(settings=active_settings)
    raw_response = await client.get_messages(bearer_token=bearer_token, top=top)
    return raw_response.get("value", [])


async def get_recent_conversation(
    bearer_token: str | None = None,
    top: int = 20,
    settings: Settings | None = None,
    skip_attachment_ocr: bool = False,
) -> ConversationThread | None:
    active_settings = settings or get_settings()
    client = GraphClient(settings=active_settings)
    raw_response = await client.get_messages(bearer_token=bearer_token, top=top)
    raw_messages = raw_response.get("value", [])
    if not raw_messages:
        return None

    messages = [_parse_message(raw) for raw in raw_messages]
    if not messages:
        return None

    thread_messages = sorted(messages, key=lambda m: m.received_at)
    thread_key = normalize_subject(thread_messages[-1].subject)

    attachments: list[ParsedAttachment] = []
    if not skip_attachment_ocr:
        ollama_client = OllamaClient(settings=active_settings)
        for message in thread_messages:
            if not message.has_attachments:
                continue
            raw_attachments = await client.get_attachments(message.id, bearer_token=bearer_token)
            for raw_attachment in raw_attachments.get("value", []):
                raw_detail = await client.get_attachment_content(message.id, raw_attachment["id"], bearer_token=bearer_token)
                attachments.append(await parse_attachment(raw_detail, ollama_client=ollama_client, source_message_id=message.id))

    return ConversationThread(
        thread_id=thread_messages[0].id,
        subject=thread_key,
        messages=thread_messages,
        attachments=attachments,
    )


async def summarize_thread(thread: ConversationThread, settings: Settings | None = None) -> EmailSummary:
    active_settings = settings or get_settings()
    attachments_by_message: dict[str, list[str]] = {}
    for attachment in thread.attachments:
        if attachment.ocr_applied and attachment.extracted_text and attachment.source_message_id:
            attachments_by_message.setdefault(attachment.source_message_id, []).append(attachment.extracted_text)

    action_candidates: list[ActionCandidate] = []
    email_summaries: list[EmailItemSummary] = []

    for message in thread.messages:
        action_candidates.extend(extract_action_candidates(message))
        attachment_texts = attachments_by_message.get(message.id, [])
        try:
            llm_result = await _generate_bu_email_llm_result(message, attachment_texts, active_settings)
        except Exception as exc:  # noqa: BLE001
            logger.error("Skipping message_id=%s after LLM failure: %s", message.id, exc)
            continue
        email_summaries.append(
            EmailItemSummary(
                message_id=message.id,
                subject=message.subject,
                from_name=message.from_name,
                from_address=message.from_address,
                received_at=message.received_at,
                summary_text=llm_result["summary_text"],
                action_items=llm_result["action_items"],
                top_questions=llm_result["top_questions"],
            )
        )

    summary_text = "\n\n".join(
        f"{es.subject or 'Message'}:\n{es.summary_text}" for es in email_summaries if es.summary_text
    )
    action_items = [item for es in email_summaries for item in es.action_items]
    top_questions = [q for es in email_summaries for q in es.top_questions][:6]

    return EmailSummary(
        thread_id=thread.thread_id,
        subject=thread.subject,
        generated_at=datetime.now(timezone.utc),
        email_summaries=email_summaries,
        summary_text=summary_text,
        action_candidates=action_candidates,
        action_items=action_items,
        top_questions=top_questions,
    )


async def get_or_generate_summary_for_thread(
    thread: ConversationThread,
    storage: StorageBackend | None = None,
    settings: Settings | None = None,
) -> EmailSummary:
    active_storage = storage or get_storage()
    stored = await active_storage.summaries.get_latest_summary(thread.thread_id)
    if stored is not None:
        return EmailSummary.model_validate(stored)
    summary = await summarize_thread(thread, settings=settings)
    await active_storage.summaries.save_summary(thread.thread_id, summary.model_dump(mode="json"))
    return summary


async def _generate_bu_email_llm_result(message: EmailMessage, attachment_texts: list[str], settings: Settings) -> dict:
    # Check if pipeline is enabled before making Ollama calls
    if not settings.bu_email_pipeline_enabled:
        logger.warning("[FLAG-CHECK] BU_EMAIL_PIPELINE_ENABLED=false; skipping LLM call for email summarization")
        return _build_parse_failure_fallback(ValueError("Pipeline disabled"), "")
    
    ollama_client = OllamaClient(settings=settings)
    prompt = build_bu_email_prompt(message.subject, message.body_text, attachment_texts)
    max_retries = 2
    last_error: Exception | None = None
    last_response_text = ""
    for attempt in range(max_retries + 1):
        try:
            last_response_text = await ollama_client.chat(prompt, system_prompt=BU_EMAIL_SYSTEM_PROMPT)
            PipelineMetricsCollector.record_llm_call(
                "email_summarization",
                tokens_in=len(prompt.split()),
                tokens_out=len(last_response_text.split()),
            )
            return parse_bu_email_response(last_response_text)
        except ValueError as e:
            last_error = e
            PipelineMetricsCollector.record_llm_call("email_summarization", error=True, retry_exhausted=(attempt >= max_retries))
            if attempt >= max_retries:
                return _build_parse_failure_fallback(last_error, last_response_text)
    return _build_parse_failure_fallback(last_error or ValueError("Unknown failure"), last_response_text)


def _build_parse_failure_fallback(error: Exception, raw_response_text: str) -> dict:
    raw_preview = raw_response_text.strip()[:4000]
    summary_text = (
        "\u2022 [PARSE_FAILED] Could not parse model JSON after retries "
        f"({error}). Raw output preserved:\n{raw_preview}"
    )
    return {"summary_text": summary_text, "action_items": [], "top_questions": []}


async def _run_one_email_summarizer_test(
    message: EmailMessage,
    client: GraphClient,
    bearer_token: str | None,
    settings: Settings,
) -> EmailSummarizerTestResult:
    attachment_types: list[str] = []
    attachment_texts: list[str] = []
    raw_response: str | None = None
    attempts = 0
    try:
        if message.has_attachments:
            raw_attachments = await client.get_attachments(message.id, bearer_token=bearer_token)
            for raw_attachment in raw_attachments.get("value", []):
                raw_detail = await client.get_attachment_content(message.id, raw_attachment["id"], bearer_token=bearer_token)
                attachment_type, attachment_text = await extract_mom_attachment_text(raw_detail, settings=settings)
                if attachment_type != "unsupported":
                    attachment_types.append(attachment_type)
                if attachment_text:
                    attachment_texts.append(attachment_text)
        prompt = build_bu_email_prompt(message.subject, message.body_text, attachment_texts)
        ollama_client = OllamaClient(settings=settings)
        parsed: dict | None = None
        parse_error: Exception | None = None
        for attempt in range(3):
            attempts = attempt + 1
            raw_response = await ollama_client.chat(prompt, system_prompt=BU_EMAIL_SYSTEM_PROMPT)
            try:
                parsed = parse_bu_email_response(raw_response)
                parse_error = None
                break
            except ValueError as exc:
                parse_error = exc
        source_text = "\n".join([message.subject, message.body_text, *attachment_texts])
        quality_data = evaluate_bu_response_quality(raw_response or "", source_text=source_text)
        if parsed is None:
            raise parse_error or ValueError("The LLM response could not be parsed.")
        result = EmailSummarizerTestResult(
            timestamp=datetime.now(timezone.utc),
            message_id=message.id,
            email_subject=message.subject,
            from_name=message.from_name,
            from_address=message.from_address,
            received_at=message.received_at,
            input_preview=re.sub(r"\s+", " ", message.body_text).strip()[:500],
            attachment_types=list(dict.fromkeys(attachment_types)),
            raw_llm_response=raw_response,
            parsed_result=EmailSummarizerParsedResult(**parsed),
            quality=EmailSummarizerQualityReport(**quality_data),
            parse_success=True,
            attempts=attempts,
        )
    except Exception as exc:  # noqa: BLE001
        quality_data = evaluate_bu_response_quality(raw_response or "")
        result = EmailSummarizerTestResult(
            timestamp=datetime.now(timezone.utc),
            message_id=message.id,
            email_subject=message.subject,
            from_name=message.from_name,
            from_address=message.from_address,
            received_at=message.received_at,
            input_preview=re.sub(r"\s+", " ", message.body_text).strip()[:500],
            attachment_types=list(dict.fromkeys(attachment_types)),
            raw_llm_response=raw_response,
            parsed_result=None,
            quality=EmailSummarizerQualityReport(**quality_data),
            parse_success=False,
            attempts=attempts,
            error=str(exc),
        )
    await test_log_store.append_test_result(result, settings=settings)
    return result


async def run_email_summarizer_test(
    request: EmailSummarizerTestRequest,
    bearer_token: str | None = None,
    settings: Settings | None = None,
) -> EmailSummarizerTestResponse:
    active_settings = settings or get_settings()
    if not active_settings.email_summarizer_test_log_enabled:
        raise RuntimeError("Email summarizer test logging is disabled.")
    client = GraphClient(settings=active_settings)
    if request.message_id is not None:
        raw_messages = [await client.get_message_detail(request.message_id.strip(), bearer_token=bearer_token)]
    else:
        requested_count = min(request.latest_n or 1, active_settings.email_pipeline_fetch_top)
        response = await client.get_messages(bearer_token=bearer_token, top=requested_count)
        raw_messages = response.get("value", [])
    messages = [_parse_message(raw) for raw in raw_messages]
    results = await asyncio.gather(
        *(_run_one_email_summarizer_test(message, client, bearer_token, active_settings) for message in messages)
    )
    succeeded = sum(1 for result in results if result.parse_success)
    return EmailSummarizerTestResponse(
        requested=1 if request.message_id is not None else (request.latest_n or 1),
        processed=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        log_file=str(test_log_store.get_test_log_path(active_settings)),
        results=results,
    )


def _snap_to_valid_department(name: str, valid_departments: list[str]) -> str:
    from agent_email.attachment_filter import _partial_ratio

    raw = str(name).strip()
    if not raw:
        return raw
    raw_lower = raw.lower()
    for dept in valid_departments:
        if dept.lower() == raw_lower:
            return dept
    substring_hits = [dept for dept in valid_departments if raw_lower in dept.lower() or dept.lower() in raw_lower]
    if len(substring_hits) == 1:
        return substring_hits[0]
    scored = sorted(
        ((max(_partial_ratio(raw_lower, dept.lower()), _partial_ratio(dept.lower(), raw_lower)), dept) for dept in valid_departments),
        key=lambda t: t[0],
        reverse=True,
    )
    if scored:
        best_score, best_dept = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= 75 and (best_score - runner_up) >= 10:
            return best_dept
    return raw


def _normalize_deadline(raw: str) -> str:
    """
    Collapse a deadline string down to a plain date (YYYY-MM-DD) when the
    LLM emits a full ISO timestamp (e.g. "2026-09-16T21:49:48+00:00")
    instead of a date. Leaves already-clean values (plain dates, "TBA",
    unparseable text) untouched.
    """
    raw = raw.strip()
    match = re.match(r"^(\d{4}-\d{2}-\d{2})[T ]\d{2}:\d{2}:\d{2}", raw)
    if match:
        return match.group(1)
    return raw


def _format_action_item(ai: dict) -> str:
    owner = str(ai.get("owner") or "TBA").strip() or "TBA"
    task = str(ai.get("task", "")).strip()
    deadline = _normalize_deadline(str(ai.get("deadline") or "TBA").strip() or "TBA")
    if not task:
        return ""
    return f"{owner}: {task} (by {deadline})"


def _coerce_action_items(raw_items) -> list[dict]:
    out: list[dict] = []
    if not isinstance(raw_items, list):
        return out
    for item in raw_items:
        if isinstance(item, dict):
            task = str(item.get("task", "")).strip()
            if task:
                out.append({
                    "owner": str(item.get("owner") or "TBA").strip() or "TBA",
                    "task": task,
                    "deadline": str(item.get("deadline") or "TBA").strip() or "TBA",
                })
        elif isinstance(item, str) and item.strip():
            out.append({"owner": "TBA", "task": item.strip(), "deadline": "TBA"})
    return out


async def _synthesize_cluster_batch(
    candidates: list[EmailCandidate],
    indices: list[int],
    settings: Settings,
    run_id: str,
    batch_no: int,
    assigned_department: str = "",
    meeting_id: str = "",
) -> dict | None:
    # Check if pipeline is enabled before making Ollama calls
    if not settings.bu_email_pipeline_enabled:
        logger.warning("[FLAG-CHECK] BU_EMAIL_PIPELINE_ENABLED=false; skipping LLM cluster synthesis")
        return None
    
    ollama_client = OllamaClient(settings=settings)
    departments = settings.bu_departments
    valid_names = settings.bu_department_names
    
    # Pre-fetch notes to avoid N+1 queries - fetch once per cluster
    notes_data: list[dict] = []
    if meeting_id:
        from agent_email import postgres_store
        try:
            notes_data = await postgres_store.read_meeting_notes(meeting_id, settings=settings)
            logger.info(
                "[CLUSTER-SYNTH-INPUT run=%s] batch=%d meeting_id=%s notes fetched=%d",
                run_id, batch_no, meeting_id, len(notes_data),
            )
        except Exception as exc:
            logger.warning(
                "[CLUSTER-SYNTH-INPUT run=%s] batch=%d failed to fetch notes for meeting_id=%s: %s",
                run_id, batch_no, meeting_id, exc,
            )
    
    appended = format_emails_for_synthesis(
        candidates, indices, settings=settings,
        meeting_id=meeting_id, notes_data=notes_data,
    )
    prompt = build_cluster_synthesis_prompt(
        appended,
        departments,
        assigned_department=assigned_department,
        email_count=len(indices),
        phrases_by_dept=settings.department_tagging_example_phrases,
    )
    logger.info(
        "[CLUSTER-SYNTH-INPUT run=%s] batch=%d emails=%d notes=%d prompt_chars=%d assigned_dept=%r",
        run_id, batch_no, len(indices), len(notes_data), len(prompt), assigned_department,
    )
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            raw = await ollama_client.chat(
                prompt,
                system_prompt=CLUSTER_SYNTHESIS_SYSTEM_PROMPT,
                num_predict=int(
                    getattr(settings, "ollama_cluster_synth_num_predict", 0)
                    or getattr(settings, "ollama_num_predict", 0)
                    or 2048
                ),
            )
            parsed = parse_cluster_synthesis_response(raw, source_text=appended)
            if assigned_department:
                parsed["department"] = assigned_department
            else:
                parsed["department"] = _snap_to_valid_department(parsed.get("department", ""), valid_names)
            parsed["action_items"] = _coerce_action_items(parsed.get("action_items", []))
            logger.info(
                "[CLUSTER-SYNTH run=%s] batch=%d attempt=%d dept=%r topic=%r summary_n=%d questions_n=%d actions_n=%d",
                run_id, batch_no, attempt + 1, parsed.get("department"), parsed.get("topic"),
                len(parsed.get("summary") or []), len(parsed.get("top_questions") or []),
                len(parsed.get("action_items") or []),
            )
            PipelineMetricsCollector.record_llm_call(
                "cluster_synthesis",
                tokens_in=len(prompt.split()),
                tokens_out=len(raw.split()),
            )
            return parsed
        except ValueError as exc:
            logger.warning("[CLUSTER-SYNTH run=%s] batch=%d parse failure attempt=%d: %s", run_id, batch_no, attempt + 1, exc)
            PipelineMetricsCollector.record_llm_call(
                "cluster_synthesis",
                error=True,
                retry_exhausted=(attempt >= max_retries),
            )
    return None


_NO_CONTENT_SUMMARY_FALLBACK = "No relevant email discussion available for this topic"


def _merge_synthesis_results(results: list[dict]) -> dict:
    if not results:
        return {"department": "", "topic": "", "summary": [], "top_questions": [], "action_items": []}
    if len(results) == 1:
        return results[0]
    dept = results[0].get("department", "")
    topic = results[0].get("topic", "")
    summary: list[str] = []
    questions: list[str] = []
    actions: list[dict] = []
    seen_s: set[str] = set()
    seen_q: set[str] = set()
    seen_a: set[str] = set()
    for r in results:
        if not dept and r.get("department"):
            dept = r["department"]
        if not topic and r.get("topic"):
            topic = r["topic"]
        for line in r.get("summary", []):
            # A batch's no-content fallback sentence only applies to that
            # batch - never mix it into a merged cluster that has real
            # content from other batches. Dropped here; re-added below
            # only if NO batch contributed any real summary content.
            if line.strip().lower() == _NO_CONTENT_SUMMARY_FALLBACK.lower():
                continue
            key = line.lower()
            if key not in seen_s:
                seen_s.add(key)
                summary.append(line)
        for q in r.get("top_questions", []):
            key = q.lower()
            if key not in seen_q:
                seen_q.add(key)
                questions.append(q)
        for ai in r.get("action_items", []):
            key = _format_action_item(ai)
            if key and key not in seen_a:
                seen_a.add(key)
                actions.append(ai)
    if not summary:
        # Every batch was content-free (or all its bullets were filtered
        # out) - surface the fallback sentence instead of an empty summary,
        # and keep questions/actions empty in that case for consistency.
        summary = [_NO_CONTENT_SUMMARY_FALLBACK]
        questions = []
        actions = []
    return {
        "department": dept,
        "topic": topic,
        "summary": summary[:5],
        "top_questions": questions[:4],
        "action_items": actions,
    }


def _email_refs_for_indices(candidates: list[EmailCandidate], indices: list[int]) -> list[ClusterEmailRef]:
    refs: list[ClusterEmailRef] = []
    for idx in indices:
        msg = candidates[idx].message
        refs.append(ClusterEmailRef(
            # Stable Graph message id so a membership can later be resolved
            # back to an actual message. `EmailMessage.id` is populated from
            # the raw Graph `id` in `_parse_message`; it is never blank for a
            # fetched message, but guard with "" to keep the ref well-formed.
            message_id=msg.id or "",
            subject=msg.subject or "",
            from_address=msg.from_address or "",
            date=msg.received_at.isoformat() if msg.received_at else "",
        ))
    return refs


async def run_email_cluster_pipeline(
    bearer_token: str | None = None,
    settings: Settings | None = None,
    token_provider=None,
) -> int:
    """
    Full cluster synthesis pipeline: fetch -> relevance filter -> embed ->
    cluster -> synthesize -> write email_clusters (incremental append).

    `token_provider`, when supplied (delegated mode), is an async callable
    that mints a fresh delegated access token from the stored refresh token.
    It is handed to the single `GraphClient` this run uses so that if the
    access token expires PART-WAY through a long run, a 401 triggers an
    in-flight refresh + retry instead of failing the run. It changes NO
    clustering, relevance, or matching behaviour - only how an expired token
    is recovered. `None` preserves the previous static-token behaviour.
    """
    active_settings = settings or get_settings()
    run_now = datetime.now(timezone.utc)
    run_id = run_now.strftime("%Y%m%dT%H%M%SZ")
    
    # Start metrics collection
    PipelineMetricsCollector.start_run(run_id)
    
    client = GraphClient(settings=active_settings, token_provider=token_provider)

    logger.info(
        "[PIPELINE run=%s] storage backend=postgres (row data -> PostgreSQL, "
        "embeddings -> Weaviate at http=%s:%s grpc=%s:%s)",
        run_id, active_settings.weaviate_http_host, active_settings.weaviate_http_port,
        active_settings.weaviate_grpc_host, active_settings.weaviate_grpc_port,
    )

    # -----------------------------------------------------------------
    # Start-of-run scoped rebuild of the rebuildable copies.
    #
    # Weaviate (vectors) and Redis (read cache) are both rebuildable from
    # PostgreSQL (the system of record). Clear THIS application's data in
    # each — strictly scoped to our own collections / key prefix, never
    # PostgreSQL and never anyone else's data — then repopulate from
    # PostgreSQL so the run starts from a clean, authoritative copy:
    #   * Weaviate: only ACTIVE clusters (is_active=1) with embeddings are
    #     restored, preserving the "only live clusters are searchable"
    #     contract.
    #   * Redis: only the meeting-matching / pre-brief input mappings
    #     (clusters + meeting embeddings + meeting history) are reloaded;
    #     the derived meeting_clusters:* cache is cleared and recomputed
    #     lazily on demand.
    # Best-effort: both helpers are non-fatal, because the pipeline's own
    # writes later in the run re-mirror vectors and refresh the cache.
    try:
        from agent_email.weaviate_sync import clear_and_restore_weaviate_from_postgres
        await clear_and_restore_weaviate_from_postgres(settings=active_settings)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[PIPELINE run=%s] Weaviate scoped rebuild failed (continuing): %s",
            run_id, exc,
        )
    if getattr(active_settings, "redis_enabled", False):
        try:
            from agent_email import redis_cache
            await redis_cache.clear_and_refill_from_source(settings=active_settings)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[PIPELINE run=%s] Redis scoped rebuild failed (continuing): %s",
                run_id, exc,
            )

    lookback_days = active_settings.resolved_email_fetch_lookback_days
    received_after: str | None = None
    if lookback_days > 0:
        fetch_from = run_now - timedelta(days=lookback_days)
        received_after = fetch_from.strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info(
            "[POOL-FETCH run=%s] received_after=%s (lookback=%d day(s))",
            run_id, received_after, lookback_days,
        )

    raw_response = await client.get_messages(
        bearer_token=bearer_token,
        top=active_settings.email_pipeline_fetch_top,
        received_after=received_after,
    )
    raw_messages = raw_response.get("value", [])
    logger.info("[POOL-FETCH run=%s] fetched %d raw message(s)", run_id, len(raw_messages))
    PipelineMetricsCollector.record_mailbox_fetch(len(raw_messages))
    
    if not raw_messages:
        logger.info("[PIPELINE run=%s] no emails fetched; pipeline complete", run_id)
        PipelineMetricsCollector.end_run("completed")
        from agent_email.pipeline_freshness import mark_pipeline_complete
        mark_pipeline_complete()
        return 0

    messages = [_parse_message(raw) for raw in raw_messages]
    PipelineMetricsCollector.record_normalization(len(messages), 0)
    
    PipelineMetricsCollector.record_deduplication(len(messages), 0)

    # -----------------------------------------------------------------
    # Populate bu_emails — the chatbot's raw-message source table.
    # Written early so the chatbot has the data even if clustering
    # fails later. Every fetched email is saved.
    # -----------------------------------------------------------------
    try:
        bu_pairs = [(msg, []) for msg in messages]
        bu_written = await postgres_store.save_emails_to_bu_emails(
            bu_pairs, settings=active_settings,
        )
        logger.info(
            "[BU-EMAILS run=%s] wrote %d row(s) to '%s'",
            run_id, bu_written, active_settings.chatbot_pg_table,
        )

        # Trigger an incremental chatbot reindex so newly written emails
        # become searchable without restarting the backend.
        if bu_written > 0:
            from chatbot.refresh import trigger_reindex
            reindexed = await trigger_reindex(reason=f"email-pipeline-run-{run_id}")
            logger.info(
                "[BU-EMAILS run=%s] chatbot reindex ingested %d object(s)",
                run_id, reindexed,
            )
    except Exception as exc:  # noqa: BLE001
        # Never let a bu_emails failure break the cluster pipeline.
        logger.error(
            "[BU-EMAILS run=%s] write/reindex failed (%s); pipeline unaffected",
            run_id, exc,
        )

    # Specific-meeting relevance requires the structured context for each
    # occurrence. Department weights do not participate in this gate.
    meeting_occurrences: list[dict] | None = None
    from agent_email.meeting_anchor import load_meeting_occurrences

    try:
        await refresh_meeting_history(
            bearer_token=bearer_token,
            settings=active_settings,
            for_relevance=True,
        )
        meeting_occurrences = await load_meeting_occurrences(settings=active_settings)
        logger.info(
            "[POOL-SCORE run=%s] specific-meeting relevance ON: %d occurrence(s) loaded",
            run_id, len(meeting_occurrences),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[POOL-SCORE run=%s] specific-meeting relevance unavailable (%s); failing closed",
            run_id, exc,
        )
        meeting_occurrences = None

    # Precompute the meeting_embeddings table that the pre-meeting-brief
    # meeting->cluster match reads via `_resolve_meeting_query_vec`. This is
    # the ONLY writer of that table (apart from the live calendar fetch and
    # local-mode startup), so without this a pipeline/rebuild run leaves it
    # empty and every brief logs "no meeting embedding; cannot anchor
    # retrieval" and falls back to the department dump.
    #
    # Embed EXACTLY the meeting_history set — the same meetings
    # `prepare_stored_meeting_matches` (below) iterates to request briefs —
    # NOT a separately-fetched calendar window. Keying off a different window
    # left a gap: briefs are requested per meeting_history row, but a window
    # fetch only embeds whichever occurrences fall inside it, so upcoming
    # meetings in history were never embedded and still logged "no meeting
    # embedding". Shaping each stored row into the {id, subject, bodyPreview}
    # dict `precompute_meeting_embeddings` expects guarantees the embedded set
    # == the brief-requested set. Best-effort: never fail the run on it.
    try:
        history_rows = await store.read_meeting_history(settings=active_settings)
        history_events = [
            {
                "id": meeting_id,
                "subject": row.get("subject") or "",
                "bodyPreview": row.get("business_context") or "",
            }
            for meeting_id, row in (history_rows or {}).items()
            if meeting_id
        ]
        persisted = await precompute_meeting_embeddings(history_events, settings=active_settings)
        logger.info(
            "[MEETING-EMBED run=%s] precomputed meeting embeddings over %d history occurrence(s); "
            "table now holds %d meeting(s)",
            run_id, len(history_events), persisted,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[MEETING-EMBED run=%s] precompute failed (non-fatal): %s",
            run_id, exc,
        )

    shells = [EmailCandidate(message=message) for message in messages]
    await enrich_candidates_with_attachments(
        shells, client, bearer_token, settings=active_settings,
    )
    attachment_by_id = {
        shell.message.id: shell.attachment_text
        for shell in shells
        if shell.message.id and shell.attachment_text
    }
    candidates = await score_candidate_pool(
        messages,
        settings=active_settings,
        run_id=run_id,
        meeting_occurrences=meeting_occurrences,
        run_now=run_now,
        attachment_by_id=attachment_by_id,
    )
    if not candidates:
        logger.info("[POOL-SCORE run=%s] no candidates after relevance filter", run_id)
        PipelineMetricsCollector.record_candidate_retrieval(len(messages), 0)
        try:
            prepared = await prepare_stored_meeting_matches(active_settings)
            logger.info("[MEETING-MATCH run=%s] stored %d prepared brief(s) in Redis", run_id, prepared)
        except Exception as exc:  # noqa: BLE001
            logger.error("[MEETING-MATCH run=%s] preparing briefs failed: %s", run_id, exc)
        PipelineMetricsCollector.end_run("completed")
        from agent_email.pipeline_freshness import mark_pipeline_complete
        mark_pipeline_complete()
        return 0

    PipelineMetricsCollector.record_candidate_retrieval(len(messages), len(candidates))

    await tag_candidates_with_topics(candidates, settings=active_settings, run_id=run_id)

    vectors = await embed_candidates(candidates, settings=active_settings)
    logger.info("[CLUSTER-EMBED run=%s] embedded %d candidate(s)", run_id, len(vectors))
    PipelineMetricsCollector.record_embedding(len(vectors), 0)

    # Meeting-anchored clustering: pull past occurrences (once/throttled),
    # then map each candidate to the meeting occurrence it belongs to so a
    # recurring series splits into one cluster per meeting.
    if active_settings.meeting_anchored_clustering_enabled:
        from agent_email.meeting_anchor import assign_candidates_to_meetings

        await assign_candidates_to_meetings(
            candidates, vectors, settings=active_settings, run_id=run_id, run_now=run_now,
        )

    meeting_ids = [c.meeting_id for c in candidates]
    index_groups = cluster_candidates(
        vectors,
        settings=active_settings,
        run_id=run_id,
        subjects=[c.message.subject for c in candidates],
        bodies=[c.message.body_text or "" for c in candidates],
        meeting_ids=meeting_ids,
        received_ats=[c.message.received_at for c in candidates],
        conversation_ids=[c.message.conversation_id for c in candidates],
    )
    for group_idx, indices in enumerate(index_groups):
        logger.info(
            "[CLUSTER-MEMBERS run=%s] cluster=%d n=%d subjects=%s",
            run_id, group_idx, len(indices),
            [candidates[i].message.subject for i in indices],
        )
    
    PipelineMetricsCollector.record_clustering(len(index_groups), algorithm=active_settings.email_cluster_algo)
    
    dept_index = await build_department_index(active_settings)
    max_size = active_settings.email_cluster_max_size
    cluster_records: list[EmailClusterRecord] = []
    # Third tuple element is the RAW anchored calendar meeting_id (the Graph
    # occurrence id) kept separate from the persisted occurrence identity so
    # CEO notes — which are keyed by the real calendar meeting_id — can still
    # be fetched even though the cluster row's meeting_id column now stores
    # the per-occurrence tag.
    pending: list[tuple[list[int], str, str]] = []

    # Clusters are upserted incrementally (per group) rather than truncated.

    for group_idx, indices in enumerate(index_groups):
        group_vectors = [vectors[i] for i in indices]
        cluster_vec = mean_embedding(group_vectors)
        assigned_dept, dept_scores = pick_department(
            cluster_vec,
            dept_index["centroids"],
            prototypes=dept_index["prototypes"],
            cluster_tokens=meaningful_tokens(_cluster_match_text(candidates, indices)),
            dept_tokens=dept_index["dept_tokens"],
            margin=active_settings.email_dept_assignment_margin,
            min_score=active_settings.email_dept_assignment_min_score,
        )
        logger.info(
            "[CLUSTER-DEPT run=%s] cluster=%d assigned=%r scores=%s",
            run_id, group_idx, assigned_dept,
            {name: round(score, 1) for name, score in sorted(dept_scores.items(), key=lambda kv: -kv[1])},
        )
        refs = _email_refs_for_indices(candidates, indices)
        # A post-split cluster is one occurrence of (at most) one series.
        # Fold that occurrence identity into the cluster_key so each
        # occurrence of a recurring series (April / May / ... of the same
        # series) is always a DISTINCT row, even when every member still
        # carries the same (unreliable) anchored meeting_id.
        #
        # Occurrence identity, most reliable first:
        #   - the email-derived recurring-series key (series stem +
        #     calendar period) when the members share one, else
        #   - the anchored meeting_id when members agree on exactly one.
        series_keys = {
            series_occurrence_key(
                candidates[i].message.subject, candidates[i].message.received_at,
            )
            for i in indices
        }
        series_keys.discard("")
        member_meeting_ids = {candidates[i].meeting_id for i in indices if candidates[i].meeting_id}

        cluster_series_key = next(iter(series_keys)) if len(series_keys) == 1 else ""
        # meeting_id/start only identify the occurrence when the members
        # agree on a single meeting AND this is not a period bucket that
        # merely inherited a shared stale anchor. When a recurring-series
        # period key is present, trust it over the anchor for the start.
        cluster_meeting_id = next(iter(member_meeting_ids)) if len(member_meeting_ids) == 1 else ""
        cluster_meeting_start = ""
        cluster_meeting_end = ""
        if cluster_meeting_id:
            cluster_meeting_start = next(
                (candidates[i].meeting_start for i in indices if candidates[i].meeting_id == cluster_meeting_id),
                "",
            )
            cluster_meeting_end = next(
                (candidates[i].meeting_end for i in indices if candidates[i].meeting_id == cluster_meeting_id),
                "",
            )
        # Occurrence tag folded into the identity hash: prefer the
        # email-derived series key; fall back to the meeting_id. This is
        # what guarantees April and May are separate rows regardless of
        # whether the calendar had distinct occurrences.
        occurrence_tag = cluster_series_key or cluster_meeting_id
        cluster_key = hashlib.sha256(
            (occurrence_tag + "|" + "|".join(sorted(r.subject + r.date for r in refs))).encode("utf-8")
        ).hexdigest()[:16]
        # The persisted meeting_id must identify the OCCURRENCE, not the raw
        # anchored calendar meeting. When members of a recurring series split
        # into per-period buckets (May / June / July / Aug of the same
        # review) they all inherit the SAME anchored calendar meeting_id,
        # because the anchor stage mapped them to one occurrence. Writing
        # that shared anchor into the uniqueness-guarded meeting_id column
        # makes every period bucket collide on uq_email_clusters_active_meeting
        # so all but one get demoted to inactive. Use the per-occurrence tag
        # (the series key when present) as the stored meeting_id so each
        # period is a distinct active row. The raw calendar anchor is still
        # preserved for note lookup / display via anchored_meeting_id.
        persisted_meeting_id = occurrence_tag
        topic_seed = normalize_subject(candidates[indices[0]].message.subject)
        logger.info(
            "[CLUSTER-OCCURRENCE run=%s] cluster=%d occurrence=%r meeting_id=%r anchored_meeting_id=%r meeting_start=%s members=%d",
            run_id, group_idx, occurrence_tag or "<unanchored>",
            persisted_meeting_id or "<none>", cluster_meeting_id or "<none>",
            cluster_meeting_start or "-", len(indices),
        )
        cluster_records.append(EmailClusterRecord(
            cluster_id=f"cluster_{cluster_key}",
            department=assigned_dept,
            topic=topic_seed,
            summary="",
            top_questions="",
            action_items="",
            email_refs=json.dumps([r.model_dump() for r in refs]),
            email_count=len(indices),
            cluster_embedding=json.dumps(cluster_vec.tolist()),
            meeting_id=persisted_meeting_id,
            anchored_meeting_id=cluster_meeting_id,
            meeting_start=cluster_meeting_start,
            meeting_end=cluster_meeting_end,
            updated_at=datetime.now(timezone.utc).isoformat(),
        ))
        pending.append((indices, assigned_dept, cluster_meeting_id))
        written = await store.write_cluster_records(cluster_records, settings=active_settings)
        logger.info(
            "[CLUSTER-WRITE run=%s] skeleton cluster=%d persisted %d cluster(s)",
            run_id, group_idx, written,
        )

    synthesis_started_at = time.perf_counter()

    # Cluster synthesis is parallelized across a bounded, queue-based worker
    # pool (core.workers) instead of the previous one-cluster-at-a-time loop.
    # Each cluster is an independent unit of work: it owns its own
    # EmailClusterRecord object (distinct per cluster), so workers mutate
    # different records with no shared-state races. Clustering and anchoring
    # already completed above and are untouched — only the per-cluster
    # synthesis body below runs concurrently. The number of concurrent
    # clusters is EMAIL_PIPELINE_CLUSTER_WORKERS; the real cap on
    # simultaneous Ollama calls is still MAX_CONCURRENT_OLLAMA_CALLS (every
    # _synthesize_cluster_batch -> OllamaClient.chat acquires that
    # semaphore), so more workers never over-drive the model host.
    #
    # Persistence writes the WHOLE shared cluster_records list, so each
    # worker's write is serialized behind a lock to keep the incremental
    # "persist after each cluster" behaviour correct under concurrency.
    worker_count = getattr(active_settings, "email_pipeline_cluster_workers", 2)
    write_lock = asyncio.Lock()

    async def _synthesize_one_cluster(group_idx: int, work_item) -> None:
        (indices, assigned_dept, anchored_meeting_id), record = work_item
        cluster_started_at = time.perf_counter()
        batches = split_for_synthesis(indices, max_size)
        batch_results: list[dict] = []
        # Fetch CEO notes by the RAW anchored calendar meeting_id (not the
        # cluster row's persisted occurrence tag), since notes are keyed by
        # the real Graph meeting occurrence id.
        meeting_id = anchored_meeting_id or ""
        for batch_no, batch_indices in enumerate(batches, start=1):
            try:
                result = await _synthesize_cluster_batch(
                    candidates, batch_indices, active_settings, run_id, batch_no,
                    assigned_department=assigned_dept,
                    meeting_id=meeting_id,
                )
                if result:
                    batch_results.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.error("[CLUSTER-SYNTH run=%s] batch failed: %s", run_id, exc)
                PipelineMetricsCollector.record_llm_call("cluster_synthesis", error=True)

        merged = _merge_synthesis_results(batch_results)
        if assigned_dept:
            merged["department"] = assigned_dept
        if merged.get("summary") or merged.get("topic"):
            record.department = merged.get("department", record.department)
            record.topic = merged.get("topic") or record.topic
            record.summary = "|".join(merged.get("summary", []))
            record.top_questions = "|".join(merged.get("top_questions", []))
            record.action_items = "|".join(
                _format_action_item(ai) for ai in merged.get("action_items", []) if _format_action_item(ai)
            )
            record.updated_at = datetime.now(timezone.utc).isoformat()
            PipelineMetricsCollector.record_summarization(1, 0)
        else:
            logger.warning(
                "[CLUSTER-MERGE run=%s] cluster %d produced no usable synthesis; keeping skeleton",
                run_id, group_idx,
            )
            PipelineMetricsCollector.record_summarization(0, 1)

        # Serialize persistence: write_cluster_records persists the shared
        # cluster_records list, so only one worker may be writing at a time.
        async with write_lock:
            written = await store.write_cluster_records(cluster_records, settings=active_settings)
        logger.info(
            "[CLUSTER-WRITE run=%s] cluster=%d persisted %d cluster(s)",
            run_id, group_idx, written,
        )
        cluster_elapsed = time.perf_counter() - cluster_started_at
        logger.info(
            "[CLUSTER-TIMING run=%s] cluster=%d topic=%r emails=%d batches=%d elapsed=%.2fs",
            run_id, group_idx, record.topic, len(indices), len(batches), cluster_elapsed,
        )

    await run_in_worker_pool(
        list(zip(pending, cluster_records)),
        _synthesize_one_cluster,
        workers=worker_count,
        label="cluster-synth",
    )

    synthesis_elapsed = time.perf_counter() - synthesis_started_at
    logger.info(
        "[CLUSTER-TIMING run=%s] total_synthesis_elapsed=%.2fs clusters=%d",
        run_id, synthesis_elapsed, len(cluster_records),
    )
    logger.info("[CLUSTER-WRITE run=%s] persisted %d cluster(s)", run_id, len(cluster_records))

    try:
        prepared = await prepare_stored_meeting_matches(active_settings)
        logger.info("[MEETING-MATCH run=%s] stored %d prepared brief(s) in Redis", run_id, prepared)
    except Exception as exc:  # noqa: BLE001
        logger.error("[MEETING-MATCH run=%s] preparing briefs failed: %s", run_id, exc)

    # End metrics collection
    PipelineMetricsCollector.end_run("completed")

    from agent_email.pipeline_freshness import mark_pipeline_complete
    mark_pipeline_complete()

    return len(cluster_records)


async def get_email_cluster_records(
    department: str | None = None,
    settings: Settings | None = None,
) -> list[EmailClusterRecord]:
    records = await store.read_all_cluster_records(settings=settings)
    if department:
        dept_lower = department.lower()
        records = [r for r in records if r.department.lower() == dept_lower]
    return records


async def precompute_meeting_embeddings(
    raw_events: list[dict],
    settings: Settings | None = None,
) -> int:
    """
    Embed calendar meetings and persist them to the meeting_embeddings store.

    Reads existing rows first. A meeting is embedded only when its `meeting_id`
    has no usable vector yet. Existing rows are left untouched (including
    meetings not on today's calendar). The file is rewritten only when at
    least one new embedding is created.
    """
    active_settings = settings or get_settings()
    existing = await store.read_meeting_embeddings(settings=active_settings)
    existing_depts = await store.read_meeting_departments(settings=active_settings)

    # Department index for tagging each meeting at CAPTURE time (same
    # nearest-prototype scoring the clusters use, behind the dedicated meeting
    # confidence gate). Built once per call; reused for every meeting below.
    dept_index = await build_department_index(active_settings)

    items: list[tuple[str, str, str]] = []
    for event in raw_events or []:
        meeting_id = str(event.get("id") or "").strip()
        subject = (event.get("subject") or "").strip()
        if not meeting_id:
            continue
        body = event.get("body") or {}
        body_text = html_to_text(
            body.get("content", event.get("bodyPreview", "")),
            body.get("contentType", ""),
        )
        items.append((meeting_id, subject, strip_email_noise(f"{subject}\n{body_text}".strip())))

    if not items and not existing:
        return 0

    # Fast path: skip the rebuild ONLY when every incoming meeting already has
    # a usable cached embedding AND a department already tagged. A meeting that
    # was embedded before department tagging existed still has dept == "", so
    # it must fall through to be labeled (otherwise the pool filter can never
    # engage for pre-existing rows).
    if items:
        all_ready = True
        for meeting_id, _subject, _query_text in items:
            _cached_subject, cached_embedding = existing.get(meeting_id, ("", ""))
            if not cached_embedding or cached_embedding.strip() in ("", "[]"):
                all_ready = False
                break
            if not (existing_depts.get(meeting_id) or "").strip():
                all_ready = False
                break
        if all_ready:
            logger.info(
                "[MEETING-EMBED] all %d meeting(s) already embedded + dept-tagged; skipping",
                len(items),
            )
            return len(existing)

    records_by_id: dict[str, dict[str, str]] = {}
    for meeting_id, (subject, embedding) in existing.items():
        if embedding and embedding.strip() not in ("", "[]"):
            records_by_id[meeting_id] = {
                "meeting_id": meeting_id,
                "subject": subject,
                "department": existing_depts.get(meeting_id, ""),
                "embedding": embedding,
                "updated_at": "",
            }

    def _tag_dept(vec: np.ndarray, subject: str, query_text: str, label: str = "") -> str:
        return _identify_meeting_department(
            vec, dept_index, f"{subject}\n{query_text}", active_settings,
            label=label or subject,
        )

    written = 0
    reused = 0
    now = datetime.now(timezone.utc).isoformat()
    for meeting_id, subject, query_text in items:
        cached = records_by_id.get(meeting_id)
        if cached and cached.get("embedding"):
            if subject and cached.get("subject") != subject:
                cached["subject"] = subject
            # Backfill the department on a reused row from its stored embedding
            # when it has not been tagged yet (row predates dept tagging). No
            # re-embed on this path; the vector is decoded from the cache.
            if not (cached.get("department") or "").strip():
                cached_vec = _decode_json_embedding(cached.get("embedding", ""))
                if cached_vec is not None:
                    cached["department"] = _tag_dept(cached_vec, subject, query_text)
            reused += 1
            continue
        try:
            vec = await embed_text(query_text[:2000], active_settings)
            records_by_id[meeting_id] = {
                "meeting_id": meeting_id,
                "subject": subject,
                "department": _tag_dept(vec, subject, query_text),
                "embedding": json.dumps(vec.tolist()),
                "updated_at": now,
            }
            written += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MEETING-EMBED] failed meeting_id=%s subject=%r: %s", meeting_id, subject, exc)

    # Always persist: even when no NEW embedding was created, a reused row may
    # have just gained a department backfill that must be written (the content
    # hash now folds in department, so unchanged rows are still skipped).
    for row in records_by_id.values():
        if not row.get("updated_at"):
            row["updated_at"] = now
    await store.write_meeting_embeddings(list(records_by_id.values()), settings=active_settings)

    # Department-coverage summary: how many of the captured meetings carry a
    # confident department (so the brief's pool filter can engage) vs fell
    # below the gate (unlabeled -> pool left unfiltered). Printed per-dept so
    # a clean-start rebuild can be verified at a glance from this one line.
    labeled = 0
    per_dept: dict[str, int] = {}
    for row in records_by_id.values():
        dept = (row.get("department") or "").strip()
        if dept:
            labeled += 1
            per_dept[dept] = per_dept.get(dept, 0) + 1
    total = len(records_by_id)
    logger.info(
        "[MEETING-DEPT-COVERAGE] capture: %d/%d meeting(s) got a confident department "
        "(min_score=%.1f); %d below gate (unlabeled -> pool NOT filtered); by_dept=%s",
        labeled, total, active_settings.meeting_dept_assignment_min_score,
        total - labeled,
        {d: per_dept[d] for d in sorted(per_dept, key=lambda k: -per_dept[k])},
    )
    logger.info(
        "[MEETING-EMBED] persisted %d meeting(s) (%d newly embedded, %d reused/backfilled)",
        len(records_by_id), written, reused,
    )
    return len(records_by_id)


# Timestamp of the last successful historical calendar pull, used to
# throttle re-pulls (see `meeting_history_refresh_hours`). Process-local.
_last_meeting_history_pull_at: datetime | None = None


# Prefix used to mark a persisted business_context as LOW_CONTEXT: the invite
# body was empty, too thin, or the extraction degenerated into a title echo /
# generic boilerplate. Consumers (relevance.py) detect this prefix to keep the
# meeting in the candidate pool while matching on its literal subject instead
# of treating the (discarded) fabricated text as real facts.
LOW_CONTEXT_PREFIX = "[LOW_CONTEXT]"

# Generic corporate boilerplate phrases that indicate the extractor invented
# content instead of grounding it in real invite text.
_GENERIC_BOILERPLATE_PHRASES = (
    "improve performance",
    "review operations",
    "optimize risk management",
    "improve profitability",
    "improve customer satisfaction",
    "align with current market conditions",
    "ensure alignment with current market conditions",
    "regulatory requirements",
    "drive business growth",
    "enhance efficiency",
    "optimize processes",
)


def _looks_like_title_echo(context: str, subject: str) -> bool:
    """
    Best-effort detector for a degenerate business context that just restates
    the meeting subject (e.g. objective == title) instead of extracting real
    facts from the invite body.
    """
    if not context or not subject:
        return False
    objective_line = next(
        (line for line in context.splitlines() if line.startswith("Objective:")),
        "",
    )
    objective_text = objective_line.split(":", 1)[1].strip().lower() if ":" in objective_line else ""
    subject_norm = subject.strip().lower()
    if not objective_text or not subject_norm:
        return False
    return objective_text == subject_norm or subject_norm in objective_text


def _looks_like_generic_boilerplate(context: str) -> bool:
    """Detect fabricated generic-corporate-speak the extraction prompt forbids."""
    if not context:
        return False
    lowered = context.lower()
    return any(phrase in lowered for phrase in _GENERIC_BOILERPLATE_PHRASES)


def _is_low_context(context: str, subject: str, body_text: str) -> bool:
    """
    Decide whether the extracted business context should be treated as
    LOW_CONTEXT: the invite body was empty/too thin, or the extraction
    produced nothing but a title echo or generic boilerplate rather than
    real facts. Used to prevent fake/invented business context from ever
    being persisted or presented to the relevance matcher as if it were
    grounded fact.
    """
    if not (body_text or "").strip():
        return True
    if not (context or "").strip():
        return True
    if _looks_like_title_echo(context, subject):
        return True
    if _looks_like_generic_boilerplate(context):
        return True
    return False


async def _extract_meeting_semantics(
    subject: str,
    body_text: str,
    settings: Settings,
) -> tuple[str, dict]:
    """Extract a calendar occurrence's business context ONCE and reuse it.

    This is the single structured extraction for a meeting occurrence. It runs
    the meeting-specific ``MEETING_CONTEXT_SYSTEM_PROMPT`` (which forbids
    echoing the title as the objective and forbids pulling in facts from other
    meetings) exactly once, parses it once, and returns BOTH:

      * ``business_context`` — the rendered, LOW_CONTEXT-gated text persisted
        for relevance matching (identical semantics to the previous
        ``_extract_meeting_business_context``), and
      * ``parsed`` — the structured dict the embedding text is built from.

    The previous design ran this extractor AND a second ``TOPIC_TAG`` LLM call
    (in ``_meeting_embed_text``) over the same subject+body; both are now one
    call. The caller renders the embedding text from ``parsed`` via
    ``_meeting_embed_text_from_semantics`` (no additional LLM call).

    Retries a transient LLM/parse failure (``max_retries=2``) before giving
    up, mirroring the email-side single extractor. On empty body or total
    failure returns ``("" or LOW_CONTEXT_PREFIX, {})`` so the caller still
    embeds the raw subject/body and matches on the literal subject.
    """
    if not (body_text or "").strip():
        # No body to ground an extraction — LOW_CONTEXT, match on subject.
        return LOW_CONTEXT_PREFIX, {}

    prompt = build_relevance_context_prompt(subject, body_text)
    client = OllamaClient(settings=settings)
    max_retries = 2
    parsed: dict = {}
    for attempt in range(max_retries + 1):
        try:
            raw = await client.chat(
                prompt,
                system_prompt=MEETING_CONTEXT_SYSTEM_PROMPT,
                num_predict=512,
            )
            parsed = parse_relevance_context(raw)
            PipelineMetricsCollector.record_llm_call(
                "meeting_context",
                tokens_in=len(prompt.split()),
                tokens_out=len(raw.split()),
            )
            break
        except Exception as exc:  # noqa: BLE001
            PipelineMetricsCollector.record_llm_call(
                "meeting_context",
                error=True,
                retry_exhausted=(attempt >= max_retries),
            )
            logger.warning(
                "[MEETING-HISTORY] context extraction subject=%r attempt=%d failed: %s",
                subject, attempt + 1, exc,
            )
            if attempt >= max_retries:
                # Fail soft: no context, no parsed dict — caller falls back to
                # raw subject/body embedding and subject-only matching.
                return "", {}

    context = render_relevance_context(parsed)
    if _is_low_context(context, subject, body_text):
        logger.warning(
            "[MEETING-HISTORY] business context for subject=%r is LOW_CONTEXT "
            "(thin/empty body or title-echo/boilerplate extraction); discarding "
            "fabricated context=%r and matching on subject only",
            subject, context[:200],
        )
        # Keep the parsed dict so the embedding can still use whatever
        # structure was extracted, but persist the LOW_CONTEXT sentinel as the
        # relevance-facing business_context.
        return LOW_CONTEXT_PREFIX, parsed
    return context, parsed


def _identify_meeting_department(
    vec: np.ndarray,
    dept_index: dict,
    text: str,
    settings: Settings,
    label: str = "",
) -> str:
    """
    Confidently identify a meeting occurrence's department from its OWN
    embedding, reusing the same nearest-prototype scoring the clusters use.

    This is a CONFIDENCE GATE: it returns a department ONLY when the best
    prototype score clears the dedicated meeting floor
    (`meeting_dept_assignment_min_score`, stricter than the cluster floor).
    Below the gate it returns "" so the caller leaves the meeting unlabeled
    and applies NO department filtering — never a low-confidence guess that
    could prune away the right clusters.

    The meeting vector already lives in the same Objective/Topic/Entities
    space as cluster vectors (see `_meeting_embed_text`), so `pick_department`
    is directly comparable to the cluster dept tagging.

    `label` is a human-readable meeting identifier (subject / meeting_id) used
    only in the log line so a batch of per-meeting decisions is readable.
    """
    dept, scores = pick_department(
        vec,
        dept_index["centroids"],
        prototypes=dept_index["prototypes"],
        cluster_tokens=meaningful_tokens(text),
        dept_tokens=dept_index["dept_tokens"],
        margin=settings.meeting_dept_assignment_margin,
        min_score=settings.meeting_dept_assignment_min_score,
    )
    ranked = sorted(scores.items(), key=lambda kv: -kv[1]) if scores else []
    best_name, best_score = ranked[0] if ranked else ("", 0.0)
    min_score = settings.meeting_dept_assignment_min_score
    # One line that makes the gate decision unambiguous: the winning dept and
    # its score vs the floor, PASS when a dept was returned, GATED when the
    # best candidate existed but fell below the floor (so no label is applied).
    logger.info(
        "[MEETING-DEPT] meeting=%r decision=%s identified=%r best=%r best_score=%.1f "
        "min_score=%.1f margin=%.1f scores=%s",
        label or "(unlabeled)",
        "PASS" if dept else "GATED",
        dept,
        best_name,
        best_score,
        min_score,
        settings.meeting_dept_assignment_margin,
        {name: round(score, 1) for name, score in ranked},
    )
    return dept


def _meeting_embed_text_from_semantics(
    subject: str,
    body_text: str,
    parsed: dict,
    settings: Settings,
) -> str:
    """Build the text to embed for a calendar occurrence — NO LLM call.

    Renders the structured semantics already extracted by
    ``_extract_meeting_semantics`` through ``build_semantic_context_text`` so
    the occurrence embedding lives in the same Objective/Topic/Entities/
    Actions space as the email clustering embeddings (matching compares
    business CONTEXT, not generic corporate-prose similarity). This replaces
    the former second ``TOPIC_TAG`` LLM call: the single meeting extraction is
    reused here.

    Falls back to the raw noise-stripped subject+body when the pipeline is
    disabled or the extraction yielded no structure — so an occurrence is
    never left without embeddable text.
    """
    raw_text = strip_email_noise(f"{subject}\n{body_text}".strip())[:2000]

    if not settings.bu_email_pipeline_enabled or not parsed:
        return raw_text

    weight = int(getattr(settings, "email_embed_objective_weight", 2) or 2)
    context = build_semantic_context_text(parsed, objective_weight=weight)
    if context:
        logger.info(
            "[MEETING-HISTORY] structured context subject=%r objective=%r topics=%s",
            subject, parsed.get("business_objective", ""), parsed.get("topics", []),
        )
        # Append a short raw grounding slice so concrete names/figures in the
        # invite still anchor the vector (mirrors the email path's Context
        # grounding), capped so it cannot dominate the structure.
        return f"{context}\nContext: {raw_text[:600]}".strip()
    return raw_text


async def refresh_meeting_history(
    bearer_token: str | None = None,
    settings: Settings | None = None,
    force: bool = False,
    for_relevance: bool = False,
) -> int:
    """
    One-time (throttled) pull of PAST calendar occurrences for anchoring.

    Pulls `meeting_history_lookback_days` of calendar history via a single
    `calendarView` call (which expands each recurring instance into its
    own dated occurrence), embeds every occurrence, and persists them to
    the `meeting_history` store. Already-embedded occurrences are reused so the
    embed only ever runs once per occurrence.

    This is deliberately decoupled from the ~5-min "today's meetings" poll:
    it re-runs at most once per `meeting_history_refresh_hours` (0 = once
    per process). Pass `force=True` to override the throttle.
    """
    global _last_meeting_history_pull_at
    active_settings = settings or get_settings()

    if not active_settings.meeting_anchored_clustering_enabled and not for_relevance:
        logger.info("[MEETING-HISTORY] anchored clustering disabled; skipping historical pull")
        return 0

    now = datetime.now(timezone.utc)
    refresh_hours = float(active_settings.meeting_history_refresh_hours or 0.0)
    if not force and _last_meeting_history_pull_at is not None:
        if refresh_hours <= 0:
            logger.info("[MEETING-HISTORY] already pulled this process (refresh_hours=0); skipping")
            return -1
        age_hours = (now - _last_meeting_history_pull_at).total_seconds() / 3600.0
        if age_hours < refresh_hours:
            logger.info(
                "[MEETING-HISTORY] last pull %.2fh ago < %.2fh refresh window; skipping",
                age_hours, refresh_hours,
            )
            return -1

    lookback_days = int(active_settings.meeting_history_lookback_days or 60)
    fetch_top = int(active_settings.meeting_history_fetch_top or 250)
    window_start = now - timedelta(days=lookback_days)
    # End the window at the run time itself, NOT end-of-today. Pulling through
    # end of today swept in today's not-yet-started calendar events and let
    # them be treated as PAST meetings for anchoring. Ending at `now` keeps the
    # historical pool to genuinely-past occurrences only.
    window_end = now

    logger.info(
        "[MEETING-HISTORY] one calendar pull, past %d day(s) up to run time: %s -> %s (top=%d)",
        lookback_days, window_start.isoformat(), window_end.isoformat(), fetch_top,
    )

    client = GraphClient(settings=active_settings)
    try:
        raw_response = await client.get_calendar_view(
            start_date_time=window_start.isoformat(),
            end_date_time=window_end.isoformat(),
            bearer_token=bearer_token,
            top=fetch_top,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[MEETING-HISTORY] calendar pull failed: %s", exc)
        return 0

    raw_events = raw_response.get("value", [])
    logger.info("[MEETING-HISTORY] fetched %d raw occurrence(s)", len(raw_events))

    existing = await store.read_meeting_history(settings=active_settings)
    records_by_id: dict[str, dict[str, str]] = dict(existing)
    embedded = 0
    reused = 0
    now_iso = now.isoformat()

    # Build the department index once for this pull so each occurrence can be
    # tagged with its own department from its embedding (same nearest-prototype
    # scoring the clusters use, behind the dedicated meeting confidence gate).
    dept_index = await build_department_index(active_settings)

    for event in raw_events:
        meeting_id = str(event.get("id") or "").strip()
        if not meeting_id:
            continue
        subject = (event.get("subject") or "").strip()
        start_obj = event.get("start") or {}
        end_obj = event.get("end") or {}
        start_dt = str(start_obj.get("dateTime") or "").strip()
        end_dt = str(end_obj.get("dateTime") or "").strip()
        cached = existing.get(meeting_id)
        body = event.get("body") or {}
        body_text = html_to_text(
            body.get("content", event.get("bodyPreview", "")),
            body.get("contentType", ""),
        )
        has_vector = (cached or {}).get("embedding", "").strip() not in ("", "[]")
        business_context = (cached or {}).get("business_context", "").strip()
        # Single structured extraction per occurrence, reused for BOTH the
        # persisted business_context AND the embedding text (when a new vector
        # is needed below). Only runs when the business_context is missing OR
        # a fresh embedding must be built; a fully-cached occurrence skips it.
        meeting_parsed: dict = {}
        if not business_context or not has_vector:
            try:
                business_context, meeting_parsed = await _extract_meeting_semantics(
                    subject, body_text, active_settings,
                )
                logger.info(
                    "[MEETING-HISTORY] relevance context subject=%r context=%r",
                    subject, business_context[:300],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[MEETING-HISTORY] relevance context failed subject=%r: %s",
                    subject, exc,
                )
                business_context = business_context or ""
                meeting_parsed = {}

        if has_vector:
            cached["subject"] = subject or cached.get("subject", "")
            cached["start_datetime"] = start_dt or cached.get("start_datetime", "")
            cached["end_datetime"] = end_dt or cached.get("end_datetime", "")
            cached["business_context"] = business_context or cached.get("business_context", "")
            # Backfill / refresh the department on the reused occurrence from
            # its already-stored embedding (no re-embed on this path). This
            # labels pre-existing rows on the first pull after this change.
            cached_vec = _decode_json_embedding((cached or {}).get("embedding", ""))
            if cached_vec is not None:
                cached["department"] = _identify_meeting_department(
                    cached_vec, dept_index, f"{subject}\n{body_text}", active_settings,
                    label=subject or meeting_id,
                )
            else:
                cached["department"] = cached.get("department", "")
            records_by_id[meeting_id] = cached
            reused += 1
            continue
        try:
            query_text = _meeting_embed_text_from_semantics(
                subject, body_text, meeting_parsed, active_settings,
            )
            vec = await embed_text(query_text, active_settings)
            department = _identify_meeting_department(
                vec, dept_index, f"{subject}\n{body_text}", active_settings,
                label=subject or meeting_id,
            )
            records_by_id[meeting_id] = {
                "meeting_id": meeting_id,
                "subject": subject,
                "start_datetime": start_dt,
                "end_datetime": end_dt,
                "business_context": business_context,
                "department": department,
                "embedding": json.dumps(vec.tolist()),
                "updated_at": now_iso,
            }
            embedded += 1
            logger.info(
                "[MEETING-HISTORY] embedded occurrence id=%s subject=%r start=%s end=%s",
                meeting_id, subject, start_dt, end_dt,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[MEETING-HISTORY] embed failed id=%s subject=%r: %s",
                meeting_id, subject, exc,
            )

    # ------------------------------------------------------------------
    # Window-scoped reconciliation: the calendar view is the authoritative
    # set of occurrences for [window_start, window_end]. Any STORED
    # occurrence whose start falls inside that window but was NOT returned
    # by this pull has been deleted/cancelled upstream, so retire it from
    # every layer. This keeps the merge from accumulating ghost meetings
    # that otherwise pollute anchoring and surface empty briefs.
    #
    # Safety guards (fail toward keeping data):
    #   * Only reconcile after a NON-EMPTY fetch. An empty result is treated
    #     as a transient/partial read, never as "the whole window is gone".
    #   * Only prune rows whose start is PARSEABLE and provably in-window.
    #     Unparseable/missing starts and anything outside the window are
    #     left untouched.
    fetched_ids = {
        str(e.get("id") or "").strip()
        for e in raw_events
        if str(e.get("id") or "").strip()
    }

    def _in_window_start(value: str) -> bool:
        raw = (value or "").strip()
        if not raw:
            return False
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return window_start <= parsed < window_end

    stale_ids: list[str] = []
    if raw_events:  # guard: never prune off a transient empty pull
        for mid, row in existing.items():
            if mid in fetched_ids:
                continue
            if _in_window_start(row.get("start_datetime", "")):
                stale_ids.append(mid)

    if stale_ids:
        for mid in stale_ids:
            records_by_id.pop(mid, None)
        logger.info(
            "[MEETING-HISTORY] reconciliation: pruning %d stale in-window "
            "occurrence(s) absent from calendar: %s",
            len(stale_ids), stale_ids,
        )
        try:
            await store.delete_meeting_occurrences(stale_ids, settings=active_settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[MEETING-HISTORY] reconciliation delete failed (will retry next pull): %s",
                exc,
            )

    if records_by_id:
        await store.write_meeting_history(list(records_by_id.values()), settings=active_settings)
    _last_meeting_history_pull_at = now

    # Department-coverage summary for the historical occurrence set (mirrors
    # the capture-time summary in precompute_meeting_embeddings).
    hist_labeled = 0
    hist_per_dept: dict[str, int] = {}
    for row in records_by_id.values():
        dept = (row.get("department") or "").strip()
        if dept:
            hist_labeled += 1
            hist_per_dept[dept] = hist_per_dept.get(dept, 0) + 1
    hist_total = len(records_by_id)
    logger.info(
        "[MEETING-DEPT-COVERAGE] history: %d/%d occurrence(s) got a confident department "
        "(min_score=%.1f); %d below gate (unlabeled); by_dept=%s",
        hist_labeled, hist_total, active_settings.meeting_dept_assignment_min_score,
        hist_total - hist_labeled,
        {d: hist_per_dept[d] for d in sorted(hist_per_dept, key=lambda k: -hist_per_dept[k])},
    )
    logger.info(
        "[MEETING-HISTORY] persisted %d occurrence(s) (%d newly embedded, %d reused, %d pruned)",
        len(records_by_id), embedded, reused, len(stale_ids),
    )
    return len(records_by_id)


def _decode_json_embedding(raw: str) -> np.ndarray | None:
    if not raw:
        return None
    try:
        values = json.loads(raw)
        if not isinstance(values, list) or not values:
            return None
        return np.array(values, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None


def _decode_cluster_embedding(record: EmailClusterRecord) -> np.ndarray | None:
    return _decode_json_embedding(record.cluster_embedding or "")


async def _resolve_meeting_query_vec(
    meeting_id: str | None,
    query_vec: np.ndarray | None,
    settings: Settings,
) -> np.ndarray | None:
    """
    Resolve the embedding that represents the live meeting — the real query
    the whole brief is built around.

    Resolution order (no runtime embedding; the pipeline precomputes these):
      1) an explicitly supplied `query_vec`, else
      2) the stored meeting embedding keyed by `meeting_id`.

    Returns None when neither is available (e.g. pipeline disabled so no
    embedding was persisted). Callers treat None as "cannot anchor to the
    meeting" and fall back accordingly.
    """
    if query_vec is not None:
        return query_vec
    if meeting_id:
        stored = await store.read_meeting_embeddings(settings=settings)
        cached = stored.get(meeting_id)
        if cached:
            return _decode_json_embedding(cached[1])
    return None


async def _resolve_meeting_department(
    meeting_id: str | None,
    settings: Settings,
) -> str:
    """
    Resolve the meeting occurrence's CONFIDENT department.

    Resolution order (both values are produced by the SAME confidence gate
    `_identify_meeting_department`, so an empty string always means "no
    confident label" and the caller leaves the candidate pool unfiltered):
      1) the department tagged on the ``meeting_embeddings`` row at
         today's-meetings CAPTURE time — the primary source, since this is
         the same table the brief match resolves its vector from, else
      2) the department tagged on the ``meeting_history`` row (historical
         occurrence pull) as a fallback.

    No runtime embedding here; both values are read straight from the system
    of record. Returns "" when there is no meeting_id or neither store carries
    a confident department.
    """
    if not meeting_id:
        return ""

    # 1) Capture-time department on the meeting_embeddings row.
    try:
        depts = await store.read_meeting_departments(settings=settings)
        dept = (depts.get(meeting_id) or "").strip()
        if dept:
            return dept
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[MEETING-MATCH] could not read meeting_embeddings department (%s); "
            "falling back to meeting_history",
            exc,
        )

    # 2) Fallback: historical-occurrence department.
    try:
        history = await store.read_meeting_history(settings=settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[MEETING-MATCH] could not read meeting_history for department (%s); "
            "treating meeting as unlabeled",
            exc,
        )
        return ""
    row = history.get(meeting_id) or {}
    return (row.get("department") or "").strip()


async def _recompute_and_cache_meeting_match(
    meeting_id: str,
    meeting_title: str = "",
    meeting_body: str = "",
    settings: Settings | None = None,
) -> list[dict]:
    """Recompute ONE meeting's matched-cluster list and (re)write it to Redis.

    This is the self-healing primitive shared by the startup rebuild, the
    pipeline's ``prepare_stored_meeting_matches``, and the recompute-on-miss
    fallback in the read path. It is DELIBERATELY cheap: it only runs the
    meeting-to-cluster MATCHING step (cosine / Weaviate search over clusters
    that already exist in PostgreSQL). It performs NO clustering, NO
    summarization, and NO LLM synthesis — all of that durable work is read
    straight from the system of record. The only thing it rebuilds is the
    derived ``meeting_clusters:*`` cache entry.

    Returns the stored match-entry list (``[{cluster_id, score, record}]``).
    """
    active = settings or get_settings()
    from agent_email import redis_cache

    result, scores = await _compute_meeting_clusters(
        meeting_title,
        meeting_body,
        settings=active,
        meeting_id=meeting_id,
    )
    matches = [
        {
            "cluster_id": record.cluster_id,
            "score": scores.get(record.cluster_id, _NO_SIMILARITY_SCORE),
            "record": record.model_dump(),
        }
        for record in result
    ]
    await redis_cache.set_meeting_clusters(meeting_id, matches, active, durable=True)
    return matches


async def prepare_stored_meeting_matches(settings: Settings | None = None) -> int:
    """Compute every meeting's cluster list once and store it in Redis.

    The calendar screen and the pre-meeting brief only read this key. They
    do not run cosine search or the activity model.

    This is pure MATCHING over data that already exists in PostgreSQL
    (clusters + meeting embeddings) — it never re-runs clustering or
    summarization. That is why it is safe to call at startup to rebuild the
    derived brief cache after an accidental restart, without paying the cost
    of the full pipeline.
    """
    active = settings or get_settings()
    history = await store.read_meeting_history(settings=active)
    if not history:
        return 0

    prepared = 0
    for meeting_id, row in history.items():
        await _recompute_and_cache_meeting_match(
            meeting_id,
            meeting_title=row.get("subject") or "",
            meeting_body="",
            settings=active,
        )
        prepared += 1
    return prepared


async def match_meeting_to_cluster(
    meeting_title: str,
    meeting_body: str = "",
    settings: Settings | None = None,
    meeting_id: str | None = None,
    query_vec: np.ndarray | None = None,
) -> EmailClusterRecord | None:
    """Return the top cluster for this meeting.

    Reads from Redis; on a true cache miss it recomputes this meeting's match
    from PostgreSQL (matching only — no re-clustering or re-summarization) and
    caches it, so the brief self-heals instead of returning empty.
    """
    _ = query_vec
    active_settings = settings or get_settings()
    if not meeting_id:
        logger.info("[MEETING-MATCH] no meeting_id; prepared brief cannot be read")
        return None

    from agent_email import redis_cache

    cached = await redis_cache.get_meeting_clusters(meeting_id, active_settings)
    if not cached:
        # Self-heal: the derived brief cache was empty (first run, cleared by
        # a meeting-embedding write, or wiped on an accidental restart). Rebuild
        # just THIS meeting's match from PostgreSQL data — pure matching, no
        # re-clustering / re-summarization — and cache it so the next read hits.
        logger.info(
            "[MEETING-MATCH] meeting_id=%s cache miss -> recomputing match on demand",
            meeting_id,
        )
        cached = await _recompute_and_cache_meeting_match(
            meeting_id,
            meeting_title=meeting_title,
            meeting_body=meeting_body,
            settings=active_settings,
        )
    if not cached:
        logger.info(
            "[MEETING-MATCH] meeting_id=%s has no matching cluster after recompute",
            meeting_id,
        )
        return None

    best_entry = cached[0]
    best = EmailClusterRecord.model_validate(best_entry["record"])
    logger.info(
        "[MEETING-MATCH] meeting_id=%s -> prepared cluster=%s score=%.1f topic=%r",
        meeting_id, best.cluster_id, best_entry.get("score", -1.0), best.topic,
    )
    return best



def _cluster_ref_subjects(record: EmailClusterRecord) -> list[str]:
    """Member email subjects for a cluster, decoded from its email_refs JSON."""
    if not record.email_refs:
        return []
    try:
        payload = json.loads(record.email_refs)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, list):
        return []
    subjects: list[str] = []
    for item in payload:
        if isinstance(item, dict):
            subject = (item.get("subject") or "").strip()
            if subject:
                subjects.append(subject)
    return subjects


def _cluster_business_context(record: EmailClusterRecord) -> str:
    """
    Readable business-context view of a cluster for the relevance gate:
    its topics, summary, member subjects, and action items. This is the
    cluster-side counterpart to the meeting business context, built from
    the structured fields the pipeline already synthesizes — no keyword
    or stem heuristic.
    """
    parts: list[str] = []
    if record.topic:
        parts.append(f"Topics: {record.topic}")
    if record.summary:
        parts.append("Summary: " + record.summary.replace("|", "; "))
    subjects = _cluster_ref_subjects(record)
    if subjects:
        parts.append("Subjects: " + " | ".join(subjects[:12]))
    if record.action_items:
        parts.append("Actions/decisions: " + record.action_items.replace("|", "; "))
    return "\n".join(parts).strip()


def _meeting_business_context(
    meeting_title: str,
    meeting_body: str,
) -> str:
    """
    Business context of the live meeting for relevance matching.

    This is built STRICTLY from the complete meeting context available —
    the meeting's own TITLE and any available INVITE/BODY details. It
    deliberately does NOT include seed-cluster context (seed topic, summary,
    subjects, actions, or embeddings) or any other cluster's context. Cluster
    eligibility must be judged against the meeting alone, so a candidate can
    never be pulled in on the strength of what the seed cluster happens to
    discuss. When the meeting has no invite body, the title alone carries the
    full meeting intent.
    """
    parts: list[str] = []
    if meeting_title:
        parts.append(f"Meeting title: {meeting_title}")
    body = (meeting_body or "").strip()
    if body:
        parts.append(f"Meeting invite/body: {body[:2000]}")
    return "\n".join(parts).strip()


def _score_clusters_against_meeting(
    meeting_vec: np.ndarray,
    records: list[EmailClusterRecord],
) -> list[tuple[float, EmailClusterRecord]]:
    """
    Stage-1 semantic candidate retrieval, MEETING-ANCHORED: cosine similarity
    (0-100) of each cluster's embedding against the MEETING embedding itself —
    not against a single intermediary "seed" cluster.

    Scoring every cluster directly against the live meeting removes the
    single-seed bottleneck: a genuinely relevant cluster is proposed on the
    strength of its own distance to the meeting, so it can no longer be
    excluded just because it happens to sit far from whichever cluster used
    to win the top-cosine race. This is the same embedding-similarity
    mechanism used elsewhere, applied to the real query.

    Returns (score, record) pairs sorted high-to-low; empty when there is no
    usable meeting embedding or no clusters carry embeddings.
    """
    if meeting_vec is None:
        return []
    scored: list[tuple[float, EmailClusterRecord]] = []
    for record in records:
        vec = _decode_cluster_embedding(record)
        if vec is None:
            continue
        score = max(0.0, min(1.0, cosine_similarity(meeting_vec, vec))) * 100.0
        scored.append((score, record))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


async def _search_clusters_against_meeting(
    meeting_vec: np.ndarray,
    records: list[EmailClusterRecord],
    settings: Settings,
) -> list[tuple[float, EmailClusterRecord]] | None:
    """
    Stage-1 candidate retrieval THROUGH WEAVIATE (near_vector).

    This is the search entry point the UI / meeting-matching path uses: the
    live meeting embedding is sent to Weaviate's ``clusters`` collection and
    the nearest active objects come back as ``(weaviate_id, score)`` pairs.
    Because only ACTIVE clusters are mirrored into Weaviate, the search can
    only ever surface live, one-per-meeting clusters — a superseded cluster
    is simply not in the index. The returned ids are hydrated back to full
    rows from the already-active ``records`` (PostgreSQL/Redis) so the
    business fields come from the system of record.

    Returns the scored candidates (high-to-low), or ``None`` when Weaviate
    is unreachable so the caller can fall back to the in-memory cosine scan.
    An empty list is a valid "searched, nothing matched" result.
    """
    if meeting_vec is None:
        return []
    by_weaviate_id = {
        stable_object_id(r.cluster_id): r for r in records
    }
    cap = max(1, int(getattr(settings, "meeting_related_candidate_max", 12) or 12))
    # Over-fetch beyond the display cap so the adaptive cut downstream has a
    # representative score distribution to work from.
    limit = min(len(by_weaviate_id) or cap, max(cap * 4, cap))
    try:
        hits = await asyncio.to_thread(
            wvs.search_vectors, NAMESPACE_CLUSTERS, meeting_vec, limit, settings
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[MEETING-MATCH] Weaviate near_vector search failed (%s); "
            "falling back to in-memory cosine over active records",
            exc,
        )
        return None

    scored: list[tuple[float, EmailClusterRecord]] = []
    for weaviate_id, score in hits:
        record = by_weaviate_id.get(weaviate_id)
        if record is None:
            # A vector in Weaviate with no matching active row — skip it
            # rather than fabricating a record. Self-healing reconciles the
            # two stores separately.
            continue
        scored.append((max(0.0, min(1.0, float(score))) * 100.0, record))
    scored.sort(key=lambda t: t[0], reverse=True)
    logger.info(
        "[MEETING-MATCH] Weaviate search hits=%d hydrated=%d (active records=%d)",
        len(hits), len(scored), len(records),
    )
    return scored


def _semantic_candidate_clusters(
    scored: list[tuple[float, EmailClusterRecord]],
    settings: Settings,
) -> list[tuple[float, EmailClusterRecord]]:
    """
    Stage-1 candidate set: keep clusters whose similarity TO THE MEETING clears
    an adaptive cut (Otsu natural break, clamped to a configured floor — the
    same adaptive-thresholding helper used by the email relevance gate),
    capped at a configured maximum. Reuses the existing distribution-relative
    mechanism rather than a new hard-coded number.

    The adaptive break still comes purely from the candidate-score
    distribution (it separates the highly-matched clusters from the broader,
    lower-matched ones). A configurable tuning offset
    (`meeting_related_adaptive_cutoff_adjustment`) is then applied so the
    effective cutoff can be nudged up (fewer, closer matches) or down (more
    clusters) without hard-coding a fixed threshold. The result is clamped
    to the configured floor.

    Candidate scores here are MEETING-anchored (distance to the live meeting),
    so the candidate set is proposed by proximity to the real query rather
    than to a single intermediary seed cluster.
    """
    if not scored:
        return []
    floor = float(getattr(settings, "meeting_related_candidate_min_score", 45.0) or 0.0)
    scores = [s for s, _ in scored]
    if bool(getattr(settings, "email_relevance_adaptive_cutoff_enabled", True)):
        adaptive_cutoff, reason = compute_adaptive_cutoff(scores, floor=floor, strictness=0.0)
    else:
        adaptive_cutoff, reason = floor, "floor-only"

    # Operator tuning dial: shift the adaptive break up/down, then re-clamp to
    # the floor so the adjustment can never drop below the configured minimum.
    adjustment = float(getattr(settings, "meeting_related_adaptive_cutoff_adjustment", 0.0) or 0.0)
    effective_cutoff = max(floor, adaptive_cutoff + adjustment)

    cap = max(1, int(getattr(settings, "meeting_related_candidate_max", 12) or 12))

    # Log the full candidate-score distribution feeding the adaptive decision.
    logger.info(
        "[MEETING-MATCH-RELATED] candidate_scores=%s",
        [round(s, 1) for s in scores],
    )
    # Log the adaptive threshold, the tuning adjustment, and the resulting
    # effective threshold so the cutoff is fully traceable.
    logger.info(
        "[MEETING-MATCH-RELATED] adaptive_threshold=%.1f (%s) adjustment=%+.1f "
        "effective_threshold=%.1f floor=%.1f",
        adaptive_cutoff, reason, adjustment, effective_cutoff, floor,
    )

    retained: list[tuple[float, EmailClusterRecord]] = []
    for s, record in scored:
        if s >= effective_cutoff and len(retained) < cap:
            retained.append((s, record))
            logger.info(
                "[MEETING-MATCH-RELATED] RETAIN cluster=%s score=%.1f >= %.1f topic=%r",
                record.cluster_id, s, effective_cutoff, record.topic,
            )
        else:
            why = "below-threshold" if s < effective_cutoff else "cap-reached"
            logger.info(
                "[MEETING-MATCH-RELATED] REJECT cluster=%s score=%.1f (%s thr=%.1f) topic=%r",
                record.cluster_id, s, why, effective_cutoff, record.topic,
            )

    logger.info(
        "[MEETING-MATCH-RELATED] stage1 candidates=%d effective_cutoff=%.1f "
        "(adaptive=%.1f %s adjustment=%+.1f) floor=%.1f cap=%d",
        len(retained), effective_cutoff, adaptive_cutoff, reason,
        adjustment, floor, cap,
    )
    return retained


def _parse_relevant_cluster_keys(raw: str, allowed: set[str]) -> tuple[list[str], str, bool]:
    """Validate the cluster relevance gate response as the strict JSON contract."""
    text = (raw or "").strip()
    fence = re.search(r"```(?:\w+)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text:
        return [], "empty relevance decision", False
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return [], "invalid relevance decision", False
        try:
            data = json.loads(match.group(0))
        except (TypeError, ValueError):
            return [], "invalid relevance decision", False
    if not isinstance(data, dict) or set(data) != {"relevant_cluster_keys", "reason"}:
        return [], "invalid relevance decision schema", False
    keys = data["relevant_cluster_keys"]
    reason = data["reason"]
    if (
        not isinstance(keys, list)
        or not all(isinstance(k, str) for k in keys)
        or not isinstance(reason, str)
    ):
        return [], "invalid relevance decision types", False
    matched = [k for k in keys if k in allowed]
    return matched, (reason.strip() or "no reason"), True


def _conservative_relevance_fallback(
    candidates: list[tuple[float, EmailClusterRecord]],
    settings: Settings,
    meeting_title: str = "",
) -> list[tuple[float, EmailClusterRecord]]:
    """
    Conservative fallback used ONLY when the LLM relevance gate cannot be
    trusted (call failed or returned invalid output after retries).

    This deliberately does NOT fail open (returning all stage-1 candidates).
    Stage-1 similarity is a broad semantic net; keeping every candidate on
    gate failure would let generically-similar clusters through. Instead we
    keep only candidates with STRONG business-context evidence — those whose
    stage-1 semantic score clears a strict high bar
    (`meeting_related_fallback_min_score`, default 70.0). This reuses the
    already-computed stage-1 scores (no new Stage-1 scoring/cutoff) and
    preserves stage-1 rank. When nothing clears the strict bar, return an
    empty set rather than broadening the meeting with weak matches.
    """
    strong_floor = float(
        getattr(settings, "meeting_related_fallback_min_score", 70.0) or 0.0
    )
    kept: list[tuple[float, EmailClusterRecord]] = []
    for score, record in candidates:
        if score >= strong_floor:
            kept.append((score, record))
            logger.info(
                "[MEETING-MATCH-RELATED] fallback KEEP cluster=%s score=%.1f >= strong_floor=%.1f topic=%r",
                record.cluster_id, score, strong_floor, record.topic,
            )
        else:
            logger.info(
                "[MEETING-MATCH-RELATED] fallback DROP cluster=%s score=%.1f < strong_floor=%.1f topic=%r",
                record.cluster_id, score, strong_floor, record.topic,
            )
    logger.warning(
        "[MEETING-MATCH-RELATED] CONSERVATIVE FALLBACK used meeting=%r: kept %d/%d candidate(s) "
        "with strong business-context evidence (score >= %.1f); NOT failing open",
        meeting_title, len(kept), len(candidates), strong_floor,
    )
    return kept


async def _business_context_relevant_clusters(
    meeting_context: str,
    candidates: list[tuple[float, EmailClusterRecord]],
    settings: Settings,
    meeting_title: str = "",
) -> tuple[list[tuple[float, EmailClusterRecord]], bool]:
    """
    Stage-2 business-context relevance gate: ask the strict LLM relevance
    classifier which stage-1 candidate clusters discuss the SAME SPECIFIC
    business activity/initiative/project/product/objective/decision as the
    meeting. Eligibility is judged against the MEETING context ONLY (title +
    any available meeting details); `meeting_context` here never carries
    seed-cluster context. Each candidate is evaluated INDEPENDENTLY against
    the meeting. Returns `(kept, used_fallback)` where `kept` preserves
    stage-1 (semantic) rank.

    On failure (call error, or invalid output after retries) this does NOT
    fail open. It uses `_conservative_relevance_fallback`, which keeps only
    candidates with strong business-context evidence, and reports
    `used_fallback=True` for logging.
    """
    if not candidates:
        return [], False
    if not bool(getattr(settings, "meeting_related_relevance_gate_enabled", True)):
        logger.info(
            "[MEETING-MATCH-RELATED] LLM relevance gate DISABLED "
            "(MEETING_RELATED_RELEVANCE_GATE_ENABLED=false) meeting=%r -> keeping %d "
            "stage-1 semantic candidate(s) without LLM call",
            meeting_title, len(candidates),
        )
        return candidates, False
    if not bool(getattr(settings, "bu_email_pipeline_enabled", True)):
        logger.info(
            "[MEETING-MATCH-RELATED] LLM relevance gate SKIPPED "
            "(BU_EMAIL_PIPELINE_ENABLED=false) meeting=%r -> keeping %d stage-1 candidate(s)",
            meeting_title, len(candidates),
        )
        return candidates, False

    logger.info(
        "[MEETING-MATCH-RELATED] LLM relevance gate CALLED meeting=%r candidates=%d",
        meeting_title, len(candidates),
    )
    logger.info(
        "[MEETING-MATCH-RELATED] relevance-gate meeting=%r MEETING BUSINESS CONTEXT:\n%s",
        meeting_title, meeting_context,
    )

    key_to_pair = {
        f"C{index}": pair for index, pair in enumerate(candidates, start=1)
    }
    payload = [
        {
            "cluster_key": key,
            "business_context": _cluster_business_context(record)
            or (record.topic or "cluster"),
        }
        for key, (_score, record) in key_to_pair.items()
    ]
    # Log each candidate's business context so an accept/reject can be traced
    # back to exactly what the gate saw for that cluster.
    for entry, (_score, record) in zip(payload, key_to_pair.values()):
        logger.info(
            "[MEETING-MATCH-RELATED] candidate %s cluster=%s business_context=%r",
            entry["cluster_key"], record.cluster_id, entry["business_context"],
        )
    prompt = (
        "MEETING BUSINESS CONTEXT (this is the COMPLETE context for the meeting — judge every "
        "candidate against THIS and nothing else):\n"
        f"{meeting_context}\n\n"
        "CANDIDATE CLUSTERS (evaluate EACH cluster_key INDEPENDENTLY against the meeting context "
        "above; keep a cluster ONLY if it is about the SAME SPECIFIC business activity/initiative/"
        "project/product/objective/decision as the meeting — not merely the same department, "
        "generic banking/corporate terms, shared entity names, generic words, or broad topic "
        "similarity. Do NOT keep a cluster just because it resembles another candidate cluster; the "
        "keys below are the ONLY valid keys for this request):\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "OUTPUT CONTRACT (write your own reason; do not copy this shape's wording):\n"
        '{"relevant_cluster_keys": [<zero or more of the exact cluster_key values listed above>], '
        '"reason": "<one sentence naming the concrete shared specific activity/initiative/project/'
        'product/objective/decision, or stating there is no materially relevant cluster>"}\n'
        "Return only the one-line JSON object."
    )
    client = OllamaClient(settings=settings)
    allowed = set(key_to_pair)
    for attempt in range(1, 3):
        attempt_prompt = prompt
        if attempt > 1:
            attempt_prompt += (
                "\nYour previous response was empty, malformed, or referenced a cluster_key not in "
                "the candidate list. Retry using exactly the two-field JSON contract and only the "
                "candidate cluster_key values above."
            )
        try:
            raw = await client.chat(
                attempt_prompt,
                system_prompt=MEETING_CLUSTER_RELEVANCE_SYSTEM_PROMPT,
                num_predict=512,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[MEETING-MATCH-RELATED] relevance gate call failed meeting=%r: %s; using conservative fallback",
                meeting_title, exc,
            )
            return _conservative_relevance_fallback(candidates, settings, meeting_title), True
        matched_keys, reason, valid = _parse_relevant_cluster_keys(raw, allowed)
        logger.info(
            "[MEETING-MATCH-RELATED] relevance-gate meeting=%r attempt=%d valid=%s matched=%s reason=%r",
            meeting_title, attempt, valid, matched_keys, reason,
        )
        if valid:
            # key_to_pair maps each cluster_key back to its (score, record),
            # so the kept set preserves stage-1 semantic ordering.
            kept = [key_to_pair[key] for key in matched_keys]
            # Trace the accept/reject decision for every candidate.
            kept_keys = set(matched_keys)
            for key, (_score, record) in key_to_pair.items():
                decision = "ACCEPT" if key in kept_keys else "REJECT"
                logger.info(
                    "[MEETING-MATCH-RELATED] gate %s %s cluster=%s topic=%r (reason=%r)",
                    decision, key, record.cluster_id, record.topic, reason,
                )
            return kept, False
    logger.warning(
        "[MEETING-MATCH-RELATED] relevance gate exhausted retries meeting=%r; using conservative fallback",
        meeting_title,
    )
    return _conservative_relevance_fallback(candidates, settings, meeting_title), True


async def _find_relevant_clusters(
    meeting_vec: np.ndarray,
    records: list[EmailClusterRecord],
    settings: Settings,
    meeting_title: str = "",
    meeting_body: str = "",
    meeting_department: str = "",
) -> tuple[list[EmailClusterRecord], bool, dict[str, float]]:
    """
    MEETING-ANCHORED relevant-cluster retrieval via the same two-stage,
    business-context approach used by search relevancy — NOT subject/keyword
    overlap, and NOT anchored to a single intermediary "seed" cluster:

      Stage 1 — Semantic candidate retrieval: embedding cosine between the
        LIVE MEETING and each cluster, kept by the adaptive cut. This scores
        every cluster against the real query, so a relevant cluster is
        proposed on its own merits instead of having to be close to whichever
        cluster used to win the top-cosine race. It only proposes the
        CANDIDATE SET; score margin/rank never decide inclusion.
      Stage 2 — Business-context relevance: the strict same-business-activity
        LLM gate judges EVERY candidate INDEPENDENTLY against the MEETING
        context ONLY (title + any available meeting details). Every candidate
        that passes is kept; every candidate that fails is rejected. No
        cluster's context is ever used to decide another cluster's eligibility.

    If there is no usable meeting context (no title/body), we STOP and return
    an empty relevant set rather than expanding the meeting using cluster
    context.

    Returns `(relevant, used_fallback, scores)`, relevant clusters ranked by
    stage-1 (meeting) semantic rank among the gate-approved set.
    `used_fallback` is True when the LLM gate failed and the conservative
    fallback decided the relevant set. `scores` maps each returned cluster_id
    to its stage-1 meeting-semantic similarity (0-100 cosine), so callers can
    surface/cache the match score without recomputing the search.
    """
    # CONFIDENCE-GATED DEPARTMENT POOL REDUCTION (always on, fail-closed):
    # When the meeting carries a CONFIDENT department (identified at
    # meeting-history time behind its own min-score gate), reduce the
    # candidate pool to clusters of that same department BEFORE stage-1
    # retrieval. This is the core false-positive fix: an unrelated-department
    # cluster that merely sits near the meeting vector can no longer enter the
    # candidate set.
    #
    # The gate FAILS CLOSED — it never expands to a full cross-department
    # pool. There are exactly two empty-result cases:
    #   * meeting department is UNKNOWN (below the confidence gate,
    #     meeting_department == "") -> STOP with an empty relevant set. We
    #     cannot trust a department constraint, and showing all clusters just
    #     to surface something is worse than showing nothing.
    #   * meeting HAS a confident department but ZERO clusters carry it -> a
    #     genuine DEPARTMENT MISMATCH, STOP empty. A Wholesale meeting must
    #     NOT surface the only existing cluster when that cluster is BB&RL.
    # An empty brief is the correct outcome in both cases.
    # Department-scoped matching is ALWAYS applied — it is the correct
    # approach and has no kill switch. There are exactly two outcomes:
    #   * meeting HAS a confident department -> reduce the pool to
    #     same-department clusters (or STOP empty when none exist), and
    #   * meeting department is UNKNOWN (below the confidence gate) -> STOP
    #     with an empty relevant set. We do NOT fall back to the full
    #     cross-department pool just to show something; an unlabeled meeting
    #     yields no brief rather than a mismatched one.
    if not meeting_department:
        logger.info(
            "[MEETING-MATCH-RELATED] dept filter STOP meeting=%r has no confident department; "
            "returning EMPTY relevant set (no full-pool fallback)",
            meeting_title,
        )
        return [], False, {}

    dept_records = [
        r for r in records
        if (r.department or "").lower() == meeting_department.lower()
    ]
    if not dept_records:
        # Known department, no same-department cluster -> nothing relevant.
        logger.info(
            "[MEETING-MATCH-RELATED] dept filter STOP meeting=%r dept=%r matched 0 "
            "same-department cluster(s) out of %d; returning EMPTY relevant set "
            "(department mismatch, not falling back to cross-dept pool)",
            meeting_title, meeting_department, len(records),
        )
        return [], False, {}

    search_records = dept_records
    logger.info(
        "[MEETING-MATCH-RELATED] dept filter ON meeting=%r dept=%r pool %d -> %d cluster(s)",
        meeting_title, meeting_department, len(records), len(dept_records),
    )

    # Stage-1 candidate retrieval goes THROUGH WEAVIATE (near_vector search
    # over the active, one-per-meeting cluster vectors). If Weaviate is
    # unreachable we fall back to the equivalent in-memory cosine scan over
    # the same active records so matching still works.
    logger.info(
        "[MEETING-MATCH-RELATED] cold path: running stage-1 cosine search meeting=%r",
        meeting_title,
    )
    scored = await _search_clusters_against_meeting(meeting_vec, search_records, settings)
    if scored is None:
        scored = _score_clusters_against_meeting(meeting_vec, search_records)
    candidates = _semantic_candidate_clusters(scored, settings)

    if not candidates:
        return [], False, {}

    # Eligibility is judged against the MEETING context only. Build it from
    # the meeting title + available details — never from any cluster.
    meeting_context = _meeting_business_context(meeting_title, meeting_body)
    if not meeting_context:
        # No meeting context to match against. Do NOT fall back to cluster
        # context to expand the meeting — stop with an empty relevant set.
        logger.info(
            "[MEETING-MATCH-RELATED] no meeting context (title/body); "
            "STOP — not expanding via cluster context",
        )
        return [], False, {}

    relevant, used_fallback = await _business_context_relevant_clusters(
        meeting_context, candidates, settings, meeting_title=meeting_title
    )

    ordered: list[EmailClusterRecord] = []
    scores: dict[str, float] = {}
    for score, record in relevant:
        ordered.append(record)
        # Keep the stage-1 meeting-semantic similarity per cluster so the
        # match (cluster_id -> score) can be surfaced and cached without
        # re-running the Weaviate/cosine search.
        scores[record.cluster_id] = round(float(score), 2)
        logger.info(
            "[MEETING-MATCH-RELATED] relevant=%s topic=%r meeting_semantic=%.1f",
            record.cluster_id, record.topic, score,
        )
    logger.info(
        "[MEETING-MATCH-RELATED] FINAL relevant clusters=%s used_fallback=%s scores=%s",
        [r.cluster_id for r in ordered],
        used_fallback,
        {cid: scores[cid] for cid in (r.cluster_id for r in ordered)},
    )
    return ordered, used_fallback, scores



def _record_series_stem(record: EmailClusterRecord) -> frozenset[str]:
    """
    Recurring-series stem for a cluster, derived generically from its
    member email subjects (calendar tokens / doc-role words stripped, see
    `series_subject_tokens`). No meeting/product name is hardcoded.

    NOTE: This is a subject-stem signal retained for email-clustering use.
    It is NO LONGER the recurrence decision for a live meeting — that is now
    made by `_assess_recurrence` from repeated business context across
    distinct periods. A non-empty/empty stem must not, on its own, be read
    as recurring/one-off.
    """
    subjects = _cluster_ref_subjects(record)
    if not subjects:
        subjects = [record.topic] if record.topic else []
    if not subjects:
        return frozenset()
    stem = series_subject_tokens(subjects[0])
    for subject in subjects[1:]:
        stem &= series_subject_tokens(subject)
    return frozenset(stem)


def _record_period_sort_key(record: EmailClusterRecord) -> tuple:
    """
    Chronological sort key for occurrences of one series: prefer the
    calendar period parsed from the topic/subject, fall back to
    meeting_start, then updated_at. Purely generic (period tokens).
    """
    subject = record.topic or ""
    subjects = _cluster_ref_subjects(record)
    if subjects:
        subject = subjects[0]
    period = series_period_token(subject)
    return (period or "", record.meeting_start or "", record.updated_at or "")


def _record_period_bucket(record: EmailClusterRecord) -> str:
    """
    The calendar period a cluster belongs to (e.g. "2026-04"), used only to
    judge whether repeated business context is separated across DIFFERENT
    periods. Prefers the period parsed from the subject/topic, then the
    meeting_start date, then the updated_at date. Returns "" when unknown.
    """
    subjects = _cluster_ref_subjects(record)
    subject = subjects[0] if subjects else (record.topic or "")
    period = series_period_token(subject)
    if period:
        return period
    for raw in (record.meeting_start, record.updated_at):
        text = (raw or "").strip()
        if len(text) >= 7:
            return text[:7]
    return ""


def _assess_recurrence(
    relevant: list[EmailClusterRecord],
    settings: Settings,
) -> tuple[bool, dict]:
    """
    Decide RECURRING vs ONE-OFF from REPEATED BUSINESS CONTEXT across history,
    not from subject-stem words and NOT from similarity to any seed cluster.

    The input `relevant` set has ALREADY passed the meeting-context relevance
    gate, so every member is confirmed on-topic for this meeting. The only
    remaining question is whether that on-topic evidence is REPEATED OVER
    TIME. A meeting is RECURRING only when:

      * there are at least `meeting_recurrence_min_occurrences` relevant
        clusters (one similar cluster is never enough), AND
      * those relevant clusters are separated across at least
        `meeting_recurrence_min_distinct_periods` DISTINCT calendar periods
        (repeated context over time, not many clusters in one period).

    There is no seed-similarity re-check: the relevance gate already decided
    these clusters belong to the meeting, so recurrence is purely a
    period-counting decision over that gate-approved set. When the evidence
    is weak or ambiguous, this returns False (ONE-OFF).
    """
    min_occ = max(1, int(getattr(settings, "meeting_recurrence_min_occurrences", 2) or 2))
    min_periods = max(1, int(getattr(settings, "meeting_recurrence_min_distinct_periods", 2) or 2))

    occurrences: list[EmailClusterRecord] = []
    periods: set[str] = set()
    for record in relevant:
        occurrences.append(record)
        bucket = _record_period_bucket(record)
        if bucket:
            periods.add(bucket)
        logger.info(
            "[MEETING-MATCH-RECURRENCE] COUNT cluster=%s period=%r topic=%r",
            record.cluster_id, bucket or "unknown", record.topic,
        )

    distinct_periods = len(periods)
    is_recurring = (
        len(occurrences) >= min_occ and distinct_periods >= min_periods
    )
    if is_recurring:
        reason = (
            f"{len(occurrences)} relevant clusters (>= {min_occ}) across "
            f"{distinct_periods} distinct periods (>= {min_periods}): repeated business "
            f"context over time"
        )
    elif len(occurrences) < min_occ:
        reason = (
            f"only {len(occurrences)} relevant cluster(s) "
            f"(need >= {min_occ}); not enough repeated business context"
        )
    else:
        reason = (
            f"{len(occurrences)} relevant cluster(s) but only {distinct_periods} distinct "
            f"period(s) (need >= {min_periods}); not spread across time"
        )
    evidence = {
        "relevant_occurrences": len(occurrences),
        "distinct_periods": distinct_periods,
        "periods": sorted(periods),
        "min_occurrences": min_occ,
        "min_distinct_periods": min_periods,
        "reason": reason,
    }
    logger.info(
        "[MEETING-MATCH-RECURRENCE] decision=%s reason=%s evidence=%s",
        "RECURRING" if is_recurring else "ONE-OFF", reason, evidence,
    )
    return is_recurring, evidence


# Sentinel similarity score for clusters surfaced WITHOUT a cosine search —
# i.e. exact-anchor occurrence rows and department-fallback rows. These are
# included by identity/fallback, not by meeting-semantic similarity, so there
# is no real score to report. Negative so it can never be confused with a real
# 0-100 cosine score.
_NO_SIMILARITY_SCORE = -1.0


async def match_meeting_to_clusters(
    meeting_title: str,
    meeting_body: str = "",
    settings: Settings | None = None,
    meeting_id: str | None = None,
    query_vec: np.ndarray | None = None,
) -> list[EmailClusterRecord]:
    """Read the cluster list the pipeline stored for this meeting.

    A click reads from Redis. On a true cache miss the match is recomputed
    on demand from PostgreSQL (matching only — no re-clustering or
    re-summarization) and cached, so the brief self-heals instead of coming
    back empty after a restart or cache clear.
    """
    _ = query_vec
    active_settings = settings or get_settings()
    if not meeting_id:
        logger.info("[MEETING-MATCH] meeting=%r has no id; brief is empty", meeting_title)
        return []

    from agent_email import redis_cache

    cached = await redis_cache.get_meeting_clusters(meeting_id, active_settings)
    if cached is None:
        # Self-heal on a true cache miss (key absent): rebuild this meeting's
        # match from PostgreSQL (matching only, no re-cluster/re-summarize) and
        # cache it. A cached empty list ([]) is a real "matched nothing" result
        # and is NOT a miss, so it is returned as-is below.
        logger.info(
            "[MEETING-MATCH] meeting=%r meeting_id=%s cache miss -> recomputing match on demand",
            meeting_title, meeting_id,
        )
        cached = await _recompute_and_cache_meeting_match(
            meeting_id,
            meeting_title=meeting_title,
            meeting_body=meeting_body,
            settings=active_settings,
        )
    logger.info(
        "[MEETING-MATCH] meeting=%r meeting_id=%s -> prepared clusters=%s",
        meeting_title, meeting_id, [m["cluster_id"] for m in cached],
    )
    return [EmailClusterRecord.model_validate(m["record"]) for m in cached]


async def _compute_meeting_clusters(
    meeting_title: str,
    meeting_body: str = "",
    settings: Settings | None = None,
    meeting_id: str | None = None,
    query_vec: np.ndarray | None = None,
) -> tuple[list[EmailClusterRecord], dict[str, float]]:
    """
    Meeting -> ordered list of precomputed clusters to show as cards, plus a
    `cluster_id -> similarity score` map for the returned clusters.

    The second element lets the caching wrapper persist the match score
    alongside each matched cluster. Exact-anchor and department-fallback rows
    are included by identity/fallback rather than cosine similarity, so they
    carry the sentinel ``_NO_SIMILARITY_SCORE`` (they were not selected by a
    similarity search).

    Behaviour (generic, no hardcoded names; MEETING-ANCHORED — retrieval is
    anchored to the live meeting itself, never to a single intermediary
    "seed" cluster; recurrence is decided from repeated business context,
    NOT from subject-stem words):
      1) Resolve the meeting's own embedding (exact `meeting_id` embedding /
         stored embedding). An exact `meeting_id` occurrence cluster is always
         kept as a guaranteed anchor so today's occurrence is never dropped.
      2) Find the business-context-relevant historical clusters with the same
         two-stage approach as search relevancy, anchored to the MEETING:
         stage-1 semantic candidate retrieval scores EVERY cluster against the
         MEETING embedding (adaptive cut) to propose the candidate set; stage-2
         is a per-candidate business-context relevance gate (strict LLM
         classifier) that keeps EVERY candidate about the same specific
         business activity and rejects the rest, independent of cosine score.
      3) Classify RECURRING vs ONE-OFF from that gate-approved relevant set:
         RECURRING only when enough relevant clusters are separated across
         distinct calendar periods (repeated evidence over time). Otherwise
         ONE-OFF; prefer ONE-OFF when evidence is ambiguous. No seed-similarity
         re-check — the gate already decided relevance.
      4) RECURRING -> all relevant occurrences ordered by calendar period.
         ONE-OFF -> relevant clusters ranked by meeting-contextual relevance.
      5) If nothing is relevant, return an EMPTY list — there is no
         whole-department fallback. A meeting with an unknown department, or a
         confident department with no matching cluster, yields no pre-meeting
         brief rather than a mismatched or cross-department one. (An exact
         meeting-id anchor cluster, when present, is still always surfaced.)
    """
    active_settings = settings or get_settings()
    records = await get_email_cluster_records(settings=active_settings)
    if not records:
        logger.debug("[MEETING-MATCH] no cluster records available")
        return [], {}

    # The exact meeting occurrence cluster(s) are a guaranteed anchor:
    # today's occurrence must always appear even if nothing else clears the
    # relevance gate. The incoming meeting_id is the RAW Microsoft Graph
    # occurrence id, so match it against BOTH the stored meeting_id (which
    # for one-off clusters still holds the raw Graph id) AND the stored
    # anchored_meeting_id (which always holds the raw Graph id, even when
    # meeting_id was rewritten to a per-occurrence series tag). Collect ALL
    # such rows — a recurring series anchored to one calendar occurrence can
    # have several active period clusters, and every one of them should
    # surface for this meeting.
    anchored_clusters: list[EmailClusterRecord] = []
    if meeting_id:
        anchored_clusters = [
            r for r in records
            if getattr(r, "meeting_id", "") == meeting_id
            or getattr(r, "anchored_meeting_id", "") == meeting_id
        ]
        if anchored_clusters:
            logger.info(
                "[MEETING-MATCH] exact meeting anchor id=%s -> %d cluster(s): %s",
                meeting_id, len(anchored_clusters),
                [c.cluster_id for c in anchored_clusters],
            )

    # Resolve the embedding that represents the live meeting — the real query.
    meeting_vec = await _resolve_meeting_query_vec(meeting_id, query_vec, active_settings)

    # Resolve the meeting's own CONFIDENT department (identified and persisted
    # at meeting-history time behind its own min-score gate). It is used to
    # reduce the candidate pool to same-department clusters before stage-1
    # retrieval — the primary false-positive reduction. Empty string means
    # "no confident label": the downstream filter then leaves the pool full.
    meeting_department = await _resolve_meeting_department(meeting_id, active_settings)
    logger.info(
        "[MEETING-MATCH] meeting=%r meeting_id=%s resolved department=%r",
        meeting_title, meeting_id, meeting_department,
    )

    relevant: list[EmailClusterRecord] = []
    used_fallback = False
    # cluster_id -> stage-1 meeting-semantic similarity (0-100). Exact-anchor
    # and department-fallback rows are included WITHOUT a cosine score, so
    # they carry the sentinel _NO_SIMILARITY_SCORE below.
    scores: dict[str, float] = {}
    if meeting_vec is not None:
        # MEETING-ANCHORED retrieval: candidates are proposed by proximity to
        # the meeting, then filtered by the meeting-context relevance gate.
        relevant, used_fallback, scores = await _find_relevant_clusters(
            meeting_vec,
            records,
            active_settings,
            meeting_title=meeting_title,
            meeting_body=meeting_body,
            meeting_department=meeting_department,
        )
    else:
        logger.warning(
            "[MEETING-MATCH] meeting=%r -> no meeting embedding; cannot anchor retrieval",
            meeting_title,
        )

    # Always include the exact-id occurrence anchor(s) in the relevant set.
    # Prepend any anchored cluster not already surfaced by the relevance
    # gate, preserving order, so every active period cluster of a recurring
    # series that anchored to this meeting appears.
    if anchored_clusters:
        existing_ids = {r.cluster_id for r in relevant}
        missing_anchors = [
            c for c in anchored_clusters if c.cluster_id not in existing_ids
        ]
        if missing_anchors:
            relevant = [*missing_anchors, *relevant]

    if not relevant:
        # Department-scoped retrieval in `_find_relevant_clusters` already made
        # the authoritative decision. Whether the meeting had a confident
        # department (searched ONLY that department) or an unknown one (STOPped
        # empty), an empty relevant set here means there is genuinely nothing
        # to show. We NEVER fall back to a whole-department dump — not for a
        # confident-but-empty department, and not for an unlabeled meeting.
        # An unlabeled or no-match meeting yields no pre-meeting brief rather
        # than a mismatched or cross-department one.
        logger.info(
            "[MEETING-MATCH] meeting=%r -> nothing relevant (dept=%r); returning no clusters "
            "(no whole-department fallback)",
            meeting_title, meeting_department,
        )
        return [], {}

    # Recurrence is assessed ONLY on the gate-approved relevant set, purely
    # from how many distinct calendar periods it spans.
    is_recurring, evidence = _assess_recurrence(relevant, active_settings)
    logger.info(
        "[MEETING-MATCH] meeting=%r final_relevant_clusters=%s recurrence=%s "
        "reason=%r used_fallback=%s evidence=%s",
        meeting_title,
        [r.cluster_id for r in relevant],
        "RECURRING" if is_recurring else "ONE-OFF",
        evidence.get("reason", ""), used_fallback, evidence,
    )

    if not is_recurring:
        # ONE-OFF: relevant clusters ranked by meeting-contextual relevance.
        logger.info(
            "[MEETING-MATCH] meeting=%r -> ONE-OFF with %d relevant cluster(s) used_fallback=%s",
            meeting_title, len(relevant), used_fallback,
        )
        # Every returned cluster gets a score entry; prepended exact-anchor
        # rows that were not similarity-scored get the sentinel.
        out_scores = {
            r.cluster_id: scores.get(r.cluster_id, _NO_SIMILARITY_SCORE)
            for r in relevant
        }
        return relevant, out_scores

    # RECURRING: enough repeated business context across distinct periods.
    # Surface all relevant occurrences ordered chronologically by calendar
    # period so the ongoing activity reads as a timeline.
    series = sorted(relevant, key=_record_period_sort_key)
    logger.info(
        "[MEETING-MATCH] meeting=%r -> RECURRING clusters=%d (occurrences=%d "
        "distinct_periods=%d) used_fallback=%s",
        meeting_title, len(series),
        evidence["relevant_occurrences"], evidence["distinct_periods"], used_fallback,
    )
    out_scores = {
        r.cluster_id: scores.get(r.cluster_id, _NO_SIMILARITY_SCORE)
        for r in series
    }
    return series, out_scores


async def get_latest_summaries(storage: StorageBackend | None = None) -> list[EmailSummary]:
    active_storage = storage or get_storage()
    stored = await active_storage.summaries.list_summaries()
    return [EmailSummary.model_validate(item) for item in stored]


# ---------------------------------------------------------------------------
# CEO Meeting Notes - Action Item Extraction
# ---------------------------------------------------------------------------

async def extract_action_items_from_text(text: str, settings: Settings) -> list[dict]:
    """
    Extract action items from CEO meeting notes using LLM.
    
    This is called when CEO saves notes without pre-extracted action items.
    Returns list of dicts with keys: owner, task, deadline.
    """
    if not text or not text.strip():
        logger.debug("[ACTION-ITEMS-EXTRACT] Empty text, returning empty list")
        return []
    
    # Check if pipeline is enabled before making Ollama calls
    if not settings.bu_email_pipeline_enabled:
        logger.warning("[FLAG-CHECK] BU_EMAIL_PIPELINE_ENABLED=false; skipping action items extraction LLM call")
        return []
    
    ollama_client = OllamaClient(settings=settings)
    
    prompt = (
        "Extract action items from the following meeting notes. "
        "Return ONLY valid JSON array with objects containing: owner (name or 'Unassigned'), "
        "task (specific deliverable), deadline (date string or null). "
        "\n\nNotes:\n" + text + "\n\nReturn ONLY the JSON array, nothing else."
    )
    
    logger.info("[ACTION-ITEMS-EXTRACT] extracting from %d chars", len(text))
    
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            raw = await ollama_client.chat(prompt, system_prompt="Return valid JSON only.")
            logger.info("[ACTION-ITEMS-EXTRACT] raw response length=%d attempt=%d", len(raw), attempt + 1)
            
            # Parse JSON array
            import json as json_module
            import re as re_module
            json_match = re_module.search(r"```(?:json)?\s*\n?\s*(\[.*\])\s*\n?\s*```", raw, re_module.DOTALL)
            if json_match:
                raw = json_match.group(1).strip()
            
            data = json_module.loads(raw)
            if not isinstance(data, list):
                raise ValueError("Expected JSON array")
            
            # Validate and normalize each action item
            out: list[dict] = []
            for item in data:
                if isinstance(item, dict):
                    task = str(item.get("task", "")).strip()
                    if task:
                        out.append({
                            "owner": str(item.get("owner") or "Unassigned").strip() or "Unassigned",
                            "task": task,
                            "deadline": str(item.get("deadline") or "TBD").strip() or "TBD",
                        })
                elif isinstance(item, str) and item.strip():
                    out.append({"owner": "Unassigned", "task": item.strip(), "deadline": "TBD"})
            
            logger.info("[ACTION-ITEMS-EXTRACT] extracted %d action item(s)", len(out))
            PipelineMetricsCollector.record_llm_call(
                "action_extraction",
                tokens_in=len(prompt.split()),
                tokens_out=len(raw.split()),
            )
            PipelineMetricsCollector.record_action_extraction(len(out), 0)
            return out
            
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[ACTION-ITEMS-EXTRACT] attempt=%d failed: %s",
                attempt + 1, exc,
            )
            PipelineMetricsCollector.record_llm_call(
                "action_extraction",
                error=True,
                retry_exhausted=(attempt >= max_retries),
            )
    
    logger.warning("[ACTION-ITEMS-EXTRACT] all attempts failed, returning empty list")
    PipelineMetricsCollector.record_action_extraction(0, 1)
    return []
