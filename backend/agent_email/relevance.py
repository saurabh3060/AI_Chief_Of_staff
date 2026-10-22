"""
agent_email/relevance.py
========================
Specific-meeting business-context relevance gate for fetched mailbox mail.

The primary decision is a strict structured-context comparison against each
individual in-window meeting. Generic department or corporate similarity is
not a pass signal. A relevance-only cosine score and adaptive cutoff are used
only after the primary meeting match has succeeded.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger
from core.ollama_client.client import OllamaClient
from core.text_match import strip_email_noise
from core.workers import run_in_worker_pool

from agent_email.prompts import (
    RELEVANCE_CONTEXT_SYSTEM_PROMPT,
    SPECIFIC_MEETING_RELEVANCE_SYSTEM_PROMPT,
    build_relevance_context_prompt,
    parse_relevance_context,
    render_relevance_context,
)
from agent_email.pipeline_cache import chat_with_cache
from agent_email.pipeline_metrics import PipelineMetricsCollector
from agent_email.schema import EmailCandidate, EmailMessage
from agent_email.semantic_match import cosine_similarity, embed_text

logger = get_agent_logger("agent_email")

# Kept in sync with service.py's LOW_CONTEXT_PREFIX. A meeting whose stored
# business_context carries this sentinel had no usable invite-body facts;
# the LOW_CONTEXT rule keeps it in the candidate pool (never dropped) but
# compares emails against its literal subject/title instead of treating the
# (never-persisted) fabricated context as real fact.
LOW_CONTEXT_PREFIX = "[LOW_CONTEXT]"


def _is_low_context_meeting(meeting: dict) -> bool:
    return meeting.get("business_context", "").strip().startswith(LOW_CONTEXT_PREFIX)


def _message_text(message: EmailMessage) -> str:
    parts = [
        message.subject or "",
        strip_email_noise(message.body_text or ""),
        " ".join(message.to_recipients),
        " ".join(message.cc_recipients),
        message.from_address or "",
    ]
    return " ".join(part for part in parts if part).strip()


def _current_meetings(
    occurrences: list[dict] | None,
    window_days: float,
    run_now: datetime,
) -> list[dict]:
    """Return usable occurrences in the configured window.

    LOW_CONTEXT rule: a meeting whose business_context is empty or carries
    the LOW_CONTEXT sentinel is still kept (it is never excluded from the
    candidate pool) — only a missing embedding and a completely missing
    subject disqualify an occurrence, since matching can still proceed on
    the meeting's literal subject/title.
    """
    meetings: list[dict] = []
    for occurrence in occurrences or []:
        if occurrence.get("vec") is None:
            continue
        if not occurrence.get("business_context", "").strip() and not occurrence.get("subject", "").strip():
            continue
        if window_days > 0:
            when = occurrence.get("boundary") or occurrence.get("start")
            if when is not None:
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                if abs((when - run_now).total_seconds()) > window_days * 86400.0:
                    continue
        meetings.append(occurrence)
    return meetings


def _otsu_threshold(scores: list[float], bins: int = 64) -> float | None:
    """Find the natural score-distribution break using Otsu's method."""
    vals = [float(score) for score in scores]
    if len(vals) < 3:
        return None
    low, high = min(vals), max(vals)
    if high - low < 1e-6:
        return None

    bins = max(2, int(bins))
    counts = [0] * bins
    width = (high - low) / bins
    for value in vals:
        index = min(int((value - low) / width), bins - 1)
        counts[index] += 1

    total = float(len(vals))
    centres = [low + (index + 0.5) * width for index in range(bins)]
    sum_all = sum(counts[index] * centres[index] for index in range(bins))
    best_between = -1.0
    best_threshold: float | None = None
    background_weight = 0.0
    background_sum = 0.0
    for index in range(bins):
        background_weight += counts[index]
        if background_weight == 0.0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0.0:
            break
        background_sum += counts[index] * centres[index]
        background_mean = background_sum / background_weight
        foreground_mean = (sum_all - background_sum) / foreground_weight
        between = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if between > best_between:
            best_between = between
            best_threshold = low + (index + 1) * width
    return best_threshold


def compute_adaptive_cutoff(
    scores: list[float],
    floor: float = 0.0,
    strictness: float = 1.0,
) -> tuple[float, str]:
    """Derive a secondary cutoff from the primary-matched score distribution."""
    threshold = _otsu_threshold(scores)
    if threshold is None:
        return max(0.0, floor), "insufficient-distribution"

    threshold -= 10.0
    
    if strictness and strictness > 0:
        high = max(scores)
        threshold += max(0.0, min(1.0, strictness)) * max(0.0, high - threshold)
        reason = "otsu+strictness"
    else:
        reason = "otsu"
    if threshold < floor:
        return floor, f"{reason}<floor"
    return threshold, reason


async def _extract_email_context(
    message: EmailMessage,
    settings: Settings,
    attachment: str = "",
    run_id: str = "",
) -> dict:
    """One structured read of the email, including minutes when they were opened.

    This is now the SINGLE structured extraction for an email — clustering
    reuses its output instead of running a second topic-tag LLM call — so it
    retries a transient LLM/parse failure (mirroring the cluster-synthesis
    retry) rather than failing the email on the first hiccup. A non-empty
    business context is required to count as success; an all-empty parse is
    retried too, since that degenerate result would otherwise drop the email
    at the gate AND leave clustering with nothing to embed.
    """
    prompt = build_relevance_context_prompt(
        message.subject,
        message.body_text or "",
        attachment,
    )
    client = OllamaClient(settings=settings)
    max_retries = 2
    last_error: Exception | None = None
    use_cache = True  # turned off once a cached reply fails validation (a retry must re-ask the model)
    for attempt in range(max_retries + 1):
        try:
            raw, ticket = await chat_with_cache(
                client,
                settings,
                task="relevance_context",
                prompt=prompt,
                system_prompt=RELEVANCE_CONTEXT_SYSTEM_PROMPT,
                num_predict=1024,
                use_cache=use_cache,
            )
            parsed = parse_relevance_context(raw)
            has_content = bool(
                parsed.get("business_objective")
                or parsed.get("topics")
                or parsed.get("entities")
                or parsed.get("actions")
            )
            PipelineMetricsCollector.record_llm_call(
                "relevance_context",
                tokens_in=len(prompt.split()),
                tokens_out=len(raw.split()),
                error=not has_content,
                retry_exhausted=(not has_content and attempt >= max_retries),
            )
            if has_content:
                await ticket.commit(settings, raw)  # only a validated reply is cached
            elif ticket.hit:
                use_cache = False
            if has_content or attempt >= max_retries:
                return parsed
            logger.info(
                "[POOL-EXTRACT run=%s] subject=%r attempt=%d produced empty context; retrying",
                run_id, message.subject, attempt + 1,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            use_cache = False
            PipelineMetricsCollector.record_llm_call(
                "relevance_context",
                error=True,
                retry_exhausted=(attempt >= max_retries),
            )
            logger.warning(
                "[POOL-EXTRACT run=%s] subject=%r attempt=%d failed: %s",
                run_id, message.subject, attempt + 1, exc,
            )
            if attempt >= max_retries:
                raise
    # Unreachable in practice (loop either returns or raises), but keep a
    # defensive raise so the type checker knows this never falls through.
    if last_error is not None:
        raise last_error
    return parse_relevance_context("")


def _is_verbatim_example_echo(raw_keys: list[str], reason: str) -> bool:
    """
    Detect the specific failure mode where the model copies the prompt's own
    output-contract example instead of making an independent decision (e.g.
    always returning key "M2" with the literal reason text "Brief reason").
    Used only for diagnostics/logging; does not change matching behavior.
    """
    return list(raw_keys) == ["M2"] and reason.strip().lower() == "brief reason"


def _parse_relevant_meeting_keys(
    raw: str,
    allowed_keys: set[str],
) -> tuple[list[str], str, bool, list[str]]:
    """Validate the matcher response as the exact two-field JSON contract.

    Returns (matched_keys, reason, is_valid, unknown_keys). unknown_keys is
    populated (and the response rejected) when the model returns a key that
    was not part of the candidate set actually offered to it.
    """
    text = (raw or "").strip()
    if not text:
        return [], "empty relevance decision", False, []
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return [], "invalid relevance decision", False, []
    if not isinstance(data, dict) or set(data) != {"relevant_meeting_keys", "reason"}:
        return [], "invalid relevance decision schema", False, []

    raw_keys = data["relevant_meeting_keys"]
    reason = data["reason"]
    if (
        not isinstance(raw_keys, list)
        or not all(isinstance(value, str) for value in raw_keys)
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        return [], "invalid relevance decision schema", False, []

    matched: list[str] = []
    unknown: list[str] = []
    for meeting_key in raw_keys:
        if meeting_key not in allowed_keys:
            unknown.append(meeting_key)
            continue
        if meeting_key not in matched:
            matched.append(meeting_key)
    if unknown:
        return [], f"unknown meeting key(s): {unknown}", False, unknown

    reason = reason.strip()
    if not matched and reason != "No specific meeting match":
        return [], "invalid no-match reason", False, []
    return matched, reason, True, []


async def _match_specific_meetings(
    email_context: str,
    meetings: list[dict],
    settings: Settings,
    run_id: str = "",
    subject: str = "",
    email_id: str = "",
) -> tuple[list[str], str]:
    """Require valid strict JSON identifying exact context-matched meetings."""
    key_to_id = {
        f"M{index}": str(meeting["meeting_id"])
        for index, meeting in enumerate(meetings, start=1)
    }
    id_to_key = {meeting_id: key for key, meeting_id in key_to_id.items()}
    meeting_payload = []
    low_context_keys: list[str] = []
    for meeting in meetings:
        key = id_to_key[str(meeting["meeting_id"])]
        meeting_subject = meeting.get("subject", "")
        if _is_low_context_meeting(meeting):
            low_context_keys.append(key)
            meeting_payload.append(
                {
                    "meeting_key": key,
                    "subject": meeting_subject,
                    "business_context": "LOW_CONTEXT: no invite-body facts available; "
                    "match this meeting on its literal subject/title text only.",
                }
            )
        else:
            meeting_payload.append(
                {
                    "meeting_key": key,
                    "subject": meeting_subject,
                    "business_context": meeting.get("business_context", ""),
                }
            )
    candidate_keys = list(key_to_id)
    low_context_note = (
        "\nSome candidates are marked LOW_CONTEXT: their business_context is a placeholder, not "
        "real fact. For those, decide relevance by comparing the email's content against the "
        "meeting's subject/title text directly (exact or clearly synonymous topic/entity match), "
        "not against invented context."
        if low_context_keys
        else ""
    )
    base_prompt = (
        "EMAIL BUSINESS CONTEXT:\n"
        f"{email_context}\n\n"
        "CANDIDATE MEETINGS (evaluate the email against EACH meeting_key below, independently, on "
        "its own facts — the keys below are the ONLY valid keys for this request):\n"
        f"{json.dumps(meeting_payload, ensure_ascii=False)}\n"
        f"{low_context_note}\n\n"
        "OUTPUT CONTRACT (write your own reason; do not copy this shape's wording):\n"
        '{"relevant_meeting_keys": [<zero or more of the exact meeting_key values listed above>], '
        '"reason": "<one sentence you write, naming the concrete matching topic/entity/action, or '
        'stating there is no specific meeting match>"}\n'
        "Return only the one-line JSON object."
    )
    client = OllamaClient(settings=settings)
    validation_reason = "invalid relevance decision"
    use_cache = True  # turned off once a cached reply fails validation (a retry must re-ask the model)
    for attempt in range(1, 3):
        prompt = base_prompt
        if attempt > 1:
            prompt += (
                "\nYour previous response was empty, malformed, or referenced a meeting_key not in "
                "the candidate list above. Retry now using exactly the two-field JSON contract, only "
                "the candidate meeting_key values above, with no markdown or surrounding text."
            )
        try:
            raw, ticket = await chat_with_cache(
                client,
                settings,
                task="meeting_relevance_match",
                prompt=prompt,
                system_prompt=SPECIFIC_MEETING_RELEVANCE_SYSTEM_PROMPT,
                num_predict=512,
                use_cache=use_cache,
            )
            PipelineMetricsCollector.record_llm_call(
                "meeting_relevance_match",
                tokens_in=len(prompt.split()),
                tokens_out=len(raw.split()),
            )
        except Exception:
            PipelineMetricsCollector.record_llm_call("meeting_relevance_match", error=True)
            raise
        logger.info(
            "[POOL-RELEVANCE-LLM run=%s] email_id=%r subject=%r attempt=%d candidate_meetings=%s raw=%r",
            run_id,
            email_id,
            subject,
            attempt,
            candidate_keys,
            raw,
        )
        matched_keys, reason, valid, unknown_keys = _parse_relevant_meeting_keys(raw, set(key_to_id))
        if unknown_keys:
            logger.warning(
                "[POOL-RELEVANCE-LLM run=%s] email_id=%r subject=%r attempt=%d returned "
                "unknown meeting key(s) not in candidate set: %s (candidates=%s)",
                run_id, email_id, subject, attempt, unknown_keys, candidate_keys,
            )
        if valid:
            if _is_verbatim_example_echo(matched_keys, reason):
                logger.warning(
                    "[POOL-RELEVANCE-LLM run=%s] email_id=%r subject=%r response matches the "
                    "prompt's own example verbatim (key=%s reason=%r); treating as untrusted",
                    run_id, email_id, subject, matched_keys, reason,
                )
                validation_reason = "response matched prompt example verbatim"
                use_cache = False
                # Retry once; if it happens again on the final attempt, fail closed
                # rather than trust a decision that looks copied, not reasoned.
                if attempt < 2:
                    continue
                return [], validation_reason
            selected_ids = [key_to_id[key] for key in matched_keys]
            logger.info(
                "[POOL-RELEVANCE-LLM run=%s] email_id=%r subject=%r -> selected_meeting_id=%s reason=%r",
                run_id, email_id, subject, selected_ids or "none", reason,
            )
            await ticket.commit(settings, raw)  # only an accepted decision is cached
            return selected_ids, reason
        validation_reason = reason
        use_cache = False
        logger.warning(
            "[POOL-RELEVANCE-LLM run=%s] subject=%r attempt=%d malformed=%r%s",
            run_id,
            subject,
            attempt,
            reason,
            "; retrying" if attempt == 1 else "; retries exhausted",
        )
    return [], validation_reason


async def score_candidate_pool(
    messages: list[EmailMessage],
    settings: Settings | None = None,
    run_id: str = "",
    meeting_occurrences: list[dict] | None = None,
    run_now: datetime | None = None,
    attachment_by_id: dict[str, str] | None = None,
) -> list[EmailCandidate]:
    """
    Keep mail only when its structured business context matches at least one
    specific meeting. The later cosine/adaptive gate is secondary and cannot
    make an email pass the primary context decision.
    """
    active = settings or get_settings()
    if not messages:
        return []

    now = run_now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    window_days = float(getattr(active, "email_relevance_meeting_window_days", 0.0) or 0.0)
    meetings = _current_meetings(meeting_occurrences, window_days, now)
    if not meetings:
        logger.warning(
            "[POOL-FILTER run=%s] no context-bearing specific meetings; failing closed",
            run_id,
        )
        return []

    logger.info(
        "[POOL-SCORE run=%s] specific-meeting context gate ON: %d meeting(s) window_days=%.1f",
        run_id,
        len(meetings),
        window_days,
    )

    async def evaluate(_index: int, message: EmailMessage):
        try:
            attachment = (attachment_by_id or {}).get(message.id or "", "")
            parsed = await _extract_email_context(message, active, attachment, run_id=run_id)
            email_context = render_relevance_context(parsed)
            if not email_context:
                return message, "", [], "empty email business context", None, parsed, None, attachment
            matched_ids, reason = await _match_specific_meetings(
                email_context,
                meetings,
                active,
                run_id=run_id,
                subject=message.subject,
                email_id=message.id or "",
            )
            if not matched_ids:
                return message, email_context, [], reason, None, parsed, None, attachment
            email_vector = await embed_text(email_context[:8000], active)
            meeting_by_id = {str(meeting["meeting_id"]): meeting for meeting in meetings}
            score = max(
                max(0.0, min(1.0, cosine_similarity(email_vector, meeting_by_id[mid]["vec"])))
                * 100.0
                for mid in matched_ids
            )
            return message, email_context, matched_ids, reason, score, parsed, email_vector, attachment
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[POOL-FILTER run=%s] context evaluation failed subject=%r: %s; failing closed",
                run_id,
                message.subject,
                exc,
            )
            # The exception is logged above; the returned reason is constant text.
            return message, "", [], "context evaluation failed", None, {}, None, ""

    # Fan the per-email extraction + meeting-match out over the shared bounded
    # worker pool (default 1 worker). This gives relevance extraction the same
    # explicit, tunable concurrency + per-item failure isolation the cluster
    # synthesis stage uses, instead of scheduling every email at once via
    # gather. The real cap on concurrent Ollama calls is still the
    # MAX_CONCURRENT_OLLAMA_CALLS semaphore, so the worker count only governs
    # how many emails are in flight. `evaluate` never raises (it fails each
    # email closed and returns a no-match tuple), so every slot has a value;
    # a defensive no-match tuple covers the should-not-happen pool-error case.
    relevance_workers = max(1, int(getattr(active, "email_pipeline_relevance_workers", 1) or 1))
    pool_results = await run_in_worker_pool(
        list(messages),
        evaluate,
        workers=relevance_workers,
        label="pool-relevance",
    )
    results = [
        wr.value
        if (wr.ok and wr.value is not None)
        else (messages[wr.index], "", [], "worker pool error", None, {}, None, "")
        for wr in pool_results
    ]

    # Diagnostic-only: flag when every matched email collapsed onto the same
    # single meeting, which is the signature of the matcher defaulting to one
    # key instead of evaluating each email independently. Does not alter
    # which emails pass or how they are scored.
    matched_id_sets = [
        tuple(sorted(matched_ids))
        for _message, _context, matched_ids, _reason, score, _parsed, _vec, _attachment in results
        if score is not None and matched_ids
    ]
    if len(matched_id_sets) >= 2 and len(meetings) > 1 and len(set(matched_id_sets)) == 1:
        logger.warning(
            "[POOL-RELEVANCE-LLM run=%s] every matched email (%d) resolved to the same "
            "meeting_id=%s out of %d candidate meeting(s); possible relevance-matcher collapse",
            run_id, len(matched_id_sets), matched_id_sets[0], len(meetings),
        )

    primary_passed: list[tuple[float, EmailMessage, list[str], dict, np.ndarray | None, str]] = []
    for message, _context, matched_ids, reason, score, parsed, vector, attachment in results:
        if score is None:
            logger.info(
                "[POOL-FILTER run=%s] DROP primary-specific-context subject=%r reason=%r",
                run_id,
                message.subject,
                reason,
            )
            continue
        primary_passed.append((score, message, matched_ids, parsed, vector, attachment))
        logger.info(
            "[POOL-SCORE run=%s] specific_score=%.1f matched_meeting_ids=%s subject=%r reason=%r",
            run_id,
            score,
            matched_ids,
            message.subject,
            reason,
        )

    if not primary_passed:
        logger.info(
            "[POOL-FILTER run=%s] scored=%d primary_specific_matches=0 pool=0",
            run_id,
            len(messages),
        )
        return []

    primary_passed.sort(key=lambda row: row[0], reverse=True)
    scores = [row[0] for row in primary_passed]
    floor = float(getattr(active, "email_relevance_min_score", 0.0) or 0.0)
    if bool(getattr(active, "email_relevance_adaptive_cutoff_enabled", True)):
        strictness = float(getattr(active, "email_relevance_adaptive_strictness", 0.0) or 0.0)
        cutoff, reason = compute_adaptive_cutoff(scores, floor=floor, strictness=strictness)
    else:
        cutoff, reason = floor, "floor-only"

    kept = [row for row in primary_passed if row[0] >= cutoff]
    for score, message, matched_ids, _parsed, _vector, _attachment in primary_passed:
        if score < cutoff:
            logger.info(
                "[POOL-FILTER run=%s] DROP secondary score=%.1f < cutoff=%.1f (%s) "
                "matched_meeting_ids=%s subject=%r",
                run_id,
                score,
                cutoff,
                reason,
                matched_ids,
                message.subject,
            )

    top_k = max(1, int(getattr(active, "email_pipeline_candidate_pool_size", 20) or 20))
    top = kept[:top_k]
    logger.info(
        "[POOL-FILTER run=%s] scored=%d primary_specific_matches=%d "
        "kept_after_secondary=%d pool=%d (cutoff=%.1f rule=%s floor=%.1f top_k=%d)",
        run_id,
        len(messages),
        len(primary_passed),
        len(kept),
        len(top),
        cutoff,
        reason,
        floor,
        top_k,
    )

    candidates: list[EmailCandidate] = []
    for score, message, matched_ids, parsed, vector, attachment in top:
        parsed = parsed or {}
        # Carry the SINGLE structured extraction onto the candidate so the
        # clustering stage reuses it instead of running a second topic-tag
        # LLM call. topic_key is derived here (same helper the old topic-tag
        # stage used) so the display/CSV value is unchanged.
        from agent_email.prompts import topic_key_from_semantics
        candidate = EmailCandidate(
            message=message,
            relevance_score=round(score, 2),
            token_score=0.0,
            semantic_score=round(score, 2),
            attachment_text=attachment or "",
            matched_meeting_ids=list(matched_ids),
            business_objective=parsed.get("business_objective", "") or "",
            topics=list(parsed.get("topics", []) or []),
            entities=list(parsed.get("entities", []) or []),
            actions=list(parsed.get("actions", []) or []),
            subtopics=list(parsed.get("subtopics", []) or []),
            decisions=list(parsed.get("decisions", []) or []),
            context=parsed.get("context", "") or "",
            topic_key=topic_key_from_semantics(parsed),
        )
        # Preserve the gate embedding so clustering can reuse it instead of
        # re-embedding the same business context. Stored as the JSON string
        # form the candidate/record contract already uses.
        if vector is not None:
            try:
                candidate.embedding_json = json.dumps(
                    vector.tolist() if hasattr(vector, "tolist") else list(vector)
                )
            except (TypeError, ValueError):
                candidate.embedding_json = ""
        candidates.append(candidate)
    logger.info(
        "[POOL-EXTRACT run=%s] carried structured semantics onto %d candidate(s) "
        "for reuse by clustering (no second extraction)",
        run_id, len(candidates),
    )
    return candidates
