"""
agent_email/meeting_anchor.py
=============================
Maps each candidate email to a specific PAST meeting occurrence so that a
recurring meeting series (e.g. April MBR, May MBR, June MBR) is split into
one cluster per occurrence instead of collapsing into a single row.

An email anchors only to meetings that have already ended before the email
arrived, and only while the email is no later than the current pipeline run.
When multiple meetings qualify, the most recent prior matching meeting wins;
semantic similarity is the tie-breaker.

A minutes-of-meeting (MoM) email is treated as a strong signal: the semantic
gate is relaxed so the MoM and its thread reliably pin to the nearest
preceding occurrence.

Every decision is logged under the `[MEETING-ANCHOR ...]` prefix.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import numpy as np

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger

from agent_email import store
from agent_email.schema import EmailCandidate
from agent_email.semantic_match import cosine_similarity

logger = get_agent_logger("agent_email")


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def _decode_embedding(raw: str) -> np.ndarray | None:
    if not raw:
        return None
    try:
        values = json.loads(raw)
        if not isinstance(values, list) or not values:
            return None
        return np.array(values, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None


def candidate_has_mom(candidate: EmailCandidate, settings: Settings | None = None) -> bool:
    """
    True when the email carries minutes/MoM, via either an attachment
    filename match or (optionally) subject/body keyword detection.
    """
    active = settings or get_settings()

    # Attachment text is populated only for MoM-filename attachments, so a
    # non-empty attachment_text already implies a MoM attachment.
    if (candidate.attachment_text or "").strip():
        return True

    msg = candidate.message
    if not active.meeting_anchor_mom_body_detection_enabled:
        return False

    keywords = active.meeting_anchor_mom_body_keywords
    if not keywords:
        return False
    haystack = f"{msg.subject or ''}\n{(msg.body_text or '')[:2000]}".lower()
    for kw in keywords:
        # Whole-token match for very short keys like "mom" to avoid
        # matching "moment"/"momentum"; substring for multi-word phrases.
        if " " in kw:
            if kw in haystack:
                return True
        elif re.search(rf"\b{re.escape(kw)}\b", haystack):
            return True
    return False


def _meeting_boundary(start_dt: datetime, end_dt: datetime | None) -> datetime:
    """Use end time when present; otherwise fall back to start."""
    return end_dt if end_dt is not None else start_dt


async def load_meeting_occurrences(
    settings: Settings | None = None,
) -> list[dict]:
    """
    Read and decode the persisted calendar occurrences from meeting_history.

    Returns a list of dicts, each with keys: ``meeting_id``, ``subject``,
    ``start`` (tz-aware datetime), ``end`` (tz-aware datetime | None),
    ``boundary`` (end or start), and ``vec`` (float32 embedding). Rows that
    cannot be decoded (no embedding or no start) are skipped.

    Shared by meeting anchoring and by meeting-aware relevance scoring so
    both consume the exact same occurrence set and decode logic.
    """
    active = settings or get_settings()
    history = await store.read_meeting_history(settings=active)
    occurrences: list[dict] = []
    for meeting_id, row in (history or {}).items():
        vec = _decode_embedding(row.get("embedding", ""))
        start_dt = _parse_iso(row.get("start_datetime", ""))
        if vec is None or start_dt is None:
            continue
        end_dt = _parse_iso(row.get("end_datetime", ""))
        occurrences.append({
            "meeting_id": meeting_id,
            "subject": row.get("subject", ""),
            "business_context": row.get("business_context", ""),
            "start": start_dt,
            "end": end_dt,
            "boundary": _meeting_boundary(start_dt, end_dt),
            "vec": vec,
        })
    return occurrences


def _email_eligible_for_meeting(
    email_dt: datetime,
    meeting_boundary: datetime,
    run_now: datetime,
    max_post_meeting_days: float,
    meeting_start: datetime | None = None,
    max_pre_meeting_days: float = 0.0,
) -> bool:
    """
    True when the email is time-eligible to anchor to this occurrence.

    Default (post-meeting only): the email must have arrived after the
    meeting ended and no later than the current pipeline run. An optional
    ``max_post_meeting_days`` cap limits how far after the meeting a
    follow-up may still anchor.

    Pre-meeting window (opt-in via ``max_pre_meeting_days`` > 0): an email
    that arrived up to that many days BEFORE the meeting starts is also
    eligible. This supports pre-meeting briefs, where mail preceding a
    meeting should inform it. When the window is 0 the behaviour is
    unchanged.
    """
    # Never anchor to information that does not exist yet at run time.
    if email_dt > run_now:
        return False

    # Post-meeting path: email arrived at/after the meeting ended.
    if email_dt >= meeting_boundary:
        if max_post_meeting_days > 0:
            cap = meeting_boundary + timedelta(days=max_post_meeting_days)
            if email_dt > cap:
                return False
        return True

    # Pre-meeting path (opt-in): email arrived before the meeting ended.
    if max_pre_meeting_days > 0 and meeting_start is not None:
        earliest = meeting_start - timedelta(days=max_pre_meeting_days)
        if email_dt >= earliest:
            return True

    return False


def _pick_best_occurrence(
    eligible: list[tuple[dict, float]],
) -> tuple[dict, float] | None:
    """
    Among semantic-qualified occurrences, pick the BEST-MATCHING meeting
    (highest cosine score), breaking ties by recency (latest meeting
    boundary).

    Ordering rationale: anchoring must be driven by what the email is
    actually about, not merely by which qualifying meeting happened most
    recently. Sorting by recency first let any email that merely cleared the
    (loose) semantic threshold collapse onto the single most-recent occurrence
    regardless of topic, cross-contaminating clusters. Cosine-first fixes that;
    recency only decides between occurrences the email matches equally well.
    """
    if not eligible:
        return None
    eligible.sort(
        key=lambda item: (item[1], item[0]["boundary"]),
        reverse=True,
    )
    return eligible[0]


async def assign_candidates_to_meetings(
    candidates: list[EmailCandidate],
    vectors: list[np.ndarray],
    settings: Settings | None = None,
    run_id: str = "",
    run_now: datetime | None = None,
) -> None:
    """
    Anchor each candidate (in place) to a past meeting occurrence.

    Sets `candidate.meeting_id`, `candidate.meeting_start`, `candidate.meeting_end`,
    and `candidate.has_mom`. Candidates that match no occurrence keep empty
    meeting fields and later cluster purely by semantics.
    """
    active = settings or get_settings()

    if not active.meeting_anchored_clustering_enabled:
        logger.info("[MEETING-ANCHOR run=%s] disabled via config; skipping", run_id)
        return

    history = await store.read_meeting_history(settings=active)
    if not history:
        logger.info(
            "[MEETING-ANCHOR run=%s] no meeting history available; all emails stay unanchored",
            run_id,
        )
        return

    pipeline_now = run_now or datetime.now(timezone.utc)
    if pipeline_now.tzinfo is None:
        pipeline_now = pipeline_now.replace(tzinfo=timezone.utc)
    else:
        pipeline_now = pipeline_now.astimezone(timezone.utc)

    # Pre-decode occurrences once.
    occurrences: list[dict] = []
    for meeting_id, row in history.items():
        vec = _decode_embedding(row.get("embedding", ""))
        start_dt = _parse_iso(row.get("start_datetime", ""))
        if vec is None or start_dt is None:
            continue
        end_dt = _parse_iso(row.get("end_datetime", ""))
        boundary = _meeting_boundary(start_dt, end_dt)
        occurrences.append({
            "meeting_id": meeting_id,
            "subject": row.get("subject", ""),
            "start": start_dt,
            "end": end_dt,
            "boundary": boundary,
            "vec": vec,
        })

    logger.info(
        "[MEETING-ANCHOR run=%s] evaluating %d email(s) against %d usable occurrence(s) "
        "run_now=%s",
        run_id, len(candidates), len(occurrences), pipeline_now.isoformat(),
    )
    if not occurrences:
        return

    sem_threshold = float(active.meeting_anchor_semantic_threshold)
    mom_is_strong = active.meeting_anchor_mom_is_strong
    mom_threshold = float(active.meeting_anchor_mom_semantic_threshold)
    max_post_days = float(active.meeting_anchor_max_post_meeting_days or 0.0)
    max_pre_days = float(active.meeting_anchor_max_pre_meeting_days or 0.0)

    anchored = 0
    unanchored = 0

    for idx, candidate in enumerate(candidates):
        msg = candidate.message
        email_dt = msg.received_at
        if email_dt is not None and email_dt.tzinfo is None:
            email_dt = email_dt.replace(tzinfo=timezone.utc)
        elif email_dt is not None:
            email_dt = email_dt.astimezone(timezone.utc)

        has_mom = candidate_has_mom(candidate, settings=active)
        candidate.has_mom = has_mom
        effective_threshold = mom_threshold if (has_mom and mom_is_strong) else sem_threshold

        email_vec = vectors[idx] if idx < len(vectors) else None
        eligible: list[tuple[dict, float]] = []

        for occ in occurrences:
            if email_dt is None:
                continue
            if not _email_eligible_for_meeting(
                email_dt, occ["boundary"], pipeline_now, max_post_days,
                meeting_start=occ["start"], max_pre_meeting_days=max_pre_days,
            ):
                continue
            cosine = cosine_similarity(email_vec, occ["vec"]) if email_vec is not None else 0.0
            if cosine < effective_threshold:
                continue
            eligible.append((occ, cosine))

        picked = _pick_best_occurrence(eligible)
        if picked is None:
            unanchored += 1
            logger.info(
                "[MEETING-ANCHOR run=%s] email idx=%d subject=%r mom=%s -> UNANCHORED "
                "(no time-eligible occurrence clearing threshold=%.2f)",
                run_id, idx, msg.subject, has_mom, effective_threshold,
            )
            continue

        occ, cosine = picked
        candidate.meeting_id = occ["meeting_id"]
        candidate.meeting_start = occ["start"].isoformat()
        candidate.meeting_end = occ["boundary"].isoformat()
        anchored += 1
        logger.info(
            "[MEETING-ANCHOR run=%s] email idx=%d subject=%r mom=%s -> occurrence id=%s "
            "meeting=%r start=%s end=%s cosine=%.3f",
            run_id, idx, msg.subject, has_mom, occ["meeting_id"], occ["subject"],
            occ["start"].isoformat(), occ["boundary"].isoformat(), cosine,
        )

    logger.info(
        "[MEETING-ANCHOR run=%s] done: %d anchored, %d unanchored",
        run_id, anchored, unanchored,
    )
