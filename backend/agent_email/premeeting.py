"""
agent_email/premeeting.py
============================
Pre-Meeting Brief: read the cluster match the pipeline stored in Redis.
"""
 
from __future__ import annotations
 
import json
 
from pydantic import BaseModel, Field
 
from agent_email.schema import ClusterEmailRef
from agent_email.service import _resolve_meeting_department, match_meeting_to_clusters
from core.config import Settings, get_settings
from core.logging_config import get_agent_logger
 
logger = get_agent_logger("agent_email")
 
DEFAULT_LOOKBACK_DAYS = 14
 
 
class PreMeetingClusterCard(BaseModel):
    """One cluster surfaced for a meeting - rendered as a single card."""
 
    cluster_id: str = ""
    department: str = ""
    topic: str = ""
    meeting_start: str = ""
    summary: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    email_refs: list[ClusterEmailRef] = Field(default_factory=list)
 
 
class PreMeetingBriefResult(BaseModel):
    """
    Cluster-match result for one calendar meeting.
 
    `clusters` holds one card per matched cluster (a recurring series
    surfaces every occurrence as its own card; a non-series meeting
    surfaces a single card; a meeting that matches nothing falls back to
    the whole department). The flat `department`/`topic`/`summary`/... and
    `email_refs` fields mirror the FIRST card for backward compatibility
    with any existing single-cluster consumer.
    """
 
    meeting_title: str = ""
    meeting_time: str = ""
    matched_department_view: str | None = None
    matched: bool = False
    # Department identified for the MEETING itself (from its embedding, behind
    # the confidence gate) and used to reduce the candidate cluster pool before
    # matching. Empty string = no confident label (pool was NOT filtered).
    # Surfaced so the dept-filter decision is visible in the API response
    # without needing server logs.
    meeting_department: str = ""
    clusters: list[PreMeetingClusterCard] = Field(default_factory=list)
    # Back-compat mirror of clusters[0]:
    department: str = ""
    topic: str = ""
    summary: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    email_refs: list[ClusterEmailRef] = Field(default_factory=list)
 
 
def _split_pipe_field(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split("|") if part.strip()]
 
 
def _parse_email_refs(raw: str) -> list[ClusterEmailRef]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    refs: list[ClusterEmailRef] = []
    for item in payload:
        if isinstance(item, dict):
            refs.append(ClusterEmailRef.model_validate(item))
    return refs
 
 
async def generate_premeeting_brief_for_meeting(
    meeting_title: str,
    meeting_start_time: str = "",
    meeting_end_time: str = "",
    organizer_email: str = "",
    attendee_emails: list[str] | None = None,
    meeting_body_snippet: str = "",
    matched_department_view: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    settings: Settings | None = None,
    meeting_id: str | None = None,
) -> PreMeetingBriefResult:
    """
    Read the cluster match stored for this meeting. No search runs here.
    `lookback_days` is kept for route compatibility but unused — clusters
    are rebuilt on the scheduled pipeline cadence.
    """
    _ = (organizer_email, attendee_emails, lookback_days)
    active_settings = settings or get_settings()

    # Department identified for this meeting (empty when below the confidence
    # gate). Surfaced on the result so the pool-filter decision is observable
    # from the API alone. Best-effort: a resolution hiccup must never fail the
    # brief, so fall back to an empty label.
    try:
        meeting_department = await _resolve_meeting_department(meeting_id, active_settings)
    except Exception:  # noqa: BLE001
        meeting_department = ""
 
    clusters = await match_meeting_to_clusters(
        meeting_title,
        meeting_body_snippet,
        settings=active_settings,
        meeting_id=meeting_id,
    )
 
    if not clusters:
        logger.info("[MEETING-MATCH] premeeting brief: meeting=%r -> no cluster match", meeting_title)
        return PreMeetingBriefResult(
            meeting_title=meeting_title,
            meeting_time=f"{meeting_start_time} to {meeting_end_time}",
            matched_department_view=matched_department_view,
            meeting_department=meeting_department,
            matched=False,
        )
 
    cards = [
        PreMeetingClusterCard(
            cluster_id=cluster.cluster_id,
            department=cluster.department,
            topic=cluster.topic,
            meeting_start=cluster.meeting_start,
            summary=_split_pipe_field(cluster.summary),
            questions=_split_pipe_field(cluster.top_questions),
            action_items=_split_pipe_field(cluster.action_items),
            email_refs=_parse_email_refs(cluster.email_refs),
        )
        for cluster in clusters
    ]
    logger.info(
        "[MEETING-MATCH] premeeting brief: meeting=%r -> %d cluster card(s): %s",
        meeting_title, len(cards), [c.topic for c in cards],
    )
 
    first = cards[0]
    return PreMeetingBriefResult(
        meeting_title=meeting_title,
        meeting_time=f"{meeting_start_time} to {meeting_end_time}",
        matched_department_view=matched_department_view,
        meeting_department=meeting_department,
        matched=True,
        clusters=cards,
        # Back-compat mirror of the first card.
        department=first.department,
        topic=first.topic,
        summary=first.summary,
        questions=first.questions,
        action_items=first.action_items,
        email_refs=first.email_refs,
    )
 