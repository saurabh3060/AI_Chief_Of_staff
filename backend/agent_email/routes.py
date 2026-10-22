"""
agent_email/routes.py
========================
FastAPI routes for the Email & Summary agent.
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from agent_email.premeeting import DEFAULT_LOOKBACK_DAYS, generate_premeeting_brief_for_meeting
from agent_email.schema import EmailSummarizerTestRequest, EmailSummarizerTestResponse
from core.graph_client.url_safety import UnsafeGraphUrlError
from agent_email.service import (
    get_email_cluster_records,
    get_latest_summaries,
    get_raw_messages,
    run_email_cluster_pipeline,
    run_email_summarizer_test,
)
from auth.dependencies import get_bearer_token
from core.config import get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_email")

router = APIRouter(prefix="/agents/email", tags=["email"])

premeeting_brief_router = APIRouter(prefix="/agents/premeeting-brief", tags=["premeeting-brief"])


@premeeting_brief_router.get("/{meeting_id}")
async def get_premeeting_brief_for_meeting(
    meeting_id: str,
    lookback_days: int = Query(default=DEFAULT_LOOKBACK_DAYS, ge=1, le=90),
    bearer_token: str | None = Depends(get_bearer_token),
):
    """Return the cluster match for a specific meeting on today's calendar."""
    settings = get_settings()

    from agent_calendar.service import get_todays_events

    today = await get_todays_events(bearer_token=bearer_token, settings=settings)
    meeting = next((event for event in today.events if event.id == meeting_id), None)
    if meeting is None:
        raise HTTPException(status_code=404, detail=f"No meeting with id '{meeting_id}' found on today's calendar.")

    result = await generate_premeeting_brief_for_meeting(
        meeting_title=meeting.subject,
        meeting_start_time=meeting.start.isoformat() if meeting.start else "",
        meeting_end_time=meeting.end.isoformat() if meeting.end else "",
        organizer_email=meeting.organizer_email or "",
        attendee_emails=meeting.attendee_emails,
        meeting_body_snippet=meeting.body_snippet or "",
        matched_department_view=meeting.matched_view,
        lookback_days=lookback_days,
        settings=settings,
        meeting_id=meeting.id,
    )
    return {"brief": result}


@router.get("/raw-messages")
async def get_raw_messages_route(bearer_token: str | None = Depends(get_bearer_token)):
    """Debugging endpoint: raw Graph API message list."""
    settings = get_settings()
    messages = await get_raw_messages(bearer_token=bearer_token, settings=settings)
    return {"count": len(messages), "messages": messages}


@router.get("/summary")
async def get_summary(bearer_token: str | None = Depends(get_bearer_token)):
    """Return the most recently generated thread summary (read-only)."""
    summaries = await get_latest_summaries()
    if not summaries:
        return {"summary": None, "message": "No mail found to summarize."}
    summary = max(summaries, key=lambda s: s.generated_at)
    return {
        "summary": {
            "thread_id": summary.thread_id,
            "subject": summary.subject,
            "generated_at": summary.generated_at.isoformat(),
            "summary_text": summary.summary_text,
            "action_items": summary.action_items,
            "top_questions": summary.top_questions,
            "email_summaries": [
                {
                    "message_id": es.message_id,
                    "subject": es.subject,
                    "from_name": es.from_name,
                    "from_address": es.from_address,
                    "received_at": es.received_at.isoformat(),
                    "summary_text": es.summary_text,
                    "action_items": es.action_items,
                    "top_questions": es.top_questions,
                }
                for es in summary.email_summaries
            ],
            "action_candidates": [
                {"text": ac.text, "source_message_id": ac.source_message_id}
                for ac in summary.action_candidates
            ],
        }
    }


@router.post("/bu/summarizer-test", response_model=EmailSummarizerTestResponse)
async def test_bu_email_summarizer(
    request: EmailSummarizerTestRequest,
    bearer_token: str | None = Depends(get_bearer_token),
):
    """Run the production email summarizer prompt against one or N Graph messages."""
    try:
        return await run_email_summarizer_test(request, bearer_token=bearer_token)
    except UnsafeGraphUrlError as exc:
        # The user-supplied message_id failed opaque-path-segment validation
        # (CWE-918 guard). Surface it as a 400 Bad Request rather than a 500
        # so a crafted id is rejected cleanly without ever building a URL.
        # The validator's message can echo the rejected value, so it is logged only.
        logger.exception("Email summarizer test rejected an unsafe message_id")
        raise HTTPException(status_code=400, detail="Invalid message_id.") from exc
    except RuntimeError as exc:
        logger.exception("Email summarizer test could not read the requested messages")
        raise HTTPException(status_code=403, detail="Access to the requested messages is not available.") from exc


@router.get("/clusters")
async def list_email_clusters(department: str | None = None):
    """Return precomputed email cluster records, optionally filtered by department."""
    from agent_email.pipeline_freshness import is_pipeline_fresh, pipeline_ready_at

    records = await get_email_cluster_records(department=department)
    return {
        "count": len(records),
        "records": [r.model_dump() for r in records],
        "pipeline_fresh": True,
        "pipeline_ready_at": pipeline_ready_at(),
    }


@router.post("/clusters/refresh")
async def refresh_email_clusters(bearer_token: str | None = Depends(get_bearer_token)):
    """Manually run the email cluster synthesis pipeline once."""
    written = await run_email_cluster_pipeline(bearer_token=bearer_token)
    return {"clusters_written": written}


@router.get("/summaries")
async def list_summaries():
    """Return every summary currently persisted by this agent."""
    summaries = await get_latest_summaries()
    return {
        "summaries": [
            {
                "thread_id": s.thread_id,
                "subject": s.subject,
                "generated_at": s.generated_at.isoformat(),
                "summary_text": s.summary_text,
                "action_items": s.action_items,
                "top_questions": s.top_questions,
                "email_summaries": [
                    {
                        "message_id": es.message_id,
                        "subject": es.subject,
                        "from_name": es.from_name,
                        "from_address": es.from_address,
                        "received_at": es.received_at.isoformat(),
                        "summary_text": es.summary_text,
                        "action_items": es.action_items,
                        "top_questions": es.top_questions,
                    }
                    for es in s.email_summaries
                ],
                "action_candidates": [
                    {"text": ac.text, "source_message_id": ac.source_message_id}
                    for ac in s.action_candidates
                ],
            }
            for s in summaries
        ]
    }
