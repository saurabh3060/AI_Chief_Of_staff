"""
documents/sync_routes.py
===========================
FastAPI routes for the OneDrive -> document library sync feature.

Endpoints:
- POST /documents/sync          — trigger an immediate sync (runs in background)
- POST /documents/sync/file     — re-sync a single file by item id
- GET  /documents/sync/status   — last sync result + current state
- GET  /documents/sync/files    — list of all tracked files from sync state
- GET  /documents/sync/config   — current sync configuration (no secrets)
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from auth.dependencies import get_bearer_token
from core.config import get_settings
from core.logging_config import get_agent_logger, sanitize_log_value
from documents import sharepoint_sync, sync_state

logger = get_agent_logger("documents")

router = APIRouter(prefix="/documents/sync", tags=["documents-sync"])


# ── Request / response models ────────────────────────────────

class SyncRequest(BaseModel):
    force: bool = Field(
        default=False,
        description="Re-download and re-ingest ALL files, ignoring sync state.",
    )


class SingleFileSyncRequest(BaseModel):
    drive_id: str = Field(
        ...,
        description="Drive id (from /documents/sync/files or initial sync).",
    )
    item_id: str = Field(
        ...,
        description="Drive item id to re-sync.",
    )


# ── Routes ────────────────────────────────────────────────────

@router.post("", status_code=202)
async def trigger_sync(
    background_tasks: BackgroundTasks,
    bearer_token: str | None = Depends(get_bearer_token),
    body: SyncRequest = SyncRequest(),
):
    """Trigger an immediate SharePoint -> document library sync.

    In delegated mode, the user must be signed in (session cookie is
    used automatically). In app-only mode, no session is needed.

    Returns 202 Accepted immediately; the sync runs in the background.
    Check ``GET /documents/sync/status`` for progress and results.
    """
    settings = get_settings()

    if not settings.onedrive_sync_enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                "OneDrive sync is not enabled. "
                "Set ONEDRIVE_SYNC_ENABLED=true in .env."
            ),
        )

    if sharepoint_sync.is_sync_running():
        raise HTTPException(
            status_code=409,
            detail="A sync is already running. Check GET /documents/sync/status.",
        )

    # Run the sync as a background task so the HTTP response returns immediately.
    background_tasks.add_task(_run_sync_background, body.force, bearer_token)

    return {
        "status": "accepted",
        "message": "Sync started in background.",
        "force": body.force,
    }


@router.post("/file")
async def sync_single_file(
    body: SingleFileSyncRequest,
    bearer_token: str | None = Depends(get_bearer_token),
):
    """Re-sync a single file by its drive_id + item_id.

    Runs synchronously (waits for download + ingestion to complete).
    Requires a signed-in session in delegated mode.
    """
    settings = get_settings()

    if not settings.onedrive_sync_enabled:
        raise HTTPException(status_code=503, detail="OneDrive sync is not enabled.")

    if settings.is_local_mail_mode:
        raise HTTPException(
            status_code=400,
            detail="Single-file sync is not supported in local mode.",
        )

    bearer_token_val = bearer_token

    try:
        result = await sharepoint_sync.sync_single_file(
            drive_id=body.drive_id,
            item_id=body.item_id,
            bearer_token=bearer_token_val,
        )
    except HTTPException:
        raise
    except Exception as exc:
        # Log the full error server-side; return a generic message to the
        # client so internal details / stack traces are not exposed (CWE-209).
        logger.error("[SHAREPOINT-SYNC] Single-file sync failed: %s", exc)
        raise HTTPException(status_code=500, detail="Single-file sync failed.")

    # sync_single_file() embeds the raw exception string in result["error"]
    # on failure. Strip it before returning so internal details / stack
    # traces are never exposed to the client (CWE-209); the detail is
    # already logged inside sync_single_file / the sync pipeline.
    if isinstance(result, dict) and result.get("status") == "failed":
        # Sanitize the user-controlled item_id before logging so CR/LF or
        # other control characters cannot forge log entries (CWE-117).
        logger.error(
            "[SHAREPOINT-SYNC] Single-file sync failed for item %s: %s",
            sanitize_log_value(body.item_id),
            sanitize_log_value(result.get("error")),
        )
        result = {
            "status": "failed",
            "name": result.get("name", ""),
            "error": "Single-file sync failed.",
        }
    return result


@router.get("/status")
async def sync_status():
    """Return the current sync state and most recent run result.

    Always returns 200 — the ``status`` field indicates whether sync
    is enabled, idle, running, or what the last run's outcome was.
    """
    settings = get_settings()
    state = await sync_state.load_state()

    return {
        "enabled": settings.onedrive_sync_enabled,
        "running": sharepoint_sync.is_sync_running(),
        "sharepoint_folders": len(settings.sharepoint_sites_list),
        "schedule": (
            f"{settings.onedrive_sync_schedule_hour:02d}:"
            f"{settings.onedrive_sync_schedule_minute:02d} IST daily"
        ),
        "last_sync_at": state.get("last_sync_at"),
        "last_sync_status": state.get("last_sync_status"),
        "last_sync_summary": state.get("last_sync_summary"),
        "tracked_files": len(state.get("files", {})),
        "last_run_result": sharepoint_sync.last_sync_result(),
    }


@router.get("/files")
async def sync_files():
    """Return the list of all tracked files from sync state."""
    state = await sync_state.load_state()
    files = sync_state.get_tracked_files(state)
    return {
        "count": len(files),
        "files": files,
    }


@router.get("/config")
async def sync_config():
    """Return the current OneDrive sync configuration (no secrets)."""
    settings = get_settings()
    sites = settings.sharepoint_sites_list
    return {
        "enabled": settings.onedrive_sync_enabled,
        "sharepoint_folders": len(sites),
        "sharepoint_sites_configured": [url[:80] + "..." if len(url) > 80 else url for url in sites],
        "file_extensions": settings.onedrive_file_extensions,
        "schedule_hour": settings.onedrive_sync_schedule_hour,
        "schedule_minute": settings.onedrive_sync_schedule_minute,
        "use_vision": settings.onedrive_use_vision,
        "auth_mode": settings.graph_auth_mode,
        "permission": "Sites.Selected (sharing URL approach)",
    }


# ── Background helpers ────────────────────────────────────────


async def _run_sync_background(force: bool, bearer_token: str | None = None) -> None:
    """Wrapper for background task execution with error handling."""
    try:
        result = await sharepoint_sync.sync_onedrive(
            bearer_token=bearer_token,
            force=force,
        )
        logger.info(
            "[SHAREPOINT-SYNC] Background sync complete: %s",
            result,
        )
    except Exception as exc:
        logger.error(
            "[SHAREPOINT-SYNC] Background sync failed: %s",
            exc,
        )
