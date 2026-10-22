"""
documents/sync_scheduler.py
===============================
Daily scheduler for the SharePoint -> document library sync.

Uses a simple ``asyncio.Task`` loop that sleeps until the configured
IST time each day, then runs ``sync_onedrive()``. No external
dependencies (APScheduler, celery, cron) — just plain asyncio, matching
the rest of the backend's lightweight async infrastructure.

In delegated mode, the scheduler reads the stored refresh token from
AWS Secrets Manager (persisted at ``/auth/callback``) and mints a fresh
access token via ``auth.token_store.refresh_access_token`` — same
mechanism the external pipeline runner uses. A user must have signed in
at least once for this to work.

Start/stop are called from ``main.py``'s lifespan hook.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from core.config import get_settings
from core.logging_config import get_agent_logger
from documents.sharepoint_sync import sync_onedrive

logger = get_agent_logger("documents")

# IST offset (UTC+5:30)
_IST = timezone(timedelta(hours=5, minutes=30))

_scheduler_task: asyncio.Task | None = None


def start_sync_scheduler() -> None:
    """Start the daily OneDrive sync scheduler as a background asyncio task.

    No-op if ``ONEDRIVE_SYNC_ENABLED`` is False or the scheduler is
    already running.
    """
    global _scheduler_task

    settings = get_settings()
    if not settings.onedrive_sync_enabled:
        logger.info("[SHAREPOINT-SCHEDULER] Sync disabled — scheduler not started")
        return

    if _scheduler_task is not None and not _scheduler_task.done():
        logger.info("[SHAREPOINT-SCHEDULER] Scheduler already running")
        return

    _scheduler_task = asyncio.create_task(_scheduler_loop())
    logger.info(
        "[SHAREPOINT-SCHEDULER] Started — sync runs daily at %02d:%02d IST",
        settings.onedrive_sync_schedule_hour,
        settings.onedrive_sync_schedule_minute,
    )


async def stop_sync_scheduler() -> None:
    """Cancel the scheduler task gracefully."""
    global _scheduler_task

    if _scheduler_task is not None and not _scheduler_task.done():
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
        logger.info("[SHAREPOINT-SCHEDULER] Stopped")
    _scheduler_task = None


async def _scheduler_loop() -> None:
    """Main scheduler loop: sleep until the next configured time, then sync."""
    settings = get_settings()
    hour = settings.onedrive_sync_schedule_hour
    minute = settings.onedrive_sync_schedule_minute

    while True:
        try:
            # Calculate seconds until the next run
            now = datetime.now(_IST)
            next_run = now.replace(
                hour=hour, minute=minute, second=0, microsecond=0,
            )
            if next_run <= now:
                # Already past today's time — schedule for tomorrow
                next_run += timedelta(days=1)

            wait_seconds = (next_run - now).total_seconds()
            logger.info(
                "[SHAREPOINT-SCHEDULER] Next sync at %s IST (in %.0f seconds / %.1f hours)",
                next_run.strftime("%Y-%m-%d %H:%M"),
                wait_seconds,
                wait_seconds / 3600,
            )

            await asyncio.sleep(wait_seconds)

            # Time to sync
            logger.info("[SHAREPOINT-SCHEDULER] Daily sync triggered")
            try:
                # In delegated mode, get a fresh bearer token from the
                # stored refresh token in AWS Secrets Manager (persisted
                # when a user signs in via /auth/login). In app-only or
                # local mode, bearer_token stays None and GraphClient
                # handles auth internally.
                bearer_token = await _get_scheduled_bearer_token(settings)

                result = await sync_onedrive(bearer_token=bearer_token)
                logger.info(
                    "[SHAREPOINT-SCHEDULER] Daily sync complete: synced=%d skipped=%d failed=%d",
                    result.get("synced", 0),
                    result.get("skipped", 0),
                    result.get("failed", 0),
                )
            except Exception as exc:
                logger.error("[SHAREPOINT-SCHEDULER] Daily sync failed: %s", exc)

        except asyncio.CancelledError:
            logger.info("[SHAREPOINT-SCHEDULER] Scheduler cancelled")
            raise
        except Exception as exc:
            # Any unexpected error: log and retry after 60s rather than dying
            logger.error(
                "[SHAREPOINT-SCHEDULER] Unexpected error in scheduler loop: %s — "
                "retrying in 60s",
                exc,
            )
            await asyncio.sleep(60)


async def _get_scheduled_bearer_token(settings) -> str | None:
    """Obtain a bearer token for the scheduled (unattended) sync.

    - **delegated mode**: reads the stored refresh token from
      AWS Secrets Manager (persisted when a user signs in via
      ``/auth/login``) and mints a fresh access token. Same mechanism
      the external pipeline runner (``run_pipeline.py``) uses. Returns
      None (and logs a warning) if no tokens are stored — a user must
      sign in at least once.

    - **app_only / local mode**: returns None. ``GraphClient`` handles
      auth internally in these modes.
    """
    if settings.graph_auth_mode in ("app_only", "local"):
        return None

    # Delegated mode — refresh from stored tokens
    try:
        from auth.token_store import get_latest_tokens, refresh_access_token

        tokens = await get_latest_tokens(settings)
        if tokens is None:
            logger.warning(
                "[SHAREPOINT-SCHEDULER] No stored delegated tokens found in "
                "AWS Secrets Manager. A user must sign in via /auth/login at "
                "least once so a refresh token is captured. Sync will fail."
            )
            return None

        bearer = await refresh_access_token(row=tokens, settings=settings)
        if bearer is None:
            logger.warning(
                "[SHAREPOINT-SCHEDULER] Could not refresh access token. "
                "The refresh token may have expired. A user must sign in again."
            )
            return None

        logger.info("[SHAREPOINT-SCHEDULER] Got fresh bearer token from stored refresh token")
        return bearer

    except Exception as exc:
        logger.error(
            "[SHAREPOINT-SCHEDULER] Failed to obtain bearer token: %s", exc
        )
        return None
