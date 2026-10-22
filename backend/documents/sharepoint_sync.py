"""
documents/sharepoint_sync.py
================================
Sync orchestrator: resolves a SharePoint/OneDrive sharing URL via
Graph's ``/shares`` API, lists files recursively, diffs against local
sync state, downloads new/changed documents (PDF, PPTX), and feeds them
into the existing ``documents/library.add_pdf()`` ingestion pipeline.
PPTX files are converted to PDF via LibreOffice before extraction.

This approach uses the **sharing URL** trick (same as the reference
script's ``_encode_sharing_url`` + ``/shares/{shareId}/driveItem``
flow) and works with the **Sites.Selected** permission — no
Files.Read.All needed.

Entry points
~~~~~~~~~~~~
- ``sync_onedrive(bearer_token, force)`` — full incremental sync
- ``sync_single_file(drive_id, item_id, bearer_token)`` — retry one file

The sync is designed to be called from:
1. A FastAPI background task (POST /documents/sync)
2. A daily asyncio scheduled task (configured via ONEDRIVE_SYNC_SCHEDULE_*)

In local mode (GRAPH_AUTH_MODE=local), PDFs are read from the
``ONEDRIVE_LOCAL_DIR`` folder instead of calling Graph API, so the
pipeline can be tested without Azure credentials.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from core.config import get_settings
from core.graph_client.client import GraphClient
from core.logging_config import get_agent_logger
from documents import sync_state

logger = get_agent_logger("documents")

# Module-level lock to prevent concurrent sync runs.
_sync_lock = asyncio.Lock()
_sync_running = False
_last_sync_result: dict | None = None


def is_sync_running() -> bool:
    """Check if a sync is currently in progress."""
    return _sync_running


def last_sync_result() -> dict | None:
    """Return the result of the most recent sync run."""
    return _last_sync_result


async def sync_onedrive(
    bearer_token: str | None = None,
    force: bool = False,
) -> dict:
    """Run a full incremental sync from SharePoint/OneDrive to the document library.

    Parameters
    ----------
    bearer_token : str | None
        Graph API bearer token (delegated mode). Ignored in app-only/local.
    force : bool
        When True, re-download and re-ingest ALL files regardless of
        whether they've changed since last sync.

    Returns
    -------
    dict
        Summary: ``{"synced": int, "skipped": int, "failed": int,
        "errors": list[str], "duration_s": float, "files_found": int}``
    """
    global _sync_running, _last_sync_result

    if _sync_lock.locked():
        return {
            "synced": 0, "skipped": 0, "failed": 0,
            "errors": ["A sync is already running"],
            "duration_s": 0, "files_found": 0,
            "status": "rejected",
        }

    async with _sync_lock:
        _sync_running = True
        try:
            result = await _do_sync(bearer_token, force)
            _last_sync_result = result
            return result
        finally:
            _sync_running = False


async def _do_sync(
    bearer_token: str | None,
    force: bool,
) -> dict:
    """Internal sync implementation."""
    settings = get_settings()
    t0 = time.perf_counter()
    state = await sync_state.load_state()
    summary = {"synced": 0, "skipped": 0, "failed": 0, "errors": [], "files_found": 0}

    sites = settings.sharepoint_sites_list
    logger.info(
        "[SHAREPOINT-SYNC] Starting sync: %d folder(s) force=%s auth_mode=%s",
        len(sites), force, settings.graph_auth_mode,
    )

    # ── Step 1: List files from ALL configured SharePoint folders ──
    drive_items = []
    try:
        if settings.is_local_mail_mode:
            drive_items = _list_local_files(settings)
        else:
            if not sites:
                raise ValueError(
                    "No SharePoint folders configured. "
                    "Set SHAREPOINT_SITES or SHAREPOINT_SITE in .env."
                )
            client = GraphClient(settings)
            for site_url in sites:
                logger.info("[SHAREPOINT-SYNC] Listing: %s", site_url[:80])
                try:
                    items, drive_id, root_id = await client.list_sharepoint_files_recursive(
                        sharing_url=site_url,
                        bearer_token=bearer_token,
                        file_extensions=settings.onedrive_extensions_tuple,
                    )
                    drive_items.extend(items)
                    logger.info("[SHAREPOINT-SYNC] Found %d files in %s",
                                len(items), site_url[:80])
                except Exception:
                    # Exception detail goes to the server log only; the summary
                    # (returned by the status route) carries constant text.
                    logger.exception("[SHAREPOINT-SYNC] Failed to list folder %s", site_url[:80])
                    summary["errors"].append(f"Listing {site_url[:80]} failed")
    except Exception:
        logger.exception("[SHAREPOINT-SYNC] Failed to list SharePoint folders")
        summary["errors"].append("Listing failed")
        summary["duration_s"] = round(time.perf_counter() - t0, 1)
        summary["status"] = "failed"
        sync_state.update_sync_result(state, "failed", summary)
        await sync_state.save_state(state)
        return summary

    summary["files_found"] = len(drive_items)
    logger.info("[SHAREPOINT-SYNC] Found %d files matching extensions %s",
                len(drive_items), settings.onedrive_extensions_tuple)

    # ── Step 2: Diff and process each file ────────────────────
    for item in drive_items:
        item_id = item.get("id", "")
        name = item.get("name", "unknown")
        relative_path = item.get("_relative_path", "")
        display = f"{relative_path}/{name}" if relative_path else name

        if not force and not sync_state.needs_sync(item, state):
            logger.debug("[SHAREPOINT-SYNC] Skipping (unchanged): %s", display)
            summary["skipped"] += 1
            continue

        logger.info("[SHAREPOINT-SYNC] Syncing: %s (id=%s)", display, item_id)

        try:
            doc_id = await _download_and_ingest(
                item, bearer_token, settings,
            )
            sync_state.record_sync(state, item_id, item, doc_id)
            await sync_state.save_state(state)  # persist after each file
            summary["synced"] += 1
            logger.info("[SHAREPOINT-SYNC] Ingested: %s -> doc_id=%s", display, doc_id)

        except Exception:
            # Full traceback in the server log; the summary keeps constant text only.
            logger.exception("[SHAREPOINT-SYNC] Failed to sync %s", display)
            summary["errors"].append(f"{display}: sync failed")
            summary["failed"] += 1

    # ── Step 3: Finalize ──────────────────────────────────────
    duration = round(time.perf_counter() - t0, 1)
    summary["duration_s"] = duration

    if summary["failed"] == 0:
        status = "success"
    elif summary["synced"] > 0:
        status = "partial"
    else:
        status = "failed"

    summary["status"] = status
    sync_state.update_sync_result(state, status, summary)
    await sync_state.save_state(state)

    logger.info(
        "[SHAREPOINT-SYNC] Complete: synced=%d skipped=%d failed=%d duration=%.1fs",
        summary["synced"], summary["skipped"], summary["failed"], duration,
    )
    return summary


async def sync_single_file(
    drive_id: str,
    item_id: str,
    bearer_token: str | None = None,
) -> dict:
    """Re-sync a single file by its driveId + itemId.

    Useful for retrying a failed file without re-running the full sync.
    """
    settings = get_settings()
    state = await sync_state.load_state()

    if settings.is_local_mail_mode:
        return {"error": "Single-file sync not supported in local mode"}

    client = GraphClient(settings)

    # Fetch item metadata via driveId + itemId
    from core.graph_client import endpoints
    meta_url = endpoints.drive_item_by_id_url(drive_id, item_id)
    item_data = await client._get(meta_url, bearer_token)
    item_data["_drive_id"] = drive_id

    try:
        doc_id = await _download_and_ingest(item_data, bearer_token, settings)
        sync_state.record_sync(state, item_id, item_data, doc_id)
        await sync_state.save_state(state)
        return {"status": "success", "doc_id": doc_id, "name": item_data.get("name", "")}
    except Exception:
        # Detail stays in the server log; the result (returned by the route) is constant.
        logger.exception("[SHAREPOINT-SYNC] Single-file sync failed for item %s", item_id)
        return {"status": "failed", "error": "Single-file sync failed.", "name": item_data.get("name", "")}


async def _download_and_ingest(
    drive_item: dict,
    bearer_token: str | None,
    settings,
) -> str:
    """Download a single file and ingest it into the document library.

    Returns the ``doc_id`` assigned by ``library.add_pdf()``.

    ``library.add_pdf()`` is now async and writes row data to PostgreSQL
    and embedding vectors to Weaviate. CPU-heavy PDF extraction still runs
    inside ``asyncio.to_thread`` within ``add_pdf``, so it doesn't block
    the event loop.
    """
    name = drive_item.get("name", "document.pdf")
    temp_dir = settings.resolved_onedrive_temp_dir
    dest_path = temp_dir / name

    try:
        if settings.is_local_mail_mode:
            # In local mode, the "drive_item" has a _local_path key
            local_path = drive_item.get("_local_path", "")
            if not local_path or not Path(local_path).exists():
                raise FileNotFoundError(f"Local file not found: {local_path}")
            # Copy to temp dir (library.add_pdf may move/rename)
            import shutil
            shutil.copy2(local_path, dest_path)
        else:
            # Download from SharePoint via Graph API using driveId + itemId
            client = GraphClient(settings)
            drive_id = drive_item.get("_drive_id", "")
            item_id = drive_item.get("id", "")

            if not drive_id or not item_id:
                raise ValueError(
                    f"Missing driveId or itemId for file '{name}'. "
                    "Cannot download without both identifiers."
                )

            await client.download_sharepoint_file(
                drive_id=drive_id,
                item_id=item_id,
                dest_path=str(dest_path),
                bearer_token=bearer_token,
            )

        # Ingest the PDF using the existing async pipeline (writes to
        # PostgreSQL + Weaviate). CPU-heavy extraction (pdfplumber / vision)
        # runs inside asyncio.to_thread internally.
        from documents.library import add_pdf

        doc_id = await add_pdf(
            str(dest_path),
            use_vision=settings.onedrive_use_vision,
            progress=lambda msg: logger.info("[SHAREPOINT-INGEST] %s: %s", name, msg),
        )
        return doc_id

    finally:
        # Clean up the temp file
        try:
            if dest_path.exists():
                dest_path.unlink()
        except OSError as exc:
            logger.warning("Could not remove temp file %s: %s", dest_path, exc)


_MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _list_local_files(settings) -> list[dict]:
    """List document files from the local OneDrive fallback directory.

    Returns dicts shaped like Graph drive items so the rest of the sync
    pipeline works identically.
    """
    local_dir = settings.resolved_onedrive_local_dir
    if not local_dir.is_dir():
        logger.info("[SHAREPOINT-SYNC] Local dir %s does not exist — no files", local_dir)
        return []

    extensions = settings.onedrive_extensions_tuple
    items = []

    for root_path, dirs, files in os.walk(local_dir):
        root = Path(root_path)
        for filename in sorted(files):
            if not any(filename.lower().endswith(ext) for ext in extensions):
                continue

            filepath = root / filename
            stat = filepath.stat()
            relative = root.relative_to(local_dir)
            relative_path = str(relative) if str(relative) != "." else ""

            ext = Path(filename).suffix.lower()
            mime = _MIME_BY_EXT.get(ext, "application/octet-stream")

            # Shape as a Graph drive item
            items.append({
                "id": f"local-{filepath.stem}-{int(stat.st_mtime)}",
                "name": filename,
                "size": stat.st_size,
                "lastModifiedDateTime": _timestamp_to_iso(stat.st_mtime),
                "file": {"mimeType": mime},
                "_relative_path": relative_path,
                "_local_path": str(filepath),
                "_drive_id": "local",
            })

    logger.info(
        "[SHAREPOINT-SYNC] Local mode: found %d files in %s",
        len(items), local_dir,
    )
    return items


def _timestamp_to_iso(ts: float) -> str:
    """Convert a Unix timestamp to an ISO 8601 UTC string."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
