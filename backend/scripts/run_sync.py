"""
scripts/run_sync.py
======================
Manually trigger the full SharePoint -> document library sync.

Gets the bearer token from AWS Secrets Manager (same as the daily
scheduler), then calls sync_onedrive() which lists files, diffs
against state, downloads new/changed PDFs, and ingests them via
library.add_pdf().

NOTE: This must be run while the backend (uvicorn) is NOT running,
because library.add_pdf() needs the picture module which is only
available in the backend's process. Instead, this script starts a
minimal app context itself.

Usage:
    cd backend
    python -m scripts.run_sync
    python -m scripts.run_sync --force     (re-download everything)
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


async def main(force: bool = False, user_email: str | None = None):
    from core.config import get_settings
    from core.db import init_db, close_db

    settings = get_settings()
    print(f"[SYNC] Auth mode: {settings.graph_auth_mode}")
    print(f"[SYNC] Sync enabled: {settings.onedrive_sync_enabled}")
    print(f"[SYNC] Force: {force}")

    if not settings.onedrive_sync_enabled:
        print("[ERROR] ONEDRIVE_SYNC_ENABLED is not true")
        return

    if not settings.sharepoint_sites_list:
        print("[ERROR] No SharePoint folders configured (SHAREPOINT_SITES or SHAREPOINT_SITE)")
        return

    # Connect to PostgreSQL
    print("[SYNC] Connecting to PostgreSQL...")
    await init_db()

    # Get bearer token
    print("[SYNC] Getting bearer token...")
    bearer_token = None
    if settings.graph_auth_mode == "delegated":
        from auth.token_store import get_latest_tokens, get_tokens_for_user, refresh_access_token
        if user_email:
            print(f"[SYNC] Looking up token for: {user_email}")
            tokens = await get_tokens_for_user(user_email, settings)
        else:
            tokens = await get_latest_tokens(settings)
        if not tokens:
            print("[ERROR] No stored tokens. Sign in via /auth/login first.")
            await close_db()
            return
        # print(f"[SYNC] Using token from: {tokens.get('user_email')}")
        bearer_token = await refresh_access_token(row=tokens, settings=settings)
        if not bearer_token:
            print("[ERROR] Could not refresh token. Sign in again.")
            await close_db()
            return

    # Run the sync
    print("[SYNC] Starting sync...")
    from documents.sharepoint_sync import sync_onedrive
    result = await sync_onedrive(bearer_token=bearer_token, force=force)

    # Print results
    print(f"\n{'='*50}")
    print(f"  SYNC RESULT: {result.get('status', 'unknown')}")
    print(f"{'='*50}")
    print(f"  Files found:  {result.get('files_found', 0)}")
    print(f"  Synced:       {result.get('synced', 0)}")
    print(f"  Skipped:      {result.get('skipped', 0)}")
    print(f"  Failed:       {result.get('failed', 0)}")
    print(f"  Duration:     {result.get('duration_s', 0)}s")
    if result.get("errors"):
        print(f"  Errors:")
        for err in result["errors"]:
            print(f"    - {err}")
    print()

    await close_db()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually trigger SharePoint document sync")
    parser.add_argument("--force", action="store_true", help="Re-download and re-ingest all files")
    parser.add_argument("--user", default="ashutosh.verma1@rblbank.com",
                        help="Email of the user whose token to use (default: ashutosh.verma1@rblbank.com)")
    args = parser.parse_args()
    asyncio.run(main(force=args.force, user_email=args.user))
