"""
scripts/test_sharepoint_sync.py
===================================
Quick test script: reads the stored delegated token from AWS Secrets
Manager, refreshes it if stale, and triggers the SharePoint document
sync.

Usage:
    cd backend
    python -m scripts.test_sharepoint_sync

Requirements:
    - POSTGRES_DSN must be set in .env (tokens are stored in PostgreSQL)
    - A user must have signed in via /auth/login at least once (so a
      refresh token was captured)
    - ONEDRIVE_SYNC_ENABLED=true and SHAREPOINT_SITE must be set in .env
"""

import asyncio
import sys
from pathlib import Path

# Add backend/ to sys.path so imports work when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


async def main():
    from core.config import get_settings
    from core.db import init_db, close_db

    settings = get_settings()
    print(f"[INFO] Auth mode: {settings.graph_auth_mode}")
    print(f"[INFO] Sync enabled: {settings.onedrive_sync_enabled}")
    print(f"[INFO] SharePoint site configured: {bool(settings.sharepoint_site)}")

    if not settings.onedrive_sync_enabled:
        print("[ERROR] ONEDRIVE_SYNC_ENABLED is not true in .env")
        return

    if not settings.sharepoint_site:
        print("[ERROR] SHAREPOINT_SITE is not set in .env")
        return

    # Step 1: Connect to PostgreSQL
    print("\n[STEP 1] Connecting to PostgreSQL...")
    await init_db()

    # Step 2: Read stored tokens
    print("[STEP 2] Reading stored tokens from AWS Secrets Manager...")
    from auth.token_store import get_latest_tokens, refresh_access_token

    tokens = await get_latest_tokens(settings)
    if tokens is None:
        print("[ERROR] No tokens found in AWS Secrets Manager.")
        print("        A user must sign in via /auth/login first.")
        await close_db()
        return

    # print(f"  User: {tokens.get('user_email')}")
    # print(f"  Access token: {tokens.get('access_token', '')[:20]}...")
    print(f"  Refresh token present: {bool(tokens.get('refresh_token'))}")
    # print(f"  Expires at: {tokens.get('access_token_expires_at')}")

    # Step 3: Refresh the token if needed
    print("\n[STEP 3] Refreshing access token if stale...")
    bearer_token = await refresh_access_token(row=tokens, settings=settings)
    if bearer_token is None:
        print("[ERROR] Could not get a valid access token.")
        print("        The refresh token may have expired. Sign in again via /auth/login.")
        await close_db()
        return

    # print(f"  Got valid bearer token: {bearer_token[:20]}...")

    # Step 4: Test resolving the sharing URL
    print("\n[STEP 4] Resolving SharePoint sharing URL...")
    from core.graph_client.client import GraphClient

    client = GraphClient(settings)
    try:
        root = await client.resolve_sharing_url(
            settings.sharepoint_site, bearer_token, expand_children=True
        )
        drive_id = (root.get("parentReference") or {}).get("driveId", "")
        root_name = root.get("name", "?")
        children = root.get("children", [])

        print(f"  SUCCESS! Resolved to: {root_name}")
        print(f"  Drive ID: {drive_id}")
        print(f"  Children (top level): {len(children)}")
        for child in children:
            kind = "folder" if child.get("folder") else "file"
            print(f"    [{kind}] {child.get('name')} ({child.get('size', 0)} bytes)")

    except Exception as exc:
        print(f"  FAILED: {exc}")
        await close_db()
        return

    # Step 5: List all PDFs recursively
    print("\n[STEP 5] Listing all PDF files recursively...")
    try:
        files, drive_id, root_id = await client.list_sharepoint_files_recursive(
            sharing_url=settings.sharepoint_site,
            bearer_token=bearer_token,
            file_extensions=settings.onedrive_extensions_tuple,
        )
        print(f"  Found {len(files)} PDF files:")
        for f in files:
            path = f.get("_relative_path", "")
            name = f.get("name", "?")
            size_kb = (f.get("size", 0) or 0) / 1024
            display = f"{path}/{name}" if path else name
            print(f"    {display} ({size_kb:.0f} KB)")

    except Exception as exc:
        print(f"  FAILED: {exc}")
        await close_db()
        return

    # Step 6: Ask if user wants to run the full sync
    print(f"\n[STEP 6] Connection test PASSED. {len(files)} PDFs found.")
    answer = input("  Run full sync now? (y/n): ").strip().lower()
    if answer == "y":
        print("\n[SYNCING] Running full sync...")
        from documents.sharepoint_sync import sync_onedrive
        result = await sync_onedrive(bearer_token=bearer_token, force=False)
        print(f"\n[RESULT] {result}")
    else:
        print("  Skipped. You can run the sync later via POST /documents/sync")

    await close_db()
    print("\n[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
