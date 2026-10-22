"""
scripts/list_sharepoint_docs.py
================================
List all document names in the configured SharePoint/OneDrive folders
WITHOUT downloading or syncing anything.

Uses the stored delegated token from the database, refreshes it if
stale, then calls the Graph API to recursively list files.

Usage:
    cd backend
    python -m scripts.list_sharepoint_docs
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure backend/ is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


async def main() -> None:
    from core.config import get_settings
    from core.db import init_db, close_db

    settings = get_settings()

    if not settings.onedrive_sync_enabled:
        print("[ERROR] ONEDRIVE_SYNC_ENABLED is not true in .env")
        return

    sites = settings.sharepoint_sites_list
    if not sites:
        print("[ERROR] No SharePoint folders configured (SHAREPOINT_SITES / SHAREPOINT_SITE).")
        return

    print(f"[INFO] Auth mode : {settings.graph_auth_mode}")
    print(f"[INFO] Folder(s) : {len(sites)}")

    # ── 1. Connect to PostgreSQL (needed to read stored tokens) ──
    await init_db()

    try:
        # ── 2. Get a valid bearer token ──
        from auth.token_store import get_latest_tokens, refresh_access_token

        tokens = await get_latest_tokens(settings)
        if tokens is None:
            print("[ERROR] No tokens found. A user must sign in via /auth/login first.")
            return

        bearer_token = await refresh_access_token(row=tokens, settings=settings)
        if bearer_token is None:
            print("[ERROR] Could not refresh the access token. Sign in again via /auth/login.")
            return

        # print(f"[INFO] Authenticated as: {tokens.get('user_email', '?')}\n")

        # ── 3. List files from every configured SharePoint folder ──
        from core.graph_client.client import GraphClient

        client = GraphClient(settings)
        total = 0

        for idx, site_url in enumerate(sites, 1):
            print(f"{'─' * 80}")
            print(f"  Folder {idx}: {site_url[:100]}{'…' if len(site_url) > 100 else ''}")
            print(f"{'─' * 80}")

            try:
                files, _drive_id, _root_id = await client.list_sharepoint_files_recursive(
                    sharing_url=site_url,
                    bearer_token=bearer_token,
                    file_extensions=settings.onedrive_extensions_tuple,
                )
            except Exception as exc:
                print(f"  [ERROR] Failed to list folder: {exc}\n")
                continue

            if not files:
                print("  (no matching files)\n")
                continue

            for i, f in enumerate(files, 1):
                rel_path = f.get("_relative_path", "")
                name = f.get("name", "?")
                size_kb = (f.get("size", 0) or 0) / 1024
                modified = (f.get("lastModifiedDateTime") or "")[:19]
                display = f"{rel_path}/{name}" if rel_path else name
                print(f"  {i:>3}. {display}  ({size_kb:,.0f} KB, modified {modified})")

            print(f"\n  Subtotal: {len(files)} file(s)\n")
            total += len(files)

        print(f"{'═' * 80}")
        print(f"  TOTAL: {total} document(s) across {len(sites)} folder(s)")
        print(f"{'═' * 80}")

    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
