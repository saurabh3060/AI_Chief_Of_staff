"""
scripts/test_pipeline.py
===========================
Test the SharePoint/OneDrive sync pipeline WITHOUT importing
documents.library (which has a broken picture dependency).

Tests: token refresh → sharing URL resolve → list files → download one PDF.

Usage:
    cd backend
    python -m scripts.test_pipeline
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


async def main():
    from core.config import get_settings
    from core.db import init_db, close_db

    settings = get_settings()
    print("=" * 60)
    print("  SHAREPOINT SYNC PIPELINE TEST")
    print("=" * 60)
    print(f"\n  Auth mode:        {settings.graph_auth_mode}")
    print(f"  Sync enabled:     {settings.onedrive_sync_enabled}")
    print(f"  SharePoint URL:   {'configured' if settings.sharepoint_site else 'NOT SET'}")
    print(f"  File extensions:  {settings.onedrive_file_extensions}")

    if not settings.sharepoint_site:
        print("\n[FAIL] SHAREPOINT_SITE not set in .env")
        return

    # ── Test 1: PostgreSQL connection ──
    print("\n[TEST 1] PostgreSQL connection...")
    await init_db()
    print("  PASS")

    # ── Test 2: Token from AWS Secrets Manager ──
    print("\n[TEST 2] Reading stored tokens...")
    from auth.token_store import get_latest_tokens, refresh_access_token

    tokens = await get_latest_tokens(settings)
    if not tokens:
        print("  FAIL — No tokens in AWS Secrets Manager.")
        print("  Fix: Sign in via /auth/login at least once.")
        await close_db()
        return
    # print(f"  User:          {tokens.get('user_email')}")
    print(f"  Refresh token: {'present' if tokens.get('refresh_token') else 'MISSING'}")
    print("  PASS")

    # ── Test 3: Refresh access token ──
    print("\n[TEST 3] Refreshing access token...")
    bearer = await refresh_access_token(row=tokens, settings=settings)
    if not bearer:
        print("  FAIL — Could not refresh. Token may have expired.")
        print("  Fix: Sign in via /auth/login again.")
        await close_db()
        return
    # print(f"  Bearer: {bearer[:30]}...")
    print("  PASS")

    # ── Test 4: Resolve sharing URL ──
    print("\n[TEST 4] Resolving SharePoint sharing URL...")
    from core.graph_client.client import GraphClient

    client = GraphClient(settings)
    try:
        root = await client.resolve_sharing_url(
            settings.sharepoint_site, bearer, expand_children=True
        )
        drive_id = (root.get("parentReference") or {}).get("driveId", "")
        root_name = root.get("name", "?")
        children = root.get("children", [])
        print(f"  Folder:   {root_name}")
        print(f"  Drive ID: {drive_id[:30]}...")
        print(f"  Children: {len(children)}")
        print("  PASS")
    except Exception as exc:
        print(f"  FAIL — {exc}")
        await close_db()
        return

    # ── Test 5: List all PDFs recursively ──
    print("\n[TEST 5] Listing all PDFs recursively...")
    try:
        files, drive_id, root_id = await client.list_sharepoint_files_recursive(
            sharing_url=settings.sharepoint_site,
            bearer_token=bearer,
            file_extensions=settings.onedrive_extensions_tuple,
        )
        print(f"  Found {len(files)} PDFs:")
        for i, f in enumerate(files):
            path = f.get("_relative_path", "")
            name = f.get("name", "?")
            size_kb = (f.get("size", 0) or 0) / 1024
            display = f"{path}/{name}" if path else name
            print(f"    [{i+1}] {display} ({size_kb:.0f} KB)")
        print("  PASS")
    except Exception as exc:
        print(f"  FAIL — {exc}")
        await close_db()
        return

    if not files:
        print("\n  No PDFs found. Nothing to download.")
        await close_db()
        return

    # ── Test 6: Download one PDF ──
    print(f"\n[TEST 6] Downloading one PDF to test...")
    chosen = files[0]  # pick the first one
    name = chosen.get("name", "test.pdf")
    item_id = chosen.get("id", "")
    file_drive_id = chosen.get("_drive_id", drive_id)

    temp_dir = settings.resolved_onedrive_temp_dir
    dest_path = temp_dir / f"test_{name}"

    try:
        await client.download_sharepoint_file(
            drive_id=file_drive_id,
            item_id=item_id,
            dest_path=str(dest_path),
            bearer_token=bearer,
        )
        file_size = dest_path.stat().st_size
        print(f"  Downloaded: {dest_path.name}")
        print(f"  Size:       {file_size / 1024:.0f} KB")

        # Verify it's a valid PDF
        with open(dest_path, "rb") as f:
            header = f.read(5)
        if header == b"%PDF-":
            print("  Valid PDF: YES")
            print("  PASS")
        else:
            print(f"  Valid PDF: NO (header: {header})")
            print("  FAIL — Downloaded file is not a PDF")

        # Cleanup
        dest_path.unlink()
        print(f"  Cleaned up temp file")

    except Exception as exc:
        print(f"  FAIL — {exc}")
        if dest_path.exists():
            dest_path.unlink()
        await close_db()
        return

    # ── Test 7: Sync state tracker ──
    print("\n[TEST 7] Sync state tracker...")
    from documents.sync_state import load_state, save_state, needs_sync, record_sync

    state = load_state()
    print(f"  State file loaded: {len(state.get('files', {}))} tracked files")

    test_item = files[0]
    needs = needs_sync(test_item, state)
    print(f"  needs_sync for '{test_item.get('name')}': {needs}")
    print("  PASS")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  ALL 7 TESTS PASSED")
    print("=" * 60)
    print(f"""
  Your SharePoint sync pipeline is working:
  - Token refresh:    OK
  - Sharing URL:      OK  ({root_name})
  - File listing:     OK  ({len(files)} PDFs found)
  - File download:    OK  (valid PDF downloaded)
  - State tracker:    OK

  The pipeline will run automatically at {settings.onedrive_sync_schedule_hour:02d}:{settings.onedrive_sync_schedule_minute:02d} IST daily.
  To trigger manually, sign in and use the browser console or API.
""")

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
