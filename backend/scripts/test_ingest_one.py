"""
scripts/test_ingest_one.py
==============================
Download ONE PDF from SharePoint/OneDrive and ingest it into the
document library. For testing the full pipeline end-to-end.

Usage:
    cd backend
    python -m scripts.test_ingest_one
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

    # Step 1: Connect
    print("[1] Connecting to PostgreSQL...")
    await init_db()

    # Step 2: Get token
    print("[2] Getting bearer token...")
    from auth.token_store import get_latest_tokens, refresh_access_token
    tokens = await get_latest_tokens(settings)
    if not tokens:
        print("ERROR: No stored tokens. Sign in via /auth/login first.")
        await close_db()
        return
    bearer_token = await refresh_access_token(row=tokens, settings=settings)
    if not bearer_token:
        print("ERROR: Could not refresh token.")
        await close_db()
        return
    # print(f"  Token OK: {bearer_token[:20]}...")

    # Step 3: List files
    print("[3] Listing PDFs from SharePoint...")
    from core.graph_client.client import GraphClient
    client = GraphClient(settings)
    files, drive_id, root_id = await client.list_sharepoint_files_recursive(
        sharing_url=settings.sharepoint_site,
        bearer_token=bearer_token,
        file_extensions=settings.onedrive_extensions_tuple,
    )
    print(f"  Found {len(files)} PDFs:\n")
    for i, f in enumerate(files):
        path = f.get("_relative_path", "")
        name = f.get("name", "?")
        size_kb = (f.get("size", 0) or 0) / 1024
        display = f"{path}/{name}" if path else name
        print(f"  [{i+1}] {display} ({size_kb:.0f} KB)")

    # Step 4: Pick one
    print()
    choice = input("  Enter number to ingest (or q to quit): ").strip()
    if choice.lower() == "q":
        await close_db()
        return
    idx = int(choice) - 1
    if idx < 0 or idx >= len(files):
        print("Invalid choice.")
        await close_db()
        return

    chosen = files[idx]
    name = chosen.get("name", "document.pdf")
    item_id = chosen.get("id", "")
    file_drive_id = chosen.get("_drive_id", drive_id)
    print(f"\n[4] Downloading: {name}...")

    # Step 5: Download
    temp_dir = settings.resolved_onedrive_temp_dir
    dest_path = temp_dir / name
    await client.download_sharepoint_file(
        drive_id=file_drive_id,
        item_id=item_id,
        dest_path=str(dest_path),
        bearer_token=bearer_token,
    )
    print(f"  Downloaded to: {dest_path} ({dest_path.stat().st_size / 1024:.0f} KB)")

    # Step 6: Ingest
    print(f"\n[5] Ingesting {name} into document library...")
    from documents.library import add_pdf
    doc_id = add_pdf(
        str(dest_path),
        use_vision=settings.onedrive_use_vision,
        progress=lambda msg: print(f"  {msg}"),
    )
    print(f"\n  SUCCESS! doc_id = {doc_id}")

    # Step 7: Cleanup
    try:
        dest_path.unlink()
    except OSError:
        pass

    # Step 8: Verify
    print(f"\n[6] Verifying in library...")
    from documents import store
    con = store.connect()
    docs = store.documents(con)
    for d in docs:
        if d["doc_id"] == doc_id:
            print(f"  File: {d['file_name']}")
            print(f"  Title: {d['title']}")
            print(f"  Status: {d['status']}")
            print(f"  Pages: {d['pages']}")
            n_facts = con.execute(
                "SELECT count(*) FROM facts WHERE doc_id=?", (doc_id,)
            ).fetchone()[0]
            print(f"  Facts: {n_facts}")
            break

    await close_db()
    print("\n[DONE] You can now query this document via the chatbot with mode='document'")


if __name__ == "__main__":
    asyncio.run(main())
