"""
scripts/weaviate_admin.py
=========================
The ONLY place a Weaviate collection may be deleted or rebuilt.

Deleting a Weaviate collection is destructive and the Weaviate server may
be shared with other projects, so the application itself has no code path
that deletes a collection (``reset_namespace`` is disabled). This explicit
operator script is the single exception, and it is deliberately hard to
misuse:

* It refuses to touch any collection that is not one of THIS project's own
  collections (derived from ``WEAVIATE_EMAIL_COLLECTION_PREFIX``). It will
  never create, change, or delete a class that is not ours.
* A delete requires an explicit ``--confirm`` flag. Without it the script
  prints what it WOULD do and exits without changing anything.
* After a delete you can rebuild from PostgreSQL (the system of record)
  with ``--rebuild``, which uses the STORED embeddings — no re-embedding,
  no LLM calls.

Usage (run from the backend/ directory)
---------------------------------------
    # Show owned collections + current object counts (read-only):
    python scripts/weaviate_admin.py --status

    # Dry-run a delete (prints, changes nothing):
    python scripts/weaviate_admin.py --delete clusters

    # Actually delete (requires --confirm):
    python scripts/weaviate_admin.py --delete clusters --confirm

    # Rebuild a namespace from PostgreSQL's stored embeddings:
    python scripts/weaviate_admin.py --rebuild clusters

    # Delete then rebuild in one go:
    python scripts/weaviate_admin.py --delete clusters --confirm --rebuild clusters

Valid namespaces are exactly this project's own: "clusters", "meetings".
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from core.config import get_settings  # noqa: E402
from core.db import init_db, close_db  # noqa: E402
from agent_email import weaviate_vector_store as wvs  # noqa: E402
from agent_email.weaviate_vector_store import (  # noqa: E402
    OWNED_NAMESPACES,
    owned_collection_names,
)
from agent_email import postgres_store  # noqa: E402


def _print_status(settings) -> None:
    print("Owned collections (prefix "
          f"{settings.weaviate_email_collection_prefix!r}):")
    for namespace, name in owned_collection_names(settings).items():
        cnt = wvs.object_count(namespace, settings)
        state = "missing" if cnt < 0 else f"{cnt} objects"
        print(f"  {namespace:10s} -> {name:30s} [{state}]")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gated Weaviate admin: delete/rebuild THIS project's own collections only."
    )
    parser.add_argument("--status", action="store_true", help="Show owned collections + counts (read-only).")
    parser.add_argument("--delete", metavar="NAMESPACE", help="Delete the collection for a namespace (requires --confirm).")
    parser.add_argument("--rebuild", metavar="NAMESPACE", help="Rebuild a namespace from PostgreSQL stored embeddings (no re-embedding).")
    parser.add_argument("--confirm", action="store_true", help="Actually perform the delete (otherwise dry-run).")
    args = parser.parse_args()

    settings = get_settings()

    if not any([args.status, args.delete, args.rebuild]):
        parser.print_help()
        return

    for ns in (args.delete, args.rebuild):
        if ns is not None and ns not in OWNED_NAMESPACES:
            print(f"ERROR: {ns!r} is not one of this project's namespaces: {list(OWNED_NAMESPACES)}")
            sys.exit(2)

    if args.status:
        _print_status(settings)

    if args.delete:
        name = owned_collection_names(settings)[args.delete]
        if not args.confirm:
            print(f"DRY-RUN: would delete collection {name!r} (namespace={args.delete}).")
            print("Re-run with --confirm to actually delete.")
        else:
            deleted = wvs._admin_delete_collection(args.delete, settings)
            print(f"{'Deleted' if deleted else 'Nothing to delete for'} collection {name!r}.")

    if args.rebuild:
        # Rebuild reads embeddings from PostgreSQL, so the pool must be up.
        await init_db()
        try:
            restored = await postgres_store.restore_weaviate_from_postgres([args.rebuild], settings)
            count = restored.get(args.rebuild, 0)
            print(f"Rebuilt namespace={args.rebuild} from PostgreSQL: {count} vector(s) restored.")
        finally:
            await close_db()
            from core.weaviate.client import close_client
            close_client()


if __name__ == "__main__":
    asyncio.run(main())
