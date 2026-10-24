"""
scripts/wipe_all_data.py
========================
DESTRUCTIVE maintenance script — removes ALL of this project's data from
every storage layer:

* PostgreSQL  — TRUNCATE every table this app owns (RESTART IDENTITY), so
                row data and sequences are reset. The tables themselves are
                left in place (the companion ``rebuild_all.py`` / app
                startup recreate/verify schema anyway).
* Vectors     — the email agent's vector store, backend-aware (follows
                ``VECTOR_STORE_BACKEND``):
                  - weaviate: DELETE this project's owned collections
                    entirely, dropping their objects AND schema/vector
                    indexes. Only collections whose names match this
                    project's prefix are touched; a shared Weaviate server's
                    other classes are never deleted (same gated primitive as
                    weaviate_admin.py).
                  - faiss: CLEAR the local on-disk FAISS indexes for the
                    owned namespaces (objects removed, empty index files
                    kept).
                The chatbot's own collection is always a Weaviate collection
                regardless of the email backend, so it is dropped separately
                (best-effort — skipped if Weaviate is unreachable).
* Redis       — scoped flush of every key under this app's key prefix
                (``clear_all_scoped``). Never ``FLUSHDB``/``FLUSHALL``, so a
                shared Redis instance keeps every other project's keys.

This is hard to reverse: PostgreSQL is the system of record, so once it is
truncated the only way back is to re-run the mailbox pipeline
(``run_pipeline.py``). Because of that, the script is DRY-RUN by default
and makes NO changes unless you pass ``--confirm``.

Usage (run from the backend/ directory, venv active)
----------------------------------------------------
    # Dry run — prints exactly what WOULD be wiped, changes nothing:
    python scripts/wipe_all_data.py

    # Actually wipe everything (all three layers):
    python scripts/wipe_all_data.py --confirm

    # Limit to specific layers (vector layer named "vectors"; "weaviate"
    # and "faiss" are accepted as aliases):
    python scripts/wipe_all_data.py --confirm --only postgres,redis
    python scripts/wipe_all_data.py --confirm --only vectors

Exit code 0 on success (including a clean dry-run), non-zero on error.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from core.config import get_settings  # noqa: E402

# Every Postgres table this application owns. Trackers + email agent +
# chatbot source/history. bu_emails is interpolated from settings so a
# custom CHATBOT_PG_TABLE is honoured.
_STATIC_OWNED_TABLES = [
    "email_clusters",
    "email_topics",
    "meeting_embeddings",
    "meeting_history",
    "meeting_notes",
    "email_summaries",
    "cluster_publish_staging",
    "tracker_item_overrides",
    "tracker_completed_items",
    "tracker_deleted_items",
    "chatbot_history",
    "chatbot_embeddings",
    # NOTE: delegated Graph OAuth tokens are no longer stored in PostgreSQL —
    # both the access and refresh tokens live in AWS Secrets Manager, which
    # this script never touches. Wiping them (to force a fresh sign-in) is a
    # separate, deliberate action against the secret, not a table truncate.
]


def _section(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def _owned_tables(settings, include_tokens: bool) -> list[str]:
    # `include_tokens` is retained for CLI backwards-compatibility but is now
    # a no-op: delegated Graph OAuth tokens live in AWS Secrets Manager, not
    # in any PostgreSQL table, so there is nothing token-related to truncate.
    tables = list(_STATIC_OWNED_TABLES)
    # Chatbot RAG source table (configurable name).
    bu = (settings.chatbot_pg_table or "bu_emails").strip()
    if bu and bu not in tables:
        tables.append(bu)
    return tables


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------
async def wipe_postgres(settings, confirm: bool, include_tokens: bool) -> None:
    _section("PostgreSQL")
    if not settings.postgres_dsn:
        print("SKIPPED: POSTGRES_DSN is empty — Postgres disabled.")
        return

    from core.db import init_db, close_db, get_pool

    await init_db()
    try:
        pool = get_pool()
        if pool is None:
            print("SKIPPED: pool unavailable after init_db().")
            return

        async with pool.acquire() as conn:
            existing = {
                r["tablename"]
                for r in await conn.fetch(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public';"
                )
            }
            targets = [t for t in _owned_tables(settings, include_tokens) if t in existing]
            missing = [t for t in _owned_tables(settings, include_tokens) if t not in existing]

            print(f"Owned tables present ({len(targets)}): "
                  f"{', '.join(targets) if targets else '(none)'}")
            if missing:
                print(f"Not present (skipped)        : {', '.join(missing)}")

            if not targets:
                print("Nothing to truncate.")
                return

            # Show current row counts so the operator sees what is at stake.
            for t in targets:
                cnt = await conn.fetchval(f'SELECT COUNT(*) FROM "{t}";')
                print(f"  {t:26s} {cnt} row(s)")

            if not confirm:
                print("\nDRY-RUN: would TRUNCATE the tables above "
                      "(RESTART IDENTITY CASCADE). Re-run with --confirm.")
                return

            # One TRUNCATE for all tables keeps FK ordering a non-issue and
            # is atomic. RESTART IDENTITY resets BIGSERIAL sequences.
            quoted = ", ".join(f'"{t}"' for t in targets)
            await conn.execute(f"TRUNCATE {quoted} RESTART IDENTITY CASCADE;")
            print(f"\nTRUNCATED {len(targets)} table(s). All row data removed, "
                  "identities reset. Schema (tables/indexes) left intact.")
    finally:
        await close_db()


# ---------------------------------------------------------------------------
# Email vector store (Weaviate or FAISS, selected by VECTOR_STORE_BACKEND)
# ---------------------------------------------------------------------------
async def wipe_vectors(settings, confirm: bool) -> None:
    """Wipe the email agent's vector store for whichever backend is active.

    The email agent selects its vector backend at runtime via
    ``VECTOR_STORE_BACKEND`` (Weaviate by default, local on-disk FAISS when
    set to "faiss"). This wipe mirrors that choice so it removes the SAME
    store the pipeline actually writes to:

      * weaviate -> delete this project's owned email collections entirely
                    (objects + schema + HNSW index), via the gated admin
                    delete so a non-owned class can never be removed.
      * faiss    -> clear the local on-disk FAISS indexes for the owned
                    namespaces (objects removed, empty index files kept).

    The chatbot's own collection is ALWAYS a Weaviate collection regardless
    of the email backend, so it is handled separately (best-effort — a FAISS
    deployment may not have Weaviate reachable at all).
    """
    from agent_email import vector_store_backend as wvs

    backend = wvs.active_backend()
    _section(f"Email vector store (backend={backend})")

    if backend == "faiss":
        await _wipe_vectors_faiss(settings, confirm, wvs)
    else:
        await _wipe_vectors_weaviate(settings, confirm, wvs)

    # The chatbot collection is Weaviate-only in every configuration.
    await _wipe_chatbot_collection(settings, confirm)


async def _wipe_vectors_faiss(settings, confirm: bool, wvs) -> None:
    owned = wvs.OWNED_NAMESPACES
    print(f"Local FAISS index dir: {settings.resolved_faiss_index_dir}")
    print("Owned email namespaces (objects will be CLEARED, empty index kept):")
    for namespace in owned:
        cnt = await asyncio.to_thread(wvs.object_count, namespace, settings)
        state = "missing" if cnt < 0 else f"{cnt} vectors"
        print(f"  {namespace:10s} [{state}]")

    if not confirm:
        print("\nDRY-RUN: would clear every vector in the owned FAISS namespaces "
              "above (index files kept, emptied). Re-run with --confirm.")
        return

    for namespace in owned:
        cleared = await asyncio.to_thread(
            wvs.clear_namespace_objects, namespace, settings
        )
        print(f"  {namespace:10s} -> cleared {cleared} vector(s)")
    print("\nCLEARED all owned FAISS namespaces (local index files emptied).")


async def _wipe_vectors_weaviate(settings, confirm: bool, wvs) -> None:
    from core.weaviate.client import get_client, close_client

    # `wvs` is the backend-neutral dispatch seam
    # (agent_email.vector_store_backend). It only re-exports the functions
    # that exist in BOTH the Weaviate and FAISS backends, so Weaviate-only
    # admin helpers (`owned_collection_names`, `_admin_delete_collection`)
    # are NOT attributes on it. Resolve those from the concrete Weaviate
    # module directly — the same module scripts/weaviate_admin.py uses.
    from agent_email import weaviate_vector_store as wv_admin

    try:
        await asyncio.to_thread(get_client, settings)
    except Exception as exc:  # noqa: BLE001
        print(f"SKIPPED/ERROR: could not connect to Weaviate: "
              f"{type(exc).__name__}: {exc}")
        return

    try:
        owned = wv_admin.owned_collection_names(settings)
        print("Owned email collections (will be DELETED entirely — objects + "
              "schema + vector index):")
        for namespace, name in owned.items():
            cnt = await asyncio.to_thread(wv_admin.object_count, namespace, settings)
            state = "missing" if cnt < 0 else f"{cnt} objects"
            print(f"  {namespace:10s} -> {name:28s} [{state}]")

        if not confirm:
            print("\nDRY-RUN: would DELETE the collections above (dropping their "
                  "objects, schema and HNSW vector indexes). Re-run with --confirm.")
            return

        # Email namespaces: use the gated admin delete so a non-owned class
        # can never be removed, even by a bug.
        for namespace in owned:
            deleted = await asyncio.to_thread(
                wv_admin._admin_delete_collection, namespace, settings
            )
            print(f"  {namespace:10s} -> "
                  f"{'deleted' if deleted else 'nothing to delete'}")

        print("\nDELETED all owned Weaviate email collections (schema + vector "
              "indexes gone).")
    finally:
        await asyncio.to_thread(close_client)


async def _wipe_chatbot_collection(settings, confirm: bool) -> None:
    """Delete the chatbot's Weaviate collection (always Weaviate).

    Best-effort: a FAISS email deployment may not run Weaviate at all, in
    which case this reports that it could not connect and moves on rather
    than failing the whole wipe.
    """
    from core.weaviate.client import get_client, close_client

    chatbot_name = settings.chatbot_weaviate_collection
    try:
        client = await asyncio.to_thread(get_client, settings)
    except Exception as exc:  # noqa: BLE001
        print(f"Chatbot collection ({chatbot_name}): SKIPPED — Weaviate not "
              f"reachable ({type(exc).__name__}: {exc}).")
        return

    try:
        try:
            chatbot_exists = await asyncio.to_thread(
                client.collections.exists, chatbot_name
            )
        except Exception:  # noqa: BLE001
            chatbot_exists = False
        print(f"Chatbot collection           : {chatbot_name} "
              f"[{'exists' if chatbot_exists else 'missing'}]")

        if not confirm:
            print("DRY-RUN: would DELETE the chatbot collection above. "
                  "Re-run with --confirm.")
            return

        if chatbot_exists:
            await asyncio.to_thread(client.collections.delete, chatbot_name)
            print(f"  chatbot    -> deleted {chatbot_name!r}")
        else:
            print(f"  chatbot    -> nothing to delete")
    finally:
        await asyncio.to_thread(close_client)


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
async def wipe_redis(settings, confirm: bool) -> None:
    _section("Redis")
    if not settings.redis_enabled:
        print("SKIPPED: REDIS_HOST is empty — Redis disabled.")
        return

    from core.redis.client import get_redis, ping as redis_ping, close_redis
    from agent_email import redis_cache

    try:
        if not await redis_ping(settings):
            print("SKIPPED: Redis configured but not reachable (PING failed).")
            return

        client = await get_redis(settings)
        owned = 0
        async for _ in client.scan_iter(match=f"{settings.redis_key_prefix}*"):
            owned += 1
        print(f"Owned keys (prefix {settings.redis_key_prefix!r}): {owned}")

        if not confirm:
            print("\nDRY-RUN: would delete every key under this prefix "
                  "(scoped — never FLUSHDB). Re-run with --confirm.")
            return

        deleted = await redis_cache.clear_all_scoped(settings)
        print(f"\nDELETED {deleted} key(s) under {settings.redis_key_prefix!r}. "
              "Other projects' keys on this Redis were untouched.")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}")
    finally:
        await close_redis()


# The vector layer can be named "vectors" (backend-neutral) or, for
# backward compatibility, "weaviate"/"faiss". All three resolve to the same
# layer; the active backend is chosen at runtime by VECTOR_STORE_BACKEND.
_VECTOR_LAYER_ALIASES = {"vectors", "weaviate", "faiss"}


def _parse_only(value: str | None) -> set[str]:
    valid = {"postgres", "redis"} | _VECTOR_LAYER_ALIASES
    if not value:
        # Default: all three logical layers (vector layer named "vectors").
        return {"postgres", "vectors", "redis"}
    chosen = {part.strip().lower() for part in value.split(",") if part.strip()}
    bad = chosen - valid
    if bad:
        print(f"ERROR: unknown layer(s) in --only: {', '.join(sorted(bad))}. "
              f"Valid: {', '.join(sorted(valid))}.")
        sys.exit(2)
    # Collapse any vector-layer alias to the canonical "vectors".
    if chosen & _VECTOR_LAYER_ALIASES:
        chosen -= _VECTOR_LAYER_ALIASES
        chosen.add("vectors")
    return chosen


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="DESTRUCTIVE: wipe all of this project's data from "
                    "PostgreSQL, the email vector store (Weaviate or FAISS, "
                    "incl. indexes) and Redis."
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Actually perform the wipe. Without it, the script is a dry-run.",
    )
    parser.add_argument(
        "--only", metavar="LAYERS",
        help="Comma-separated subset to wipe: postgres,vectors,redis "
             "(default: all three). The 'vectors' layer targets the email "
             "agent's active backend (Weaviate or FAISS); 'weaviate'/'faiss' "
             "are accepted as aliases for it.",
    )
    parser.add_argument(
        "--include-tokens", action="store_true",
        help="Deprecated no-op. Delegated Graph tokens now live in AWS "
             "Secrets Manager, not PostgreSQL, so there is no token table to "
             "truncate. Flag kept so existing automation does not break.",
    )
    args = parser.parse_args()

    settings = get_settings()
    layers = _parse_only(args.only)

    mode = "LIVE (changes will be made)" if args.confirm else "DRY-RUN (no changes)"
    print(f"wipe_all_data — mode: {mode}")
    print(f"Layers        : {', '.join(sorted(layers))}")
    if args.include_tokens:
        print("Note          : --include-tokens is a no-op; delegated tokens "
              "live in AWS Secrets Manager, not PostgreSQL.")

    if "postgres" in layers:
        await wipe_postgres(settings, args.confirm, args.include_tokens)
    if "vectors" in layers:
        await wipe_vectors(settings, args.confirm)
    if "redis" in layers:
        await wipe_redis(settings, args.confirm)

    _section("DONE")
    if args.confirm:
        print("Wipe complete. Run scripts/rebuild_all.py (or run_pipeline.py) "
              "to repopulate.")
    else:
        print("Dry-run complete. Nothing was changed. Add --confirm to wipe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
