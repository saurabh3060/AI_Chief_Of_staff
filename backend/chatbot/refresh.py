"""
chatbot/refresh.py
==================
Incremental reindex entry point, called by the email pipeline after it
writes freshly fetched mail into the chatbot's source table.

Why this exists
---------------
``chatbot/routes.py:_initialize_weaviate`` only ingests when the
collection is EMPTY (``count == 0``). That is the right behaviour at
startup — it avoids re-embedding the whole mailbox on every boot — but it
means rows the email pipeline adds later are never picked up: the
collection is non-empty, so ingestion is skipped forever and the chatbot
stays frozen on whatever it indexed first.

``trigger_reindex`` closes that gap. The email pipeline calls it once per
run, right after upserting into ``bu_emails``
(``agent_email.postgres_store.save_emails_to_bu_emails``), so new mail
becomes answerable without restarting the backend.

Design notes
------------
* **Upsert, not rebuild.** Uses ``recreate=False``, so the authoritative
  row set is written over the existing collection using the deterministic
  per-message UUIDs (``uuid5`` of the message id). Unchanged messages keep
  their object id and are simply overwritten in place, so the object count
  tracks the row count instead of growing. Orphans (messages deleted at
  the source) are deliberately left alone — clearing those is the job of
  the explicit ``POST /chatbot/index`` rebuild, which passes
  ``recreate=True``.
* **Never raises.** A reindex failure leaves the previous index serving
  rather than knocking the chatbot offline mid-pipeline.
* **Serialised.** A module-level lock prevents two overlapping pipeline
  runs from ingesting concurrently.
* **Weaviate only.** The reindex updates the Weaviate collection used
  by the chatbot's hybrid retrieval backend.
"""

import asyncio
import time

from core.config import get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("chatbot")

# Guards against concurrent ingestion if two pipeline runs overlap.
# Created lazily so importing this module never touches an event loop.
_reindex_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _reindex_lock
    if _reindex_lock is None:
        _reindex_lock = asyncio.Lock()
    return _reindex_lock


async def trigger_reindex(reason: str = "") -> int:
    """Re-embed the chatbot's current source rows into Weaviate.

    Args:
        reason: short free-text label for the logs (e.g. "email-pipeline"),
            so it is obvious what prompted the reindex.

    Returns:
        Number of objects ingested, or 0 if the reindex was skipped or
        failed. Never raises.
    """
    settings = get_settings()

    if not settings.chatbot_auto_reindex:
        logger.debug(
            "[REINDEX] skipped (CHATBOT_AUTO_REINDEX disabled) reason=%s", reason
        )
        return 0

    # Imported lazily: keeps this module importable (and the email agent
    # free of a hard chatbot dependency) even when Weaviate extras are
    # unavailable, and avoids a circular import with chatbot.routes.
    try:
        from chatbot.bm25_index import build_bm25
        from chatbot.indexer import build_weaviate_index
        from chatbot.retriever import fetch_all_chunks_weaviate
        from chatbot.routes import get_chatbot_state
        from core.weaviate.client import get_client
    except Exception as e:
        logger.error("[REINDEX] could not load chatbot components: %s", e)
        return 0

    async with _get_lock():
        t0 = time.perf_counter()
        try:
            client = get_client()

            # recreate=False: upsert the live row set over the existing
            # collection. Deterministic UUIDs mean unchanged messages are
            # overwritten rather than duplicated.
            try:
                inserted = await build_weaviate_index(client, recreate=False)
            except RuntimeError as e:
                # build_weaviate_index asserts stored_objects == source_rows.
                # An incremental upsert cannot satisfy that once the
                # collection holds orphans - objects whose source row was
                # deleted - because upserting never removes anything. Left
                # alone this would wedge every future reindex, so fall back
                # to one full rebuild, which drops the collection first and
                # clears the orphans.
                logger.warning(
                    "[REINDEX] incremental upsert failed validation (%s); "
                    "falling back to a full rebuild to clear orphans", e,
                )
                inserted = await build_weaviate_index(client, recreate=True)

            # Sync Weaviate with Postgres: remove deactivated emails,
            # restore any objects that are in Postgres but missing from
            # Weaviate. No embedding model calls.
            try:
                from chatbot.weaviate_sync import ensure_weaviate_consistent
                sync_result = await ensure_weaviate_consistent(client)
                logger.info(
                    "[REINDEX] sync: added=%d removed=%d already_ok=%d",
                    sync_result.get("added", 0),
                    sync_result.get("removed", 0),
                    sync_result.get("already_ok", 0),
                )
            except Exception as e:
                logger.warning("[REINDEX] sync failed (non-fatal): %s", e)

            # BM25 re-ranks whatever Weaviate returns, so it has to be
            # rebuilt from the new message set or re-ranking would score
            # against a stale corpus.
            chunks = await asyncio.to_thread(fetch_all_chunks_weaviate, client)

            state = get_chatbot_state()
            state["wv_client"] = client
            state["bm25"] = build_bm25(chunks)
            state["ready"] = True

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "[REINDEX] complete reason=%s objects=%d bm25_messages=%d in %.0fms",
                reason or "unspecified", inserted, len(chunks), elapsed,
            )
            return inserted
        except Exception as e:
            # Leave the previous index (and `ready` flag) untouched so the
            # chatbot keeps answering from the last good state.
            logger.error(
                "[REINDEX] failed reason=%s: %s - keeping previous index",
                reason or "unspecified", e,
            )
            return 0
