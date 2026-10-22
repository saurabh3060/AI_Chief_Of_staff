"""
chatbot/weaviate_sync.py
============================
Weaviate sync / self-healing restore for the chatbot.

Weaviate is a rebuildable copy of PostgreSQL, never the master. These
helpers (re)build Weaviate from the embeddings ALREADY STORED in
``chatbot_embeddings`` — they never call the embedding model. The vector
ids are the deterministic UUIDs already written by the indexer, so
restored objects line up exactly with what was there before.

Modelled on ``agent_email/weaviate_sync.py``.
"""

from __future__ import annotations

import asyncio
import uuid

import numpy as np

from core.config import get_settings
from core.db import get_pool
from core.logging_config import get_agent_logger
from agent_email.embedding_json import json_to_vec, jsonb_to_embedding_str

logger = get_agent_logger("chatbot")


async def _load_active_embeddings_from_pg() -> list[dict]:
    """Load embeddings for active bu_emails rows from chatbot_embeddings.

    Joins chatbot_embeddings with bu_emails on message_id, filtering
    to is_active = 1 only. Returns dicts with message_id, vector,
    weaviate_id, and the full chunk properties needed by Weaviate.
    """
    pool = get_pool()
    if pool is None:
        return []

    settings = get_settings()
    table = settings.chatbot_pg_table

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT ce.message_id, ce.embedding, ce.weaviate_id "
                f"FROM chatbot_embeddings ce "
                f"INNER JOIN {table} be ON ce.message_id = be.message_id "
                f"WHERE be.is_active = 1 "
                f"  AND ce.embedding IS NOT NULL"
            )
    except Exception as e:
        logger.error("[CHATBOT-SYNC] could not load embeddings from PG: %s", e)
        return []

    out: list[dict] = []
    for r in rows:
        vec = json_to_vec(jsonb_to_embedding_str(r["embedding"]))
        if vec is not None:
            out.append({
                "message_id": r["message_id"],
                "vector": vec,
                "weaviate_id": r["weaviate_id"],
            })
    return out


async def _active_message_ids_from_pg() -> set[str]:
    """Return the set of message_ids that are active in bu_emails.

    Checks chatbot_embeddings first (the usual case after at least one
    indexing run).  Falls back to querying bu_emails directly when the
    embeddings table is empty — this prevents the sync from treating
    every Weaviate object as stale on a fresh deployment where
    build_weaviate_index populated Weaviate but chatbot_embeddings
    hasn't been filled yet.
    """
    pool = get_pool()
    if pool is None:
        return set()

    settings = get_settings()
    table = settings.chatbot_pg_table

    try:
        async with pool.acquire() as conn:
            # Primary path: message_ids with both an embedding AND an active source row.
            rows = await conn.fetch(
                f"SELECT ce.message_id "
                f"FROM chatbot_embeddings ce "
                f"INNER JOIN {table} be ON ce.message_id = be.message_id "
                f"WHERE be.is_active = 1 "
                f"  AND ce.embedding IS NOT NULL"
            )
            if rows:
                return {r["message_id"] for r in rows}

            # Fallback: chatbot_embeddings is empty (fresh deploy / first run).
            # Use bu_emails directly so we don't nuke valid Weaviate objects.
            logger.info(
                "[CHATBOT-SYNC] chatbot_embeddings is empty — "
                "falling back to bu_emails for active message_ids"
            )
            rows = await conn.fetch(
                f"SELECT message_id FROM {table} "
                f"WHERE is_active = 1 "
                f"  AND NULLIF(BTRIM(message_id), '') IS NOT NULL"
            )
            return {r["message_id"] for r in rows}
    except Exception as e:
        logger.error("[CHATBOT-SYNC] could not load active ids: %s", e)
        return set()


def _weaviate_object_ids(client) -> dict[str, str]:
    """Return {message_id: weaviate_uuid} for all objects in the chatbot collection."""
    from chatbot._vec_backend import vec_store
    collection = vec_store().get_collection(client)
    result: dict[str, str] = {}
    offset = 0
    batch_size = 100

    while True:
        response = collection.query.fetch_objects(limit=batch_size, offset=offset)
        if not response.objects:
            break
        for obj in response.objects:
            msg_id = obj.properties.get("message_id", "")
            if msg_id:
                result[msg_id] = str(obj.uuid)
        if len(response.objects) < batch_size:
            break
        offset += batch_size

    return result


async def ensure_weaviate_consistent(client, settings=None) -> dict:
    """Compare Weaviate with Postgres and fix discrepancies.

    - Objects in Weaviate whose message_id is not active in Postgres → delete
    - Embeddings in Postgres with no Weaviate object → insert from stored vector

    Returns a summary dict: {added: int, removed: int, already_ok: int}.
    """
    from chatbot._vec_backend import vec_store
    vs = vec_store()
    vs.ensure_collection(client, settings)

    # Load both sides
    wv_objects = await asyncio.to_thread(_weaviate_object_ids, client)
    active_ids = await _active_message_ids_from_pg()

    wv_ids = set(wv_objects.keys())
    to_remove = wv_ids - active_ids   # in Weaviate but not active in PG
    to_add = active_ids - wv_ids      # in PG but missing from Weaviate
    already_ok = wv_ids & active_ids

    summary = {"added": 0, "removed": 0, "already_ok": len(already_ok)}

    # Remove stale objects (deactivated emails)
    if to_remove:
        collection = vs.get_collection(client)
        for msg_id in to_remove:
            try:
                wv_uuid = wv_objects[msg_id]
                collection.data.delete_by_id(wv_uuid)
            except Exception as e:
                logger.warning("[CHATBOT-SYNC] could not delete %s: %s", msg_id, e)
        summary["removed"] = len(to_remove)
        logger.info("[CHATBOT-SYNC] removed %d stale object(s) from Weaviate", len(to_remove))

    # Add missing objects from Postgres embeddings
    if to_add:
        pg_embeddings = await _load_active_embeddings_from_pg()
        # Filter to only the ones actually missing
        missing = {e["message_id"]: e for e in pg_embeddings if e["message_id"] in to_add}

        if missing:
            # We need the full chunk properties to insert into Weaviate.
            # Load them from bu_emails.
            pool = get_pool()
            active = settings or get_settings()
            table = active.chatbot_pg_table

            try:
                async with pool.acquire() as conn:
                    placeholders = ", ".join(f"${i+1}" for i in range(len(missing)))
                    rows = await conn.fetch(
                        f"SELECT message_id, conversation_id, subject, raw_content, "
                        f"attachment_data, received_at, from_address "
                        f"FROM {table} WHERE message_id = ANY($1::text[])",
                        list(missing.keys()),
                    )
            except Exception as e:
                logger.error("[CHATBOT-SYNC] could not load bu_emails rows: %s", e)
                rows = []

            collection = vs.get_collection(client)
            inserted = 0
            for row in rows:
                msg_id = row["message_id"]
                emb = missing.get(msg_id)
                if not emb:
                    continue

                wv_uuid = emb["weaviate_id"] or str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"rbl-chatbot-message:{msg_id}")
                )

                # Minimal properties — enough for the chatbot's retrieval
                # to return usable results. A full reindex (POST /chatbot/index)
                # will populate all fields including thread_raw_content.
                properties = {
                    "message_id": msg_id,
                    "conversation_id": row["conversation_id"] or "",
                    "subject": row["subject"] or "",
                    "body": row["raw_content"] or "",
                    "received_at": str(row["received_at"] or ""),
                    "from_address": row["from_address"] or "",
                    "source_type": "email_message",
                }

                try:
                    collection.data.insert(
                        properties=properties,
                        vector=emb["vector"].tolist(),
                        uuid=wv_uuid,
                    )
                    inserted += 1
                except Exception as e:
                    logger.warning("[CHATBOT-SYNC] could not insert %s: %s", msg_id, e)

            summary["added"] = inserted
            logger.info("[CHATBOT-SYNC] added %d missing object(s) to Weaviate from Postgres", inserted)

    if not to_remove and not to_add:
        logger.info(
            "[CHATBOT-SYNC] Weaviate is consistent with Postgres (%d object(s))",
            len(already_ok),
        )

    return summary
