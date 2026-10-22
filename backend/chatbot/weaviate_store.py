"""
chatbot/weaviate_store.py
============================
Weaviate collection management + ingestion for the chatbot's message-level
hybrid retrieval (vector + BM25 search via Weaviate's `hybrid()` query).

Reuses the backend's shared Weaviate client (core.weaviate.client) rather
than opening a second connection - the chatbot and the email cluster
pipeline (agent_email/weaviate_vector_store.py) talk to the same Weaviate
deployment but use entirely separate collections, so neither feature's
data can collide with the other's:

    - agent_email: "EmailAgentClusters" / "EmailAgentMeetings"
    - chatbot:      settings.chatbot_weaviate_collection (default
                     "ChatbotEmailMessage")

One Weaviate object = one individual email message. Each message carries
its parent thread's conversation_id and a full rendering of every
message in that thread (messages_json / thread_raw_content) so the
complete conversation can be reconstructed at answer time without a
second lookup, mirroring chatbot_final's design.
"""

import uuid

import weaviate
from weaviate.classes.config import Configure, DataType, Property

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger
from core.weaviate.client import get_client

logger = get_agent_logger("chatbot")


def _collection_name(settings: Settings | None = None) -> str:
    active = settings or get_settings()
    return active.chatbot_weaviate_collection


def ensure_collection(client: weaviate.WeaviateClient, settings: Settings | None = None) -> None:
    """Create the message-level collection if it does not exist."""
    name = _collection_name(settings)
    if client.collections.exists(name):
        logger.info("Weaviate collection '%s' already exists - skipping create.", name)
        return

    logger.info("Creating Weaviate collection '%s'...", name)
    client.collections.create(
        name=name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            # ── Message identity ──────────────────────────────
            Property(name="message_id",         data_type=DataType.TEXT),
            Property(name="conversation_id",    data_type=DataType.TEXT),
            Property(name="message_index",      data_type=DataType.INT),

            # ── Per-message content (what was embedded) ───────
            Property(name="subject",            data_type=DataType.TEXT),
            Property(name="body",               data_type=DataType.TEXT),
            Property(name="attachment_data",    data_type=DataType.TEXT),
            Property(name="metadata_summary",   data_type=DataType.TEXT),
            Property(name="received_at",        data_type=DataType.TEXT),
            Property(name="from_address",       data_type=DataType.TEXT),
            Property(name="has_attachment",     data_type=DataType.BOOL),

            # ── Thread-level context (denormalised onto every message) ──
            Property(name="thread_subject",       data_type=DataType.TEXT),
            Property(name="thread_raw_content",   data_type=DataType.TEXT),
            Property(name="first_received_at",    data_type=DataType.TEXT),
            Property(name="last_received_at",     data_type=DataType.TEXT),
            Property(name="participants",         data_type=DataType.TEXT_ARRAY),
            Property(name="message_ids",          data_type=DataType.TEXT_ARRAY),
            Property(name="message_count",        data_type=DataType.INT),
            Property(name="messages_json",        data_type=DataType.TEXT),

            Property(name="source_type",        data_type=DataType.TEXT),
            Property(name="email_id",           data_type=DataType.INT),
        ],
    )
    logger.info("Weaviate collection '%s' created.", name)


def recreate_collection(client: weaviate.WeaviateClient, settings: Settings | None = None) -> None:
    """Delete and recreate the collection for a clean full rebuild."""
    name = _collection_name(settings)
    if client.collections.exists(name):
        logger.info("Deleting collection '%s' for clean rebuild...", name)
        client.collections.delete(name)
    ensure_collection(client, settings)


def get_collection(client: weaviate.WeaviateClient, settings: Settings | None = None):
    """Return the configured chatbot message collection handle."""
    return client.collections.get(_collection_name(settings))


def ensure_collection_exists(client: weaviate.WeaviateClient, settings: Settings | None = None) -> bool:
    """Self-heal: re-create AND refill the chatbot's Weaviate collection if
    it has gone missing underneath a long-running backend process.

    `ensure_collection()` only runs once, at startup. If Weaviate is later
    restarted/reset without this process also restarting (e.g. a dev
    container recreated without a persistent volume), the backend keeps
    believing the collection is ready while every query actually fails
    with "could not find class ... in schema". Calling this cheaply
    before each query means a missing collection is recreated AND
    repopulated from PostgreSQL (see _refill_from_postgres below) before
    the user's question is answered, instead of throwing straight at
    them or quietly answering from an empty collection.

    Returns True if the collection had to be (re)created, False if it
    was already present.
    """
    name = _collection_name(settings)
    if client.collections.exists(name):
        return False
    logger.warning(
        "Weaviate collection '%s' is missing (likely a Weaviate restart/reset "
        "since this process last checked) - recreating it now.",
        name,
    )
    ensure_collection(client, settings)
    _refill_from_postgres(client, settings)
    return True


def _refill_from_postgres(client: weaviate.WeaviateClient, settings: Settings | None = None) -> None:
    """Repopulate a freshly (re)created collection from PostgreSQL's
    already-computed embeddings, so the very query that triggered the
    self-heal still gets real results instead of 'nothing found' from
    an empty collection. PostgreSQL is the system of record; Weaviate
    is a rebuildable copy (see chatbot.weaviate_sync).

    Runs from a sync call site, so the async restore coroutine is
    scheduled onto the main event loop the connection pool was created
    on (see core.db.get_main_loop()) and awaited here with a generous
    timeout. Never raises: a failed refill still leaves the collection
    usable (just empty until the next POST /chatbot/index or background
    sync), which is strictly better than crashing the request.
    """
    import asyncio

    from core.db import get_main_loop
    from chatbot.weaviate_sync import ensure_weaviate_consistent

    try:
        loop = get_main_loop()
        if loop is None:
            logger.error(
                "Could not refill Weaviate collection '%s' from PostgreSQL: "
                "no main event loop available.", _collection_name(settings),
            )
            return
        future = asyncio.run_coroutine_threadsafe(
            ensure_weaviate_consistent(client, settings), loop,
        )
        summary = future.result(timeout=120)
        logger.info(
            "Refilled Weaviate collection '%s' from PostgreSQL: added=%d already_ok=%d removed=%d",
            _collection_name(settings),
            summary.get("added", 0), summary.get("already_ok", 0), summary.get("removed", 0),
        )
    except Exception as e:
        logger.error(
            "Could not refill Weaviate collection '%s' from PostgreSQL: %s",
            _collection_name(settings), e,
        )


def ingest(
    client: weaviate.WeaviateClient,
    chunks: list[dict],
    embeddings,
    recreate: bool = False,
    settings: Settings | None = None,
) -> int:
    """Write one deterministic Weaviate object per email message.

    Args:
        client:     Connected Weaviate client (core.weaviate.client.get_client()).
        chunks:     List of message chunk dicts from chatbot.indexer.load_emails().
        embeddings: numpy array of shape (n, dim) - same order as chunks.

    Returns:
        Number of objects successfully inserted.
    """
    if recreate:
        recreate_collection(client, settings)
    else:
        ensure_collection(client, settings)
    collection = get_collection(client, settings)

    total = len(chunks)
    name = _collection_name(settings)
    logger.info("Ingesting %d messages into Weaviate collection '%s'...", total, name)

    with collection.batch.dynamic() as batch:
        for i, chunk in enumerate(chunks):
            vector = embeddings[i].tolist()
            message_id = chunk["message_id"]

            properties = {
                "message_id":        message_id,
                "conversation_id":   chunk.get("conversation_id", ""),
                "message_index":     int(chunk.get("message_index", 0)),

                "subject":           chunk.get("subject", ""),
                "body":              chunk.get("body", ""),
                "attachment_data":   chunk.get("attachment_data", ""),
                "metadata_summary":  chunk.get("metadata_summary", ""),
                "received_at":       chunk.get("received_at", ""),
                "from_address":      chunk.get("from_address", ""),
                "has_attachment":    bool(chunk.get("has_attachment", False)),

                "thread_subject":      chunk.get("thread_subject", ""),
                "thread_raw_content":  chunk.get("thread_raw_content", ""),
                "first_received_at":   chunk.get("first_received_at", ""),
                "last_received_at":    chunk.get("last_received_at", ""),
                "participants":        chunk.get("participants", []),
                "message_ids":         chunk.get("message_ids", []),
                "message_count":       int(chunk.get("message_count", 0)),
                "messages_json":       chunk.get("messages_json", "[]"),

                "source_type":       chunk.get("source_type", "email_message"),
                "email_id":          int(chunk.get("email_id", i)),
            }

            object_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"rbl-chatbot-message:{message_id}")
            batch.add_object(properties=properties, vector=vector, uuid=object_uuid)

            if (i + 1) % 50 == 0:
                logger.info("  Ingested %d/%d...", i + 1, total)

    failed = collection.batch.failed_objects
    if failed:
        first_errors = "; ".join(str(error) for error in failed[:3])
        raise RuntimeError(
            f"Weaviate rejected {len(failed)} of {total} objects. First errors: {first_errors}"
        )

    logger.info("Weaviate message ingestion complete: %d/%d objects inserted.", total, total)
    return total


def collection_count(client: weaviate.WeaviateClient, settings: Settings | None = None) -> int:
    """Return total number of objects currently stored in the collection.

    Returns 0 if the collection does not exist yet.
    """
    name = _collection_name(settings)
    if not client.collections.exists(name):
        return 0
    collection = get_collection(client, settings)
    response = collection.aggregate.over_all(total_count=True)
    return response.total_count or 0


def fetch_all_chunks(client: weaviate.WeaviateClient, settings: Settings | None = None) -> list[dict]:
    """Fetch all message chunks from Weaviate to build a local BM25 index.

    Called once at startup so BM25 re-ranking can run locally after
    Weaviate returns its top-k results.
    """
    collection = get_collection(client, settings)

    all_chunks: list[dict] = []
    offset = 0
    batch_size = 100

    while True:
        response = collection.query.fetch_objects(limit=batch_size, offset=offset)
        objects = response.objects
        if not objects:
            break

        for obj in objects:
            props = obj.properties
            chunk = {
                "message_id":        props.get("message_id", ""),
                "conversation_id":   props.get("conversation_id", ""),
                "message_index":     props.get("message_index", 0) or 0,
                "subject":           props.get("subject", ""),
                "body":              props.get("body", ""),
                "attachment_data":   props.get("attachment_data", ""),
                "metadata_summary":  props.get("metadata_summary", ""),
                "received_at":       props.get("received_at", ""),
                "from_address":      props.get("from_address", ""),
                "has_attachment":    props.get("has_attachment", False) or False,
                "thread_subject":      props.get("thread_subject", ""),
                "thread_raw_content":  props.get("thread_raw_content", ""),
                "raw_content":         props.get("thread_raw_content", ""),
                "first_received_at":   props.get("first_received_at", ""),
                "last_received_at":    props.get("last_received_at", ""),
                "participants":        props.get("participants", []) or [],
                "message_ids":         props.get("message_ids", []) or [],
                "message_count":       props.get("message_count", 0) or 0,
                "messages_json":       props.get("messages_json", "[]"),
                "source_type":      props.get("source_type", "email_message"),
                "email_id":         props.get("email_id", 0),
            }
            if chunk.get("message_id") and (chunk.get("thread_raw_content") or "").strip():
                all_chunks.append(chunk)

        offset += batch_size
        if len(objects) < batch_size:
            break

    logger.info("Fetched %d messages from Weaviate for BM25.", len(all_chunks))
    return all_chunks
