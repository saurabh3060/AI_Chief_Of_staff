"""
chatbot/indexer.py
======================
Load email messages, aggregate conversations, and persist vector indexes
for the chatbot's RAG pipeline.

Ported from the standalone chatbot_final project's indexer.py, adapted to:
  - read tuning knobs and paths from core.config.get_settings() instead
    of a module-local config.py
  - fetch raw rows via chatbot.postgres_store.fetch_email_rows() (the
    shared asyncpg pool) instead of a dedicated psycopg2 connection -
    this is always used now; the standalone project's CSV fallback is
    dropped since this backend has no equivalent local email_data.csv
  - write Weaviate objects via chatbot.weaviate_store (the already-ported
    collection management module) instead of a module-local
    weaviate_client.py
  - use this project's centralised logger

Retrieval backend: Weaviate (hybrid vector + BM25 search via the shared
Weaviate deployment). The chatbot's dedicated collection is managed by
chatbot.weaviate_store.
"""

import hashlib
import json
import os
import re
import uuid

import numpy as np
import ollama

from chatbot.metadata_extractor import extract_metadata
from core.config import get_settings
from core.db import get_pool
from core.logging_config import get_agent_logger
from agent_email.embedding_json import vec_to_json, json_to_vec, jsonb_to_embedding_str
from agent_email.storage_ids import stable_object_id

logger = get_agent_logger("chatbot")


def _chatbot_content_hash(text: str) -> str:
    """SHA-256 hash of the searchable text used for embedding cache lookup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _load_cached_embeddings() -> dict[str, dict]:
    """Load all rows from chatbot_embeddings into a dict keyed by message_id."""
    pool = get_pool()
    if pool is None:
        return {}
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT message_id, embedding, embedding_model, content_hash, weaviate_id "
                "FROM chatbot_embeddings"
            )
        return {
            r["message_id"]: {
                "embedding": jsonb_to_embedding_str(r["embedding"]),
                "model": r["embedding_model"] or "",
                "hash": r["content_hash"] or "",
                "weaviate_id": r["weaviate_id"] or "",
            }
            for r in rows
        }
    except Exception as e:
        logger.warning("Could not load cached embeddings: %s", e)
        return {}


async def _save_embeddings_batch(records: list[dict]) -> int:
    """Upsert embedding rows into chatbot_embeddings. Returns count written."""
    pool = get_pool()
    if pool is None or not records:
        return 0
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    "INSERT INTO chatbot_embeddings "
                    "(message_id, embedding, embedding_model, embedding_dim, content_hash, weaviate_id, updated_at) "
                    "VALUES ($1, $2::jsonb, $3, $4, $5, $6, now()) "
                    "ON CONFLICT (message_id) DO UPDATE SET "
                    "embedding = EXCLUDED.embedding, embedding_model = EXCLUDED.embedding_model, "
                    "embedding_dim = EXCLUDED.embedding_dim, content_hash = EXCLUDED.content_hash, "
                    "weaviate_id = EXCLUDED.weaviate_id, updated_at = now()",
                    [
                        (
                            r["message_id"],
                            r["embedding_json"],
                            r["model"],
                            r["dim"],
                            r["hash"],
                            r["weaviate_id"],
                        )
                        for r in records
                    ],
                )
        return len(records)
    except Exception as e:
        logger.error("Could not save embeddings batch: %s", e)
        return 0


# Pre-compiled pattern to detect pre-concatenated thread blobs (bu_emails format)
_THREAD_HEADER_RE = re.compile(
    r"---\s*Message from\s+.+?\s+on\s+\d{4}-\d{2}-\d{2}T[\d:+\-Z.]+\s*---",
    re.IGNORECASE,
)

# Prefix injected by email gateway — strip before embedding so it doesn't pollute vectors
_EXTERNAL_DISCLAIMER = "EXTERNAL EMAIL: Do not click any links or open any attachments unless you trust the sender and know the content is safe."


def _clean_content(text: str) -> str:
    """Strip the external-email disclaimer and excess whitespace."""
    text = text.replace(_EXTERNAL_DISCLAIMER, "").strip()
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


# nomic-embed-text's model maximum is 2048 tokens. Passing a larger num_ctx
# makes Ollama log "requested context size too large" and clamp it anyway, so
# cap the embedding context here instead of inheriting any larger chat default.
_EMBED_NUM_CTX = 2048


def _embed(text: str) -> list[float]:
    """Generate a single embedding vector for text using Ollama."""
    settings = get_settings()
    client = ollama.Client(host=settings.ollama_host)
    response = client.embed(
        model=settings.embedding_model,
        input=text,
        options={"num_ctx": _EMBED_NUM_CTX},
    )
    return response["embeddings"][0]


def _normalise_for_comparison(text: str) -> str:
    """Normalize text only for duplicate detection; stored text is preserved."""
    text = re.sub(r"(?m)^\s*>+\s?", "", text or "")
    return re.sub(r"\s+", " ", text).strip().casefold()


def _strip_verified_quoted_suffix(body: str, earlier_bodies: list[str]) -> str:
    """Remove only a trailing quoted block proven to repeat an earlier body."""
    body = (body or "").strip()
    if not body:
        return ""

    all_earlier = [
        normalized for value in earlier_bodies
        if (normalized := _normalise_for_comparison(value))
    ]
    earlier = [normalized for normalized in all_earlier if len(normalized) >= 200]
    normalized_body = _normalise_for_comparison(body)
    if normalized_body in all_earlier:
        return ""

    boundaries = []
    for pattern in (
        r"(?im)^\s*-{2,}\s*original message\s*-{2,}\s*$",
        r"(?im)^\s*on .{5,250} wrote:\s*$",
    ):
        boundaries.extend(match.start() for match in re.finditer(pattern, body))

    for match in re.finditer(r"(?im)^\s*from:\s*.+$", body):
        header = body[match.start():match.start() + 800]
        if all(re.search(rf"(?im)^\s*{field}:\s*.+$", header) for field in ("sent", "to", "subject")):
            boundaries.append(match.start())

    for boundary in sorted(set(boundaries)):
        quoted = _normalise_for_comparison(body[boundary:])
        if len(quoted) >= 200 and any(previous in quoted for previous in earlier):
            return body[:boundary].strip()

    return body


def _canonical_subject(subjects: list[str]) -> str:
    """Choose the earliest subject and remove repeated reply/forward prefixes."""
    subject = next((value.strip() for value in subjects if value and value.strip()), "No Subject")
    return re.sub(r"^(?:(?:re|fw|fwd)\s*:\s*)+", "", subject, flags=re.IGNORECASE).strip() or subject


def _split_thread_into_messages(
    thread_body: str,
    conversation_id: str,
    thread_subject: str,
    thread_received_at: str,
    thread_from_address: str,
) -> list[dict]:
    """Split a pre-concatenated email thread (bu_emails.csv format) into
    individual message dicts that the rest of load_emails() can process.

    The source stores the full reply chain in one ``email_content``/
    ``raw_content`` cell using two separator patterns:

      1. ``--- Message from <Name> on <ISO-datetime> ---``
         The outermost / most-recent message written by the collector.

      2. ``-----Original Message-----``  (5+ dashes, "Original Message", 5+ dashes)
         Each quoted reply/forward block that follows.

    Messages are returned in ascending chronological order (oldest first) so
    the existing de-duplication and quote-stripping logic in load_emails()
    works correctly.
    """

    text = _clean_content(thread_body)
    if not text:
        return []

    outer_pattern = re.compile(
        r"---\s*Message from\s+(.+?)\s+on\s+(\d{4}-\d{2}-\d{2}T[\d:+\-Z.]+)\s*---",
        re.IGNORECASE,
    )
    original_pattern = re.compile(
        r"-{5,}\s*Original Message\s*-{5,}",
        re.IGNORECASE,
    )

    splits: list[tuple[int, str]] = []  # (char_offset, type)

    for m in outer_pattern.finditer(text):
        splits.append((m.start(), "outer", m.group(1).strip(), m.group(2).strip()))

    for m in original_pattern.finditer(text):
        splits.append((m.start(), "original", "", ""))

    splits.sort(key=lambda x: x[0])

    if not splits:
        return [{
            "message_id": f"pg-{hashlib.sha256((conversation_id + thread_body[:200]).encode()).hexdigest()[:32]}",
            "subject": thread_subject,
            "body": text,
            "attachment_data": "",
            "received_at": thread_received_at,
            "from_address": thread_from_address,
        }]

    messages_raw: list[dict] = []
    preamble = text[:splits[0][0]].strip()

    for idx, split_info in enumerate(splits):
        start = split_info[0]
        end = splits[idx + 1][0] if idx + 1 < len(splits) else len(text)
        segment = text[start:end].strip()

        split_type = split_info[1]
        detected_from = split_info[2] if len(split_info) > 2 else ""
        detected_date = split_info[3] if len(split_info) > 3 else ""

        if split_type == "outer":
            body = outer_pattern.sub("", segment, count=1).strip()
            if idx == 0 and preamble:
                body = preamble + "\n\n" + body if body else preamble
            from_addr = _extract_email_address(detected_from) or thread_from_address
            received = detected_date or thread_received_at
        else:
            body, from_addr, received, subj = _parse_original_message_block(segment)
            if subj:
                thread_subject = subj

        body = body.strip()
        if not body:
            continue

        identity = f"{conversation_id}:{idx}:{body[:100]}"
        msg_id = f"pg-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]}"

        messages_raw.append({
            "message_id": msg_id,
            "subject": thread_subject,
            "body": body,
            "attachment_data": "",
            "received_at": received or thread_received_at,
            "from_address": from_addr or thread_from_address,
        })

    if not messages_raw:
        return [{
            "message_id": f"pg-{hashlib.sha256((conversation_id + thread_body[:200]).encode()).hexdigest()[:32]}",
            "subject": thread_subject,
            "body": text,
            "attachment_data": "",
            "received_at": thread_received_at,
            "from_address": thread_from_address,
        }]

    messages_raw.reverse()
    return messages_raw


def _extract_email_address(text: str) -> str:
    """Extract a bare email address from a display-name + address string."""
    m = re.search(r"[\w.+\-]+@[\w.\-]+", text or "")
    return m.group(0) if m else ""


def _parse_original_message_block(block: str) -> tuple[str, str, str, str]:
    """Parse a '-----Original Message-----' block into (body, from, date, subject)."""
    block = re.sub(r"-{5,}\s*Original Message\s*-{5,}", "", block, count=1).strip()

    from_addr = ""
    received = ""
    subject = ""

    lines = block.splitlines()
    header_end = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            header_end = i + 1
            break
        m_from = re.match(r"^From:\s*(.+)$", stripped, re.IGNORECASE)
        if m_from:
            raw = m_from.group(1).strip()
            from_addr = _extract_email_address(raw) or raw
            continue
        m_sent = re.match(r"^Sent:\s*(.+)$", stripped, re.IGNORECASE)
        if m_sent:
            received = m_sent.group(1).strip()
            continue
        m_subj = re.match(r"^Subject:\s*(.+)$", stripped, re.IGNORECASE)
        if m_subj:
            subject = re.sub(r"^(?:(?:re|fw|fwd)\s*:\s*)+", "", m_subj.group(1).strip(), flags=re.IGNORECASE).strip()
            continue
    else:
        header_end = len(lines)

    body = "\n".join(lines[header_end:]).strip()
    return body, from_addr, received, subject


def _render_conversation(messages: list[dict]) -> str:
    """Render one chronological, provenance-preserving LLM context document."""
    sections = []
    total = len(messages)
    for position, message in enumerate(messages, 1):
        lines = [
            f"=== Message {position} of {total} ===",
            f"Message-ID: {message['message_id']}",
            f"Date: {message['received_at'] or 'N/A'}",
            f"From: {message['from_address'] or 'N/A'}",
            f"Subject: {message['subject'] or 'No Subject'}",
        ]
        if message["body"]:
            lines.extend(["", message["body"]])
        if message["attachment_data"]:
            lines.extend(["", f"--- Attachment text for message {position} ---", message["attachment_data"]])
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


async def load_emails() -> tuple[list[dict], np.ndarray]:
    """Build one searchable chunk per individual message, with the full
    parent thread (messages_json / thread_raw_content) denormalised onto
    every message so retrieval can reconstruct the whole conversation
    from a single matched message without a second lookup.

    Source rows come from the shared PostgreSQL pool
    (chatbot.postgres_store.fetch_email_rows) - this backend has no CSV
    fallback for the chatbot's data source.
    """
    from chatbot.postgres_store import fetch_email_rows

    logger.info("Loading email conversations from Postgres...")
    rows = await fetch_email_rows()

    # Load cached embeddings so we can skip re-embedding unchanged chunks.
    settings = get_settings()
    embedding_model = settings.embedding_model
    cached = await _load_cached_embeddings()
    cache_hits = 0
    cache_misses = 0
    new_embeddings: list[dict] = []  # records to save back to Postgres

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        conversation_id = (row.get("conversation_id") or "").strip()
        if not conversation_id:
            identity = "|".join([
                row.get("subject") or "",
                row.get("received_at") or "",
                (row.get("from_address") or "").lower(),
            ])
            conversation_id = f"conv-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]}"
            logger.warning("Row has no conversation_id - generated synthetic id %r", conversation_id)

        message_id = (row.get("message_id") or "").strip()
        subject = (row.get("subject") or "").strip()
        body = _clean_content(row.get("raw_content") or "")
        attachment_data = _clean_content(row.get("attachment_data") or "")
        received_at = str(row.get("received_at") or "").strip()
        from_address = (row.get("from_address") or "").strip()

        if not message_id:
            identity = "|".join((subject, received_at, from_address.lower(), body, attachment_data))
            message_id = f"pg-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]}"

        # Detect pre-concatenated thread blob - a row stores the full
        # thread as one blob and needs splitting into individual messages.
        if _THREAD_HEADER_RE.search(body):
            split_messages = _split_thread_into_messages(
                body, conversation_id, subject, received_at, from_address,
            )
            if attachment_data and split_messages:
                split_messages[0]["attachment_data"] = attachment_data
            for msg in split_messages:
                grouped.setdefault(conversation_id, []).append(msg)
            logger.debug("Thread '%s' -> %d messages", subject[:60], len(split_messages))
            continue

        grouped.setdefault(conversation_id, []).append({
            "message_id": message_id,
            "subject": subject,
            "body": body,
            "attachment_data": attachment_data,
            "received_at": received_at,
            "from_address": from_address,
        })

    chunks: list[dict] = []
    embeddings: list[list[float]] = []

    for conversation_index, (conversation_id, source_messages) in enumerate(sorted(grouped.items())):
        source_messages.sort(key=lambda message: (
            not bool(message["received_at"]),
            message["received_at"],
            message["message_id"],
        ))

        messages = []
        earlier_bodies: list[str] = []
        seen_message_ids = set()
        seen_attachment_hashes = set()
        participants = []

        for source in source_messages:
            if source["message_id"] in seen_message_ids:
                logger.warning("Duplicate message_id '%s' - skipping", source["message_id"])
                continue
            seen_message_ids.add(source["message_id"])

            body = _strip_verified_quoted_suffix(source["body"], earlier_bodies)
            if body:
                earlier_bodies.append(body)

            attachment_data = source["attachment_data"]
            if attachment_data:
                attachment_hash = hashlib.sha256(
                    _normalise_for_comparison(attachment_data).encode("utf-8")
                ).hexdigest()
                if attachment_hash in seen_attachment_hashes:
                    attachment_data = ""
                else:
                    seen_attachment_hashes.add(attachment_hash)

            if source["from_address"] and source["from_address"] not in participants:
                participants.append(source["from_address"])

            messages.append({
                "message_id": source["message_id"],
                "subject": source["subject"],
                "from_address": source["from_address"],
                "received_at": source["received_at"],
                "body": body,
                "attachment_data": attachment_data,
                "has_attachment": bool(attachment_data),
            })

        if not messages or not any(message["body"] or message["attachment_data"] for message in messages):
            logger.warning("Conversation '%s' has no usable content - skipping", conversation_id)
            continue

        subject = _canonical_subject([message["subject"] for message in messages])
        thread_raw_content = _render_conversation(messages)
        messages_json = json.dumps(messages, ensure_ascii=False, sort_keys=True)

        first_received_at = next((message["received_at"] for message in messages if message["received_at"]), "")
        last_received_at = next((message["received_at"] for message in reversed(messages) if message["received_at"]), "")
        message_ids = [message["message_id"] for message in messages]

        for position, message in enumerate(messages, 1):
            if not (message["body"] or message["attachment_data"]):
                continue

            message_metadata_summary = extract_metadata(
                message["subject"] or subject,
                "\n\n".join(part for part in (message["body"], message["attachment_data"]) if part),
            )

            searchable_text = "\n\n".join(part for part in (
                message_metadata_summary,
                f"Conversation subject: {subject}",
                message["subject"],
                message["body"],
                message["attachment_data"],
            ) if part)

            if not searchable_text.strip():
                continue

            # Check embedding cache: reuse if content hash + model match.
            msg_id = message["message_id"]
            content_hash = _chatbot_content_hash(searchable_text)
            cached_entry = cached.get(msg_id)

            if (
                cached_entry
                and cached_entry["hash"] == content_hash
                and cached_entry["model"] == embedding_model
            ):
                vec = json_to_vec(cached_entry["embedding"])
                if vec is not None:
                    vector = vec.tolist()
                    cache_hits += 1
                else:
                    vector = _embed(searchable_text)
                    cache_misses += 1
            else:
                vector = _embed(searchable_text)
                cache_misses += 1

            # Queue new/changed embeddings for persistence.
            if not cached_entry or cached_entry["hash"] != content_hash or cached_entry["model"] != embedding_model:
                wv_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"rbl-chatbot-message:{msg_id}"))
                vec_array = np.array(vector, dtype=np.float32)
                new_embeddings.append({
                    "message_id": msg_id,
                    "embedding_json": vec_to_json(vec_array),
                    "model": embedding_model,
                    "dim": len(vector),
                    "hash": content_hash,
                    "weaviate_id": wv_id,
                })

            chunks.append({
                "email_id": conversation_index,
                "message_id": message["message_id"],
                "conversation_id": conversation_id,
                "message_index": position,

                "subject": message["subject"],
                "body": message["body"],
                "attachment_data": message["attachment_data"],
                "metadata_summary": message_metadata_summary,
                "received_at": message["received_at"],
                "from_address": message["from_address"],
                "has_attachment": message["has_attachment"],

                "thread_subject": subject,
                "thread_raw_content": thread_raw_content,
                "raw_content": thread_raw_content,
                "first_received_at": first_received_at,
                "last_received_at": last_received_at,
                "participants": participants,
                "message_ids": message_ids,
                "message_count": len(messages),
                "messages_json": messages_json,

                "source_type": "email_message",
            })
            embeddings.append(vector)

    total_messages = sum(len(v) for v in grouped.values())
    logger.info(
        "Loaded %d messages from %d conversations (%d source rows)",
        total_messages, len(grouped), len(rows),
    )

    # Persist newly generated embeddings to Postgres.
    if new_embeddings:
        saved = await _save_embeddings_batch(new_embeddings)
        logger.info(
            "[CHATBOT-EMBED] generated=%d reused=%d saved=%d",
            cache_misses, cache_hits, saved,
        )
    else:
        logger.info(
            "[CHATBOT-EMBED] generated=%d reused=%d (nothing new to save)",
            cache_misses, cache_hits,
        )

    array = np.asarray(embeddings, dtype=np.float32) if embeddings else np.empty((0, 0), dtype=np.float32)
    return chunks, array


# ── Weaviate ingestion ────────────────────────────────────────

async def build_weaviate_index(client, recreate: bool = False) -> int:
    """Load, embed, and write the authoritative message set into Weaviate.

    With ``recreate=True``, the old collection is removed only after all
    source rows have embedded successfully, clearing legacy duplicates
    before the new set is inserted.
    """
    import time as _time

    from chatbot._vec_backend import vec_store
    vs = vec_store()

    t0 = _time.perf_counter()
    logger.info("Starting vector index build (backend=%s)...",
                "faiss" if hasattr(vs, "search_vectors") and not hasattr(vs, "get_client") else "weaviate")

    chunks, embeddings = await load_emails()
    if not chunks:
        raise ValueError("No emails found. Check the chatbot's Postgres table.")

    inserted = vs.ingest(client, chunks, embeddings, recreate=recreate)
    stored = vs.collection_count(client)
    if stored != len(chunks):
        raise RuntimeError(
            f"Message index count mismatch: expected {len(chunks)}, found {stored}"
        )

    elapsed = _time.perf_counter() - t0
    logger.info("Vector index build complete: %d objects in %.1fs", inserted, elapsed)
    return inserted


def weaviate_collection_count(client) -> int:
    """Return total number of message objects in the chatbot's collection."""
    from chatbot._vec_backend import vec_store
    try:
        return vec_store().collection_count(client)
    except Exception:
        return 0
