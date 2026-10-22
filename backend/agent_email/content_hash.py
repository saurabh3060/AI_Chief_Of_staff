"""
agent_email/content_hash.py
===========================
Stable content hashing for the email agent's persisted records.
 
A record's ``content_hash`` is a deterministic SHA-256 over the record's
STABLE business fields only. It is used by the storage layer to answer one
question cheaply: *has this record actually changed since it was last
written?* If the hash matches what is already stored, the write (and, for
embedding-bearing records, any embedding/LLM work the caller would do) can
be SKIPPED — nothing new to persist.
 
Design rules
------------
* **Stable fields only.** Volatile bookkeeping fields are excluded so an
  unchanged record always hashes to the same value: ``updated_at`` (always
  moves), ``run_id`` (changes every pipeline run), and the embedding vector
  itself (a derived artifact of the stable text, not an input to identity).
* **Order-independent for the payload, order-sensitive within a field.**
  We build an ordered list of ``(field, value)`` pairs in a FIXED order and
  JSON-encode it, so the hash is reproducible across processes and Python
  versions (``sort_keys`` + no whitespace).
* **No model/dim in the hash.** ``embedding_model`` / ``embedding_dim`` are
  recorded next to the embedding for provenance, but they are not part of
  content identity: the same text under a different model is still the same
  *record*, it just needs a re-embed. Re-embedding is driven by the caller,
  not by this hash.
 
The hash is intentionally independent of PostgreSQL vs CSV vs Weaviate — it
is computed from the in-memory record shape so every store compares the
same value.
"""
 
from __future__ import annotations
 
import hashlib
import json
from typing import Any
 
 
def _stable_hash(payload: list[tuple[str, Any]]) -> str:
    """
    Return a hex SHA-256 of an ordered ``(field, value)`` payload.
 
    Values are JSON-encoded with sorted keys and compact separators so the
    encoding is byte-for-byte reproducible regardless of dict insertion
    order or interpreter. ``ensure_ascii`` is left on so non-ASCII subjects
    hash identically everywhere.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
 
 
def cluster_content_hash(
    *,
    cluster_id: str,
    department: str,
    topic: str,
    summary: str,
    top_questions: str,
    action_items: str,
    email_refs: str,
    email_count: int,
    meeting_id: str,
    meeting_start: str,
    is_active: int = 1,
) -> str:
    """
    Content hash for one ``email_clusters`` row.
 
    Excludes ``updated_at`` and the embedding. ``email_refs`` is hashed as
    its stored string form (the JSON text the row carries), so a change in
    referenced emails changes the hash. ``is_active`` is part of identity
    so that flipping a cluster active<->inactive is detected as a change
    and actually persisted (the write-skip would otherwise ignore it).
    """
    return _stable_hash(
        [
            ("cluster_id", cluster_id or ""),
            ("department", department or ""),
            ("topic", topic or ""),
            ("summary", summary or ""),
            ("top_questions", top_questions or ""),
            ("action_items", action_items or ""),
            ("email_refs", email_refs or ""),
            ("email_count", int(email_count or 0)),
            ("meeting_id", meeting_id or ""),
            ("meeting_start", meeting_start or ""),
            ("is_active", int(is_active if is_active is not None else 1)),
        ]
    )


def meeting_embedding_content_hash(
    *,
    meeting_id: str,
    subject: str,
    department: str = "",
) -> str:
    """
    Content hash for one ``meeting_embeddings`` row.
 
    The only stable identity fields are the meeting id and its subject;
    the embedding is derived from the subject/query text, so a subject
    change is what should trigger a re-embed.
    """
    return _stable_hash(
        [
            ("meeting_id", meeting_id or ""),
            ("subject", subject or ""),
            ("department", department or ""),
        ]
    )
 
 
def meeting_history_content_hash(
    *,
    meeting_id: str,
    subject: str,
    start_datetime: str,
    business_context: str,
    end_datetime: str = "",
    department: str = "",
) -> str:
    """
    Content hash for one ``meeting_history`` row.
 
    Excludes ``updated_at`` and the embedding. ``business_context`` is part
    of identity because it is part of what the occurrence embedding is
    built from.
    """
    return _stable_hash(
        [
            ("meeting_id", meeting_id or ""),
            ("subject", subject or ""),
            ("start_datetime", start_datetime or ""),
            ("end_datetime", end_datetime or ""),
            ("business_context", business_context or ""),
            ("department", department or ""),
        ]
    )