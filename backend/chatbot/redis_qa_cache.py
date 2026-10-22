"""
chatbot/redis_qa_cache.py
===========================
Cache-first lookup for chatbot answers (email + document modes).

If the exact same question (after normalisation) was already answered, the
stored answer is returned straight from Redis instead of re-running
retrieval + LLM.

* Exact match only: lowercase, collapse whitespace, strip trailing punctuation.
* Document mode keys also include the sorted `doc_ids`, so different
  document scopes never share an entry.
* No TTL: the cache is wiped on every pipeline run / index rebuild via
  `clear_qa_cache()`.
* Miss-safe: any Redis problem is logged and treated as a cache miss.
"""

import hashlib
import json
import re
import time

from chatbot.redis_history import _get_redis_client
from core.logging_config import get_agent_logger

logger = get_agent_logger("chatbot")

KEY_PREFIX = "chatbot:qa_cache:"


def normalize_question(query: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation."""
    text = re.sub(r"\s+", " ", (query or "").strip().lower())
    return text.rstrip("?!.,;: ")


def _make_key(query: str, mode: str = "email", doc_ids: list[str] | None = None) -> str:
    raw = normalize_question(query)
    if mode == "document":
        scope = ",".join(sorted(str(d) for d in (doc_ids or [])))
        raw = f"{raw}|{scope}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()  # noqa: S324 - cache key only
    return f"{KEY_PREFIX}{mode}:{digest}"


def get_cached_answer(query: str, mode: str = "email", doc_ids: list[str] | None = None) -> dict | None:
    """Return the cached response dict, or None on miss / Redis problem."""
    client = _get_redis_client()
    if client is None:
        logger.info("QA cache DISABLED: Redis client is None (REDIS_HOST empty?)")
        return None
    try:
        key = _make_key(query, mode, doc_ids)
        raw = client.get(key)
        if raw is None:
            logger.info("QA cache MISS (%s): key=%s", mode, key)
            return None
        data = json.loads(raw)
        if not isinstance(data, dict) or not data.get("answer"):
            return None
        return data
    except Exception as e:
        logger.warning("Could not read QA cache from Redis: %s", e)
        return None


def set_cached_answer(
    query: str,
    response_data: dict,
    mode: str = "email",
    doc_ids: list[str] | None = None,
) -> None:
    """Store a response dict (answer, sources, follow_up_questions, tool_used, ...).

    Empty answers and "Not found" answers are not cached, since they may
    become answerable after the next pipeline run.
    """
    answer = (response_data or {}).get("answer", "")
    if not answer or answer.strip().lower().startswith("not found in the provided"):
        logger.info("QA cache SKIP (%s): empty or 'Not found' answer is not cached", mode)
        return
    client = _get_redis_client()
    if client is None:
        logger.info("QA cache SKIP (%s): Redis client is None", mode)
        return
    try:
        payload = dict(response_data)
        payload["timestamp"] = time.time()
        key = _make_key(query, mode, doc_ids)
        client.set(key, json.dumps(payload, default=str))
        logger.info("QA cache STORED (%s): key=%s", mode, key)
    except Exception as e:
        logger.warning("Could not write QA cache to Redis: %s", e)


def clear_qa_cache() -> int:
    """Delete every `chatbot:qa_cache:*` key. Never raises. Returns count deleted."""
    client = _get_redis_client()
    if client is None:
        return 0
    try:
        keys = list(client.scan_iter(match=f"{KEY_PREFIX}*"))
        deleted = client.delete(*keys) if keys else 0
        logger.info("Chatbot QA cache cleared (%d keys).", deleted)
        return deleted
    except Exception as e:
        logger.warning("Could not clear QA cache from Redis: %s", e)
        return 0
