"""
chatbot/redis_history.py
===========================
Redis client for storing and retrieving chatbot chat history.

Single-user setup: all messages are stored under one list key
(`chat:history`). Each entry is a JSON string: {question, answer, timestamp}.

Redis is treated as a nice-to-have for this feature - if the Redis
server is unreachable (or REDIS_HOST is unset), chat keeps working, just
without history persistence. No other agent/module in this backend uses
Redis, so an outage here cannot affect anything else.

Ported from the standalone chatbot_final project, adapted to read
connection settings from core.config.get_settings() instead of a
module-local config.py.
"""

import json
import time

import redis

from core.config import get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("chatbot")

HISTORY_KEY = "chatbot:history"
MAX_HISTORY = 100  # keep only the most recent N Q&A pairs

_client: "redis.Redis | None" = None
_client_initialised = False


def _get_redis_client() -> "redis.Redis | None":
    """Return a cached Redis connection, creating it on first use.

    Returns None if REDIS_HOST is not configured, so callers can skip
    the attempt entirely rather than hitting a connection error on every
    call.
    """
    global _client, _client_initialised
    if _client_initialised:
        return _client

    settings = get_settings()
    if not settings.redis_host:
        logger.info("REDIS_HOST is empty - chat history persistence disabled.")
        _client_initialised = True
        return None

    _client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    _client_initialised = True
    return _client


def save_message(question: str, answer: str) -> None:
    """Append a Q&A pair to the chat history list in Redis.

    Trims the list to the last MAX_HISTORY entries. Never raises - logs
    a warning and returns silently if Redis is unreachable, so a Redis
    outage never breaks the chat response.
    """
    client = _get_redis_client()
    if client is None:
        return
    try:
        entry = json.dumps({
            "question": question,
            "answer": answer,
            "timestamp": time.time(),
        })
        client.rpush(HISTORY_KEY, entry)
        client.ltrim(HISTORY_KEY, -MAX_HISTORY, -1)
    except Exception as e:
        logger.warning("Could not save message to Redis: %s", e)


def get_history(limit: int = 50) -> list[dict]:
    """Return the last `limit` Q&A pairs from Redis, oldest first.

    Returns an empty list if Redis is unreachable, unconfigured, or
    history is empty.
    """
    client = _get_redis_client()
    if client is None:
        return []
    try:
        raw_entries = client.lrange(HISTORY_KEY, -limit, -1)
        return [json.loads(e) for e in raw_entries]
    except Exception as e:
        logger.warning("Could not fetch history from Redis: %s", e)
        return []


def clear_history() -> bool:
    """Delete all chatbot Q&A history from Redis.

    Called by run_pipeline.py at the start of each pipeline run so
    stale Q&A pairs don't carry over. Never raises.

    Returns True when the history is gone (including when Redis is
    disabled, since there is nothing to clear) and False when Redis is
    configured but the delete failed.
    """
    client = _get_redis_client()
    if client is None:
        return True
    try:
        client.delete(HISTORY_KEY)
        logger.info("Chatbot Redis history cleared.")
        return True
    except Exception as e:
        logger.warning("Could not clear chatbot history from Redis: %s", e)
        return False
