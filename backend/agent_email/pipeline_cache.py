"""
agent_email/pipeline_cache.py
=============================
Exact-match Redis caches used ONLY by the email pipeline.

* **LLM replies** - key = sha256(model + system prompt + full prompt + every
  generation parameter). A hit returns the reply the same model already gave
  for the identical input, so the call is not repeated.
* **Embeddings** - key = sha256(model + num_ctx + exact text).

Safety properties (each one is relied on by the callers):

* Separate key space. Keys live under ``EMAIL_PIPELINE_CACHE_PREFIX``
  (default ``emailagent:``): never ``chatbot:*`` and never under
  ``REDIS_KEY_PREFIX`` (which the pipeline wipes at the start of a run). A
  prefix that would collide with either disables the cache with a warning.
* Never deletes. Writes are plain ``SET ... EX`` (an existing entry is only
  overwritten); nothing here issues DEL / UNLINK / FLUSH.
* Miss-safe. Redis absent, down, slow or returning garbage is a miss and the
  caller makes the normal call.
* Only validated replies are stored (see ``LlmCacheTicket.commit``): the
  callers retry an invalid reply with the same prompt, so caching one would
  make every retry return the same bad text.
"""

from __future__ import annotations

import hashlib
import json

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger
from core.redis.client import get_redis

logger = get_agent_logger("agent_email")

# Bump to invalidate every entry if the key/value layout ever changes.
_KEY_VERSION = "v1"
_LLM_NAMESPACE = "llmcache:"

_llm_cache_hits = 0
_prefix_warning_logged = False


def llm_cache_hits() -> int:
    """LLM replies served from the cache so far in this process."""
    return _llm_cache_hits


def _base_prefix(settings: Settings) -> str | None:
    """The cache key prefix, or ``None`` when caching is off or the prefix is unsafe."""
    global _prefix_warning_logged
    if not getattr(settings, "email_pipeline_cache_enabled", False):
        return None
    prefix = (getattr(settings, "email_pipeline_cache_prefix", "") or "").strip()
    if not prefix:
        return None
    app_prefix = getattr(settings, "redis_key_prefix", "") or ""
    if prefix.startswith("chatbot:") or (app_prefix and prefix.startswith(app_prefix)):
        if not _prefix_warning_logged:
            _prefix_warning_logged = True
            logger.warning(
                "[PIPELINE-CACHE] EMAIL_PIPELINE_CACHE_PREFIX=%r collides with the chatbot or "
                "REDIS_KEY_PREFIX key space - pipeline cache DISABLED", prefix,
            )
        return None
    return prefix


# ---------------------------------------------------------------------------
# LLM replies
# ---------------------------------------------------------------------------

def llm_cache_key(
    settings: Settings,
    *,
    task: str,
    system_prompt: str | None,
    prompt: str,
    params: dict,
) -> str | None:
    """Exact-match key for one LLM call, or ``None`` when caching is off.

    ``params`` is the dict from ``OllamaClient.generation_params()`` (model,
    temperature, num_ctx, num_predict): any change to any of them is a
    different key, as is any change to either prompt.
    """
    prefix = _base_prefix(settings)
    if prefix is None:
        return None
    payload = json.dumps(
        {"v": _KEY_VERSION, "system": system_prompt or "", "prompt": prompt, "params": params},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}{_LLM_NAMESPACE}{task}:{digest}"


async def _llm_get(settings: Settings, key: str) -> str | None:
    client = await get_redis(settings)
    if client is None:
        return None
    try:
        raw = await client.get(key)
        if raw is None:
            return None
        value = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - any cache problem is a miss
        logger.warning("[PIPELINE-CACHE] LLM GET failed (treating as miss): %s", exc)
        return None
    text = value.get("text") if isinstance(value, dict) else None
    return text if isinstance(text, str) and text else None


async def _llm_set(settings: Settings, key: str, text: str) -> None:
    client = await get_redis(settings)
    if client is None or not text:
        return
    ttl = int(getattr(settings, "email_pipeline_llm_cache_ttl_seconds", 0) or 0)
    try:
        if ttl > 0:
            await client.set(key, json.dumps({"text": text}), ex=ttl)
        else:
            await client.set(key, json.dumps({"text": text}))
    except Exception as exc:  # noqa: BLE001 - the cache must never break a pipeline run
        logger.warning("[PIPELINE-CACHE] LLM SET failed (not cached): %s", exc)


class LlmCacheTicket:
    """What ``chat_with_cache`` hands back so the caller can store a validated reply."""

    def __init__(self, key: str | None, hit: bool) -> None:
        self.key = key
        self.hit = hit

    async def commit(self, settings: Settings, raw: str) -> None:
        """Store ``raw`` - call ONLY after the reply passed the caller's validation.

        Nothing is written for a reply that came from the cache. A reply that
        replaces a cached one the caller rejected is simply overwritten.
        """
        if self.key is not None and not self.hit:
            await _llm_set(settings, self.key, raw)


async def chat_with_cache(
    client,
    settings: Settings,
    *,
    task: str,
    prompt: str,
    system_prompt: str | None = None,
    num_predict: int | None = None,
    use_cache: bool = True,
) -> tuple[str, LlmCacheTicket]:
    """``client.chat(...)`` with an exact-match cache in front of it.

    Returns ``(reply, ticket)``. On a hit the reply is the stored one and no
    model call is made; on a miss (or ``use_cache=False``, which callers pass
    after a cached reply failed their validation) the model is called exactly as
    before. Call ``await ticket.commit(settings, reply)`` once the reply is
    accepted.
    """
    key = llm_cache_key(
        settings, task=task, system_prompt=system_prompt, prompt=prompt,
        params=client.generation_params(num_predict),
    )
    if key is not None and use_cache:
        cached = await _llm_get(settings, key)
        if cached is not None:
            global _llm_cache_hits
            _llm_cache_hits += 1
            logger.info(
                "[PIPELINE-CACHE] LLM HIT task=%s prompt_chars=%d reply_chars=%d - model call skipped",
                task, len(prompt), len(cached),
            )
            return cached, LlmCacheTicket(key, hit=True)
    reply = await client.chat(prompt, system_prompt=system_prompt, num_predict=num_predict)
    return reply, LlmCacheTicket(key, hit=False)
