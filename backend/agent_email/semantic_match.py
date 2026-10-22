"""
agent_email/semantic_match.py
=================================
On-demand embeddings and cosine similarity for email clustering,
department tagging, and meeting-to-cluster matching.

Deliberately separate from `chatbot/indexer.py` + `chatbot/retriever.py`:
those build/query a persistent FAISS index over raw `.eml` files for
the general Q&A chatbot. This module embeds texts as needed and
compares them directly - no index to keep in sync. Uses the same
underlying embedding model (`nomic-embed-text` via Ollama) so results
are comparable in spirit, without introducing a second heavy
dependency.
"""

import asyncio
import threading

import numpy as np
import ollama

from agent_email import pipeline_cache
from core.config import Settings, get_settings
from core.utils import parse_ollama_keep_alive

# A dedicated, owned Ollama client for embeddings.
#
# The embedding path previously called the module-level ``ollama.embed``,
# which routes through the ``ollama`` package's shared default client. That
# default client lazily opens an httpx keep-alive connection pool to the
# Ollama host and is NEVER closed, so every embedding call could strand a
# socket to ``:11434`` until interpreter shutdown — exactly the
# ``ResourceWarning: unclosed <socket ... raddr=(..., 11434)>`` seen when the
# pipeline runner exits. By holding our OWN ``ollama.Client`` we can close it
# deterministically from ``close_ollama`` on shutdown and release those
# sockets. Created lazily (first embed) and guarded by a lock so concurrent
# worker coroutines running embeds via ``run_in_executor`` share one pool.
_client_lock = threading.Lock()
_embed_client: "ollama.Client | None" = None


def _get_embed_client() -> "ollama.Client":
    """Return the process-wide embedding client, creating it on first use."""
    global _embed_client
    with _client_lock:
        if _embed_client is None:
            settings = get_settings()
            _embed_client = ollama.Client(host=settings.ollama_host)
        return _embed_client


def close_embed_client() -> None:
    """
    Close the owned embedding client's underlying HTTP connections, if one
    was created. Safe to call multiple times and when embeddings were never
    used. Invoked from ``core.ollama_client.close_ollama`` on shutdown.
    """
    global _embed_client
    with _client_lock:
        client = _embed_client
        _embed_client = None
    if client is None:
        return
    # The ollama.Client wraps an httpx.Client at ``_client``; close it so its
    # pooled sockets to :11434 are released deterministically.
    inner = getattr(client, "_client", None)
    closer = getattr(inner, "close", None)
    if closer is not None:
        try:
            closer()
        except Exception:  # noqa: BLE001 - teardown must never raise
            pass

# Default embedding model, used only if settings don't resolve one. The
# ACTIVE model comes from `Settings.embedding_model` (env EMBEDDING_MODEL)
# so it can be swapped without a code change; this constant is just the
# fallback and keeps parity with the chatbot's FAISS index default
# (`chatbot/indexer.py`). Kept as a local constant rather than an import
# to avoid a cross-agent dependency between `agent_email` and `chatbot`.
EMBEDDING_MODEL = "nomic-embed-text"


def _resolve_embedding_model(settings: Settings | None) -> str:
    """Return the configured embedding model name, falling back to the module default."""
    active = settings or get_settings()
    return getattr(active, "embedding_model", "") or EMBEDDING_MODEL


def _embed_sync(text: str, num_ctx: int, model: str) -> np.ndarray:
    """Blocking call to Ollama's embedding endpoint for one text."""
    # Use our OWN client (closed on shutdown) rather than the ollama module's
    # shared default client, so embedding sockets to :11434 don't leak.
    settings = get_settings()
    options = {"num_ctx": num_ctx}
    # Opt-in runtime options (EMAIL_PIPELINE_OLLAMA_*): only sent when set, so the default
    # request is unchanged. They change speed, not the vector.
    num_thread = int(getattr(settings, "email_pipeline_ollama_num_thread", 0) or 0)
    if num_thread > 0:
        options["num_thread"] = num_thread
    extra = {}
    keep_alive = (getattr(settings, "email_pipeline_ollama_keep_alive", "") or "").strip()
    if keep_alive:
        extra["keep_alive"] = parse_ollama_keep_alive(keep_alive)
    response = _get_embed_client().embed(model=model, input=text, options=options, **extra)
    return np.array(response["embeddings"][0], dtype=np.float32)


async def embed_text(text: str, settings: Settings | None = None) -> np.ndarray:
    """
    Public async embedding helper: embed a single text with the
    configured embedding model, off the event loop. Shared by clustering,
    department assignment, and meeting-cluster matching so all use
    identical embedding settings. Raises on failure (callers that need
    best-effort behaviour must catch).
    """
    active = settings or get_settings()
    
    # Check if pipeline is enabled before making Ollama embedding calls
    if not active.bu_email_pipeline_enabled:
        import logging
        logger = logging.getLogger("agent_email.semantic_match")
        logger.debug("[FLAG-CHECK] BU_EMAIL_PIPELINE_ENABLED=false; skipping embedding LLM call")
        # Return zero vector as fallback
        return np.zeros(384, dtype=np.float32)
    
    model = _resolve_embedding_model(active)

    # Exact-match cache (pipeline-only Redis keys): the same model + context size +
    # text always embeds to the same vector, so an already-computed one is reused.
    cache_key = pipeline_cache.embedding_cache_key(
        active, model=model, num_ctx=active.ollama_num_ctx, text=text,
    )
    if cache_key is not None:
        cached = await pipeline_cache.embedding_get(active, cache_key)
        if cached is not None:
            return cached

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _embed_sync, text, active.ollama_num_ctx, model)
    
    # Record embedding call
    from agent_email.pipeline_metrics import PipelineMetricsCollector
    PipelineMetricsCollector.record_llm_call("embedding", tokens_in=len(text.split()))

    if cache_key is not None:
        await pipeline_cache.embedding_set(active, cache_key, result)
    
    return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors; 0.0 if either has zero magnitude."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
