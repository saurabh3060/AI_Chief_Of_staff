"""
core/weaviate/client.py
========================
Weaviate connection management, shared core infrastructure. Currently
consumed by the email agent's embedding vector store
(``agent_email/weaviate_vector_store.py``), but lives in ``core`` since
Weaviate connectivity is cross-cutting infrastructure, not something
specific to the email agent.

Connects to the bank's internal Weaviate deployment via
``weaviate.connect_to_custom()`` with independent REST and gRPC endpoints:

* REST (``http_*``)  -> TLS-terminating hostname (``weaviateui.rbl.bank.in``,
  port 443, secure).
* gRPC (``grpc_*``)  -> internal NLB hostname, a separate port, no TLS.

This mirrors the bank's own connectivity test script exactly:

    client = weaviate.connect_to_custom(
        http_host=..., http_port=443, http_secure=True,
        grpc_host=..., grpc_port=..., grpc_secure=False,
    )

Lifecycle
---------
The FastAPI process is long-running, so we keep a single process-wide
client (lazily connected, cached) rather than reconnecting on every call —
``weaviate.connect_to_custom()`` opens a gRPC channel + HTTP session that
is meant to be reused, and the client docs call ``client.close()`` only
once, at shutdown. Call :func:`close_client` from the app's shutdown hook
to release the connection cleanly.

Entirely separate from the chatbot's local FAISS index — the chatbot does
not import this module and is not affected by Weaviate being
unreachable.
"""

from __future__ import annotations

import threading

import weaviate

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_email")

_lock = threading.Lock()
_client: weaviate.WeaviateClient | None = None

# Callbacks invoked whenever the process-wide client is torn down
# (``close_client``). Modules that cache per-collection handles bound to the
# live client (e.g. ``agent_email/weaviate_vector_store.py``'s ``_collections``)
# register a reset here so a close+reconnect never leaves a handle pointing at
# a dead connection — reusing such a stale handle is what surfaces the gRPC
# "could not find class ... in schema" desync. Kept as a registry (instead of
# importing the email module here) so ``core`` has no dependency on an agent.
_on_close_callbacks: list = []


def register_on_close(callback) -> None:
    """
    Register `callback` (a zero-arg callable) to run inside ``close_client``
    after the client socket is closed. Used by modules that cache handles
    bound to the client so they can drop those handles on teardown. Idempotent
    per callable (the same function is not registered twice).
    """
    with _lock:
        if callback not in _on_close_callbacks:
            _on_close_callbacks.append(callback)


def _connect(settings: Settings) -> weaviate.WeaviateClient:
    auth = None
    if settings.weaviate_api_key:
        # Imported lazily so the common case (anonymous access, matching
        # the bank's connectivity test script) never needs this symbol.
        from weaviate.classes.init import Auth
        auth = Auth.api_key(settings.weaviate_api_key)
    client = weaviate.connect_to_custom(
        http_host=settings.weaviate_http_host,
        http_port=settings.weaviate_http_port,
        http_secure=settings.weaviate_http_secure,
        grpc_host=settings.weaviate_grpc_host,
        grpc_port=settings.weaviate_grpc_port,
        grpc_secure=settings.weaviate_grpc_secure,
        auth_credentials=auth,
    )
    if not client.is_ready():
        client.close()
        raise RuntimeError(
            f"Weaviate is not ready at http={settings.weaviate_http_host}:"
            f"{settings.weaviate_http_port} grpc={settings.weaviate_grpc_host}:"
            f"{settings.weaviate_grpc_port}"
        )
    # Warm the schema once, right after connecting. `is_ready()` is only a
    # liveness ping; it does NOT load the collection schema that the gRPC
    # query path resolves class names against. Without this, the FIRST gRPC
    # query in a freshly-connected process (aggregate / near_vector) can fail
    # with "could not find class ... in schema" even though REST sees the
    # class — the cold-start desync. `collections.list_all()` forces the
    # client to load the schema so gRPC can resolve classes immediately.
    # Best-effort: a warm-up hiccup must not fail the connection (the per-call
    # retry/readiness logic still covers a slow server).
    try:
        client.collections.list_all()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[EMAIL-WEAVIATE] schema warm-up (list_all) failed, continuing: %s", exc
        )
    logger.info(
        "[EMAIL-WEAVIATE] connected http=%s:%s secure=%s grpc=%s:%s secure=%s",
        settings.weaviate_http_host, settings.weaviate_http_port, settings.weaviate_http_secure,
        settings.weaviate_grpc_host, settings.weaviate_grpc_port, settings.weaviate_grpc_secure,
    )
    return client


def get_client(settings: Settings | None = None) -> weaviate.WeaviateClient:
    """
    Return the process-wide Weaviate client, connecting on first use.

    Raises if the connection cannot be established or the deployment does
    not report ready — callers must not silently fall back to any other
    vector store on failure.
    """
    global _client
    with _lock:
        if _client is not None:
            return _client
        active = settings or get_settings()
        _client = _connect(active)
        return _client


def close_client() -> None:
    """
    Close the process-wide Weaviate client, if one is open. Safe to call
    multiple times. Intended for FastAPI's shutdown/lifespan hook.
    """
    global _client
    with _lock:
        if _client is not None:
            try:
                _client.close()
                logger.info("[EMAIL-WEAVIATE] connection closed")
            finally:
                _client = None
        # Always run teardown callbacks (even if no client was open) so any
        # cached per-collection handles are dropped and cannot be reused
        # against a closed/reconnected client. Run under the lock we already
        # hold; callbacks must be cheap and must NOT call back into
        # get_client/close_client (would deadlock).
        for callback in list(_on_close_callbacks):
            try:
                callback()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[EMAIL-WEAVIATE] on-close callback failed: %s", exc)


def reset_client_for_tests() -> None:
    """
    Drop the cached client without closing the underlying connection
    gracefully. Only intended for test teardown when a mock/stub client
    was injected in place of a real connection.
    """
    global _client
    with _lock:
        _client = None
