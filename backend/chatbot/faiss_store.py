"""
chatbot/faiss_store.py
=======================
Local FAISS vector store for the chatbot email retrieval — drop-in
alternative to ``chatbot/weaviate_store.py``.

Activated by ``CHATBOT_VECTOR_DB=faiss`` in ``.env``.

Stores:
* ``chatbot_messages.faiss`` — IndexFlatIP (inner-product on L2-normalised
  vectors ≈ cosine similarity)
* ``chatbot_messages_chunks.json`` — the full chunk dicts so
  ``fetch_all_chunks()`` can return them without Weaviate

The public API surface matches ``chatbot/weaviate_store.py`` so callers
can swap transparently.  Functions that take a ``client`` parameter
ignore it (FAISS is local, no Weaviate client needed).
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

import faiss
import numpy as np

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("chatbot")

_lock = threading.Lock()
_state: dict | None = None  # {"index": faiss.IndexFlatIP, "chunks": [...], "id_map": [...]}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _index_dir() -> Path:
    d = Path(get_settings().faiss_index_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _paths() -> tuple[Path, Path]:
    d = _index_dir()
    return d / "chatbot_messages.faiss", d / "chatbot_messages_chunks.json"


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def _normalize(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr


def _load() -> dict:
    global _state
    with _lock:
        if _state is not None:
            return _state

    faiss_path, chunks_path = _paths()
    from documents.config import EMBED_DIM as dim

    if faiss_path.exists() and chunks_path.exists():
        index = faiss.read_index(str(faiss_path))
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    else:
        index = faiss.IndexFlatIP(dim)
        chunks = []

    entry = {"index": index, "chunks": chunks}
    with _lock:
        _state = entry
    return entry


def _save() -> None:
    if _state is None:
        return
    faiss_path, chunks_path = _paths()
    faiss.write_index(_state["index"], str(faiss_path))
    chunks_path.write_text(
        json.dumps(_state["chunks"], ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Public API  (mirrors chatbot/weaviate_store.py signatures)
# ---------------------------------------------------------------------------

def ensure_collection(client=None, settings: Settings | None = None) -> None:
    """Create index files if they don't exist yet (no-op if they do)."""
    _load()
    _save()
    logger.info("[CHATBOT-FAISS] ensure_collection: index ready")


def recreate_collection(client=None, settings: Settings | None = None) -> None:
    """Delete and recreate for a clean rebuild."""
    global _state
    faiss_path, chunks_path = _paths()
    faiss_path.unlink(missing_ok=True)
    chunks_path.unlink(missing_ok=True)
    with _lock:
        _state = None
    ensure_collection(client, settings)
    logger.info("[CHATBOT-FAISS] recreate_collection: clean rebuild")


def get_collection(client=None, settings: Settings | None = None):
    """Return the loaded state dict (used by retriever for vector search)."""
    return _load()


def ensure_collection_exists(client=None, settings: Settings | None = None) -> bool:
    """Self-heal: create if missing.  Returns True if it was freshly created."""
    faiss_path, _ = _paths()
    was_missing = not faiss_path.exists()
    _load()
    if was_missing:
        _save()
    return was_missing


def ingest(
    client=None,
    chunks: list[dict] | None = None,
    embeddings=None,
    recreate: bool = False,
    settings: Settings | None = None,
) -> int:
    """Write message chunks + their vectors to local FAISS + JSON.

    Signature matches ``weaviate_store.ingest()``.
    """
    global _state
    chunks = chunks or []
    if recreate:
        recreate_collection(client, settings)

    from documents.config import EMBED_DIM as dim
    index = faiss.IndexFlatIP(dim)
    stored_chunks: list[dict] = []

    for i, chunk in enumerate(chunks):
        vec = _normalize(embeddings[i])
        index.add(vec.reshape(1, -1))
        stored_chunks.append(chunk)

    with _lock:
        _state = {"index": index, "chunks": stored_chunks}
    _save()

    logger.info("[CHATBOT-FAISS] ingested %d messages", len(chunks))
    return len(chunks)


def collection_count(client=None, settings: Settings | None = None) -> int:
    entry = _load()
    return entry["index"].ntotal


def fetch_all_chunks(client=None, settings: Settings | None = None) -> list[dict]:
    """Return all stored chunks for the local BM25 index."""
    entry = _load()
    out = []
    for chunk in entry["chunks"]:
        if chunk.get("message_id") and (chunk.get("thread_raw_content") or "").strip():
            # Add raw_content alias expected by the BM25 builder
            c = dict(chunk)
            c.setdefault("raw_content", c.get("thread_raw_content", ""))
            out.append(c)
    logger.info("[CHATBOT-FAISS] fetch_all_chunks: %d messages", len(out))
    return out


# ---------------------------------------------------------------------------
# Vector search (used by retriever when FAISS is active)
# ---------------------------------------------------------------------------

def search_vectors(query_vector, limit: int = 50) -> list[tuple[dict, float]]:
    """k-NN search.  Returns [(chunk_dict, score), ...] best-first.

    ``score`` is inner-product similarity (higher is better, ≈ cosine
    since vectors are L2-normalised).  The retriever converts these to
    the same shape as Weaviate hybrid results.
    """
    entry = _load()
    index = entry["index"]
    chunks = entry["chunks"]
    if index.ntotal == 0:
        return []

    qvec = _normalize(query_vector).reshape(1, -1)
    k = min(limit, index.ntotal)
    scores, ids = index.search(qvec, k)

    out: list[tuple[dict, float]] = []
    for score, idx in zip(scores[0], ids[0]):
        if 0 <= idx < len(chunks):
            out.append((chunks[idx], float(score)))
    return out


def clear_cache() -> None:
    global _state
    with _lock:
        _state = None
