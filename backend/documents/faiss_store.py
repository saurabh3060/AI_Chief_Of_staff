"""
documents/faiss_store.py
=========================
Local FAISS vector store for the documents bot — drop-in replacement for
``documents/weaviate_store.py``.

Activated by ``CHATBOT_VECTOR_DB=faiss`` in ``.env``.  Two FAISS indexes
(inner-product on L2-normalised vectors ≈ cosine similarity):

* ``doc_blocks.faiss``  + ``doc_blocks_meta.json``
* ``doc_series.faiss``  + ``doc_series_meta.json``

Metadata JSON maps FAISS internal position → ``{row_id, doc_id}`` so
we can translate results back and filter by ``doc_id`` at query time.

All functions are **synchronous** (same as ``weaviate_store.py``);
callers wrap them in ``asyncio.to_thread``.

The public API surface matches ``weaviate_store.py`` exactly so the
dispatch layer in ``store.py`` can swap them transparently.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from core.config import get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("documents")

NAMESPACE_BLOCKS = "Blocks"
NAMESPACE_SERIES = "Series"

_lock = threading.Lock()

# In-memory state per namespace: {"Blocks": {"index": faiss.Index, "meta": [...]}, ...}
_indexes: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _index_dir() -> Path:
    settings = get_settings()
    d = Path(settings.faiss_index_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _paths(namespace: str) -> tuple[Path, Path]:
    d = _index_dir()
    tag = "doc_blocks" if namespace == NAMESPACE_BLOCKS else "doc_series"
    return d / f"{tag}.faiss", d / f"{tag}_meta.json"


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def _load(namespace: str) -> dict:
    """Load or create an index + metadata for a namespace.  Returns the
    cached dict ``{"index": faiss.IndexFlatIP, "meta": [...]}``."""
    with _lock:
        if namespace in _indexes:
            return _indexes[namespace]

    faiss_path, meta_path = _paths(namespace)
    from . import config
    dim = config.EMBED_DIM

    if faiss_path.exists() and meta_path.exists():
        index = faiss.read_index(str(faiss_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        index = faiss.IndexFlatIP(dim)
        meta = []  # list of {"row_id": int, "doc_id": str}

    entry = {"index": index, "meta": meta}
    with _lock:
        _indexes[namespace] = entry
    return entry


def _save(namespace: str) -> None:
    entry = _indexes.get(namespace)
    if entry is None:
        return
    faiss_path, meta_path = _paths(namespace)
    faiss.write_index(entry["index"], str(faiss_path))
    meta_path.write_text(json.dumps(entry["meta"], ensure_ascii=False), encoding="utf-8")


def _normalize(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr


# ---------------------------------------------------------------------------
# Public API  (matches weaviate_store.py signatures)
# ---------------------------------------------------------------------------

def ensure_collections(settings=None) -> dict[str, bool]:
    """Create index files if they don't exist yet."""
    created: dict[str, bool] = {}
    for ns in (NAMESPACE_BLOCKS, NAMESPACE_SERIES):
        fp, mp = _paths(ns)
        was_new = not fp.exists()
        _load(ns)
        if was_new:
            _save(ns)
        created[ns] = was_new
    logger.info("[DOC-FAISS] ensure_collections -> %s", created)
    return created


def clear_cache() -> None:
    with _lock:
        _indexes.clear()


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _upsert(namespace: str, items: Iterable[tuple[int, str, "np.ndarray | list[float]"]]) -> int:
    materialized = list(items)
    if not materialized:
        return 0
    entry = _load(namespace)
    index: faiss.IndexFlatIP = entry["index"]
    meta: list[dict] = entry["meta"]

    # Build a set of existing row_ids for fast lookup
    existing = {m["row_id"]: pos for pos, m in enumerate(meta)}

    # For simplicity: remove existing entries for these row_ids, then append all.
    # FAISS IndexFlatIP doesn't support in-place update, so we rebuild when there
    # are removals; otherwise just append.
    remove_positions = set()
    for row_id, _, _ in materialized:
        if row_id in existing:
            remove_positions.add(existing[row_id])

    if remove_positions:
        # Rebuild index without the removed positions
        keep = [i for i in range(len(meta)) if i not in remove_positions]
        if keep:
            vecs = np.vstack([index.reconstruct(i) for i in keep]).astype(np.float32)
            new_meta = [meta[i] for i in keep]
        else:
            from . import config
            vecs = np.empty((0, config.EMBED_DIM), dtype=np.float32)
            new_meta = []
        new_index = faiss.IndexFlatIP(index.d)
        if vecs.shape[0] > 0:
            new_index.add(vecs)
        entry["index"] = new_index
        entry["meta"] = new_meta
        index = new_index
        meta = new_meta

    # Append new vectors
    new_vecs = np.vstack([_normalize(vec) for _, _, vec in materialized]).astype(np.float32)
    index.add(new_vecs)
    for row_id, doc_id, _ in materialized:
        meta.append({"row_id": int(row_id), "doc_id": str(doc_id)})

    _save(namespace)
    return len(materialized)


def upsert_block_embeddings(
    items: Iterable[tuple[int, str, "np.ndarray | list[float]"]],
    settings=None,
) -> int:
    n = _upsert(NAMESPACE_BLOCKS, items)
    logger.info("[DOC-FAISS] upserted %d block vectors", n)
    return n


def upsert_series_embeddings(
    items: Iterable[tuple[int, str, "np.ndarray | list[float]"]],
    settings=None,
) -> int:
    n = _upsert(NAMESPACE_SERIES, items)
    logger.info("[DOC-FAISS] upserted %d series vectors", n)
    return n


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _search(namespace: str, qvec, k: int, doc_ids: list[str] | None) -> list[tuple[int, float]]:
    entry = _load(namespace)
    index: faiss.IndexFlatIP = entry["index"]
    meta: list[dict] = entry["meta"]
    if index.ntotal == 0:
        return []

    query = _normalize(qvec).reshape(1, -1)
    # Search more than k when filtering by doc_ids, since some results
    # will be filtered out.
    search_k = min(index.ntotal, k * 5 if doc_ids else k)
    scores, ids = index.search(query, search_k)

    out: list[tuple[int, float]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx]
        if doc_ids and m["doc_id"] not in doc_ids:
            continue
        # Convert inner-product similarity to distance (1 - similarity)
        # so the caller gets the same "lower is better" semantics as Weaviate.
        distance = 1.0 - float(score)
        out.append((m["row_id"], distance))
        if len(out) >= k:
            break
    return out


def search_blocks(qvec, k: int, doc_ids: list[str] | None = None,
                  settings=None) -> list[tuple[int, float]]:
    return _search(NAMESPACE_BLOCKS, qvec, k, doc_ids)


def search_series(qvec, k: int, doc_ids: list[str] | None = None,
                  settings=None) -> list[tuple[int, float]]:
    return _search(NAMESPACE_SERIES, qvec, k, doc_ids)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def _delete_by_row_ids(namespace: str, row_ids: list[int]) -> int:
    if not row_ids:
        return 0
    entry = _load(namespace)
    index: faiss.IndexFlatIP = entry["index"]
    meta: list[dict] = entry["meta"]

    to_remove = set(row_ids)
    keep = [i for i, m in enumerate(meta) if m["row_id"] not in to_remove]
    removed = len(meta) - len(keep)

    if removed > 0:
        if keep:
            vecs = np.vstack([index.reconstruct(i) for i in keep]).astype(np.float32)
            new_meta = [meta[i] for i in keep]
        else:
            from . import config
            vecs = np.empty((0, config.EMBED_DIM), dtype=np.float32)
            new_meta = []
        new_index = faiss.IndexFlatIP(index.d)
        if vecs.shape[0] > 0:
            new_index.add(vecs)
        entry["index"] = new_index
        entry["meta"] = new_meta
        _save(namespace)

    return removed


def delete_block_ids(block_ids: list[int], settings=None) -> int:
    return _delete_by_row_ids(NAMESPACE_BLOCKS, block_ids)


def delete_series_ids(series_ids: list[int], settings=None) -> int:
    return _delete_by_row_ids(NAMESPACE_SERIES, series_ids)


def delete_by_doc_id(doc_id: str, settings=None) -> dict[str, int]:
    counts = {}
    for namespace in (NAMESPACE_BLOCKS, NAMESPACE_SERIES):
        entry = _load(namespace)
        ids_to_remove = [m["row_id"] for m in entry["meta"] if m["doc_id"] == doc_id]
        counts[namespace] = _delete_by_row_ids(namespace, ids_to_remove)
    logger.info("[DOC-FAISS] delete_by_doc_id doc_id=%s counts=%s", doc_id, counts)
    return counts


def owned_collection_names(settings=None) -> dict[str, str]:
    return {
        NAMESPACE_BLOCKS: "doc_blocks.faiss",
        NAMESPACE_SERIES: "doc_series.faiss",
    }
