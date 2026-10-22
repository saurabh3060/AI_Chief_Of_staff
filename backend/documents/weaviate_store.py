"""
documents/weaviate_store.py
============================
Weaviate vector store for the documents bot.

Replaces the SQLite ``blocks_vec`` and ``series_vec`` virtual tables
(sqlite-vec) with two Weaviate collections:

* ``<prefix>Blocks``   — one vector per block (passage), id derived from block_id
* ``<prefix>Series``   — one vector per series (metric), id derived from series_id

Row data (blocks, series) still lives in PostgreSQL; this module only
holds the embedding vectors.  The PostgreSQL row id (``block_id``,
``series_id``) is converted to a deterministic UUID v5 for use as the
Weaviate object id, so repeat inserts for the same row cleanly upsert
and ``delete_by_doc_id`` can address them by id.

Vectors are self-provided (``Vectorizer.none()``) and L2-normalized
before insertion so cosine distance behaves consistently.

All Weaviate calls are blocking (gRPC); callers wrap them in
``asyncio.to_thread`` since this module is used from async code.
"""

from __future__ import annotations

import threading
import uuid
from typing import Iterable

import numpy as np
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter, MetadataQuery

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger
from core.weaviate.client import get_client

logger = get_agent_logger("documents")

# Namespace suffix appended to settings.weaviate_doc_collection_prefix.
NAMESPACE_BLOCKS = "Blocks"
NAMESPACE_SERIES = "Series"

# Stable UUID namespace for mapping (block_id, series_id) -> uuid.
# Fixed constant — do NOT change, or every stored vector id becomes
# unreachable.
_UUID_NS_BLOCKS = uuid.UUID("5d2cfd1c-7d1d-4b7f-9f4f-9d1b1c3f0001")
_UUID_NS_SERIES = uuid.UUID("5d2cfd1c-7d1d-4b7f-9f4f-9d1b1c3f0002")

# Property name used to store the natural numeric id alongside the vector.
_KEY_PROPERTY = "row_id"
# Property used so we can delete every vector belonging to a document in one call.
_DOC_PROPERTY = "doc_id"

_lock = threading.Lock()
_collections: dict[str, "weaviate.collections.Collection"] = {}


# ---------------------------------------------------------------------------
# Collection naming + management
# ---------------------------------------------------------------------------

def _prefix(settings: Settings | None = None) -> str:
    active = settings or get_settings()
    # Fall back to a sensible default if the setting is not defined, so
    # the module keeps working on an older config.
    return getattr(active, "weaviate_doc_collection_prefix", "DocLibraryDev")


def _collection_name(namespace: str, settings: Settings | None = None) -> str:
    return f"{_prefix(settings)}{namespace}"


def owned_collection_names(settings: Settings | None = None) -> dict[str, str]:
    return {
        NAMESPACE_BLOCKS: _collection_name(NAMESPACE_BLOCKS, settings),
        NAMESPACE_SERIES: _collection_name(NAMESPACE_SERIES, settings),
    }


def _create_collection_if_missing(client, name: str) -> bool:
    if client.collections.exists(name):
        return False
    client.collections.create(
        name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name=_KEY_PROPERTY, data_type=DataType.INT),
            Property(name=_DOC_PROPERTY, data_type=DataType.TEXT),
        ],
    )
    logger.info("[DOC-WEAVIATE] created collection %s", name)
    return True


def _get_collection(namespace: str, settings: Settings | None = None):
    with _lock:
        if namespace in _collections:
            return _collections[namespace]
        active = settings or get_settings()
        client = get_client(active)
        name = _collection_name(namespace, active)
        _create_collection_if_missing(client, name)
        collection = client.collections.get(name)
        _collections[namespace] = collection
        return collection


def ensure_collections(settings: Settings | None = None) -> dict[str, bool]:
    """Create the Blocks and Series collections if they don't exist."""
    active = settings or get_settings()
    client = get_client(active)
    created: dict[str, bool] = {}
    for namespace, name in owned_collection_names(active).items():
        was_created = _create_collection_if_missing(client, name)
        created[name] = was_created
        with _lock:
            _collections[namespace] = client.collections.get(name)
    logger.info("[DOC-WEAVIATE] ensure_collections -> %s", created)
    return created


def clear_cache() -> None:
    """Drop cached collection handles (used by tests)."""
    with _lock:
        _collections.clear()


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def block_uuid(block_id: int) -> str:
    return str(uuid.uuid5(_UUID_NS_BLOCKS, str(int(block_id))))


def series_uuid(series_id: int) -> str:
    return str(uuid.uuid5(_UUID_NS_SERIES, str(int(series_id))))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _upsert(namespace: str, items: Iterable[tuple[int, str, "np.ndarray | list[float]"]],
            settings: Settings | None = None) -> int:
    """items: iterable of (row_id, doc_id, vector)."""
    materialized = list(items)
    if not materialized:
        return 0
    collection = _get_collection(namespace, settings)
    uuid_fn = block_uuid if namespace == NAMESPACE_BLOCKS else series_uuid
    objects = [
        DataObject(
            uuid=uuid_fn(row_id),
            properties={_KEY_PROPERTY: int(row_id), _DOC_PROPERTY: str(doc_id)},
            vector=_normalize(vec).tolist(),
        )
        for row_id, doc_id, vec in materialized
    ]
    # insert_many with explicit uuids = upsert (replace-if-exists).
    collection.data.insert_many(objects)
    return len(objects)


def upsert_block_embeddings(
    items: Iterable[tuple[int, str, "np.ndarray | list[float]"]],
    settings: Settings | None = None,
) -> int:
    """items: iterable of (block_id, doc_id, embedding)."""
    n = _upsert(NAMESPACE_BLOCKS, items, settings)
    logger.info("[DOC-WEAVIATE] upserted %d block vectors", n)
    return n


def upsert_series_embeddings(
    items: Iterable[tuple[int, str, "np.ndarray | list[float]"]],
    settings: Settings | None = None,
) -> int:
    """items: iterable of (series_id, doc_id, embedding)."""
    n = _upsert(NAMESPACE_SERIES, items, settings)
    logger.info("[DOC-WEAVIATE] upserted %d series vectors", n)
    return n


# ---------------------------------------------------------------------------
# Search (k-NN by cosine distance)
# ---------------------------------------------------------------------------

def _search(namespace: str, qvec, k: int, doc_ids: list[str] | None,
            settings: Settings | None = None) -> list[tuple[int, float]]:
    collection = _get_collection(namespace, settings)
    query_vec = _normalize(qvec).tolist()
    kwargs = dict(
        near_vector=query_vec,
        limit=int(k),
        return_metadata=MetadataQuery(distance=True),
        return_properties=[_KEY_PROPERTY, _DOC_PROPERTY],
    )
    if doc_ids:
        kwargs["filters"] = Filter.by_property(_DOC_PROPERTY).contains_any(list(doc_ids))
    res = collection.query.near_vector(**kwargs)
    out: list[tuple[int, float]] = []
    for obj in res.objects:
        row_id = obj.properties.get(_KEY_PROPERTY)
        if row_id is None:
            continue
        dist = float(obj.metadata.distance) if obj.metadata and obj.metadata.distance is not None else 0.0
        out.append((int(row_id), dist))
    return out


def search_blocks(qvec, k: int, doc_ids: list[str] | None = None,
                  settings: Settings | None = None) -> list[tuple[int, float]]:
    """Return [(block_id, distance), ...] in ascending distance order."""
    return _search(NAMESPACE_BLOCKS, qvec, k, doc_ids, settings)


def search_series(qvec, k: int, doc_ids: list[str] | None = None,
                  settings: Settings | None = None) -> list[tuple[int, float]]:
    """Return [(series_id, distance), ...] in ascending distance order."""
    return _search(NAMESPACE_SERIES, qvec, k, doc_ids, settings)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def delete_block_ids(block_ids: list[int], settings: Settings | None = None) -> int:
    if not block_ids:
        return 0
    collection = _get_collection(NAMESPACE_BLOCKS, settings)
    uuids = [block_uuid(i) for i in block_ids]
    collection.data.delete_many(where=Filter.by_id().contains_any(uuids))
    return len(uuids)


def delete_series_ids(series_ids: list[int], settings: Settings | None = None) -> int:
    if not series_ids:
        return 0
    collection = _get_collection(NAMESPACE_SERIES, settings)
    uuids = [series_uuid(i) for i in series_ids]
    collection.data.delete_many(where=Filter.by_id().contains_any(uuids))
    return len(uuids)


def delete_by_doc_id(doc_id: str, settings: Settings | None = None) -> dict[str, int]:
    """Remove every block and series vector that belongs to `doc_id`."""
    counts = {}
    for namespace in (NAMESPACE_BLOCKS, NAMESPACE_SERIES):
        collection = _get_collection(namespace, settings)
        res = collection.data.delete_many(
            where=Filter.by_property(_DOC_PROPERTY).equal(str(doc_id))
        )
        # Weaviate v4 returns a result object with .matches / .successful.
        counts[namespace] = int(getattr(res, "successful", 0) or getattr(res, "matches", 0) or 0)
    logger.info("[DOC-WEAVIATE] delete_by_doc_id doc_id=%s counts=%s", doc_id, counts)
    return counts
