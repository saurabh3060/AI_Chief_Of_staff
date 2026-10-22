"""
core/faiss/vector_store.py
==========================
TEMPORARY local FAISS implementation of the email agent's vector-store
contract (see ``agent_email/weaviate_vector_store.py``).

This module mirrors — function name for function name, signature for
signature — the public surface of ``agent_email.weaviate_vector_store`` so
``agent_email.vector_store_backend`` can select one or the other purely by
config (``Settings.vector_store_backend``) without any caller change. Only
the EMAIL CLUSTER pipeline uses it; the chatbot is untouched.

Design notes
------------
* **Namespaces**: ``"clusters"`` and ``"meetings"``, each a SEPARATE
  on-disk FAISS index (one ``<namespace>.faiss`` + one ``<namespace>.json``
  id-map), mirroring the two Weaviate collections. They never share an
  index, so the two namespaces can never collide — the same contract the
  Weaviate store provides via per-namespace collections.

* **IDs are kept strictly separate from Weaviate ids.** Callers pass the
  SAME UUID-string ids the Weaviate path uses (``storage_ids.stable_object_id``),
  but FAISS indexes vectors by int64 ids only. This module keeps a private,
  per-namespace bijection ``{uuid_str <-> int64}`` and translates at the
  boundary: UUID strings go in and come out of the public API; the int64
  FAISS ids never leak out and are never persisted anywhere a Weaviate id
  is stored. The two id spaces do not intermix.

* **Geometry matches Weaviate**: vectors are L2-normalized and stored in an
  inner-product index (``IndexFlatIP`` wrapped in ``IndexIDMap2``), so a
  near-vector search returns cosine similarity in ``[0, 1]`` just like the
  Weaviate ``1 - cosine_distance`` score.

* ``faiss`` is imported lazily (inside ``_require_faiss``) so importing this
  module never fails when ``faiss-cpu`` is not installed — only actually
  USING the FAISS backend requires the library.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_email")

# ---------------------------------------------------------------------------
# Namespaces + sentinels — identical names/values to weaviate_vector_store so
# callers that import these constants keep working regardless of backend.
# ---------------------------------------------------------------------------
NAMESPACE_CLUSTERS = "clusters"
NAMESPACE_MEETINGS = "meetings"
OWNED_NAMESPACES: tuple[str, ...] = (NAMESPACE_CLUSTERS, NAMESPACE_MEETINGS)

COUNT_MISSING = -1
COUNT_UNRESOLVABLE = -2

# One lock guards the per-process index cache + the on-disk read/write of the
# index and id-map files, so concurrent pipeline threads don't corrupt either.
_lock = threading.RLock()

# namespace -> _NamespaceIndex (lazy, cached per process).
_indexes: dict[str, "_NamespaceIndex"] = {}


def _require_faiss():
    """
    Import and return the ``faiss`` module, with a clear error if it is not
    installed. Imported lazily so the Weaviate path (and plain imports of
    this module) never need ``faiss-cpu`` present.
    """
    try:
        import faiss  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "VECTOR_STORE_BACKEND='faiss' requires the 'faiss-cpu' package. "
            "Install it (pip install faiss-cpu) or set VECTOR_STORE_BACKEND='weaviate'."
        ) from exc
    return faiss


def _uuid_to_int64(uuid_str: str) -> int:
    """
    Deterministically map a UUID-string id to a signed 63-bit int64 FAISS id.

    FAISS id maps need int64 ids; our external ids are UUID strings. We hash
    the string and take 63 bits (always non-negative, fits a signed int64),
    so the same UUID always maps to the same FAISS id within and across runs.
    The reverse direction is served by the persisted id-map (below), not by
    inverting the hash. This int64 is PRIVATE to FAISS and never stored where
    a Weaviate id would be.

    SHA-256 is used purely as a deterministic string->int mapping (not for
    any security purpose); we take the first 8 bytes and mask to 63 bits.
    """
    digest = hashlib.sha256(uuid_str.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Per-namespace index + id map
# ---------------------------------------------------------------------------
class _NamespaceIndex:
    """
    One namespace's FAISS index plus its ``{uuid_str <-> int64}`` id map,
    backed by two files under the configured FAISS index dir:
        <dir>/<namespace>.faiss   (the vectors, keyed by int64 FAISS ids)
        <dir>/<namespace>.json    (the uuid<->int64 bijection + dim)
    """

    def __init__(self, namespace: str, settings: Settings):
        self.namespace = namespace
        self._settings = settings
        base = settings.resolved_faiss_index_dir
        self._index_path: Path = base / f"{namespace}.faiss"
        self._map_path: Path = base / f"{namespace}.json"
        self._index = None  # faiss.IndexIDMap2, created lazily
        self._dim: int | None = None
        # uuid string -> int64 faiss id, and the reverse.
        self._uuid_to_id: dict[str, int] = {}
        self._id_to_uuid: dict[int, str] = {}
        self._load()

    # ---- persistence ----------------------------------------------------
    def _load(self) -> None:
        faiss = _require_faiss()
        if self._map_path.exists():
            try:
                data = json.loads(self._map_path.read_text(encoding="utf-8"))
                self._dim = int(data.get("dim")) if data.get("dim") else None
                self._uuid_to_id = {str(k): int(v) for k, v in data.get("uuid_to_id", {}).items()}
                self._id_to_uuid = {int(k): str(v) for k, v in data.get("id_to_uuid", {}).items()}
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[EMAIL-FAISS] failed to read id map %s (%s); starting empty",
                    self._map_path, exc,
                )
                self._dim, self._uuid_to_id, self._id_to_uuid = None, {}, {}
        if self._index_path.exists():
            try:
                self._index = faiss.read_index(str(self._index_path))
                if self._dim is None:
                    self._dim = int(self._index.d)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[EMAIL-FAISS] failed to read index %s (%s); starting empty",
                    self._index_path, exc,
                )
                self._index = None

    def _persist(self) -> None:
        faiss = _require_faiss()
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(self._index_path))
        payload = {
            "dim": self._dim,
            "uuid_to_id": self._uuid_to_id,
            # json keys must be strings; cast the int64 keys for storage.
            "id_to_uuid": {str(k): v for k, v in self._id_to_uuid.items()},
        }
        self._map_path.write_text(json.dumps(payload), encoding="utf-8")

    # ---- index helpers --------------------------------------------------
    def _ensure_index(self, dim: int):
        """Create the IndexIDMap2(IndexFlatIP) on first write, fixing the dim."""
        faiss = _require_faiss()
        if self._index is None:
            self._dim = dim
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        elif self._dim is not None and dim != self._dim:
            raise ValueError(
                f"[EMAIL-FAISS] namespace={self.namespace} vector dim {dim} does not "
                f"match existing index dim {self._dim}"
            )
        return self._index

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr

    # ---- operations -----------------------------------------------------
    def upsert(self, items: list[tuple[str, np.ndarray]]) -> int:
        if not items:
            return 0
        faiss = _require_faiss()
        dim = int(np.asarray(items[0][1]).reshape(-1).shape[0])
        index = self._ensure_index(dim)

        # Replace-if-exists semantics (upsert): remove any existing FAISS ids
        # for these uuids first, then add the fresh vectors.
        existing_ids = [
            self._uuid_to_id[uuid_str]
            for uuid_str, _ in items
            if uuid_str in self._uuid_to_id
        ]
        if existing_ids:
            index.remove_ids(np.asarray(existing_ids, dtype=np.int64))

        vectors = np.vstack([self._normalize(vec) for _, vec in items]).astype(np.float32)
        int_ids = np.empty(len(items), dtype=np.int64)
        for i, (uuid_str, _) in enumerate(items):
            fid = _uuid_to_int64(uuid_str)
            int_ids[i] = fid
            self._uuid_to_id[uuid_str] = fid
            self._id_to_uuid[fid] = uuid_str
        index.add_with_ids(vectors, int_ids)
        self._persist()
        return len(items)

    def get(self, uuid_str: str) -> np.ndarray | None:
        if not uuid_str or self._index is None:
            return None
        fid = self._uuid_to_id.get(uuid_str)
        if fid is None:
            return None
        try:
            vec = self._index.reconstruct(int(fid))
        except Exception:  # noqa: BLE001 - id not present in the index
            return None
        return np.asarray(vec, dtype=np.float32)

    def get_many(self, uuid_strs: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for uuid_str in uuid_strs:
            vec = self.get(uuid_str)
            if vec is not None:
                out[uuid_str] = vec
        return out

    def search(self, query_vec: np.ndarray, limit: int) -> list[tuple[str, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        arr = self._normalize(query_vec)
        if arr.size == 0 or float(np.linalg.norm(arr)) == 0.0:
            return []
        k = min(max(1, int(limit)), int(self._index.ntotal))
        scores, ids = self._index.search(arr.reshape(1, -1), k)
        out: list[tuple[str, float]] = []
        for score, fid in zip(scores[0], ids[0]):
            if fid == -1:
                continue
            uuid_str = self._id_to_uuid.get(int(fid))
            if uuid_str is None:
                continue
            out.append((uuid_str, float(score)))
        return out

    def remove(self, uuid_strs: list[str]) -> int:
        ids = [self._uuid_to_id[u] for u in uuid_strs if u in self._uuid_to_id]
        if not ids or self._index is None:
            return len([u for u in uuid_strs if u])
        self._index.remove_ids(np.asarray(ids, dtype=np.int64))
        for uuid_str in uuid_strs:
            fid = self._uuid_to_id.pop(uuid_str, None)
            if fid is not None:
                self._id_to_uuid.pop(fid, None)
        self._persist()
        return len([u for u in uuid_strs if u])

    def clear(self) -> int:
        """Delete every vector in this namespace, keeping the (empty) index."""
        before = int(self._index.ntotal) if self._index is not None else 0
        faiss = _require_faiss()
        dim = self._dim or int(self._settings.embedding_dim)
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._dim = dim
        self._uuid_to_id.clear()
        self._id_to_uuid.clear()
        self._persist()
        return before

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0

    @property
    def exists(self) -> bool:
        return self._index_path.exists()


def _get_ns(namespace: str, settings: Settings | None = None) -> _NamespaceIndex:
    if namespace not in OWNED_NAMESPACES:
        raise ValueError(
            f"[EMAIL-FAISS] unknown namespace {namespace!r}; expected one of {OWNED_NAMESPACES}"
        )
    active = settings or get_settings()
    with _lock:
        ns = _indexes.get(namespace)
        if ns is None:
            ns = _NamespaceIndex(namespace, active)
            _indexes[namespace] = ns
        return ns


# ---------------------------------------------------------------------------
# Public API — mirrors agent_email.weaviate_vector_store exactly.
# ---------------------------------------------------------------------------
def upsert_vectors(
    namespace: str,
    items: list[tuple[str, np.ndarray]],
    settings: Settings | None = None,
) -> int:
    """Insert or replace ``(uuid_str, vector)`` pairs. Returns count written."""
    if not items:
        return 0
    with _lock:
        n = _get_ns(namespace, settings).upsert(items)
    logger.info("[EMAIL-FAISS] upserted namespace=%s vectors=%d", namespace, n)
    return n


def get_vector(
    namespace: str,
    weaviate_id: str,
    settings: Settings | None = None,
) -> np.ndarray | None:
    """Return the stored (L2-normalized) vector for `weaviate_id`, or None."""
    with _lock:
        return _get_ns(namespace, settings).get(weaviate_id)


def get_vectors(
    namespace: str,
    weaviate_ids: list[str],
    settings: Settings | None = None,
) -> dict[str, np.ndarray]:
    """Batch variant of :func:`get_vector`. Returns only ids that were found."""
    ids = [wid for wid in weaviate_ids if wid]
    if not ids:
        return {}
    with _lock:
        return _get_ns(namespace, settings).get_many(ids)


def search_vectors(
    namespace: str,
    query_vec: np.ndarray,
    limit: int = 12,
    settings: Settings | None = None,
) -> list[tuple[str, float]]:
    """
    Cosine search over `namespace`. Returns up to `limit`
    ``(uuid_str, score)`` pairs best-first, score in ``[0, 1]`` — the same
    contract as the Weaviate ``search_vectors``.
    """
    with _lock:
        hits = _get_ns(namespace, settings).search(query_vec, limit)
    logger.info(
        "[EMAIL-FAISS] search namespace=%s limit=%d hits=%d", namespace, limit, len(hits)
    )
    return hits


def remove_vectors(
    namespace: str,
    weaviate_ids: list[str],
    settings: Settings | None = None,
) -> int:
    """Remove the given ids from `namespace`. Missing ids are ignored."""
    ids = [wid for wid in weaviate_ids if wid]
    if not ids:
        return 0
    with _lock:
        n = _get_ns(namespace, settings).remove(ids)
    logger.info("[EMAIL-FAISS] removed namespace=%s requested=%d", namespace, n)
    return n


def clear_namespace_objects(namespace: str, settings: Settings | None = None) -> int:
    """Delete every vector in `namespace`, keeping the (now empty) index."""
    if namespace not in OWNED_NAMESPACES:
        raise ValueError(
            f"clear_namespace_objects refuses namespace {namespace!r}: not one of "
            f"this project's owned namespaces {OWNED_NAMESPACES}"
        )
    with _lock:
        n = _get_ns(namespace, settings).clear()
    logger.info("[EMAIL-FAISS] cleared namespace=%s objects=%d", namespace, n)
    return n


def ensure_schema(settings: Settings | None = None) -> dict[str, bool]:
    """
    Create each namespace's on-disk index if missing. Returns
    ``{namespace: created_now}``. For parity with the Weaviate contract;
    for FAISS "schema" just means the index file exists.
    """
    active = settings or get_settings()
    created: dict[str, bool] = {}
    for namespace in OWNED_NAMESPACES:
        with _lock:
            ns = _get_ns(namespace, active)
            was_missing = not ns.exists
            if was_missing:
                # Materialize an empty index on disk so later reads find it.
                ns.clear()
            created[namespace] = was_missing
    return created


def ensure_schema_ready(
    settings: Settings | None = None,
    attempts: int = 5,
    base_delay: float = 0.5,
) -> dict[str, dict]:
    """
    Readiness gate parity with the Weaviate store. FAISS is local and has no
    REST/gRPC desync, so a single ``ensure_schema`` pass is always "ready".
    Returns ``{namespace: {"ready", "objects", "created_now", "attempts"}}``.
    """
    active = settings or get_settings()
    created = ensure_schema(active)
    report: dict[str, dict] = {}
    for namespace in OWNED_NAMESPACES:
        with _lock:
            objects = _get_ns(namespace, active).ntotal
        report[namespace] = {
            "ready": True,
            "objects": objects,
            "created_now": created.get(namespace, False),
            "attempts": 1,
        }
    return report


def object_count(namespace: str, settings: Settings | None = None) -> int:
    """Number of vectors in `namespace`, or COUNT_MISSING if no index yet."""
    active = settings or get_settings()
    with _lock:
        ns = _get_ns(namespace, active)
        if not ns.exists and ns.ntotal == 0:
            return COUNT_MISSING
        return ns.ntotal


def count(namespace: str, settings: Settings | None = None) -> int:
    """Number of vectors currently stored in `namespace`."""
    with _lock:
        return _get_ns(namespace, settings).ntotal


def collection_exists(namespace: str, settings: Settings | None = None) -> bool:
    """True iff the on-disk index for `namespace` exists."""
    with _lock:
        return _get_ns(namespace, settings).exists


def cluster_node_count(settings: Settings | None = None) -> int:
    """FAISS is local/in-process — there is no cluster. Always 1 (parity)."""
    return 1


def startup_report(settings: Settings | None = None) -> dict:
    """
    Build + log a startup report in the SAME shape the Weaviate
    ``startup_report`` returns, so the shared startup/health code works
    unchanged under the FAISS backend.
    """
    active = settings or get_settings()
    per_collection: dict[str, dict] = {}
    for namespace in OWNED_NAMESPACES:
        cnt = object_count(namespace, active)
        exists = cnt >= 0
        per_collection[namespace] = {
            "collection": f"faiss:{namespace}",
            "exists": exists,
            "unresolvable": False,
            "objects": max(cnt, 0),
        }
        logger.info(
            "[EMAIL-FAISS][STARTUP] namespace=%s exists=%s objects=%d",
            namespace, exists, max(cnt, 0),
        )
    logger.info(
        "[EMAIL-FAISS][STARTUP] index_dir=%s", active.resolved_faiss_index_dir
    )
    return {"nodes": 1, "collections": per_collection}


def clear_cache() -> None:
    """
    Drop the in-process index cache (does NOT delete anything on disk).
    Parity with the Weaviate store's handle-cache reset; also used by tests
    that repoint the FAISS index dir.
    """
    with _lock:
        _indexes.clear()
