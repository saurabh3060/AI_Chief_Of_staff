"""
agent_email/weaviate_vector_store.py
=====================================
Weaviate vector store for the email cluster synthesis pipeline.

The *row* data (cluster metadata, per-mail topics, meeting metadata) and
the embeddings live in PostgreSQL (the system of record); the embedding
**vectors** are mirrored here into Weaviate as a rebuildable searchable
copy. PostgreSQL also stores a stable
``weaviate_id`` (a deterministic UUID string, see
``agent_email/storage_ids.py``) per row; the actual float vector is
fetched from / written to Weaviate by that id.

Design notes
------------
* Two independent namespaces are kept so cluster and meeting vectors never
  collide: ``"clusters"`` and ``"meetings"``. Each namespace is a separate
  Weaviate collection (``settings.weaviate_collection_name(namespace)``),
  mirroring the previous per-namespace FAISS index files.
* Vectors are provided by the caller (self-provided), not computed by
  Weaviate — collections are created with
  ``Configure.Vectorizer.none()``. We L2-normalize before insertion
  so cosine distance search behaves the same way the old FAISS inner-
  product index did.
* Collections are created lazily (on first use) if they don't already
  exist on the Weaviate instance, and cached per-namespace in-process so
  repeat calls don't re-fetch collection metadata every time.
* This module makes network calls (unlike the old FAISS module, which was
  purely local/CPU-bound). Callers already wrap calls in
  ``asyncio.to_thread`` (see ``postgres_store.py``), which still applies
  here to avoid blocking the event loop.
* There is no fallback to FAISS or any other vector store anywhere in
  this module. If Weaviate is unreachable or returns an error, calls
  raise — callers must handle that explicitly rather than silently
  degrading to a different backend.

Weaviate is the source of truth for vectors; PostgreSQL is the source of
truth for row data + the id linkage. A row without a matching Weaviate
object simply has no embedding (treated as "no embedding" by the matching
code), the same failure mode the CSV path already tolerates via
``_decode_json_embedding`` returning ``None``.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import numpy as np
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.exceptions import UnexpectedStatusCodeError

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger

from core.weaviate.client import get_client

logger = get_agent_logger("agent_email")

# Namespaces (one Weaviate collection each).
NAMESPACE_CLUSTERS = "clusters"
NAMESPACE_MEETINGS = "meetings"

# One lock guards per-process collection-handle caching. The Weaviate
# client itself is thread-safe for concurrent use; this only protects the
# `_collections` cache dict from concurrent mutation.
_lock = threading.Lock()

# namespace -> Weaviate collection handle (lazy, cached per process).
_collections: dict[str, "weaviate.collections.Collection"] = {}

# Property name used to store the natural key alongside each vector, for
# debugging / future lookups. Not used for querying; Weaviate's own
# object UUID (the deterministic weaviate_id) is the lookup key.
_KEY_PROPERTY = "natural_key"

# Sentinel used for a type-agnostic "match every object" delete filter:
# every stored object has a creation time strictly greater than the Unix
# epoch, so `creation_time > epoch` selects all of them regardless of any
# property's inferred data type. `by_creation_time()` compares against a
# datetime, so use epoch start (tz-aware) rather than the integer 0.
_EPOCH_START = datetime(1970, 1, 1, tzinfo=timezone.utc)


# The namespaces this project owns. Every collection this module may ever
# create, read, verify, or (via the gated admin script only) delete derives
# its name from one of these through ``weaviate_collection_name()``. Any
# collection whose name does not match one of ours is NEVER touched — the
# Weaviate server may be shared with other projects.
OWNED_NAMESPACES: tuple[str, ...] = (NAMESPACE_CLUSTERS, NAMESPACE_MEETINGS)


def _collection_name(namespace: str, settings: Settings | None = None) -> str:
    active = settings or get_settings()
    return active.weaviate_collection_name(namespace)


def owned_collection_names(settings: Settings | None = None) -> dict[str, str]:
    """
    Return ``{namespace: collection_name}`` for every collection this
    project owns, using the configured prefix. This is the single source
    of truth for "which collections are ours" — the startup report, the
    self-healing restore, and the admin delete script all consult it so
    nothing outside this set is ever created, verified, or deleted.
    """
    active = settings or get_settings()
    return {ns: _collection_name(ns, active) for ns in OWNED_NAMESPACES}


def is_owned_collection(name: str, settings: Settings | None = None) -> bool:
    """True iff `name` is one of this project's own collection names."""
    return name in set(owned_collection_names(settings).values())


def _create_collection_if_missing(client, name: str) -> bool:
    """
    Create the collection `name` only when it does not already exist.

    Returns True if it was created by this call, False if it already
    existed. NEVER deletes or recreates an existing collection, so a class
    that is already present (ours or, defensively, anyone else's) is left
    exactly as-is. Vectors are self-provided (``Vectorizer.none()``).

    Robustness note: on some Weaviate deployments the REST schema check
    ``client.collections.exists(name)`` reports an EXISTING collection as
    absent (its REST schema view disagrees with what the server actually
    holds, which the gRPC aggregate can still resolve). Trusting that stale
    ``False`` made this function attempt a ``create`` for a collection that
    is really there, and the server rejected it with a 422
    ``class name ... already exists`` — crashing ``ensure_schema`` at
    startup. So we also treat that specific 422 as "already existed"
    (return False) rather than letting it propagate. We still NEVER delete
    or recreate anything; this only makes the create-if-missing guard
    idempotent against a flaky existence check.
    """
    if client.collections.exists(name):
        return False
    try:
        client.collections.create(
            name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name=_KEY_PROPERTY, data_type=DataType.TEXT),
            ],
        )
    except UnexpectedStatusCodeError as exc:
        # exists() said "no" but the server says the class is already
        # there — the collection is present, nothing to create.
        if "already exists" in str(exc).lower():
            logger.info(
                "[EMAIL-WEAVIATE] collection %s already exists (REST exists() was "
                "stale); treating as present, not re-creating",
                name,
            )
            return False
        raise
    logger.info("[EMAIL-WEAVIATE] created collection %s", name)
    return True


def _get_collection(namespace: str, settings: Settings | None = None):
    """
    Return the cached collection handle for `namespace`, creating the
    collection on the Weaviate instance on first use if it doesn't exist
    yet. Raises if Weaviate is unreachable (no fallback).
    """
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


def ensure_schema(settings: Settings | None = None) -> dict[str, bool]:
    """
    Ensure every collection this project owns exists, creating only the
    missing ones. Never deletes or alters an existing collection, and only
    ever touches names that belong to this project (the configured prefix).

    After creating a collection it is verified through gRPC (an
    ``aggregate`` count call goes over the gRPC channel), so a collection
    that was created but is not actually queryable surfaces immediately
    rather than at first use.

    Returns ``{collection_name: created_now}`` so the caller can log what
    changed. Raises if Weaviate is unreachable (no silent fallback).
    """
    active = settings or get_settings()
    client = get_client(active)
    created: dict[str, bool] = {}
    for namespace, name in owned_collection_names(active).items():
        was_created = _create_collection_if_missing(client, name)
        created[name] = was_created
        # Verify over gRPC: aggregate.over_all() uses the gRPC channel, so
        # this both confirms the collection is live and that gRPC works.
        collection = client.collections.get(name)
        with _lock:
            _collections[namespace] = collection
        try:
            agg = collection.aggregate.over_all(total_count=True)
            verified = int(agg.total_count or 0)
            logger.info(
                "[EMAIL-WEAVIATE] ensure_schema collection=%s created_now=%s "
                "grpc_verified_objects=%d",
                name, was_created, verified,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[EMAIL-WEAVIATE] ensure_schema gRPC verify FAILED collection=%s: %s",
                name, exc,
            )
            raise
    return created


def ensure_schema_ready(
    settings: Settings | None = None,
    attempts: int = 5,
    base_delay: float = 0.5,
) -> dict[str, dict]:
    """
    Create every owned collection if missing, then BLOCK until each one is
    actually queryable over gRPC (or until `attempts` are exhausted).

    This is the pre-start readiness gate for Weaviate. It exists because the
    bank's Weaviate deployment can momentarily report a class as present over
    REST while the gRPC query engine still answers "could not find class ...
    in schema" (a REST/gRPC schema desync). ``ensure_schema`` fails hard on
    that transient state; this function instead RETRIES the gRPC verify with
    exponential backoff, dropping the cached collection handle between tries
    so each attempt re-resolves the class, giving the server time to converge
    before the pipeline/backend starts touching vectors.

    Never deletes or recreates an existing collection (create-if-missing
    only). Returns ``{collection_name: {"ready": bool, "objects": int,
    "created_now": bool, "attempts": int}}`` so the caller can log/decide.
    Does NOT raise on a collection that stays unresolvable — readiness is
    reported per-collection so a single flaky class never crashes startup;
    the caller logs it and lets the background monitor keep retrying.
    """
    import time

    active = settings or get_settings()
    client = get_client(active)
    report: dict[str, dict] = {}

    for namespace, name in owned_collection_names(active).items():
        created_now = _create_collection_if_missing(client, name)
        ready = False
        objects = 0
        used = 0
        for attempt in range(1, max(1, attempts) + 1):
            used = attempt
            # Drop any cached handle so this attempt re-resolves the class
            # from the server (a stale handle would keep hitting the desync).
            _drop_cached_handle(namespace)
            collection = client.collections.get(name)
            with _lock:
                _collections[namespace] = collection
            try:
                agg = collection.aggregate.over_all(total_count=True)
                objects = int(agg.total_count or 0)
                ready = True
                break
            except Exception as exc:  # noqa: BLE001
                transient = _is_missing_class_error(exc)
                logger.warning(
                    "[EMAIL-WEAVIATE] ensure_schema_ready collection=%s attempt=%d/%d "
                    "not-yet-queryable (%s): %s",
                    name, attempt, attempts,
                    "transient desync" if transient else "error", exc,
                )
                if attempt < attempts:
                    time.sleep(base_delay * (2 ** (attempt - 1)))

        report[name] = {
            "ready": ready,
            "objects": objects,
            "created_now": created_now,
            "attempts": used,
        }
        if ready:
            logger.info(
                "[EMAIL-WEAVIATE] ensure_schema_ready collection=%s READY "
                "created_now=%s objects=%d attempts=%d",
                name, created_now, objects, used,
            )
        else:
            logger.error(
                "[EMAIL-WEAVIATE] ensure_schema_ready collection=%s NOT READY after "
                "%d attempt(s) — leaving it to the background monitor to retry",
                name, used,
            )
    return report


def _is_missing_class_error(exc: Exception) -> bool:
    """
    True iff `exc` is Weaviate's "could not find class ... in schema" gRPC
    error (StatusCode.UNKNOWN). This happens when the REST schema view and
    the gRPC server disagree: ``collections.exists()`` reports the class
    present (so a handle was cached by ``_get_collection``) but a later gRPC
    op on that cached handle cannot resolve the class — e.g. the collection
    was dropped/recreated out from under a cached handle. Matched by message
    text because the client raises a generic error type for this case.
    """
    return "could not find class" in str(exc).lower()


def _drop_cached_handle(namespace: str) -> None:
    """
    Forget the cached collection handle for `namespace` so the next
    ``_get_collection`` call re-resolves it (re-running create-if-missing).
    Used to recover from a stale handle whose class vanished server-side.
    """
    with _lock:
        _collections.pop(namespace, None)


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Return a 1-D float32 L2-normalized copy of `vec`."""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr


# ---------------------------------------------------------------------------
# Public API — same contract as the old faiss_vector_store module, except
# ids are UUID strings (see agent_email/storage_ids.py) rather than int64.
# ---------------------------------------------------------------------------

def upsert_vectors(
    namespace: str,
    items: list[tuple[str, np.ndarray]],
    settings: Settings | None = None,
) -> int:
    """
    Insert or replace vectors for `namespace`.

    `items` is a list of ``(weaviate_id, vector)`` pairs, where
    `weaviate_id` is a UUID string (see ``storage_ids.stable_object_id``).
    Weaviate's ``insert_many`` with an explicit UUID performs an upsert
    (replace-if-exists), so re-running the pipeline for the same
    cluster/meeting id overwrites cleanly rather than duplicating.

    Returns the number of vectors written.
    """
    if not items:
        return 0

    collection = _get_collection(namespace, settings)
    objects = [
        DataObject(
            uuid=weaviate_id,
            properties={_KEY_PROPERTY: weaviate_id},
            vector=_normalize(vec).tolist(),
        )
        for weaviate_id, vec in items
    ]
    result = collection.data.insert_many(objects)
    if result.has_errors:
        raise RuntimeError(
            f"[EMAIL-WEAVIATE] upsert_vectors namespace={namespace} "
            f"failed for {len(result.errors)} object(s): {result.errors}"
        )
    logger.info(
        "[EMAIL-WEAVIATE] upserted namespace=%s vectors=%d", namespace, len(items),
    )
    return len(items)


def get_vector(
    namespace: str,
    weaviate_id: str,
    settings: Settings | None = None,
) -> np.ndarray | None:
    """
    Return the stored (L2-normalized) vector for `weaviate_id`, or None if
    the namespace/collection or id is not present.
    """
    if not weaviate_id:
        return None
    collection = _get_collection(namespace, settings)
    try:
        obj = collection.query.fetch_object_by_id(weaviate_id, include_vector=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[EMAIL-WEAVIATE] get_vector namespace=%s id=%s failed: %s",
            namespace, weaviate_id, exc,
        )
        return None
    if obj is None:
        return None
    vec = obj.vector.get("default") if isinstance(obj.vector, dict) else obj.vector
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32)


def get_vectors(
    namespace: str,
    weaviate_ids: list[str],
    settings: Settings | None = None,
) -> dict[str, np.ndarray]:
    """
    Batch variant of :func:`get_vector`. Returns a
    ``{weaviate_id: vector}`` dict containing only the ids that were
    found in the collection.
    """
    out: dict[str, np.ndarray] = {}
    ids = [wid for wid in weaviate_ids if wid]
    if not ids:
        return out

    collection = _get_collection(namespace, settings)
    try:
        response = collection.query.fetch_objects(
            filters=Filter.by_id().contains_any(ids),
            include_vector=True,
            limit=len(ids),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[EMAIL-WEAVIATE] get_vectors namespace=%s ids=%d failed: %s",
            namespace, len(ids), exc,
        )
        return out

    for obj in response.objects:
        vec = obj.vector.get("default") if isinstance(obj.vector, dict) else obj.vector
        if vec is None:
            continue
        out[str(obj.uuid)] = np.asarray(vec, dtype=np.float32)
    return out


def search_vectors(
    namespace: str,
    query_vec: np.ndarray,
    limit: int = 12,
    settings: Settings | None = None,
) -> list[tuple[str, float]]:
    """
    Vector (near_vector) search over `namespace`.

    Returns up to `limit` ``(weaviate_id, score)`` pairs sorted best-first,
    where ``score`` is cosine similarity in [0, 1] (derived from Weaviate's
    cosine distance as ``1 - distance``). The query vector is L2-normalized
    before the search so the geometry matches how vectors were stored.

    This is the search entry point the UI / meeting-matching path uses:
    because only ACTIVE clusters are mirrored into Weaviate, a search here
    can only ever return live, one-per-meeting clusters — a superseded
    (historical) cluster is simply not in the index to be found. The caller
    hydrates the returned ids back to full rows from PostgreSQL/Redis.

    Returns ``[]`` for an empty/invalid query vector. Raises if Weaviate is
    unreachable (no fallback), same contract as the rest of this module.
    """
    arr = _normalize(query_vec)
    if arr.size == 0 or float(np.linalg.norm(arr)) == 0.0:
        return []
    collection = _get_collection(namespace, settings)
    try:
        response = collection.query.near_vector(
            near_vector=arr.tolist(),
            limit=max(1, int(limit)),
            return_metadata=MetadataQuery(distance=True),
        )
    except Exception as exc:  # noqa: BLE001
        # REST exists()/gRPC schema desync: the handle we cached points at a
        # class the gRPC server can no longer resolve ("could not find class
        # ... in schema"). Drop the stale handle so the next call re-resolves
        # (and re-creates if genuinely missing), and treat THIS search as
        # "nothing matched" rather than a hard failure — the collection has
        # no objects to return in this state anyway. Any other error is a
        # real problem and must propagate (no silent fallback).
        if _is_missing_class_error(exc):
            logger.warning(
                "[EMAIL-WEAVIATE] search namespace=%s: collection unresolvable "
                "(%s); dropping stale handle and returning no hits",
                namespace, exc,
            )
            _drop_cached_handle(namespace)
            return []
        raise
    out: list[tuple[str, float]] = []
    for obj in response.objects:
        distance = getattr(obj.metadata, "distance", None)
        score = (1.0 - float(distance)) if distance is not None else 0.0
        out.append((str(obj.uuid), score))
    logger.info(
        "[EMAIL-WEAVIATE] search namespace=%s limit=%d hits=%d",
        namespace, limit, len(out),
    )
    return out


def remove_vectors(
    namespace: str,
    weaviate_ids: list[str],
    settings: Settings | None = None,
) -> int:
    """
    Remove the given ids from `namespace`. Returns the number of ids
    requested for removal (Weaviate's batch delete does not report an
    exact removed count; missing ids are ignored, same semantics as the
    old FAISS ``remove_ids``). No-op (returns 0) when `weaviate_ids` is
    empty.
    """
    ids = [wid for wid in weaviate_ids if wid]
    if not ids:
        return 0
    collection = _get_collection(namespace, settings)
    collection.data.delete_many(where=Filter.by_id().contains_any(ids))
    logger.info(
        "[EMAIL-WEAVIATE] removed namespace=%s requested=%d", namespace, len(ids),
    )
    return len(ids)


def clear_namespace_objects(namespace: str, settings: Settings | None = None) -> int:
    """
    Delete every OBJECT in this project's collection for `namespace`, while
    KEEPING the collection itself (schema) intact.

    This is the scoped-clear primitive used at the start of a pipeline run
    to wipe the rebuildable vector copy before it is repopulated from
    PostgreSQL. It is strictly bounded to this project's own data:

      * It only ever targets ``namespace`` in :data:`OWNED_NAMESPACES`; any
        other value raises, so a shared Weaviate server's other collections
        can never be touched.
      * It deletes objects, never the collection, so no concurrently
        running reader sees a missing class and the schema survives.
      * It never touches PostgreSQL or any non-Weaviate store.

    Returns the number of objects deleted (0 when the collection is absent
    or already empty). Raises if Weaviate is unreachable (no fallback), the
    same contract as the rest of this module.
    """
    if namespace not in OWNED_NAMESPACES:
        raise ValueError(
            f"clear_namespace_objects refuses namespace {namespace!r}: not one of "
            f"this project's owned namespaces {OWNED_NAMESPACES}"
        )
    active = settings or get_settings()
    client = get_client(active)
    name = _collection_name(namespace, active)
    # Nothing to clear if the collection does not exist yet. Do NOT create
    # it here (that is `ensure_schema`'s job) — a clear of an absent
    # collection is a no-op.
    if not client.collections.exists(name):
        return 0
    before = object_count(namespace, active)
    # `object_count` returns -1 when the class exists per the schema check
    # but cannot be resolved by the server (half-created / desynced schema).
    # There is nothing to delete in that state, and delete_many would raise
    # the same "could not find class" error, so treat it as a no-op.
    if before < 0:
        logger.warning(
            "[EMAIL-WEAVIATE] clear_namespace_objects: collection %s "
            "(namespace=%s) is present but unresolvable; nothing to clear",
            name, namespace,
        )
        return 0
    collection = _get_collection(namespace, active)
    # Delete all objects without dropping the collection. We must NOT match
    # on `natural_key`: although it is declared TEXT, every value this module
    # writes is the object's weaviate_id (a UUID string), so Weaviate infers
    # the property as a `uuid` type at query time — and the `Like` operator
    # is rejected on uuid/uuid[] props ("operator \"Like\" cannot be used on
    # uuid/uuid[] props"). Match on the always-present internal creation
    # timestamp instead, which is type-agnostic and selects every object.
    collection.data.delete_many(
        where=Filter.by_creation_time().greater_than(_EPOCH_START)
    )
    logger.info(
        "[EMAIL-WEAVIATE] cleared namespace=%s objects=%d (collection kept)",
        namespace, max(before, 0),
    )
    return max(before, 0)


def reset_namespace(namespace: str, settings: Settings | None = None) -> None:
    """
    DISABLED. Deleting a Weaviate collection is destructive and the server
    may be shared with other projects, so no code path may delete a
    collection as a side effect. Weaviate is a rebuildable copy of
    PostgreSQL; to drop/rebuild use the explicit, confirmation-gated admin
    script (``scripts/weaviate_admin.py``) which only accepts this
    project's own collection names.

    Kept as a named function so any accidental caller fails loudly instead
    of silently wiping data.
    """
    raise RuntimeError(
        "reset_namespace is disabled: deleting a Weaviate collection is only "
        "allowed via scripts/weaviate_admin.py --delete <namespace> --confirm. "
        "Weaviate is rebuilt from PostgreSQL (the system of record), never reset "
        "from a normal code path."
    )


def _admin_delete_collection(namespace: str, settings: Settings | None = None) -> bool:
    """
    GATED collection delete — intended to be called ONLY from the explicit
    admin script (``scripts/weaviate_admin.py``), never from the app.

    Refuses to delete anything that is not one of this project's own
    collections (guards a shared Weaviate server). Returns True if a
    collection was deleted, False if there was nothing to delete.
    """
    active = settings or get_settings()
    name = _collection_name(namespace, active)
    if not is_owned_collection(name, active):
        raise RuntimeError(
            f"refusing to delete collection {name!r}: not one of this project's "
            f"collections ({sorted(owned_collection_names(active).values())})"
        )
    client = get_client(active)
    with _lock:
        _collections.pop(namespace, None)
        if client.collections.exists(name):
            client.collections.delete(name)
            logger.warning(
                "[EMAIL-WEAVIATE][ADMIN] deleted collection %s (namespace=%s)",
                name, namespace,
            )
            return True
    logger.info("[EMAIL-WEAVIATE][ADMIN] collection %s did not exist, nothing to delete", name)
    return False


def count(namespace: str, settings: Settings | None = None) -> int:
    """Return the number of vectors currently stored in `namespace`."""
    collection = _get_collection(namespace, settings)
    agg = collection.aggregate.over_all(total_count=True)
    return int(agg.total_count or 0)


def collection_exists(namespace: str, settings: Settings | None = None) -> bool:
    """True iff this project's collection for `namespace` exists on Weaviate."""
    active = settings or get_settings()
    client = get_client(active)
    return bool(client.collections.exists(_collection_name(namespace, active)))


# Sentinel return values for object_count() / startup_report(), kept as
# named constants so callers (self-heal, startup report, health check) can
# tell the three NON-count states apart instead of lumping them together:
#   COUNT_MISSING      (-1): the collection genuinely does not exist (REST
#                            exists() == False). Rebuild-from-PostgreSQL is
#                            the correct response when PostgreSQL has rows.
#   COUNT_UNRESOLVABLE (-2): REST exists() == True but the gRPC aggregate
#                            cannot resolve the class ("could not find class
#                            ... in schema"). This is a TRANSIENT REST/gRPC
#                            schema desync, NOT a missing collection — the
#                            class is really there and the server converges
#                            after a moment. Rebuilding here is WRONG (it
#                            thrashes create/verify/restore); the right
#                            response is to wait/retry (see ensure_schema_ready).
COUNT_MISSING = -1
COUNT_UNRESOLVABLE = -2


def object_count(namespace: str, settings: Settings | None = None) -> int:
    """
    Object count for `namespace`, or a sentinel for the non-count states:

      * ``COUNT_MISSING`` (-1)      — the collection genuinely does not exist.
      * ``COUNT_UNRESOLVABLE`` (-2) — REST reports it present but the gRPC
        aggregate cannot resolve it yet (transient REST/gRPC schema desync).

    Distinguishing -2 from -1 is what stops the self-heal from mistaking a
    momentary desync for an empty/missing collection and needlessly
    rebuilding it (the regression this fixes). Unlike :func:`count`, this
    does not create the collection as a side effect — safe for the startup
    report / health check.
    """
    active = settings or get_settings()
    client = get_client(active)
    name = _collection_name(namespace, active)
    if not client.collections.exists(name):
        return COUNT_MISSING
    # `collections.exists()` and the gRPC aggregate can disagree when the
    # schema is in a half-created / desynced state (the REST schema check
    # reports the class present, but the server can't resolve it for the
    # aggregate — "could not find class ... in schema"). Report that as the
    # distinct UNRESOLVABLE sentinel (not MISSING) so callers wait/retry
    # instead of rebuilding a collection that is actually present. Any OTHER
    # aggregate error is also non-fatal here (-2) rather than crashing the
    # caller — the readiness gate decides whether it is transient.
    try:
        agg = client.collections.get(name).aggregate.over_all(total_count=True)
    except Exception as exc:  # noqa: BLE001
        if _is_missing_class_error(exc):
            logger.warning(
                "[EMAIL-WEAVIATE] object_count: %s (namespace=%s) is REST-present "
                "but gRPC-unresolvable (transient schema desync), reporting "
                "UNRESOLVABLE: %s",
                name, namespace, exc,
            )
        else:
            logger.warning(
                "[EMAIL-WEAVIATE] object_count: aggregate failed for %s "
                "(namespace=%s), reporting UNRESOLVABLE: %s",
                name, namespace, exc,
            )
        return COUNT_UNRESOLVABLE
    return int(agg.total_count or 0)


def cluster_node_count(settings: Settings | None = None) -> int:
    """
    Number of nodes in the Weaviate cluster, for the startup log. Returns
    0 if the server does not expose node info to this client.
    """
    active = settings or get_settings()
    client = get_client(active)
    try:
        return len(client.cluster.nodes())
    except Exception:  # noqa: BLE001
        return 0


def startup_report(settings: Settings | None = None) -> dict:
    """
    Build (and log) the startup Weaviate report WITHOUT creating anything:
    REST host, gRPC host, node count, and object count per owned
    collection. Host names are logged here for operators; the HTTP /health
    endpoint must NOT expose them (see the health route).

    Returns a dict the health endpoint can reuse (connection + per-
    collection existence/counts), minus host names.
    """
    active = settings or get_settings()
    nodes = cluster_node_count(active)
    per_collection: dict[str, dict] = {}
    for namespace, name in owned_collection_names(active).items():
        cnt = object_count(namespace, active)
        # Three distinct states, not the old binary exists/absent:
        #   cnt >= 0                 -> present and queryable (objects=cnt)
        #   cnt == COUNT_UNRESOLVABLE-> present but gRPC can't resolve it yet
        #                               (transient desync; exists=True so the
        #                               startup gate waits rather than rebuilds)
        #   cnt == COUNT_MISSING     -> genuinely absent
        unresolvable = cnt == COUNT_UNRESOLVABLE
        exists = cnt >= 0 or unresolvable
        per_collection[namespace] = {
            "collection": name,
            "exists": exists,
            "unresolvable": unresolvable,
            "objects": max(cnt, 0),
        }
        logger.info(
            "[EMAIL-WEAVIATE][STARTUP] collection=%s exists=%s unresolvable=%s objects=%d",
            name, exists, unresolvable, max(cnt, 0),
        )
    logger.info(
        "[EMAIL-WEAVIATE][STARTUP] rest_host=%s grpc_host=%s nodes=%d",
        active.weaviate_http_host, active.weaviate_grpc_host, nodes,
    )
    return {"nodes": nodes, "collections": per_collection}


def clear_cache() -> None:
    """
    Drop the in-process collection-handle cache (does NOT delete anything
    on the Weaviate instance).

    Called in two situations:
      * On client teardown (registered as a ``core.weaviate.client`` on-close
        callback below) so a close+reconnect never leaves a cached handle
        bound to a dead connection — reusing such a stale handle is what
        surfaces the gRPC "could not find class ... in schema" desync.
      * By tests that repoint Weaviate connection settings at a different
        instance, so the next access re-fetches handles from the new target.
    """
    with _lock:
        _collections.clear()


# Register the handle-cache reset to run whenever the shared Weaviate client
# is closed, so a close+reconnect (e.g. the one-shot run_pipeline.py process,
# or the app shutdown hook) can never leave `_collections` holding handles
# bound to the previous, now-closed connection.
try:
    from core.weaviate.client import register_on_close as _register_on_close

    _register_on_close(clear_cache)
except Exception:  # noqa: BLE001
    # If the client module changes shape, never let registration break import
    # of the vector store; the only cost is losing the automatic cache reset.
    pass
