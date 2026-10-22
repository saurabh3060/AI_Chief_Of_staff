"""
Unit tests for the email agent's Weaviate vector store and the
deterministic weaviate_id helper (``stable_object_id``).

These run fully offline: a fake in-memory client stands in for the real
`weaviate.WeaviateClient`, implementing just the subset of the v4 API
surface that ``agent_email/weaviate_vector_store.py`` calls (collections
create/get/exists/delete, data insert_many/delete_many, query
fetch_object_by_id/fetch_objects, aggregate.over_all). No network access,
no real Weaviate instance required.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from agent_email import weaviate_vector_store as wvs
from agent_email.weaviate_vector_store import NAMESPACE_CLUSTERS, NAMESPACE_MEETINGS
from agent_email.storage_ids import stable_object_id


# ---------------------------------------------------------------------------
# Fake in-memory Weaviate client (just enough of the v4 API surface).
# ---------------------------------------------------------------------------

class _FakeObject:
    def __init__(self, uuid, properties, vector):
        self.uuid = uuid
        self.properties = properties
        self.vector = {"default": vector}


class _FakeInsertManyResult:
    has_errors = False
    errors = {}


class _FakeCollection:
    def __init__(self):
        self._objects: dict[str, _FakeObject] = {}

    @property
    def data(self):
        return SimpleNamespace(
            insert_many=self._insert_many,
            delete_many=self._delete_many,
        )

    @property
    def query(self):
        return SimpleNamespace(
            fetch_object_by_id=self._fetch_object_by_id,
            fetch_objects=self._fetch_objects,
            near_vector=self._near_vector,
        )

    def _near_vector(self, near_vector, limit=10, return_metadata=None):
        # Rank stored objects by cosine distance to the query vector.
        q = np.asarray(near_vector, dtype=np.float32)
        scored = []
        for obj in self._objects.values():
            vec = obj.vector["default"] if isinstance(obj.vector, dict) else obj.vector
            v = np.asarray(vec, dtype=np.float32)
            denom = float(np.linalg.norm(q) * np.linalg.norm(v)) or 1.0
            cosine = float(np.dot(q, v) / denom)
            distance = 1.0 - cosine
            scored.append((distance, obj))
        scored.sort(key=lambda t: t[0])
        objects = [
            SimpleNamespace(uuid=obj.uuid, metadata=SimpleNamespace(distance=dist))
            for dist, obj in scored[: max(1, int(limit))]
        ]
        return SimpleNamespace(objects=objects)

    @property
    def aggregate(self):
        return SimpleNamespace(over_all=self._over_all)

    def _insert_many(self, objects):
        for obj in objects:
            self._objects[obj.uuid] = _FakeObject(obj.uuid, obj.properties, obj.vector)
        return _FakeInsertManyResult()

    def _delete_many(self, where):
        # Two filter shapes are used by the store:
        #   * by_id().contains_any(ids)   -> _FakeFilterById(ids)
        #   * by_property(key).like("*")  -> _FakeFilterLike (match-all clear)
        if isinstance(where, _FakeFilterLike):
            self._objects.clear()
            return
        for wid in where.ids:
            self._objects.pop(wid, None)

    def _fetch_object_by_id(self, uuid, include_vector=True):
        obj = self._objects.get(uuid)
        return obj

    def _fetch_objects(self, filters, include_vector=True, limit=None):
        found = [self._objects[wid] for wid in filters.ids if wid in self._objects]
        return SimpleNamespace(objects=found)

    def _over_all(self, total_count=True):
        return SimpleNamespace(total_count=len(self._objects))


class _FakeFilterById:
    """Stand-in for weaviate.classes.query.Filter.by_id().contains_any(ids)."""
    def __init__(self, ids):
        self.ids = ids


class _FakeFilterLike:
    """Stand-in for Filter.by_property(name).like(pattern) — match-all clear."""
    def __init__(self, prop, pattern):
        self.prop = prop
        self.pattern = pattern


class _FakeCollections:
    def __init__(self):
        self._store: dict[str, _FakeCollection] = {}

    def exists(self, name):
        return name in self._store

    def create(self, name, *args, **kwargs):
        self._store[name] = _FakeCollection()

    def get(self, name):
        return self._store[name]

    def delete(self, name):
        self._store.pop(name, None)


class _FakeCluster:
    """Stand-in for client.cluster.nodes()."""
    def nodes(self):
        return [SimpleNamespace(name="node1")]


class _FakeWeaviateClient:
    def __init__(self):
        self.collections = _FakeCollections()
        self.cluster = _FakeCluster()

    def is_ready(self):
        return True

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _fake_weaviate(monkeypatch):
    """
    Patch core.weaviate.client.get_client (as imported into
    agent_email.weaviate_vector_store) to return a fresh fake client per
    test, and patch Filter.by_id().contains_any(...) usage
    inside weaviate_vector_store to the fake equivalent. Also clears the
    module's in-process collection-handle cache before and after.
    """
    fake_client = _FakeWeaviateClient()
    monkeypatch.setattr(
        "agent_email.weaviate_vector_store.get_client", lambda settings=None: fake_client
    )

    class _FakeFilter:
        @staticmethod
        def by_id():
            return SimpleNamespace(contains_any=lambda ids: _FakeFilterById(ids))

        @staticmethod
        def by_property(name, length=False):
            return SimpleNamespace(like=lambda pattern: _FakeFilterLike(name, pattern))

    monkeypatch.setattr("agent_email.weaviate_vector_store.Filter", _FakeFilter)

    wvs.clear_cache()
    yield
    wvs.clear_cache()


def _settings():
    """Minimal settings stub exposing only what the vector store reads."""
    return SimpleNamespace(weaviate_collection_name=lambda ns: f"Test{ns.capitalize()}")


# ---------------------------------------------------------------------------
# stable_object_id
# ---------------------------------------------------------------------------

def test_stable_object_id_is_deterministic_and_valid_uuid():
    a = stable_object_id("cluster_x")
    b = stable_object_id("cluster_x")
    assert a == b
    import uuid
    uuid.UUID(a)  # does not raise
    assert stable_object_id("cluster_x") != stable_object_id("cluster_y")
    # Namespaced keys never collide across meeting vs history.
    assert stable_object_id("meeting:m1") != stable_object_id("history:m1")


# ---------------------------------------------------------------------------
# upsert / get roundtrip
# ---------------------------------------------------------------------------

def test_upsert_then_get_roundtrip():
    settings = _settings()
    wid = stable_object_id("cluster_a")
    vec = np.array([3.0, 4.0], dtype=np.float32)  # will be L2-normalized

    written = wvs.upsert_vectors(NAMESPACE_CLUSTERS, [(wid, vec)], settings=settings)
    assert written == 1
    assert wvs.count(NAMESPACE_CLUSTERS, settings=settings) == 1

    got = wvs.get_vector(NAMESPACE_CLUSTERS, wid, settings=settings)
    assert got is not None
    # Stored normalized: [3,4] / 5 = [0.6, 0.8]
    np.testing.assert_allclose(got, [0.6, 0.8], rtol=1e-5, atol=1e-6)


def test_missing_id_returns_none():
    settings = _settings()
    assert wvs.get_vector(NAMESPACE_CLUSTERS, stable_object_id("missing"), settings=settings) is None
    assert wvs.get_vectors(NAMESPACE_CLUSTERS, [stable_object_id("a"), stable_object_id("b")], settings=settings) == {}


def test_upsert_is_idempotent_replaces_not_duplicates():
    settings = _settings()
    wid = stable_object_id("cluster_a")
    wvs.upsert_vectors(NAMESPACE_CLUSTERS, [(wid, np.array([1.0, 0.0], np.float32))], settings=settings)
    wvs.upsert_vectors(NAMESPACE_CLUSTERS, [(wid, np.array([0.0, 1.0], np.float32))], settings=settings)
    assert wvs.count(NAMESPACE_CLUSTERS, settings=settings) == 1
    got = wvs.get_vector(NAMESPACE_CLUSTERS, wid, settings=settings)
    np.testing.assert_allclose(got, [0.0, 1.0], rtol=1e-5, atol=1e-6)


def test_get_vectors_batch():
    settings = _settings()
    items = [
        (stable_object_id("c1"), np.array([1.0, 0.0], np.float32)),
        (stable_object_id("c2"), np.array([0.0, 1.0], np.float32)),
    ]
    wvs.upsert_vectors(NAMESPACE_CLUSTERS, items, settings=settings)
    ids = [wid for wid, _ in items]
    out = wvs.get_vectors(NAMESPACE_CLUSTERS, ids + [stable_object_id("missing")], settings=settings)
    assert set(out.keys()) == set(ids)


def test_remove_vectors():
    settings = _settings()
    wid = stable_object_id("c1")
    wvs.upsert_vectors(NAMESPACE_CLUSTERS, [(wid, np.array([1.0, 0.0], np.float32))], settings=settings)
    removed = wvs.remove_vectors(NAMESPACE_CLUSTERS, [wid], settings=settings)
    assert removed == 1
    assert wvs.get_vector(NAMESPACE_CLUSTERS, wid, settings=settings) is None


def test_reset_namespace_is_disabled():
    """
    Deleting a collection as a side effect is forbidden now: reset_namespace
    must raise rather than wipe data. Deletion is only possible via the
    gated admin helper.
    """
    settings = _settings()
    wid = stable_object_id("c1")
    wvs.upsert_vectors(NAMESPACE_CLUSTERS, [(wid, np.array([1.0, 0.0], np.float32))], settings=settings)
    with pytest.raises(RuntimeError):
        wvs.reset_namespace(NAMESPACE_CLUSTERS, settings=settings)
    # Data is untouched.
    assert wvs.count(NAMESPACE_CLUSTERS, settings=settings) == 1


def test_admin_delete_only_owned_collection():
    """The gated admin delete removes our own collection and rejects others."""
    settings = _settings()
    wid = stable_object_id("c1")
    wvs.upsert_vectors(NAMESPACE_CLUSTERS, [(wid, np.array([1.0, 0.0], np.float32))], settings=settings)
    deleted = wvs._admin_delete_collection(NAMESPACE_CLUSTERS, settings=settings)
    assert deleted is True
    assert wvs.object_count(NAMESPACE_CLUSTERS, settings=settings) == -1


def test_ensure_schema_creates_owned_collections():
    settings = _settings()
    created = wvs.ensure_schema(settings=settings)
    names = set(wvs.owned_collection_names(settings).values())
    assert set(created.keys()) == names
    # Both collections now exist and verify over (fake) gRPC with 0 objects.
    for namespace in wvs.OWNED_NAMESPACES:
        assert wvs.object_count(namespace, settings=settings) == 0


def test_namespaces_are_isolated():
    settings = _settings()
    wid = stable_object_id("shared")
    wvs.upsert_vectors(NAMESPACE_CLUSTERS, [(wid, np.array([1.0, 0.0], np.float32))], settings=settings)
    # Same id in a different namespace must not be visible.
    assert wvs.get_vector(NAMESPACE_MEETINGS, wid, settings=settings) is None
    assert wvs.count(NAMESPACE_CLUSTERS, settings=settings) == 1
    assert wvs.count(NAMESPACE_MEETINGS, settings=settings) == 0


def test_upsert_vectors_empty_list_is_noop():
    settings = _settings()
    assert wvs.upsert_vectors(NAMESPACE_CLUSTERS, [], settings=settings) == 0


def test_remove_vectors_empty_list_is_noop():
    settings = _settings()
    assert wvs.remove_vectors(NAMESPACE_CLUSTERS, [], settings=settings) == 0


# ---------------------------------------------------------------------------
# clear_namespace_objects — scoped clear that keeps the collection
# ---------------------------------------------------------------------------

def test_clear_namespace_objects_removes_objects_keeps_collection():
    settings = _settings()
    items = [
        (stable_object_id("c1"), np.array([1.0, 0.0], np.float32)),
        (stable_object_id("c2"), np.array([0.0, 1.0], np.float32)),
    ]
    wvs.upsert_vectors(NAMESPACE_CLUSTERS, items, settings=settings)
    assert wvs.count(NAMESPACE_CLUSTERS, settings=settings) == 2

    cleared = wvs.clear_namespace_objects(NAMESPACE_CLUSTERS, settings=settings)
    assert cleared == 2
    # Objects gone...
    assert wvs.count(NAMESPACE_CLUSTERS, settings=settings) == 0
    # ...but the collection still EXISTS (object_count == 0, not -1).
    assert wvs.object_count(NAMESPACE_CLUSTERS, settings=settings) == 0


def test_clear_namespace_objects_rejects_unowned_namespace():
    settings = _settings()
    with pytest.raises(ValueError):
        wvs.clear_namespace_objects("not-ours", settings=settings)


def test_clear_namespace_objects_absent_collection_is_noop():
    settings = _settings()
    # Collection was never created for this fresh fake client.
    assert wvs.object_count(NAMESPACE_MEETINGS, settings=settings) == -1
    assert wvs.clear_namespace_objects(NAMESPACE_MEETINGS, settings=settings) == 0


def test_clear_namespace_objects_is_isolated_per_namespace():
    settings = _settings()
    wvs.upsert_vectors(
        NAMESPACE_CLUSTERS, [(stable_object_id("c1"), np.array([1.0, 0.0], np.float32))],
        settings=settings,
    )
    wvs.upsert_vectors(
        NAMESPACE_MEETINGS, [(stable_object_id("meeting:m1"), np.array([0.0, 1.0], np.float32))],
        settings=settings,
    )
    wvs.clear_namespace_objects(NAMESPACE_CLUSTERS, settings=settings)
    # Clearing clusters must not touch the meetings namespace.
    assert wvs.count(NAMESPACE_CLUSTERS, settings=settings) == 0
    assert wvs.count(NAMESPACE_MEETINGS, settings=settings) == 1


def test_search_works_after_clear_and_repopulate():
    """Vector search must still return hits once the namespace is rebuilt."""
    settings = _settings()
    wid = stable_object_id("c1")
    wvs.upsert_vectors(NAMESPACE_CLUSTERS, [(wid, np.array([1.0, 0.0], np.float32))], settings=settings)
    wvs.clear_namespace_objects(NAMESPACE_CLUSTERS, settings=settings)
    # Re-populate (as restore_weaviate_from_postgres would) and search.
    wvs.upsert_vectors(NAMESPACE_CLUSTERS, [(wid, np.array([1.0, 0.0], np.float32))], settings=settings)
    hits = wvs.search_vectors(NAMESPACE_CLUSTERS, np.array([1.0, 0.0], np.float32), limit=5, settings=settings)
    assert [h[0] for h in hits] == [wid]
