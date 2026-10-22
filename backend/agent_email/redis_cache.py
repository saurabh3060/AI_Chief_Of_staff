"""
agent_email/redis_cache.py
==========================
Redis read-cache layer for the email agent's embedding-bearing records.
 
Role in the three-layer model
------------------------------
Redis is a **disposable read cache only**. It is never the system of
record and never the vector index — PostgreSQL is the master and Weaviate
is the rebuildable vector copy. Consequences baked into this module:
 
* **No TTL.** Keys are written with a plain ``SET`` and never expire. They
  are instead kept correct by being updated after every write and refilled
  from Weaviate/PostgreSQL at startup.
* **Never authoritative.** A cache miss (or Redis being down/disabled) just
  means the caller reads from the system of record instead. This module
  never fabricates data and never serves a value it did not read back from
  Redis — if Redis is unavailable, reads return ``None`` (miss), they do
  NOT return stale defaults.
* **No LLM / embedding work.** Refilling the cache only copies already
  persisted records; it never calls the embedding model or an LLM.
* **Namespaced.** Every key is prefixed with ``settings.redis_key_prefix``
  so a shared Redis server is safe.
 
The cached payloads are the exact serialized shapes the store layer
returns, so a cache hit is a drop-in substitute for a store read:
 
* clusters         -> list of ``EmailClusterRecord.model_dump()`` dicts
* meeting_embeddings -> ``{meeting_id: [subject, embedding_json]}``
* meeting_history  -> ``{meeting_id: {..record fields..}}``
 
Both a per-record key and a bulk "all" key are maintained for each entity;
the bulk key is what the read path uses, the per-record keys exist so a
single-record write can refresh its own entry cheaply and for targeted
lookups.
"""
 
from __future__ import annotations
 
import json
 
from core.config import Settings, get_settings
from core.logging_config import get_agent_logger
from core.redis.client import get_redis
 
logger = get_agent_logger("agent_email")
 
# Logical key suffixes (joined to settings.redis_key_prefix at use time).
_BULK_CLUSTERS = "clusters:all"
_BULK_MEETING_EMB = "meeting_embeddings:all"
_BULK_MEETING_HIST = "meeting_history:all"
 
_REC_CLUSTER = "cluster:"           # + cluster_id
_REC_MEETING_EMB = "meeting_embedding:"  # + meeting_id
_REC_MEETING_HIST = "meeting_history:"   # + meeting_id
 
# Per-meeting matched-cluster result cache. Key suffix is
# ``meeting_clusters:{meeting_id}`` and the value is the ordered list of
# ``EmailClusterRecord.model_dump()`` dicts that the meeting-match path
# resolved for that meeting (exact anchors + relevant clusters / department
# fallback, in display order). This lets the meeting list and the
# pre-meeting brief reuse ONE Weaviate/cosine search (and, when the LLM gate
# is enabled, one gate decision) per meeting instead of each recomputing it.
#
# UNLIKE the other caches this one is DERIVED (it depends on clusters +
# embeddings + history), so it is explicitly INVALIDATED — never refilled —
# whenever any of those change (see ``clear_meeting_clusters``). It is also
# the only key given a bounded TTL as a safety net so a missed invalidation
# can never pin a stale result indefinitely.
_REC_MEETING_CLUSTERS = "meeting_clusters:"  # + meeting_id
 
# Safety-net TTL (seconds) for the derived per-meeting cluster cache. Explicit
# invalidation on every pipeline write is the primary freshness mechanism;
# this bound simply guarantees staleness is self-limiting if an invalidation
# is ever missed (e.g. an out-of-band DB edit). 6 hours is well under the
# pipeline cadence while still absorbing a burst of brief refreshes.
_MEETING_CLUSTERS_TTL_SECONDS = 6 * 60 * 60
 
 
def _k(settings: Settings, suffix: str) -> str:
    return f"{settings.redis_key_prefix}{suffix}"
 
 
# ---------------------------------------------------------------------------
# Low-level helpers (every one is miss-safe: Redis down/absent -> no-op)
# ---------------------------------------------------------------------------
 
async def _get_json(settings: Settings, suffix: str):
    client = await get_redis(settings)
    if client is None:
        return None
    try:
        raw = await client.get(_k(settings, suffix))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[REDIS] GET %s failed (treating as miss): %s", suffix, exc)
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[REDIS] value for %s is not valid JSON (treating as miss)", suffix)
        return None
 
 
async def _set_json(settings: Settings, suffix: str, value) -> bool:
    """SET a key with NO TTL. Returns True on success, False on any failure."""
    client = await get_redis(settings)
    if client is None:
        return False
    try:
        await client.set(_k(settings, suffix), json.dumps(value))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[REDIS] SET %s failed (cache not updated): %s", suffix, exc)
        return False
 
 
async def _set_json_ttl(settings: Settings, suffix: str, value, ttl_seconds: int) -> bool:
    """SET a key WITH a TTL (seconds). Returns True on success, False on failure.
 
    Used only for the derived per-meeting cluster cache, where the TTL is a
    safety net on top of explicit invalidation. Miss-safe: Redis down/absent
    is a no-op that returns False.
    """
    client = await get_redis(settings)
    if client is None:
        return False
    try:
        await client.set(_k(settings, suffix), json.dumps(value), ex=ttl_seconds)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[REDIS] SET(ttl) %s failed (cache not updated): %s", suffix, exc)
        return False
 
 
async def _prune_stale_record_keys(
    settings: Settings,
    prefix_suffix: str,
    keep_ids: set[str],
) -> int:
    """
    Delete per-record keys under ``prefix_suffix`` whose id is NOT in
    ``keep_ids``, so a refresh reconciles the namespace to exactly the
    current snapshot instead of accumulating orphans from superseded ids.
 
    ``prefix_suffix`` is the logical record prefix (e.g. ``"cluster:"``);
    the id is whatever follows it after the global ``redis_key_prefix``.
    Returns the number of keys deleted. Miss-safe: Redis down/absent is a
    no-op returning 0, and any failure is swallowed (the next refresh
    reconciles again), so cache eviction can never break a write path.
 
    This evicts ONLY cache entries. Redis is a disposable, rebuildable read
    cache, so pruning never touches the system of record and never changes
    clustering, matching, ordering, or any business logic — it only stops
    the per-record namespaces from growing without bound.
    """
    client = await get_redis(settings)
    if client is None:
        return 0
    full_prefix = _k(settings, prefix_suffix)  # e.g. "rbl_email:cluster:"
    pattern = f"{full_prefix}*"
    stale: list[str] = []
    try:
        async for key in client.scan_iter(match=pattern):
            rec_id = key[len(full_prefix):]
            if rec_id not in keep_ids:
                stale.append(key)
        if stale:
            await client.delete(*stale)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[REDIS] prune stale keys (pattern=%s) failed: %s", pattern, exc
        )
        return 0
    if stale:
        logger.info(
            "[REDIS][CACHE-PRUNE] prefix=%s removed=%d kept=%d",
            prefix_suffix, len(stale), len(keep_ids),
        )
    return len(stale)
 
 
# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------
 
async def get_clusters(settings: Settings | None = None):
    """Return the cached list of cluster dicts, or None on miss."""
    active = settings or get_settings()
    return await _get_json(active, _BULK_CLUSTERS)
 
 
async def set_clusters(cluster_dicts: list[dict], settings: Settings | None = None) -> bool:
    """
    Replace the cached cluster snapshot (bulk key) and refresh each
    per-record key. No TTL. Called after a cluster write and during refill.
    """
    active = settings or get_settings()
    ok = await _set_json(active, _BULK_CLUSTERS, cluster_dicts)
    keep_ids: set[str] = set()
    for rec in cluster_dicts:
        cid = rec.get("cluster_id")
        if cid:
            await _set_json(active, f"{_REC_CLUSTER}{cid}", rec)
            keep_ids.add(cid)
    # Reconcile: drop per-record keys for cluster_ids no longer in the active
    # snapshot (superseded by a content-hash change) so the namespace cannot
    # accumulate orphans across refreshes. Cache eviction only.
    await _prune_stale_record_keys(active, _REC_CLUSTER, keep_ids)
    logger.info("[REDIS][CACHE-SET] clusters bulk=%s records=%d", ok, len(cluster_dicts))
    return ok
 
 
# ---------------------------------------------------------------------------
# Meeting embeddings
# ---------------------------------------------------------------------------
 
async def get_meeting_embeddings(settings: Settings | None = None):
    """Return cached ``{meeting_id: [subject, embedding_json]}`` or None."""
    active = settings or get_settings()
    return await _get_json(active, _BULK_MEETING_EMB)
 
 
async def set_meeting_embeddings(mapping: dict, settings: Settings | None = None) -> bool:
    active = settings or get_settings()
    ok = await _set_json(active, _BULK_MEETING_EMB, mapping)
    for meeting_id, value in mapping.items():
        await _set_json(active, f"{_REC_MEETING_EMB}{meeting_id}", value)
    # Reconcile: drop per-record keys for meeting_ids no longer present (e.g.
    # occurrence ids that churned) so the namespace stays bounded. Cache
    # eviction only — the system of record is unchanged.
    await _prune_stale_record_keys(active, _REC_MEETING_EMB, set(mapping.keys()))
    logger.info("[REDIS][CACHE-SET] meeting_embeddings bulk=%s records=%d", ok, len(mapping))
    return ok
 
 
# ---------------------------------------------------------------------------
# Meeting history
# ---------------------------------------------------------------------------
 
async def get_meeting_history(settings: Settings | None = None):
    """Return cached ``{meeting_id: {..fields..}}`` or None."""
    active = settings or get_settings()
    return await _get_json(active, _BULK_MEETING_HIST)
 
 
async def set_meeting_history(mapping: dict, settings: Settings | None = None) -> bool:
    active = settings or get_settings()
    ok = await _set_json(active, _BULK_MEETING_HIST, mapping)
    for meeting_id, value in mapping.items():
        await _set_json(active, f"{_REC_MEETING_HIST}{meeting_id}", value)
    # Reconcile: drop per-record keys for meeting_ids no longer present so the
    # namespace stays bounded across refreshes. Cache eviction only.
    await _prune_stale_record_keys(active, _REC_MEETING_HIST, set(mapping.keys()))
    logger.info("[REDIS][CACHE-SET] meeting_history bulk=%s records=%d", ok, len(mapping))
    return ok
 
 
# ---------------------------------------------------------------------------
# Per-meeting matched clusters (DERIVED cache — reused by both the meeting
# list and the pre-meeting brief; invalidated, never refilled)
# ---------------------------------------------------------------------------
 
async def get_meeting_clusters(meeting_id: str, settings: Settings | None = None):
    """
    Return the cached ordered list of MATCH entries for ``meeting_id``, or
    ``None`` on miss (also when ``meeting_id`` is empty or Redis is absent).
 
    Each match entry is ``{"cluster_id": str, "score": float, "record": dict}``
    in display order, where ``score`` is the stage-1 meeting-semantic
    similarity (0-100 cosine) and ``-1.0`` for exact-anchor / department-
    fallback rows that were not similarity-selected. A hit is a drop-in
    substitute for recomputing the meeting match (exact anchors, relevant
    clusters, department fallback, order, and per-cluster scores), so no
    Weaviate/cosine search or LLM gate call is needed.
 
    Backward compatible: a value written by the earlier format (a bare list of
    cluster-record dicts, no scores) is normalized to the match-entry shape
    with the sentinel score so old cache entries still hit cleanly.
    """
    if not meeting_id:
        return None
    active = settings or get_settings()
    value = await _get_json(active, f"{_REC_MEETING_CLUSTERS}{meeting_id}")
    if value is None:
        logger.info("[REDIS][CACHE-MISS] meeting_clusters meeting_id=%s", meeting_id)
        return None
 
    matches = _normalize_meeting_match_entries(value)
    if matches is None:
        # Unrecognized/corrupt shape -> treat as a miss so the caller recomputes.
        logger.warning(
            "[REDIS] meeting_clusters meeting_id=%s has unexpected shape (treating as miss)",
            meeting_id,
        )
        return None
 
    logger.info(
        "[REDIS][CACHE-HIT] meeting_clusters meeting_id=%s records=%d scores=%s",
        meeting_id, len(matches),
        {m["cluster_id"]: m["score"] for m in matches},
    )
    return matches
 
 
def _normalize_meeting_match_entries(value) -> list[dict] | None:
    """
    Coerce a cached meeting_clusters value into the list-of-match-entries shape
    ``[{"cluster_id", "score", "record"}, ...]``.
 
    Accepts both the current format (already match entries) and the legacy
    format (a bare list of cluster-record dicts). Returns ``None`` when the
    value is not a recognizable list so the caller can treat it as a miss.
    """
    if not isinstance(value, list):
        return None
    normalized: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        if "record" in item and "cluster_id" in item:
            # Current structured format.
            normalized.append(
                {
                    "cluster_id": item.get("cluster_id", ""),
                    "score": item.get("score", -1.0),
                    "record": item.get("record", {}),
                }
            )
        else:
            # Legacy bare cluster-record dict (no score captured).
            normalized.append(
                {
                    "cluster_id": item.get("cluster_id", ""),
                    "score": -1.0,
                    "record": item,
                }
            )
    return normalized
 
 
async def set_meeting_clusters(
    meeting_id: str,
    matches: list[dict],
    settings: Settings | None = None,
    durable: bool = False,
) -> bool:
    """
    Cache the ordered MATCH entries for ``meeting_id`` with a bounded safety-net
    TTL. ``matches`` is a list of ``{"cluster_id", "score", "record"}`` dicts in
    display order (cluster IDs + similarity scores + the full cluster record).
 
    No-op (returns False) when ``meeting_id`` is empty or Redis is absent. The
    authoritative freshness mechanism is explicit invalidation on pipeline
    writes; the TTL only bounds staleness if that is ever missed.
    """
    if not meeting_id:
        return False
    active = settings or get_settings()
    suffix = f"{_REC_MEETING_CLUSTERS}{meeting_id}"
    if durable:
        ok = await _set_json(active, suffix, matches)
    else:
        ok = await _set_json_ttl(
            active,
            suffix,
            matches,
            _MEETING_CLUSTERS_TTL_SECONDS,
        )
    logger.info(
        "[REDIS][CACHE-SET] meeting_clusters meeting_id=%s ok=%s records=%d "
        "scores=%s ttl=%ds",
        meeting_id, ok, len(matches),
        {m.get("cluster_id", ""): m.get("score", -1.0) for m in matches},
        _MEETING_CLUSTERS_TTL_SECONDS,
    )
    return ok
 
 
async def clear_meeting_clusters(settings: Settings | None = None) -> int:
    """
    Invalidate EVERY per-meeting matched-cluster cache entry.
 
    Called whenever clusters / meeting embeddings / meeting history change
    (i.e. the email/meeting pipeline reran), because those are exactly the
    inputs that decide a meeting's matched clusters. Wiping the whole
    ``meeting_clusters:*`` namespace guarantees the next meeting-list or brief
    request recomputes against the new data and can never show a stale match.
 
    Miss-safe: Redis down/absent is a logged no-op returning 0.
    Returns the number of keys deleted.
    """
    active = settings or get_settings()
    client = await get_redis(active)
    if client is None:
        return 0
    pattern = _k(active, f"{_REC_MEETING_CLUSTERS}*")
    deleted = 0
    try:
        keys = [key async for key in client.scan_iter(match=pattern)]
        if keys:
            deleted = await client.delete(*keys)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[REDIS] clear meeting_clusters (pattern=%s) failed: %s", pattern, exc
        )
        return 0
    logger.info("[REDIS][CACHE-INVALIDATE] meeting_clusters cleared keys=%d", deleted)
    return deleted
 
 
# ---------------------------------------------------------------------------
# Scoped clear (this application's keys only)
# ---------------------------------------------------------------------------
 
async def clear_all_scoped(settings: Settings | None = None) -> int:
    """
    Delete EVERY key this application owns, and ONLY those keys.
 
    Scope is enforced by the global ``settings.redis_key_prefix`` (default
    ``"rbl_email:"``): the scan pattern is ``{prefix}*``, so a shared Redis
    server's other databases/keys are never matched. This wipes all of the
    email agent's cache namespaces at once — the bulk snapshots
    (``clusters:all``, ``meeting_embeddings:all``, ``meeting_history:all``),
    their per-record keys, and the derived ``meeting_clusters:*`` entries —
    so a pipeline run can start from an empty cache and repopulate it from
    PostgreSQL (the system of record).
 
    It NEVER touches PostgreSQL or Weaviate. Miss-safe: Redis down/absent is
    a logged no-op returning 0, and any failure is swallowed (the cache is
    disposable and will be refilled). Returns the number of keys deleted.
    """
    active = settings or get_settings()
    client = await get_redis(active)
    if client is None:
        logger.info("[REDIS][CLEAR] Redis disabled — nothing to clear")
        return 0
    pattern = _k(active, "*")  # e.g. "rbl_email:*" — this app's keys only
    deleted = 0
    try:
        keys = [key async for key in client.scan_iter(match=pattern)]
        if keys:
            deleted = await client.delete(*keys)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[REDIS][CLEAR] scoped clear (pattern=%s) failed: %s", pattern, exc)
        return 0
    logger.info("[REDIS][CLEAR] scoped clear removed keys=%d (pattern=%s)", deleted, pattern)
    return deleted
 
 
async def clear_and_refill_from_source(settings: Settings | None = None) -> dict:
    """
    Scoped clear of this app's Redis keys, then reload ONLY the mappings the
    meeting-matching and pre-meeting-brief paths need from PostgreSQL.
 
    The reload is exactly :func:`refill_from_source`: it repopulates the
    bulk snapshots the read/match path consumes (``clusters:all`` — the
    active one-per-meeting clusters — plus ``meeting_embeddings:all`` and
    ``meeting_history:all``, which feed the meeting-to-cluster match
    computation). The DERIVED ``meeting_clusters:*`` cache is intentionally
    NOT pre-filled: it is recomputed lazily on first request and would
    otherwise be stale, so clearing it (done by the clear above) is correct.
 
    Returns the refill counts. Non-fatal and miss-safe throughout.
    """
    active = settings or get_settings()
    await clear_all_scoped(active)
    return await refill_from_source(active)
 
 
# ---------------------------------------------------------------------------
# Startup refill (no LLM / no embedding — copies persisted records only)
# ---------------------------------------------------------------------------
 
async def refill_from_source(settings: Settings | None = None) -> dict:
    """
    Refill every cache key from the system of record at startup.
 
    Reads the current clusters / meeting embeddings / meeting history
    straight out of PostgreSQL (whose embeddings were themselves restored
    into Weaviate earlier in startup if needed) and writes them into Redis
    with no TTL. Makes NO embedding or LLM calls — it only copies records
    that already exist.
 
    Safe no-op when Redis is disabled or unreachable: it logs and returns
    zero counts rather than raising, because the cache is disposable.
 
    Returns ``{entity: count}`` for what was written.
    """
    active = settings or get_settings()
    result = {"clusters": 0, "meeting_embeddings": 0, "meeting_history": 0}
 
    client = await get_redis(active)
    if client is None:
        logger.info("[REDIS][REFILL] Redis disabled — skipping cache refill")
        return result
 
    # Import here to avoid a circular import at module load (the store
    # facade imports this module for cache writes).
    from agent_email import postgres_store
 
    try:
        clusters = await postgres_store.read_all_cluster_records(settings=active)
        await set_clusters([r.model_dump() for r in clusters], active)
        result["clusters"] = len(clusters)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[REDIS][REFILL] clusters refill failed: %s", exc)
 
    try:
        me = await postgres_store.read_meeting_embeddings(settings=active)
        await set_meeting_embeddings({mid: list(val) for mid, val in me.items()}, active)
        result["meeting_embeddings"] = len(me)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[REDIS][REFILL] meeting_embeddings refill failed: %s", exc)
 
    try:
        mh = await postgres_store.read_meeting_history(settings=active)
        await set_meeting_history(mh, active)
        result["meeting_history"] = len(mh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[REDIS][REFILL] meeting_history refill failed: %s", exc)
 
    logger.info(
        "[REDIS][REFILL] cache refilled clusters=%d meeting_embeddings=%d meeting_history=%d",
        result["clusters"], result["meeting_embeddings"], result["meeting_history"],
    )
    return result


# ---------------------------------------------------------------------------
# Emptiness detection + self-heal (used by the background health monitor)
# ---------------------------------------------------------------------------

async def clusters_snapshot_present(settings: Settings | None = None) -> bool | None:
    """
    Report whether the bulk clusters snapshot key (``clusters:all``) exists
    in Redis.

    Returns:
      * ``True``  - the key exists (cache is populated for the cluster path).
      * ``False`` - Redis is reachable but the key is absent (cache emptied /
        flushed / never filled).
      * ``None``  - Redis is disabled or unreachable, so emptiness is
        *indeterminate* and the caller must NOT treat it as "empty" (doing
        so would trigger a pointless refill against a dead cache).

    This is the cheap probe the health monitor uses to decide whether a
    refill is warranted. ``clusters:all`` is chosen as the sentinel because
    it is the key the read/match path actually consumes and the first thing
    ``refill_from_source`` writes, so its presence is a faithful proxy for
    "the cache has been filled from the system of record".
    """
    active = settings or get_settings()
    client = await get_redis(active)
    if client is None:
        return None
    try:
        exists = await client.exists(_k(active, _BULK_CLUSTERS))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[REDIS] EXISTS %s failed (treating as indeterminate): %s",
            _BULK_CLUSTERS, exc,
        )
        return None
    return bool(exists)


async def heal_if_empty(settings: Settings | None = None) -> dict | None:
    """
    Self-heal the Redis cache when it has been emptied while the system of
    record still has data.

    Mirrors ``weaviate_sync.ensure_weaviate_consistent`` for Redis: it only
    acts when the cache is genuinely empty (the ``clusters:all`` snapshot is
    absent) AND PostgreSQL actually holds cluster rows. In that case it
    refills every cache key from the system of record via
    :func:`refill_from_source` (copies persisted records only - no embedding
    or LLM work, no re-clustering) and returns the refill counts.

    Returns ``None`` (no action taken) when:
      * Redis is disabled/unreachable (emptiness indeterminate), or
      * the cache snapshot is already present (nothing to heal), or
      * PostgreSQL has no cluster rows (genuinely empty - nothing to copy, so
        an empty cache is correct and we never fabricate data).

    Safe and non-fatal throughout: any failure is logged and swallowed,
    because Redis is a disposable cache and a missed heal just means reads
    fall through to the system of record until the next tick.
    """
    active = settings or get_settings()

    present = await clusters_snapshot_present(active)
    if present is None:
        # Redis down/disabled -> emptiness unknown; do not refill.
        return None
    if present:
        # Cache already populated -> nothing to heal.
        return None

    # Cache is empty. Only refill if the system of record actually has data,
    # so a legitimately empty system (no clusters yet) is not mistaken for a
    # lost cache.
    from agent_email import postgres_store

    try:
        pg_clusters = await postgres_store.clusters_row_count(active)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[REDIS][HEAL] could not count PostgreSQL clusters: %s", exc)
        return None

    if pg_clusters <= 0:
        logger.info(
            "[REDIS][HEAL] cache empty and PostgreSQL has no clusters -> nothing to heal"
        )
        return None

    logger.warning(
        "[REDIS][HEAL] clusters snapshot absent while PostgreSQL has %d cluster "
        "row(s) -> refilling Redis from the system of record",
        pg_clusters,
    )
    try:
        return await refill_from_source(active)
    except Exception as exc:  # noqa: BLE001
        logger.error("[REDIS][HEAL] refill_from_source failed: %s", exc)
        return None
