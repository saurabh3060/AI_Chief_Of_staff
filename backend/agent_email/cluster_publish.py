"""
agent_email/cluster_publish.py
==============================
Incremental, retry-safe publish of email clusters.

The legacy pipeline (``service.run_email_cluster_pipeline``) re-clusters the
whole fetch window and rewrites every cluster in one go. When
``EMAIL_CLUSTER_INCREMENTAL_PUBLISH_ENABLED`` is on, the pipeline instead
hands its already-scored, already-embedded candidates to this module, which:

1. resumes any incomplete staged work from a previous run FIRST,
2. works out which candidates are genuinely new (not in a published cluster,
   not already staged),
3. decides for each new email whether it joins an existing cluster or forms
   a new one, by running the UNCHANGED ``clustering.cluster_candidates`` over
   (new emails + members of the nearby existing clusters),
4. records that decision as one ``cluster_publish_staging`` row per new
   email per affected cluster (one atomic insert),
5. publishes clusters one at a time (summary -> idempotent PG upsert ->
   Weaviate/Redis delta) and deletes a cluster's staging rows only after all
   of that succeeded.

The clustering algorithm itself is not modified or re-implemented here; the
only new decision logic is bookkeeping around it (which existing cluster a
joined group maps to, and how to name a new one).

Idempotency
-----------
* New cluster ids are a hash of the sorted member message ids, so redoing the
  same assignment yields the same id (no duplicate clusters).
* Updated clusters keep their existing id.
* Cluster rows are upserted by ``cluster_id`` (``ON CONFLICT``); Weaviate
  objects use a deterministic id derived from ``cluster_id``; Redis writes
  are whole-key overwrites. Re-running any step is therefore a no-op.
* If the DB row is already current on a retry (content hash unchanged, so the
  writer skips the Weaviate/Redis mirror), the downstream sync is forced.
* The summary is reused when the stored cluster already contains every staged
  email (i.e. a previous attempt got as far as the DB write).

Failure policy
--------------
A failed cluster keeps its staging rows and has ``retry_count`` incremented.
At ``EMAIL_CLUSTER_MAX_STAGING_RETRIES`` it is no longer selected; its rows
stay (so its emails are never silently dropped or re-staged) and an ERROR
alert is logged on every run. To retry it manually after fixing the cause:
``UPDATE cluster_publish_staging SET retry_count = 0 WHERE run_id = '...'``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable

import numpy as np

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger, sanitize_log_value

from agent_email import cluster_publish_store as staging
from agent_email.cluster_publish_store import NEW_CLUSTER, UPDATED_CLUSTER, ClusterStaging, StagingRow
from agent_email.pipeline_metrics import PipelineMetricsCollector
from agent_email.schema import ClusterEmailRef, EmailCandidate, EmailClusterRecord, EmailMessage
from agent_email.semantic_match import cosine_similarity

logger = get_agent_logger("agent_email")

_SAMPLE_SIZE = 5


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

def log_event(
    event: str,
    *,
    level: int = logging.INFO,
    run_id: str = "",
    cluster_id: str = "",
    email_message_id: str = "",
    count: int | None = None,
    duration_ms: float | None = None,
    error: str = "",
    **extra: object,
) -> None:
    """One consistent line: event, run_id, cluster_id, email_message_id, count, duration_ms, error."""
    fields = [
        ("event", event),
        ("run_id", run_id),
        ("cluster_id", cluster_id),
        ("email_message_id", email_message_id),
        ("count", "" if count is None else count),
        ("duration_ms", "" if duration_ms is None else f"{duration_ms:.1f}"),
        ("error", error),
    ]
    fields += list(extra.items())
    # Every value (not just str) is stringified, stripped of CR/LF/control
    # characters and length-capped, then repr-quoted, so untrusted subjects,
    # message ids or exception text can never forge or split a log line.
    line = " ".join(f"{k}={sanitize_log_value(v, 500)!r}" for k, v in fields)
    logger.log(level, "[CLUSTER-PUBLISH] %s", line)


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _sample(ids: list[str]) -> str:
    return ",".join(ids[:_SAMPLE_SIZE])


# ---------------------------------------------------------------------------
# Assignment (new email -> cluster). Pure; reuses the clustering helpers.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Assignment:
    message_id: str
    subject: str
    cluster_id: str
    action: str
    reason: str


def new_cluster_id(message_ids: list[str]) -> str:
    """Deterministic id for a brand-new cluster: same members => same id."""
    digest = hashlib.sha256("|".join(sorted(message_ids)).encode("utf-8")).hexdigest()[:16]
    return f"cluster_{digest}"


def candidate_floor(settings: Settings) -> float:
    """Loose centroid-cosine floor used ONLY to shortlist existing clusters.

    It does not decide membership (``cluster_candidates`` does); it only keeps
    obviously unrelated clusters out of the working set. It is the loosest of
    the configured clustering thresholds minus a safety margin.
    """
    values = [
        float(getattr(settings, name, 0.0) or 0.0)
        for name in (
            "email_cluster_similarity_threshold",
            "email_cluster_residual_threshold",
            "email_cluster_series_near_miss",
            "email_cluster_cite_min_cosine",
        )
    ]
    positive = [v for v in values if v > 0.0]
    return max(0.0, (min(positive) if positive else 0.5) - 0.10)


def match_reason(
    new_idx: int,
    member_idxs: list[int],
    candidates: list[EmailCandidate],
    vectors: list[np.ndarray],
    settings: Settings,
) -> str:
    """Name the first business-context rule that links a new email to a cluster."""
    from agent_email.clustering import (
        _subject_is_cited,
        department_stem_tokens,
        quoted_nested_subjects,
        series_subject_tokens,
        stems_share_series,
    )

    new = candidates[new_idx].message
    conv = (new.conversation_id or "").strip()
    if conv and any((candidates[j].message.conversation_id or "").strip() == conv for j in member_idxs):
        return "conversation_id"

    jaccard_min = float(getattr(settings, "email_cluster_series_jaccard", 0.5) or 0.5)
    dept_tokens = department_stem_tokens(settings)
    stem = series_subject_tokens(new.subject)
    if any(
        stems_share_series(stem, series_subject_tokens(candidates[j].message.subject), jaccard_min, dept_tokens)
        for j in member_idxs
    ):
        return "subject_series"

    quotes = quoted_nested_subjects(new.body_text or "")
    for j in member_idxs:
        other = candidates[j].message
        if any(_subject_is_cited(other.subject, q) for q in quotes) or any(
            _subject_is_cited(new.subject, q) for q in quoted_nested_subjects(other.body_text or "")
        ):
            return "citation_reference"

    best = max(cosine_similarity(vectors[new_idx], vectors[j]) for j in member_idxs)
    if best >= float(settings.email_cluster_similarity_threshold):
        return f"vector_similarity={best:.3f}"
    residual = float(getattr(settings, "email_cluster_residual_threshold", 0.0) or 0.0)
    if residual > 0 and best >= residual:
        return f"residual_similarity={best:.3f}"
    return f"graph_context(best_cosine={best:.3f})"


def derive_assignments(
    groups: list[list[int]],
    n_members: int,
    member_cluster: list[str],
    candidates: list[EmailCandidate],
    vectors: list[np.ndarray],
    published_ids: set[str],
    settings: Settings,
    run_id: str = "",
) -> list[Assignment]:
    """Turn ``cluster_candidates`` groups into per-new-email assignments.

    ``candidates``/``vectors`` are the working set: the first ``n_members``
    entries are existing cluster members (``member_cluster[i]`` is their
    cluster id), the rest are new emails.

    * A group with no existing members  -> one NEW cluster for all its new emails.
    * A group with existing members     -> the new emails join the existing
      cluster they are most similar to. Existing clusters are never merged
      with each other here (that would change published ids); if a group
      spans several, the merge is suppressed and logged.
    """
    out: list[Assignment] = []
    for group in groups:
        new_idxs = [i for i in group if i >= n_members]
        if not new_idxs:
            continue
        by_cluster: dict[str, list[int]] = {}
        for i in group:
            if i < n_members:
                by_cluster.setdefault(member_cluster[i], []).append(i)

        if not by_cluster:
            msg_ids = [candidates[i].message.id for i in new_idxs]
            cid = new_cluster_id(msg_ids)
            reason = "no_existing_cluster_match" + (
                f";grouped_with_{len(new_idxs) - 1}_other_new_email(s)" if len(new_idxs) > 1 else ""
            )
            for i in new_idxs:
                out.append(Assignment(candidates[i].message.id, candidates[i].message.subject or "", cid, NEW_CLUSTER, reason))
            continue

        if len(by_cluster) > 1:
            log_event(
                "cluster_merge_suppressed", level=logging.WARNING, run_id=run_id,
                count=len(by_cluster), clusters=",".join(sorted(by_cluster)),
            )
        for i in new_idxs:
            cid = max(
                by_cluster,
                key=lambda c: max(cosine_similarity(vectors[i], vectors[j]) for j in by_cluster[c]),
            )
            action = UPDATED_CLUSTER if cid in published_ids else NEW_CLUSTER
            reason = match_reason(i, by_cluster[cid], candidates, vectors, settings)
            out.append(Assignment(candidates[i].message.id, candidates[i].message.subject or "", cid, action, reason))
    return out


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------

FetchMessage = Callable[[str], Awaitable["EmailMessage | None"]]


@dataclass
class PublishSummary:
    run_id: str
    processed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    exhausted: list[str] = field(default_factory=list)
    deferred: int = 0
    remaining_rows: int = 0


class ClusterPublisher:
    """One instance per pipeline invocation. Not thread-safe; run under the lease lock."""

    def __init__(self, settings: Settings | None = None, fetch_message: FetchMessage | None = None) -> None:
        self.settings = settings or get_settings()
        self.fetch_message = fetch_message
        self.run_id = ""
        self._pending_runs: list[str] = []
        self._capacity: int | None = None            # None = unlimited
        self._attempted: set[tuple[str, str]] = set()
        self._alerted: set[tuple[str, str]] = set()
        self._deferred: set[tuple[str, str]] = set()
        self._published: dict[str, EmailClusterRecord] | None = None
        self._dept_index = None
        self._started = time.perf_counter()
        self._synth_ok = 0       # the metrics collector overwrites, so keep running totals
        self._synth_failed = 0
        self.summary = PublishSummary(run_id="")

    # -- lifecycle ---------------------------------------------------------

    async def begin(self) -> str:
        """Pick the run id: reuse the oldest incomplete run, else mint a new one."""
        self._pending_runs = await staging.list_pending_run_ids()
        resumed = bool(self._pending_runs)
        self.run_id = self._pending_runs[0] if resumed else staging.new_run_id()
        self.summary.run_id = self.run_id
        log_event(
            "run_started", run_id=self.run_id, count=len(self._pending_runs),
            resumed=resumed, pending_runs=",".join(self._pending_runs),
        )
        return self.run_id

    async def resume_pending(self) -> None:
        """Finish incomplete staged work BEFORE any new email is considered."""
        for rid in self._pending_runs:
            log_event("resume_pending_run", run_id=rid)
            await self._process_run(rid)

    async def publish_new(self, candidates: list[EmailCandidate], vectors: list[np.ndarray]) -> None:
        """Stage the genuinely new candidates, then publish the current run."""
        await self.stage_new_emails(candidates, vectors)
        await self._process_run(self.run_id)

    async def finish(self) -> PublishSummary:
        remaining = await staging.remaining_rows()
        self.summary.remaining_rows = remaining
        self.summary.deferred = len(self._deferred - self._attempted)
        log_event(
            "run_completed", run_id=self.run_id, count=len(self.summary.processed),
            duration_ms=_ms(self._started),
            processed_clusters=len(self.summary.processed),
            deferred_clusters=self.summary.deferred,
            failed_clusters=len(self.summary.failed),
            exhausted_clusters=len(self.summary.exhausted),
            remaining_staging_rows=remaining,
        )
        return self.summary

    # -- published clusters ------------------------------------------------

    async def _published_records(self, refresh: bool = False) -> dict[str, EmailClusterRecord]:
        if self._published is None or refresh:
            from agent_email import postgres_store

            records = await postgres_store.read_all_cluster_records(settings=self.settings)
            self._published = {r.cluster_id: r for r in records}
        return self._published

    @staticmethod
    def _ref_dicts(record: EmailClusterRecord) -> list[dict]:
        try:
            refs = json.loads(record.email_refs or "[]")
        except (TypeError, ValueError):
            return []
        return [r for r in refs if isinstance(r, dict)] if isinstance(refs, list) else []

    # -- staging -----------------------------------------------------------

    async def stage_new_emails(self, candidates: list[EmailCandidate], vectors: list[np.ndarray]) -> list[Assignment]:
        from agent_email.clustering import cluster_candidates

        settings = self.settings
        run_id = self.run_id
        published = await self._published_records(refresh=True)
        staged = await staging.staged_members_by_cluster()

        known: set[str] = set()
        for rec in published.values():
            known.update(r.get("message_id", "") for r in self._ref_dicts(rec))
        for members in staged.values():
            known.update(mid for mid, _ in members)

        new_idx = [i for i, c in enumerate(candidates) if c.message.id and c.message.id not in known]
        new_ids = [candidates[i].message.id for i in new_idx]
        log_event("new_emails_received", run_id=run_id, count=len(new_ids), sample=_sample(new_ids),
                  candidates_seen=len(candidates))
        if not new_idx:
            log_event("staging_rows_created", run_id=run_id, count=0, distinct_clusters=0, distinct_emails=0)
            return []

        new_cands = [candidates[i] for i in new_idx]
        new_vecs = [vectors[i] for i in new_idx]

        # ---- shortlist nearby existing clusters + load their members -------
        member_ids_by_cluster = await self._shortlist_clusters(new_cands, new_vecs, published, staged)
        member_cands, member_vecs, member_cluster = await self._load_members(member_ids_by_cluster, published)

        work_cands = member_cands + new_cands
        work_vecs = member_vecs + new_vecs
        n_members = len(member_cands)

        groups = cluster_candidates(
            work_vecs,
            settings=settings,
            run_id=run_id,
            subjects=[c.message.subject for c in work_cands],
            bodies=[c.message.body_text or "" for c in work_cands],
            meeting_ids=[c.meeting_id for c in work_cands],
            received_ats=[c.message.received_at for c in work_cands],
            conversation_ids=[c.message.conversation_id for c in work_cands],
        )
        assignments = derive_assignments(
            groups, n_members, member_cluster, work_cands, work_vecs, set(published), settings, run_id=run_id,
        )

        for a in assignments:
            if a.action == UPDATED_CLUSTER:
                log_event("email_matched_existing_cluster", run_id=run_id, cluster_id=a.cluster_id,
                          email_message_id=a.message_id, match_reason=a.reason)
            else:
                log_event("email_assigned_new_cluster", run_id=run_id, cluster_id=a.cluster_id,
                          email_message_id=a.message_id, reason=a.reason)

        rows = [
            StagingRow(run_id, a.cluster_id, a.message_id, a.subject, a.action) for a in assignments
        ]
        inserted = await staging.insert_staging_rows(rows)
        PipelineMetricsCollector.record_clustering(
            len({r.cluster_id for r in rows}), algorithm=settings.email_cluster_algo,
        )
        log_event(
            "staging_rows_created", run_id=run_id, count=len(rows), inserted=inserted,
            distinct_clusters=len({r.cluster_id for r in rows}),
            distinct_emails=len({r.email_message_id for r in rows}),
        )
        return assignments

    async def _shortlist_clusters(
        self,
        new_cands: list[EmailCandidate],
        new_vecs: list[np.ndarray],
        published: dict[str, EmailClusterRecord],
        staged: dict[str, list[tuple[str, str | None]]],
    ) -> dict[str, list[str]]:
        """``{cluster_id: [member message ids]}`` for clusters worth comparing against."""
        from agent_email.clustering import department_stem_tokens, series_subject_tokens, stems_share_series
        from agent_email.embedding_json import json_to_vec

        settings = self.settings
        floor = candidate_floor(settings)
        jaccard_min = float(getattr(settings, "email_cluster_series_jaccard", 0.5) or 0.5)
        dept_tokens = department_stem_tokens(settings)
        new_stems = [series_subject_tokens(c.message.subject) for c in new_cands]

        # message id -> clusters, for thread (conversationId) matches.
        msg_to_cluster: dict[str, str] = {}
        members: dict[str, list[str]] = {}
        for cid, rec in published.items():
            ids = [r.get("message_id", "") for r in self._ref_dicts(rec) if r.get("message_id")]
            members[cid] = list(ids)
            for mid in ids:
                msg_to_cluster[mid] = cid
        for cid, pairs in staged.items():
            ids = [mid for mid, _ in pairs]
            members.setdefault(cid, [])
            members[cid].extend(i for i in ids if i not in members[cid])
            for mid in ids:
                msg_to_cluster.setdefault(mid, cid)

        selected: set[str] = set()

        # 1. same Graph conversation as a member
        thread_hits = await staging.message_ids_for_conversations([c.message.conversation_id for c in new_cands], settings)
        for mid in thread_hits:
            if mid in msg_to_cluster:
                selected.add(msg_to_cluster[mid])

        # 2. clusters that exist only as staged (deferred) work: no centroid, always compare
        selected.update(cid for cid in staged if cid not in published)

        # 3. published clusters by centroid cosine / series stem on member subjects
        for cid, rec in published.items():
            if cid in selected:
                continue
            centroid = json_to_vec(rec.cluster_embedding or "")
            if centroid is not None and any(cosine_similarity(v, centroid) >= floor for v in new_vecs):
                selected.add(cid)
                continue
            ref_stems = [series_subject_tokens(r.get("subject", "")) for r in self._ref_dicts(rec)]
            if any(stems_share_series(ns, rs, jaccard_min, dept_tokens) for ns in new_stems for rs in ref_stems):
                selected.add(cid)

        shortlisted = {cid: members[cid] for cid in selected if members.get(cid)}
        log_event("cluster_shortlist", run_id=self.run_id, count=len(shortlisted),
                  published_clusters=len(published), staged_clusters=len(staged))
        return shortlisted

    async def _load_members(
        self,
        member_ids_by_cluster: dict[str, list[str]],
        published: dict[str, EmailClusterRecord],
    ) -> tuple[list[EmailCandidate], list[np.ndarray], list[str]]:
        from agent_email.clustering import embed_candidates

        cands: list[EmailCandidate] = []
        owner: list[str] = []
        for cid, ids in member_ids_by_cluster.items():
            anchored = published[cid].anchored_meeting_id if cid in published else ""
            loaded, _missing = await self._load_candidates(ids, meeting_id=anchored)
            cands.extend(loaded)
            owner.extend([cid] * len(loaded))
        if not cands:
            return [], [], []
        vecs = await embed_candidates(cands, settings=self.settings)
        return cands, vecs, owner

    # -- candidate loading -------------------------------------------------

    async def _load_candidates(
        self, message_ids: list[str], meeting_id: str = "",
    ) -> tuple[list[EmailCandidate], list[str]]:
        """Rebuild EmailCandidates for stored emails. Returns (candidates, missing ids)."""
        from agent_email import postgres_store

        ids = [m for m in dict.fromkeys(message_ids) if m]
        messages, attachments = await staging.read_emails_by_ids(ids, self.settings)
        if self.fetch_message:
            for mid in [m for m in ids if m not in messages]:
                try:
                    fetched = await self.fetch_message(mid)
                except Exception as exc:  # noqa: BLE001
                    log_event("email_fetch_failed", level=logging.WARNING, run_id=self.run_id,
                              email_message_id=mid, error=repr(exc))
                    continue
                if fetched is not None:
                    messages[mid] = fetched
        missing = [m for m in ids if m not in messages]

        semantics = await postgres_store.read_email_semantics(list(messages), settings=self.settings) if messages else {}
        cands: list[EmailCandidate] = []
        for mid in ids:
            msg = messages.get(mid)
            if msg is None:
                continue
            sem = semantics.get(mid, {})
            cands.append(EmailCandidate(
                message=msg,
                attachment_text=attachments.get(mid, ""),
                meeting_id=meeting_id,
                topic_key=sem.get("topic_key", ""),
                business_objective=sem.get("business_objective", ""),
                topics=sem.get("topics", []),
                entities=sem.get("entities", []),
                actions=sem.get("actions", []),
                subtopics=sem.get("subtopics", []),
                decisions=sem.get("decisions", []),
                context=sem.get("context", ""),
            ))
        return cands, missing

    # -- run processing ----------------------------------------------------

    async def _process_run(self, run_id: str) -> None:
        s = self.settings
        affected, emails = await staging.run_counts(run_id)
        if affected == 0:
            return
        groups = await staging.list_cluster_groups(run_id)
        max_retries = int(s.email_cluster_max_staging_retries)

        exhausted = [g for g in groups if g.retry_count >= max_retries]
        eligible = [
            g for g in groups
            if g.retry_count < max_retries and (run_id, g.cluster_id) not in self._attempted
        ]

        exceeded = affected > int(s.email_cluster_max_affected_clusters) or emails > int(s.email_cluster_max_new_emails)
        if exceeded and self._capacity is None:
            self._capacity = int(s.email_cluster_max_clusters_per_run)
        selected = eligible if self._capacity is None else eligible[: max(self._capacity, 0)]
        deferred = eligible[len(selected):]
        if self._capacity is not None:
            self._capacity -= len(selected)
        self._deferred.update((run_id, g.cluster_id) for g in deferred)

        log_event(
            "threshold_check", run_id=run_id, count=affected,
            affected_clusters=affected, new_emails=emails,
            max_affected_clusters=int(s.email_cluster_max_affected_clusters),
            max_new_emails=int(s.email_cluster_max_new_emails),
            max_clusters_per_run=int(s.email_cluster_max_clusters_per_run),
            batching=exceeded or self._capacity is not None,
            clusters_selected=len(selected), clusters_deferred=len(deferred),
            clusters_exhausted=len(exhausted),
            clusters_remaining=len(deferred) + len(exhausted),
        )

        for g in exhausted:
            self._alert_exhausted(g)
        for g in selected:
            await self._process_cluster(g)

    def _alert_exhausted(self, g: ClusterStaging) -> None:
        key = (g.run_id, g.cluster_id)
        if key in self._alerted:
            return
        self._alerted.add(key)
        self.summary.exhausted.append(g.cluster_id)
        log_event(
            "cluster_retries_exhausted", level=logging.ERROR, run_id=g.run_id, cluster_id=g.cluster_id,
            count=g.row_count, retry_count=g.retry_count,
            error="max staging retries reached; staging rows kept, cluster will not be retried automatically",
        )

    async def _process_cluster(self, g: ClusterStaging) -> None:
        run_id, cluster_id = g.run_id, g.cluster_id
        self._attempted.add((run_id, cluster_id))
        started = time.perf_counter()
        rows: list[StagingRow] = []
        try:
            rows = await staging.rows_for_cluster(run_id, cluster_id)
            log_event("cluster_processing_started", run_id=run_id, cluster_id=cluster_id,
                      count=len(rows), retry_count=g.retry_count)
            record, reused = await self._build_record(run_id, cluster_id, rows)
            log_event("summary_generation", run_id=run_id, cluster_id=cluster_id,
                      summary="reused" if reused else "generated")

            await self._publish_record(record)

            deleted = await staging.delete_cluster_rows(run_id, cluster_id)
            log_event("staging_rows_deleted", run_id=run_id, cluster_id=cluster_id, count=deleted)
            self._published = self._published or {}
            self._published[cluster_id] = record
            self.summary.processed.append(cluster_id)
            log_event("cluster_processing_completed", run_id=run_id, cluster_id=cluster_id,
                      count=len(rows), duration_ms=_ms(started))
        except Exception as exc:  # noqa: BLE001 - every failure must keep the rows and count a retry
            try:
                retry = await staging.bump_retry(run_id, cluster_id)
            except Exception as bump_exc:  # noqa: BLE001
                retry = g.retry_count
                log_event("retry_bump_failed", level=logging.ERROR, run_id=run_id, cluster_id=cluster_id,
                          error=repr(bump_exc))
            ids = [r.email_message_id for r in rows]
            log_event(
                "cluster_processing_failed", level=logging.ERROR, run_id=run_id, cluster_id=cluster_id,
                email_message_id=_sample(ids), count=len(ids), duration_ms=_ms(started),
                retry_count=retry, error=repr(exc),
            )
            self.summary.failed.append(cluster_id)
            if retry >= int(self.settings.email_cluster_max_staging_retries):
                self._alert_exhausted(ClusterStaging(run_id, cluster_id, len(ids), retry, datetime.now(timezone.utc)))

    # -- building the final record ----------------------------------------

    async def _build_record(
        self, run_id: str, cluster_id: str, rows: list[StagingRow],
    ) -> tuple[EmailClusterRecord, bool]:
        from agent_email.clustering import embed_candidates, mean_embedding, normalize_subject
        from agent_email.dept_centroids import build_department_index, pick_department
        from agent_email.meeting_anchor import assign_candidates_to_meetings
        from agent_email import service
        from core.text_match import meaningful_tokens

        s = self.settings
        if not rows:
            raise RuntimeError("no staging rows found for cluster")

        published = await self._published_records()
        existing = published.get(cluster_id)
        if existing is None and any(r.cluster_action == UPDATED_CLUSTER for r in rows):
            # Matched an existing cluster that is no longer active/published.
            # Building it from only the staged emails would silently shrink it.
            published = await self._published_records(refresh=True)
            existing = published.get(cluster_id)
            if existing is None:
                raise RuntimeError("target cluster for UPDATED_CLUSTER rows is no longer an active published cluster")

        staged_ids = list(dict.fromkeys(r.email_message_id for r in rows))
        existing_refs = self._ref_dicts(existing) if existing else []
        existing_ids = [r["message_id"] for r in existing_refs if r.get("message_id")]
        member_ids = list(dict.fromkeys(existing_ids + staged_ids))

        anchored = existing.anchored_meeting_id if existing else ""
        cands, missing = await self._load_candidates(member_ids, meeting_id=anchored)
        missing_staged = [m for m in staged_ids if m in set(missing)]
        if missing_staged:
            raise RuntimeError(
                f"{len(missing_staged)} staged email(s) not found in raw store or Graph: {_sample(missing_staged)}"
            )
        if missing:
            log_event("existing_members_unloadable", level=logging.WARNING, run_id=run_id,
                      cluster_id=cluster_id, count=len(missing), sample=_sample(missing))

        vectors = await embed_candidates(cands, settings=s)
        cluster_vec = mean_embedding(vectors)

        if existing is None and s.meeting_anchored_clustering_enabled:
            # Same anchoring the legacy pipeline applies before clustering. It is
            # recomputed here because staging rows do not carry anchor state.
            await assign_candidates_to_meetings(cands, vectors, settings=s, run_id=run_id,
                                                run_now=datetime.now(timezone.utc))

        indices = list(range(len(cands)))
        if self._dept_index is None:
            self._dept_index = await build_department_index(s)
        assigned_dept, _scores = pick_department(
            cluster_vec,
            self._dept_index["centroids"],
            prototypes=self._dept_index["prototypes"],
            cluster_tokens=meaningful_tokens(service._cluster_match_text(cands, indices)),
            dept_tokens=self._dept_index["dept_tokens"],
            margin=s.email_dept_assignment_margin,
            min_score=s.email_dept_assignment_min_score,
        )

        staged_set, existing_set = set(staged_ids), set(existing_ids)
        new_refs = [
            ClusterEmailRef(
                message_id=c.message.id or "",
                subject=c.message.subject or "",
                from_address=c.message.from_address or "",
                date=c.message.received_at.isoformat() if c.message.received_at else "",
            ).model_dump()
            for c in cands
            if c.message.id in staged_set and c.message.id not in existing_set
        ]
        refs = existing_refs + new_refs

        if existing is not None:
            meeting_id, anchored_id = existing.meeting_id, existing.anchored_meeting_id
            meeting_start = existing.meeting_start
        else:
            meeting_id, anchored_id, meeting_start = self._occurrence_fields(cands)

        record = EmailClusterRecord(
            cluster_id=cluster_id,
            department=assigned_dept or (existing.department if existing else ""),
            topic=(existing.topic if existing else "") or normalize_subject(cands[0].message.subject),
            summary=existing.summary if existing else "",
            top_questions=existing.top_questions if existing else "",
            action_items=existing.action_items if existing else "",
            email_refs=json.dumps(refs),
            email_count=len(refs),
            cluster_embedding=json.dumps(cluster_vec.tolist()),
            meeting_id=meeting_id,
            anchored_meeting_id=anchored_id,
            meeting_start=meeting_start,
            updated_at=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
        )

        # Reuse the stored summary when it already covers every staged email
        # (a previous attempt wrote the DB row but did not finish).
        if existing is not None and existing.summary and staged_set <= existing_set:
            record.department = existing.department or record.department
            return record, True

        await self._synthesize_into(record, cands, indices, assigned_dept, anchored_id or anchored, run_id)
        return record, False

    def _occurrence_fields(self, cands: list[EmailCandidate]) -> tuple[str, str, str]:
        """(meeting_id, anchored_meeting_id, meeting_start) for a NEW cluster.

        Mirrors the per-cluster identity logic in ``service.run_email_cluster_pipeline``.
        """
        from agent_email.clustering import series_occurrence_key

        series_keys = {series_occurrence_key(c.message.subject, c.message.received_at) for c in cands}
        series_keys.discard("")
        meeting_ids = {c.meeting_id for c in cands if c.meeting_id}
        series_key = next(iter(series_keys)) if len(series_keys) == 1 else ""
        anchored = next(iter(meeting_ids)) if len(meeting_ids) == 1 else ""
        start = ""
        if anchored:
            start = next((c.meeting_start for c in cands if c.meeting_id == anchored), "")
        return (series_key or anchored), anchored, start

    async def _synthesize_into(
        self,
        record: EmailClusterRecord,
        cands: list[EmailCandidate],
        indices: list[int],
        assigned_dept: str,
        anchored_meeting_id: str,
        run_id: str,
    ) -> None:
        from agent_email import service
        from agent_email.clustering import split_for_synthesis

        s = self.settings
        results: list[dict] = []
        for batch_no, batch in enumerate(split_for_synthesis(indices, s.email_cluster_max_size), start=1):
            result = await service._synthesize_cluster_batch(
                cands, batch, s, run_id, batch_no,
                assigned_department=assigned_dept, meeting_id=anchored_meeting_id or "",
            )
            if result:
                results.append(result)
        merged = service._merge_synthesis_results(results)
        if assigned_dept:
            merged["department"] = assigned_dept
        if merged.get("summary") or merged.get("topic"):
            record.department = merged.get("department", record.department)
            record.topic = merged.get("topic") or record.topic
            record.summary = "|".join(merged.get("summary", []))
            record.top_questions = "|".join(merged.get("top_questions", []))
            record.action_items = "|".join(
                f for f in (service._format_action_item(ai) for ai in merged.get("action_items", [])) if f
            )
            self._synth_ok += 1
            PipelineMetricsCollector.record_summarization(self._synth_ok, self._synth_failed)
            return
        self._synth_failed += 1
        PipelineMetricsCollector.record_summarization(self._synth_ok, self._synth_failed)
        if s.bu_email_pipeline_enabled:
            # Publishing an empty summary would hide the failure; retry instead.
            raise RuntimeError("cluster synthesis produced no usable output")
        log_event("summary_skipped_pipeline_disabled", level=logging.WARNING, run_id=run_id,
                  cluster_id=record.cluster_id)

    # -- publish -----------------------------------------------------------

    async def _publish_record(self, record: EmailClusterRecord) -> None:
        from agent_email import store

        started = time.perf_counter()
        written = await store.write_cluster_records([record], settings=self.settings)
        log_event("cluster_db_write", run_id=record.run_id, cluster_id=record.cluster_id, count=written,
                  duration_ms=_ms(started), result="upserted" if written else "unchanged")
        forced = False
        if written == 0:
            # Row already current (retry after a partial success). The writer skips the
            # Weaviate/Redis mirror for unchanged rows, so do it explicitly (idempotent).
            forced = True
            await self._force_downstream_sync(record)
        log_event("downstream_sync", run_id=record.run_id, cluster_id=record.cluster_id,
                  weaviate="ok", redis="ok" if getattr(self.settings, "redis_enabled", False) else "disabled",
                  forced=forced)

    async def _force_downstream_sync(self, record: EmailClusterRecord) -> None:
        from agent_email import postgres_store, redis_cache
        from agent_email import weaviate_vector_store as wvs
        from agent_email.embedding_json import json_to_vec
        from agent_email.storage_ids import stable_object_id
        from agent_email.weaviate_vector_store import NAMESPACE_CLUSTERS

        vec = json_to_vec(record.cluster_embedding or "")
        if vec is not None and int(record.is_active or 0):
            await asyncio.to_thread(
                wvs.upsert_vectors, NAMESPACE_CLUSTERS, [(stable_object_id(record.cluster_id), vec)], self.settings,
            )
        if getattr(self.settings, "redis_enabled", False):
            snapshot = await postgres_store.read_all_cluster_records(settings=self.settings)
            await redis_cache.set_clusters([r.model_dump() for r in snapshot], self.settings)
            await redis_cache.clear_meeting_clusters(self.settings)


async def drain_pending(settings: Settings | None = None) -> PublishSummary | None:
    """Publish pending staged work only (no mailbox fetch). For bulk rebuilds.

    One call honours the per-run cluster cap, so callers loop until
    ``remaining_rows`` is 0 or stops shrinking. Returns None if another run
    holds the lock.
    """
    holder: list[PublishSummary] = []

    async def body() -> int:
        pub = ClusterPublisher(settings)
        await pub.begin()
        await pub.resume_pending()
        holder.append(await pub.finish())
        return 0

    await run_with_lock(body)
    return holder[0] if holder else None


async def run_with_lock(body: Callable[[], Awaitable[int]]) -> int:
    """Run ``body`` only if no other publish run holds the lease lock."""
    async with staging.run_lock() as acquired:
        if not acquired:
            log_event("run_skipped_locked", level=logging.WARNING,
                      error="another cluster publish run holds the lock lease")
            return 0
        return await body()
