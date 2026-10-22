"""
agent_email/backfill_email_refs.py
==================================
One-off, OPT-IN backfill of the Graph ``message_id`` into legacy entries
of ``email_clusters.email_refs`` (the JSONB array of ``ClusterEmailRef``
dicts).

Background
----------
``ClusterEmailRef`` gained a ``message_id`` field after many clusters had
already been written, so legacy refs carry only ``{subject, from_address,
date}``. This module resolves the missing id by matching each legacy ref,
on the EXACT triple ``(subject, from_address, date)``, against the
``email_topics`` table — the one place whose ``received_at`` is written
with the identical ``received_at.isoformat()`` expression used for
``ClusterEmailRef.date`` (see ``service._email_refs_for_indices`` vs
``postgres_store.write_topic_records``), so no timezone / precision
reconciliation is needed.

Safety rules (hard requirements)
--------------------------------
* **Exact triple only** — never subject-alone, never fuzzy.
* **Exactly one distinct message_id** — a triple that resolves to 0 rows
  (unmatched) or to >= 2 distinct message_ids (ambiguous) is LEFT
  UNCHANGED and reported. IDs are never guessed.
* **Never overwrite** — a ref that already has a non-blank ``message_id``
  is left exactly as-is (counted as ``already_populated``).
* **Preserve everything else** — every other JSON field on a ref, every
  ref, and the ref order are preserved; only a blank ``message_id`` on a
  safely-matched ref is filled in.
* **``bu_emails`` is never read or written** — the chatbot's source table
  is untouched. The match source is ``email_topics`` only.
* **Dry-run by default** — :func:`backfill_email_refs_message_ids` writes
  NOTHING unless ``dry_run=False`` is passed explicitly. Even then each
  UPDATE carries an optimistic guard so a row changed by a concurrent
  pipeline run is skipped rather than clobbered.

This module performs no embedding/LLM work and touches only the
``email_clusters`` table (reads ``email_topics`` for the lookup map).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from core.config import Settings, get_settings
from core.db import get_pool
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_email")


# ---------------------------------------------------------------------------
# Pure helpers (no I/O — unit-testable in isolation)
# ---------------------------------------------------------------------------

def _ref_key(subject: str, from_address: str, date: str) -> tuple[str, str, str]:
    """The exact match key for a ref / topic row.

    Kept as a single function so the ref side and the topic side can never
    drift apart in how they normalize the triple. We intentionally do NOT
    lower-case or strip: both ``email_refs.date`` and
    ``email_topics.received_at`` are produced by the same
    ``received_at.isoformat()`` call and subject/from_address are stored
    verbatim, so an EXACT, byte-for-byte equality is the correct — and
    safest — comparison. Loosening it here would risk a wrong match.
    """
    return (subject or "", from_address or "", date or "")


def build_unambiguous_message_id_map(
    topic_rows: list[dict],
) -> dict[tuple[str, str, str], str]:
    """Build ``{(subject, from_address, received_at): message_id}`` keeping
    ONLY triples that resolve to exactly one distinct, non-blank message_id.

    ``topic_rows`` is a list of dicts with keys ``subject``,
    ``from_address``, ``received_at``, ``message_id`` (as read from
    ``email_topics``). A triple seen with two or more DISTINCT message_ids
    is dropped from the map entirely (ambiguous → not resolvable), so a
    later lookup for it misses and the ref is left unchanged.
    """
    # First collect the SET of distinct message_ids per triple.
    candidates: dict[tuple[str, str, str], set[str]] = {}
    for row in topic_rows:
        message_id = (row.get("message_id") or "").strip()
        if not message_id:
            continue  # a topic row without an id cannot resolve anything
        key = _ref_key(row.get("subject", ""), row.get("from_address", ""), row.get("received_at", ""))
        candidates.setdefault(key, set()).add(message_id)

    # Keep only the triples with EXACTLY one distinct id.
    resolved: dict[tuple[str, str, str], str] = {}
    for key, ids in candidates.items():
        if len(ids) == 1:
            resolved[key] = next(iter(ids))
    return resolved


@dataclass
class ClusterPlan:
    """The planned change (if any) for a single ``email_clusters`` row.

    ``new_email_refs_json`` is set only when at least one ref in the row
    was safely filled; otherwise it is ``None`` (nothing to write for this
    row). The per-category counts describe only THIS row.
    """

    cluster_id: str
    old_email_refs_json: str
    new_email_refs_json: str | None = None
    matched: int = 0
    unmatched: int = 0
    ambiguous: int = 0
    already_populated: int = 0
    # Human-readable notes for the unmatched / ambiguous refs so the report
    # can show WHICH refs were left alone and why.
    unresolved: list[dict] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.new_email_refs_json is not None


def plan_cluster_backfill(
    cluster_id: str,
    email_refs_json: str,
    resolved_map: dict[tuple[str, str, str], str],
    ambiguous_keys: set[tuple[str, str, str]] | None = None,
) -> ClusterPlan:
    """Compute the backfill plan for one cluster's ``email_refs`` JSON.

    * A ref with a non-blank ``message_id`` → ``already_populated`` (never
      touched).
    * A ref whose triple is in ``resolved_map`` → filled → ``matched``.
    * A ref whose triple is in ``ambiguous_keys`` → ``ambiguous`` (left as
      is, reported).
    * Otherwise → ``unmatched`` (left as is, reported).

    All other JSON fields and the ref ordering are preserved. Returns a
    :class:`ClusterPlan`; ``new_email_refs_json`` is populated only if the
    row actually changed.
    """
    ambiguous_keys = ambiguous_keys or set()
    plan = ClusterPlan(cluster_id=cluster_id, old_email_refs_json=email_refs_json)

    try:
        refs = json.loads(email_refs_json) if email_refs_json else []
    except (json.JSONDecodeError, TypeError):
        # A malformed email_refs is left completely untouched and reported
        # as a single unmatched entry — we never rewrite data we can't parse.
        plan.unmatched = 1
        plan.unresolved.append({"reason": "unparseable_email_refs", "cluster_id": cluster_id})
        return plan

    if not isinstance(refs, list):
        plan.unmatched = 1
        plan.unresolved.append({"reason": "email_refs_not_a_list", "cluster_id": cluster_id})
        return plan

    changed = False
    new_refs: list = []
    for ref in refs:
        # Non-dict entries are preserved verbatim and not counted.
        if not isinstance(ref, dict):
            new_refs.append(ref)
            continue

        existing_id = (ref.get("message_id") or "").strip()
        if existing_id:
            # Never overwrite an existing id.
            plan.already_populated += 1
            new_refs.append(ref)
            continue

        key = _ref_key(ref.get("subject", ""), ref.get("from_address", ""), ref.get("date", ""))
        if key in resolved_map:
            # Safe single-match: fill the id, preserve every other field and
            # the field order (build a shallow copy so the original dict in
            # memory is untouched).
            updated = dict(ref)
            updated["message_id"] = resolved_map[key]
            new_refs.append(updated)
            plan.matched += 1
            changed = True
        elif key in ambiguous_keys:
            plan.ambiguous += 1
            plan.unresolved.append({
                "reason": "ambiguous",
                "subject": ref.get("subject", ""),
                "from_address": ref.get("from_address", ""),
                "date": ref.get("date", ""),
            })
            new_refs.append(ref)
        else:
            plan.unmatched += 1
            plan.unresolved.append({
                "reason": "unmatched",
                "subject": ref.get("subject", ""),
                "from_address": ref.get("from_address", ""),
                "date": ref.get("date", ""),
            })
            new_refs.append(ref)

    if changed:
        # Compact separators keep the stored JSON tight, matching how the
        # pipeline writes it (json.dumps default is fine either way since
        # Postgres stores JSONB, not the raw text).
        plan.new_email_refs_json = json.dumps(new_refs)
    return plan


@dataclass
class BackfillReport:
    """Aggregate report across every cluster considered."""

    dry_run: bool = True
    clusters_scanned: int = 0
    clusters_changed: int = 0
    clusters_written: int = 0          # rows an UPDATE actually touched
    clusters_write_skipped: int = 0    # optimistic-guard misses (concurrent edit)
    refs_matched: int = 0
    refs_unmatched: int = 0
    refs_ambiguous: int = 0
    refs_already_populated: int = 0
    # A capped sample of unresolved refs for the operator to eyeball.
    unresolved_sample: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "dry_run": self.dry_run,
            "clusters_scanned": self.clusters_scanned,
            "clusters_changed": self.clusters_changed,
            "clusters_written": self.clusters_written,
            "clusters_write_skipped": self.clusters_write_skipped,
            "refs_matched": self.refs_matched,
            "refs_unmatched": self.refs_unmatched,
            "refs_ambiguous": self.refs_ambiguous,
            "refs_already_populated": self.refs_already_populated,
            "unresolved_sample": self.unresolved_sample,
        }


def build_report_from_plans(
    plans: list[ClusterPlan],
    dry_run: bool,
    unresolved_sample_limit: int = 50,
) -> BackfillReport:
    """Aggregate per-cluster plans into a single :class:`BackfillReport`.

    This is the pure counting step (no I/O), separated so tests can assert
    the exact matched / unmatched / ambiguous / already-populated tallies
    without a database.
    """
    report = BackfillReport(dry_run=dry_run)
    for plan in plans:
        report.clusters_scanned += 1
        report.refs_matched += plan.matched
        report.refs_unmatched += plan.unmatched
        report.refs_ambiguous += plan.ambiguous
        report.refs_already_populated += plan.already_populated
        if plan.changed:
            report.clusters_changed += 1
        for item in plan.unresolved:
            if len(report.unresolved_sample) < unresolved_sample_limit:
                report.unresolved_sample.append({"cluster_id": plan.cluster_id, **item})
    return report


# ---------------------------------------------------------------------------
# Orchestration (I/O) — dry-run by default; writes ONLY when dry_run=False
# ---------------------------------------------------------------------------

async def _load_topic_rows(conn) -> list[dict]:
    """Read the (subject, from_address, received_at, message_id) tuples from
    ``email_topics``. Read-only."""
    rows = await conn.fetch(
        "SELECT subject, from_address, received_at, message_id FROM email_topics"
    )
    return [
        {
            "subject": r["subject"] or "",
            "from_address": r["from_address"] or "",
            "received_at": r["received_at"] or "",
            "message_id": r["message_id"] or "",
        }
        for r in rows
    ]


def _ambiguous_keys_from_topics(topic_rows: list[dict]) -> set[tuple[str, str, str]]:
    """Triples that map to >= 2 distinct message_ids (reportable as ambiguous)."""
    seen: dict[tuple[str, str, str], set[str]] = {}
    for row in topic_rows:
        message_id = (row.get("message_id") or "").strip()
        if not message_id:
            continue
        key = _ref_key(row.get("subject", ""), row.get("from_address", ""), row.get("received_at", ""))
        seen.setdefault(key, set()).add(message_id)
    return {key for key, ids in seen.items() if len(ids) >= 2}


async def backfill_email_refs_message_ids(
    settings: Settings | None = None,
    dry_run: bool = True,
    unresolved_sample_limit: int = 50,
) -> BackfillReport:
    """Resolve and (optionally) persist missing ``message_id`` values in
    ``email_clusters.email_refs``.

    DRY-RUN BY DEFAULT: with ``dry_run=True`` (the default) this reads
    ``email_topics`` and ``email_clusters`` and returns a full report but
    executes NO ``UPDATE``. Pass ``dry_run=False`` ONLY after the plan has
    been reviewed/approved to actually persist the safely-matched ids.

    Scope: reads ``email_topics`` (match source) + ``email_clusters``
    (target). Never reads or writes ``bu_emails`` and never touches the
    chatbot. Each write is guarded by ``email_refs::text = <value read>``
    so a row changed by a concurrent pipeline run is skipped, never
    clobbered.
    """
    active = settings or get_settings()
    pool = get_pool()
    if pool is None:
        raise RuntimeError("backfill_email_refs_message_ids: PostgreSQL pool unavailable")

    async with pool.acquire() as conn:
        topic_rows = await _load_topic_rows(conn)
        resolved_map = build_unambiguous_message_id_map(topic_rows)
        ambiguous_keys = _ambiguous_keys_from_topics(topic_rows)
        logger.info(
            "[BACKFILL] email_topics rows=%d unambiguous_keys=%d ambiguous_keys=%d",
            len(topic_rows), len(resolved_map), len(ambiguous_keys),
        )

        # Read the full cluster-ref history (active AND inactive) so legacy
        # superseded rows get fixed too. We only read id + email_refs.
        cluster_rows = await conn.fetch(
            "SELECT cluster_id, email_refs FROM email_clusters"
        )

        plans: list[ClusterPlan] = []
        for r in cluster_rows:
            refs_val = r["email_refs"]
            # asyncpg returns JSONB as a str unless a codec is registered;
            # normalize to a JSON string for the pure planner.
            if isinstance(refs_val, (list, dict)):
                refs_json = json.dumps(refs_val)
            else:
                refs_json = refs_val or "[]"
            plans.append(
                plan_cluster_backfill(
                    cluster_id=r["cluster_id"],
                    email_refs_json=refs_json,
                    resolved_map=resolved_map,
                    ambiguous_keys=ambiguous_keys,
                )
            )

        report = build_report_from_plans(
            plans, dry_run=dry_run, unresolved_sample_limit=unresolved_sample_limit
        )

        if dry_run:
            logger.info("[BACKFILL][DRY-RUN] no writes. report=%s", report.as_dict())
            return report

        # --- APPLY PATH (only reached when dry_run is explicitly False) ---
        for plan in plans:
            if not plan.changed:
                continue
            # Optimistic guard: only update if the stored email_refs still
            # equals what we planned against. ::jsonb on both sides makes
            # the comparison insensitive to whitespace/key-order
            # reformatting that asyncpg/Postgres may apply.
            result = await conn.execute(
                """
                UPDATE email_clusters
                   SET email_refs = $1::jsonb, updated_at = now()
                 WHERE cluster_id = $2
                   AND email_refs::jsonb = $3::jsonb
                """,
                plan.new_email_refs_json, plan.cluster_id, plan.old_email_refs_json,
            )
            # asyncpg returns e.g. "UPDATE 1" / "UPDATE 0".
            affected = 0
            try:
                affected = int(str(result).split()[-1])
            except (ValueError, IndexError):
                affected = 0
            if affected > 0:
                report.clusters_written += 1
            else:
                report.clusters_write_skipped += 1
                logger.warning(
                    "[BACKFILL] cluster_id=%s not written (row changed since read / "
                    "optimistic guard miss)", plan.cluster_id,
                )

        logger.info("[BACKFILL][APPLY] report=%s", report.as_dict())
        return report
