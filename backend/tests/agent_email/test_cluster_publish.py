"""
tests/agent_email/test_cluster_publish.py
=========================================
The incremental publish flow, with the staging table replaced by an in-memory
fake. No PostgreSQL, Weaviate, Redis, Ollama or Graph is needed.

Covers: assignment (existing vs new cluster, via the real cluster_candidates),
deterministic ids, run-id reuse, threshold batching, deferral across runs,
retry accounting / exhaustion, crash-after-publish recovery, and de-duplication
of already-staged / already-published emails.
"""

import asyncio
import re
from datetime import datetime, timezone

import numpy as np
import pytest

from agent_email import cluster_publish as cp
from agent_email import cluster_publish_store as store
from agent_email.cluster_publish_store import NEW_CLUSTER, UPDATED_CLUSTER, ClusterStaging, StagingRow
from agent_email.schema import EmailCandidate, EmailClusterRecord, EmailMessage

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeStaging:
    """In-memory stand-in for the functions in cluster_publish_store."""

    def __init__(self):
        self.rows: list[dict] = []
        self.fail_delete_once = False

    def seed(self, run_id, cluster_id, emails, action=NEW_CLUSTER, retry=0):
        for e in emails:
            self.rows.append(dict(run_id=run_id, cluster_id=cluster_id, email_message_id=e,
                                  email_subject=e, cluster_action=action, retry_count=retry,
                                  created_at=NOW))

    async def insert_staging_rows(self, rows):
        added = 0
        for r in rows:
            key = (r.run_id, r.cluster_id, r.email_message_id)
            if any((x["run_id"], x["cluster_id"], x["email_message_id"]) == key for x in self.rows):
                continue
            self.rows.append(dict(run_id=r.run_id, cluster_id=r.cluster_id, email_message_id=r.email_message_id,
                                  email_subject=r.email_subject, cluster_action=r.cluster_action,
                                  retry_count=0, created_at=NOW))
            added += 1
        return added

    async def list_pending_run_ids(self):
        return list(dict.fromkeys(r["run_id"] for r in self.rows))

    async def run_counts(self, run_id):
        rs = [r for r in self.rows if r["run_id"] == run_id]
        return len({r["cluster_id"] for r in rs}), len({r["email_message_id"] for r in rs})

    async def list_cluster_groups(self, run_id):
        out = {}
        for r in self.rows:
            if r["run_id"] == run_id:
                out.setdefault(r["cluster_id"], []).append(r)
        return [ClusterStaging(run_id, cid, len(rs), max(x["retry_count"] for x in rs), NOW)
                for cid, rs in out.items()]

    async def rows_for_cluster(self, run_id, cluster_id):
        return [StagingRow(r["run_id"], r["cluster_id"], r["email_message_id"], r["email_subject"],
                           r["cluster_action"], r["retry_count"])
                for r in self.rows if r["run_id"] == run_id and r["cluster_id"] == cluster_id]

    async def staged_members_by_cluster(self):
        out = {}
        for r in self.rows:
            out.setdefault(r["cluster_id"], []).append((r["email_message_id"], r["email_subject"]))
        return out

    async def bump_retry(self, run_id, cluster_id):
        value = 0
        for r in self.rows:
            if r["run_id"] == run_id and r["cluster_id"] == cluster_id:
                r["retry_count"] += 1
                value = max(value, r["retry_count"])
        return value

    async def delete_cluster_rows(self, run_id, cluster_id):
        if self.fail_delete_once:
            self.fail_delete_once = False
            raise RuntimeError("connection reset")
        before = len(self.rows)
        self.rows = [r for r in self.rows if not (r["run_id"] == run_id and r["cluster_id"] == cluster_id)]
        return before - len(self.rows)

    async def remaining_rows(self, run_id=None):
        return len([r for r in self.rows if run_id is None or r["run_id"] == run_id])


@pytest.fixture
def fake(monkeypatch):
    f = FakeStaging()
    for name in ("insert_staging_rows", "list_pending_run_ids", "run_counts", "list_cluster_groups",
                 "rows_for_cluster", "staged_members_by_cluster", "bump_retry", "delete_cluster_rows",
                 "remaining_rows"):
        monkeypatch.setattr(store, name, getattr(f, name))
    return f


@pytest.fixture
def events(monkeypatch):
    seen = []
    monkeypatch.setattr(cp, "log_event", lambda event, **kw: seen.append((event, kw)))
    return seen


@pytest.fixture
def settings(required_env):
    from core.config import get_settings

    return get_settings().model_copy(update={
        "email_cluster_max_affected_clusters": 2,
        "email_cluster_max_new_emails": 1000,
        "email_cluster_max_clusters_per_run": 2,
        "email_cluster_max_staging_retries": 3,
    })


@pytest.fixture
def published_calls(monkeypatch):
    """Replace build/publish with recorders so only orchestration is under test."""
    calls = {"built": [], "published": [], "fail": set()}

    async def build(self, run_id, cluster_id, rows):
        if cluster_id in calls["fail"]:
            raise RuntimeError("synthesis exploded")
        calls["built"].append(cluster_id)
        return EmailClusterRecord(cluster_id=cluster_id, department="d", run_id=run_id), False

    async def publish(self, record):
        calls["published"].append(record.cluster_id)

    monkeypatch.setattr(cp.ClusterPublisher, "_build_record", build)
    monkeypatch.setattr(cp.ClusterPublisher, "_publish_record", publish)
    return calls


def run(coro):
    return asyncio.run(coro)


def cand(mid, subject, conv, vec):
    msg = EmailMessage(id=mid, conversation_id=conv, subject=subject, received_at=NOW, body_text="body")
    return EmailCandidate(message=msg), np.array(vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# Run id
# ---------------------------------------------------------------------------

def test_new_run_id_format():
    assert re.fullmatch(r"run_\d{14}_[0-9a-f]{32}", store.new_run_id())
    assert store.new_run_id() != store.new_run_id()


def test_begin_reuses_oldest_incomplete_run(fake, settings, events):
    fake.seed("run_old", "c1", ["m1"])
    fake.seed("run_newer", "c2", ["m2"])
    pub = cp.ClusterPublisher(settings)
    assert run(pub.begin()) == "run_old"


def test_begin_mints_new_run_when_nothing_pending(fake, settings, events):
    pub = cp.ClusterPublisher(settings)
    assert re.fullmatch(r"run_\d{14}_[0-9a-f]{32}", run(pub.begin()))


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def test_new_cluster_id_is_deterministic_and_order_insensitive():
    assert cp.new_cluster_id(["b", "a"]) == cp.new_cluster_id(["a", "b"])
    assert cp.new_cluster_id(["a"]) != cp.new_cluster_id(["a", "b"])


def test_derive_assignments_existing_vs_new(settings):
    from agent_email.clustering import cluster_candidates

    # working set: two existing members (clusters c1, c2) then three new emails
    pairs = [
        cand("e1", "Alpha pricing review", "t1", [1, 0, 0]),
        cand("e2", "Beta capex plan", "t2", [0, 1, 0]),
        cand("n1", "Gamma budget note", "t3", [0.99, 0.05, 0]),   # close to e1
        cand("n2", "Delta audit finding", "t4", [0, 0, 1]),        # matches nothing
        cand("n3", "Epsilon staffing memo", "t5", [0.02, 0, 0.99]),  # close to n2
    ]
    cands = [p[0] for p in pairs]
    vecs = [p[1] for p in pairs]
    groups = cluster_candidates(
        vecs, settings=settings, subjects=[c.message.subject for c in cands],
        bodies=[c.message.body_text for c in cands], meeting_ids=[""] * 5,
        received_ats=[c.message.received_at for c in cands],
        conversation_ids=[c.message.conversation_id for c in cands],
    )
    out = cp.derive_assignments(groups, 2, ["c1", "c2"], cands, vecs, {"c1", "c2"}, settings)
    by_id = {a.message_id: a for a in out}

    assert by_id["n1"].cluster_id == "c1" and by_id["n1"].action == UPDATED_CLUSTER
    assert by_id["n1"].reason.startswith(("vector_similarity", "residual_similarity", "graph_context"))
    assert by_id["n2"].action == NEW_CLUSTER and by_id["n3"].action == NEW_CLUSTER
    assert by_id["n2"].cluster_id == by_id["n3"].cluster_id == cp.new_cluster_id(["n2", "n3"])
    assert "c2" not in {a.cluster_id for a in out}          # untouched cluster gets no rows
    assert set(by_id) == {"n1", "n2", "n3"}                  # existing members never get rows


def test_conversation_id_match_reason(settings):
    cands = [cand("e1", "Alpha", "same-thread", [1, 0])[0], cand("n1", "Zzz", "same-thread", [0, 1])[0]]
    vecs = [np.array([1, 0], dtype=np.float32), np.array([0, 1], dtype=np.float32)]
    assert cp.match_reason(1, [0], cands, vecs, settings) == "conversation_id"


# ---------------------------------------------------------------------------
# Threshold + batching + deferral
# ---------------------------------------------------------------------------

def _seed_clusters(fake, run_id, n):
    for i in range(n):
        fake.seed(run_id, f"c{i}", [f"m{i}a", f"m{i}b"])


def test_over_threshold_processes_capped_clusters_and_keeps_rest(fake, settings, events, published_calls):
    _seed_clusters(fake, "R1", 5)                        # 5 > max_affected(2)
    pub = cp.ClusterPublisher(settings)
    run(pub.begin())
    run(pub.resume_pending())
    summary = run(pub.finish())

    assert summary.processed == ["c0", "c1"]             # capped at max_clusters_per_run=2
    assert summary.deferred == 3
    assert {r["cluster_id"] for r in fake.rows} == {"c2", "c3", "c4"}
    assert len(fake.rows) == 6                           # deferred rows untouched, none deleted
    check = next(kw for ev, kw in events if ev == "threshold_check")
    assert (check["clusters_selected"], check["clusters_deferred"]) == (2, 3)


def test_deferred_clusters_drain_over_following_runs(fake, settings, events, published_calls):
    _seed_clusters(fake, "R1", 5)
    for expected_left in (3, 1, 0):
        pub = cp.ClusterPublisher(settings)
        run(pub.begin())
        run(pub.resume_pending())
        run(pub.finish())
        assert len({r["cluster_id"] for r in fake.rows}) == expected_left
    assert sorted(published_calls["published"]) == ["c0", "c1", "c2", "c3", "c4"]
    assert len(published_calls["published"]) == 5        # nothing published twice


def test_under_threshold_processes_everything(fake, settings, events, published_calls):
    settings = settings.model_copy(update={"email_cluster_max_affected_clusters": 10})
    _seed_clusters(fake, "R1", 5)
    pub = cp.ClusterPublisher(settings)
    run(pub.begin())
    run(pub.resume_pending())
    assert run(pub.finish()).processed == ["c0", "c1", "c2", "c3", "c4"]
    assert fake.rows == []


def test_new_email_count_threshold_also_triggers_batching(fake, settings, events, published_calls):
    settings = settings.model_copy(update={"email_cluster_max_affected_clusters": 100,
                                           "email_cluster_max_new_emails": 3})
    _seed_clusters(fake, "R1", 3)                        # 6 emails > 3
    pub = cp.ClusterPublisher(settings)
    run(pub.begin())
    run(pub.resume_pending())
    assert len(run(pub.finish()).processed) == 2


# ---------------------------------------------------------------------------
# Failure, retry, exhaustion, crash recovery
# ---------------------------------------------------------------------------

def test_failure_keeps_rows_and_increments_retry(fake, settings, events, published_calls):
    settings = settings.model_copy(update={"email_cluster_max_affected_clusters": 10})
    fake.seed("R1", "bad", ["m1", "m2"])
    fake.seed("R1", "ok", ["m3"])
    published_calls["fail"].add("bad")
    pub = cp.ClusterPublisher(settings)
    run(pub.begin())
    run(pub.resume_pending())
    summary = run(pub.finish())

    assert summary.processed == ["ok"] and summary.failed == ["bad"]
    bad = [r for r in fake.rows if r["cluster_id"] == "bad"]
    assert len(bad) == 2 and all(r["retry_count"] == 1 for r in bad)
    failure = next(kw for ev, kw in events if ev == "cluster_processing_failed")
    assert failure["cluster_id"] == "bad" and failure["retry_count"] == 1
    assert "synthesis exploded" in failure["error"] and "m1" in failure["email_message_id"]


def test_failed_cluster_is_not_retried_twice_in_one_invocation(fake, settings, events, published_calls):
    fake.seed("R1", "bad", ["m1"])
    published_calls["fail"].add("bad")
    pub = cp.ClusterPublisher(settings)
    run(pub.begin())
    run(pub.resume_pending())
    run(pub._process_run("R1"))                          # what publish_new does afterwards
    assert fake.rows[0]["retry_count"] == 1


def test_retries_exhausted_stops_retrying_but_keeps_rows_and_alerts(fake, settings, events, published_calls):
    fake.seed("R1", "bad", ["m1"])
    published_calls["fail"].add("bad")
    for _ in range(5):
        pub = cp.ClusterPublisher(settings)
        run(pub.begin())
        run(pub.resume_pending())
        summary = run(pub.finish())

    assert fake.rows[0]["retry_count"] == 3              # capped at EMAIL_CLUSTER_MAX_STAGING_RETRIES
    assert summary.exhausted == ["bad"]
    assert len(fake.rows) == 1                           # never deleted
    alerts = [kw for ev, kw in events if ev == "cluster_retries_exhausted"]
    assert alerts and alerts[-1]["cluster_id"] == "bad"


def test_crash_after_publish_before_delete_recovers_without_duplicates(fake, settings, events, published_calls):
    fake.seed("R1", "c1", ["m1", "m2"])
    fake.fail_delete_once = True                         # DB/Weaviate/Redis done, then the process "dies"
    pub = cp.ClusterPublisher(settings)
    run(pub.begin())
    run(pub.resume_pending())
    assert len(fake.rows) == 2 and fake.rows[0]["retry_count"] == 1

    pub2 = cp.ClusterPublisher(settings)                 # next run resumes the same run id
    assert run(pub2.begin()) == "R1"
    run(pub2.resume_pending())
    assert fake.rows == []
    # The second pass republishes the same cluster id (an upsert), never a different one.
    assert published_calls["published"] == ["c1", "c1"]


def test_resume_runs_before_new_email_staging(fake, settings, events, published_calls, monkeypatch):
    order = []
    fake.seed("R1", "old", ["m1"])

    async def build(self, run_id, cluster_id, rows):
        order.append(("process", cluster_id))
        return EmailClusterRecord(cluster_id=cluster_id, department="d", run_id=run_id), False

    async def stage(self, candidates, vectors):
        order.append(("stage", ""))
        return []

    monkeypatch.setattr(cp.ClusterPublisher, "_build_record", build)
    monkeypatch.setattr(cp.ClusterPublisher, "stage_new_emails", stage)

    async def flow():
        pub = cp.ClusterPublisher(settings)
        await pub.begin()
        await pub.resume_pending()
        await pub.publish_new([], [])

    run(flow())
    assert order[0] == ("process", "old") and ("stage", "") in order


# ---------------------------------------------------------------------------
# Staging of new emails
# ---------------------------------------------------------------------------

def test_stage_skips_already_published_and_already_staged(fake, settings, events, monkeypatch):
    fake.seed("R0", "pending", ["staged-1"])
    published = {"cX": EmailClusterRecord(cluster_id="cX", department="d",
                                          email_refs='[{"message_id": "pub-1"}]')}

    async def pubs(self, refresh=False):
        return published

    async def shortlist(self, *a, **k):
        return {}

    async def members(self, *a, **k):
        return [], [], []

    monkeypatch.setattr(cp.ClusterPublisher, "_published_records", pubs)
    monkeypatch.setattr(cp.ClusterPublisher, "_shortlist_clusters", shortlist)
    monkeypatch.setattr(cp.ClusterPublisher, "_load_members", members)

    pairs = [cand("pub-1", "Alpha", "t1", [1, 0, 0]), cand("staged-1", "Beta", "t2", [0, 1, 0]),
             cand("fresh-1", "Gamma", "t3", [0, 0, 1])]

    async def flow():
        pub = cp.ClusterPublisher(settings)
        await pub.begin()
        return await pub.stage_new_emails([p[0] for p in pairs], [p[1] for p in pairs])

    out = run(flow())
    assert [a.message_id for a in out] == ["fresh-1"]
    assert out[0].action == NEW_CLUSTER
    staged_now = [r for r in fake.rows if r["email_message_id"] == "fresh-1"]
    assert len(staged_now) == 1 and staged_now[0]["run_id"] == "R0"   # joins the reused run


# ---------------------------------------------------------------------------
# Log injection
# ---------------------------------------------------------------------------

def test_log_event_neutralises_newlines_in_untrusted_values(monkeypatch):
    lines = []

    class Rec:
        def log(self, level, fmt, line):
            lines.append(fmt % line)

    monkeypatch.setattr(cp, "logger", Rec())
    cp.log_event(
        "cluster_processing_failed",
        run_id="r1\n[CLUSTER-PUBLISH] event='forged'",
        email_message_id="AAMk\r\nFAKE",
        error=RuntimeError("boom\nevent=forged"),
        sample=["a\nb"],
    )
    assert len(lines) == 1 and "\n" not in lines[0] and "\r" not in lines[0]
