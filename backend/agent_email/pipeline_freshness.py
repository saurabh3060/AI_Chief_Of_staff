"""
agent_email/pipeline_freshness.py
=====================================
In-memory, process-lifetime marker for whether the email cluster
pipeline (``run_email_cluster_pipeline``) has completed at least once
since this backend process started.

Why this exists
----------------
``email_clusters.csv`` (or its Postgres equivalent) is persistent
storage: it survives a backend restart untouched. Without this marker,
``GET /agents/email/clusters`` and ``GET /agents/open-tracker`` would
immediately serve whatever was written by the *previous* process's
pipeline run, with no way for the frontend to know that data predates
the current backend session and a fresh Ollama-driven run is still in
flight.

This module tracks a single boolean flag, reset to ``False`` on every
process start (module import), and flipped to ``True`` the first time
``run_email_cluster_pipeline`` finishes a run (success, empty mailbox,
or no candidates after filtering all count as "finished" - the run
completed without crashing).

Deliberately simple: a single module-level flag, no locking. Flag
writes are a single bool assignment (atomic under the GIL) and reads
never need to observe partial state.
"""

from __future__ import annotations

import threading

_ready: bool = False
_ready_at: str | None = None
_lock = threading.Lock()


def mark_pipeline_complete() -> None:
    """Call once a pipeline run has finished (success or empty-result).

    Idempotent - safe to call after every run, not just the first.
    """
    global _ready, _ready_at
    from datetime import datetime, timezone

    with _lock:
        _ready = True
        _ready_at = datetime.now(timezone.utc).isoformat()


def is_pipeline_fresh() -> bool:
    """True once at least one pipeline run has completed this process."""
    return _ready


def pipeline_ready_at() -> str | None:
    """ISO timestamp of when the pipeline first became fresh this session, or None."""
    return _ready_at


def reset_for_tests() -> None:
    """Reset the flag. Only intended for test isolation."""
    global _ready, _ready_at
    with _lock:
        _ready = False
        _ready_at = None
