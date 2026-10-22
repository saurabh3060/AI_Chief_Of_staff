"""
tests/security/test_regex_and_error_leaks.py
===============================================
Regression tests for two CodeQL findings, with no database or network needed:

  * py/polynomial-redos - the question-text regexes in documents.generic_metrics
    must give the same answers on normal questions and stay fast on hostile input.
  * py/stack-trace-exposure - exception text must never reach an HTTP response
    from documents.sync_routes.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from documents import generic_metrics as gm


@pytest.fixture
def sync_routes():
    """documents.sync_routes pulls in the Graph client stack; skip if it is not installed."""
    return pytest.importorskip("documents.sync_routes")


# A hostile input must finish well inside this budget (the unfixed patterns took
# far longer on inputs of this size; normal calls take a few milliseconds).
_BUDGET_S = 1.0
_HOSTILE = [
    "a" * 50000 + "!",
    "apr" * 20000,
    "between " + "a " * 20000,
    "between a and " + "b " * 20000,
    " " * 50000 + "h",
    "what is " * 6000,
    "h" + " " * 50000 + "1",
]


def _timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    assert time.perf_counter() - start < _BUDGET_S
    return result


# ── ReDoS: normal inputs keep their results ───────────────────

def test_question_scopes_normal_inputs():
    assert {"RANGE", "YTD"} <= gm._question_scopes("What was YTD growth in advances?")
    assert {"RANGE", "YTD"} <= gm._question_scopes("Show apr - nov deposits")
    assert {"RANGE", "YTD"} <= gm._question_scopes("Show April to November deposits")
    assert "Q" in gm._question_scopes("q3 advances")


def test_comparison_windows_normal_inputs():
    assert gm._comparison_windows("What changed between H1 and Oct-Feb?") == ["H1", "Oct-Feb"]
    assert gm._comparison_windows("Compare between H2 versus Apr-Jun.") == ["H2", "Apr-Jun"]
    assert gm._comparison_windows("What was the total in June?") == []


def test_window_months_normal_inputs():
    assert gm._window_months("H1") == set(range(4, 10))
    assert gm._window_months("  h 2 ") == {10, 11, 12, 1, 2, 3}
    assert gm._window_months("Oct-Feb") == {10, 11, 12, 1, 2}


def test_asked_prefix_pattern_matches_like_the_old_pattern():
    match = gm._ASKED_PREFIX_RE.search("hey, what is the npa recovery rate?")
    assert match and "hey, what is the npa recovery rate?"[match.end():] == "the npa recovery rate?"
    assert gm._ASKED_PREFIX_RE.search("npa recovery rate") is None


# ── ReDoS: hostile inputs stay fast ───────────────────────────

@pytest.mark.parametrize("text", _HOSTILE, ids=[f"hostile{i}" for i in range(len(_HOSTILE))])
def test_hostile_input_is_fast(text):
    _timed(gm._question_scopes, text)
    _timed(gm._comparison_windows, text)
    _timed(gm._window_months, text)
    _timed(gm._ASKED_PREFIX_RE.search, text)
    _timed(gm._YTD_RANGE_RE.search, text)


def test_oversized_input_is_capped_not_matched():
    assert gm._window_months("h1" + " " * (gm.MAX_REGEX_INPUT_LEN + 1)) is None
    assert gm._comparison_windows("x" * (gm.MAX_REGEX_INPUT_LEN + 1) + " between H1 and H2") == []


# ── Exception exposure: sync routes ───────────────────────────

LEAK = r"C:\secret\path password=hunter2 Traceback (most recent call last)"


def _run_single_file(sync_routes, monkeypatch, outcome):
    monkeypatch.setattr(
        sync_routes, "get_settings",
        lambda: SimpleNamespace(onedrive_sync_enabled=True, is_local_mail_mode=False),
    )

    async def fake_sync_single_file(**kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(sync_routes.sharepoint_sync, "sync_single_file", fake_sync_single_file)
    body = sync_routes.SingleFileSyncRequest(drive_id="drive", item_id="item")
    return asyncio.run(sync_routes.sync_single_file(body=body, bearer_token=None))


def test_failed_single_file_sync_never_returns_the_exception_text(sync_routes, monkeypatch):
    response = _run_single_file(
        sync_routes, monkeypatch, {"status": "failed", "error": LEAK, "name": "a.pdf"},
    )
    assert response == {"status": "failed", "name": "a.pdf", "error": "Single-file sync failed."}
    assert LEAK not in repr(response)


def test_unexpected_result_shape_is_reported_as_a_generic_failure(sync_routes, monkeypatch):
    response = _run_single_file(sync_routes, monkeypatch, {"error": LEAK})
    assert response == {"status": "failed", "name": "", "error": "Single-file sync failed."}


def test_successful_single_file_sync_returns_only_the_safe_fields(sync_routes, monkeypatch):
    response = _run_single_file(
        sync_routes, monkeypatch, {"status": "success", "doc_id": "doc1", "name": "a.pdf", "debug": LEAK},
    )
    assert response == {"status": "success", "doc_id": "doc1", "name": "a.pdf"}


def test_exception_in_single_file_sync_becomes_a_generic_500(sync_routes, monkeypatch):
    with pytest.raises(HTTPException) as caught:
        _run_single_file(sync_routes, monkeypatch, RuntimeError(LEAK))
    assert caught.value.status_code == 500
    assert caught.value.detail == "Single-file sync failed." and LEAK not in str(caught.value.detail)


def test_public_summary_keeps_counts_and_hides_error_strings(sync_routes):
    summary = {"synced": 1, "failed": 2, "status": "partial", "errors": [LEAK, LEAK]}
    public = sync_routes._public_summary(summary)
    assert public == {"synced": 1, "failed": 2, "status": "partial", "error_count": 2}
    assert "errors" in summary                       # the original is not modified
    assert sync_routes._public_summary(None) is None
    assert sync_routes._public_summary({"synced": 0}) == {"synced": 0}


def test_status_route_hides_error_strings(sync_routes, monkeypatch):
    summary = {"synced": 0, "failed": 1, "status": "failed", "errors": [LEAK]}
    monkeypatch.setattr(
        sync_routes, "get_settings",
        lambda: SimpleNamespace(
            onedrive_sync_enabled=True, sharepoint_sites_list=[],
            onedrive_sync_schedule_hour=2, onedrive_sync_schedule_minute=0,
        ),
    )

    async def fake_load_state():
        return {"last_sync_status": "failed", "last_sync_summary": summary, "files": {}}

    monkeypatch.setattr(sync_routes.sync_state, "load_state", fake_load_state)
    monkeypatch.setattr(sync_routes.sharepoint_sync, "last_sync_result", lambda: summary)

    response = asyncio.run(sync_routes.sync_status())
    assert LEAK not in repr(response)
    assert response["last_run_result"]["error_count"] == 1
    assert response["last_sync_summary"]["error_count"] == 1
