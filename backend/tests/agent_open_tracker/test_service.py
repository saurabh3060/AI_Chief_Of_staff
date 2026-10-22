"""
tests/agent_open_tracker/test_service.py
=========================================
Tests for the Open Actions Status Tracker's 7-day "upcoming" look-ahead
window (see agent_open_tracker/service.py::get_open_tracker_items).

Overdue and Due Today items are always shown. Upcoming items are only
shown if they have no due date, or their due date falls within the next
7 days (today .. today+7 inclusive). Items dated further out are hidden
from both the returned list and the summary counts.
"""

from datetime import datetime, timedelta, timezone

import pytest

from agent_email.schema import EmailClusterRecord
from agent_open_tracker import service
from agent_open_tracker.schema import OpenTrackerStatus


def _cluster(cluster_id: str, department: str, action_items: str) -> EmailClusterRecord:
    return EmailClusterRecord(
        cluster_id=cluster_id,
        department=department,
        topic="Test Topic",
        action_items=action_items,
    )


@pytest.fixture
def today() -> "datetime.date":
    return datetime.now(timezone.utc).date()


@pytest.fixture
def patch_stores(monkeypatch):
    """Stub out the CSV read and the completed/override stores so the
    service's classification + windowing logic can be tested in isolation."""

    async def _fake_completed_ids():
        return set()

    async def _fake_overrides():
        return {}

    monkeypatch.setattr(service, "load_completed_ids", _fake_completed_ids)
    monkeypatch.setattr(service, "load_item_overrides", _fake_overrides)

    def _set_clusters(clusters):
        async def _fake_read_all(*args, **kwargs):
            return clusters
        monkeypatch.setattr(service, "read_all_cluster_records", _fake_read_all)

    return _set_clusters


@pytest.mark.asyncio
async def test_upcoming_window_filters_far_future_items(patch_stores, today):
    """Overdue, due-today, tomorrow, and 7-days-out items are kept;
    8-days-out and 30-days-out are dropped; no-due-date is kept."""
    fmt = lambda d: d.strftime("%Y-%m-%d")

    clusters = [
        _cluster("c1", "BB&RL", (
            f"Alice: overdue task (by {fmt(today - timedelta(days=3))})|"
            f"Bob: due today task (by {fmt(today)})|"
            f"Carol: due tomorrow task (by {fmt(today + timedelta(days=1))})|"
            f"Dave: due in 7 days task (by {fmt(today + timedelta(days=7))})|"
            f"Erin: due in 8 days task (by {fmt(today + timedelta(days=8))})|"
            f"Frank: due in 30 days task (by {fmt(today + timedelta(days=30))})|"
            f"Grace: no due date task"
        )),
    ]
    patch_stores(clusters)

    result = await service.get_open_tracker_items()

    texts = {item.text: item for item in result.items}

    assert "overdue task" in texts
    assert "due today task" in texts
    assert "due tomorrow task" in texts
    assert "due in 7 days task" in texts
    assert "no due date task" in texts

    assert "due in 8 days task" not in texts
    assert "due in 30 days task" not in texts

    # Counts must reflect only the visible/filtered items.
    assert result.overdue_count == 1
    assert result.due_today_count == 1
    assert result.upcoming_count == 3  # tomorrow, 7-days-out, no-due-date
    assert result.total_count == 5


@pytest.mark.asyncio
async def test_upcoming_window_applies_to_pending_decisions(patch_stores, today):
    """The same 7-day window also applies to the Awaiting-Decision list."""
    fmt = lambda d: d.strftime("%Y-%m-%d")

    clusters = [
        _cluster("c1", "BB&RL", (
            f"R Subramaniakumar: near-term decision (by {fmt(today + timedelta(days=5))})|"
            f"R Subramaniakumar: far-future decision (by {fmt(today + timedelta(days=45))})"
        )),
    ]
    patch_stores(clusters)

    result = await service.get_open_tracker_items()
    texts = {p.text for p in result.pending_decisions}

    assert "near-term decision" in texts
    assert "far-future decision" not in texts


@pytest.mark.asyncio
async def test_no_due_date_items_always_kept(patch_stores):
    clusters = [
        _cluster("c1", "Wholesale Banking", "Heidi: task with no date at all"),
    ]
    patch_stores(clusters)

    result = await service.get_open_tracker_items()

    assert len(result.items) == 1
    assert result.items[0].status == OpenTrackerStatus.UPCOMING
    assert result.items[0].due_date is None
