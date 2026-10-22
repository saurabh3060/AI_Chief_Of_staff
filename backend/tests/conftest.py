"""
tests/conftest.py
==================
Shared pytest fixtures for the entire backend test suite.

This file grows incrementally as later tasks introduce new components
(GraphClient mocks, OllamaClient mocks, storage fixtures, etc.) - each
addition is grouped under a clear comment banner so it's easy to see
which task introduced which fixture.
"""

import json
import os
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Task 1: config-related fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def required_env(monkeypatch):
    """
    Set the minimal set of environment variables required for
    `Settings` to construct successfully (tenant_id/client_id/client_secret
    have no defaults by design, so tests must supply them explicitly).
    Also clears the `get_settings()` cache before and after the test so
    each test gets a fresh `Settings` instance reflecting the patched env.
    """
    from core.config import get_settings

    monkeypatch.setenv("TENANT_ID", "test-tenant-id")
    monkeypatch.setenv("CLIENT_ID", "test-client-id")
    monkeypatch.setenv("CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("GRAPH_AUTH_MODE", "delegated")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Generic fixture-data loaders
# ---------------------------------------------------------------------------

@pytest.fixture
def mail_thread_fixture() -> dict:
    """Load the sample mail thread JSON used across email/action-tracker tests."""
    with open(FIXTURES_DIR / "mail_thread.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def calendar_events_fixture_factory():
    """
    Return a factory function that builds a raw Microsoft Graph API
    calendar events response, with event start/end times computed
    relative to a given "now" - avoids hardcoding calendar dates that
    would go stale over time (see agent_calendar tests).
    """
    from datetime import timedelta

    def _build(now):
        def iso(dt):
            return dt.strftime("%Y-%m-%dT%H:%M:%S.0000000")

        return {
            "value": [
                {
                    "id": "evt-past-001",
                    "subject": "Q1 Retail Review",
                    "start": {"dateTime": iso(now - timedelta(hours=3)), "timeZone": "UTC"},
                    "end": {"dateTime": iso(now - timedelta(hours=2)), "timeZone": "UTC"},
                    "organizer": {"emailAddress": {"name": "Deepak Kumar", "address": "deepak.kumar@rblbank.com"}},
                    "attendees": [],
                },
                {
                    "id": "evt-current-002",
                    "subject": "CEO x CRO Weekly Sync",
                    "start": {"dateTime": iso(now - timedelta(minutes=15)), "timeZone": "UTC"},
                    "end": {"dateTime": iso(now + timedelta(minutes=45)), "timeZone": "UTC"},
                    "organizer": {"emailAddress": {"name": "Deepak Kumar", "address": "deepak.kumar@rblbank.com"}},
                    "attendees": [
                        {"emailAddress": {"name": "CEO Office", "address": "ceo.office@rblbank.com"}}
                    ],
                },
                {
                    "id": "evt-next-003",
                    "subject": "Wholesale Banking Review",
                    "start": {"dateTime": iso(now + timedelta(hours=3)), "timeZone": "UTC"},
                    "end": {"dateTime": iso(now + timedelta(hours=4)), "timeZone": "UTC"},
                    "organizer": {"emailAddress": {"name": "Wholesale Head", "address": "wholesale.head@rblbank.com"}},
                    "attendees": [],
                },
            ]
        }

    return _build


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """
    Generate a minimal single-page PDF in memory (via Pillow) containing
    a line of text, so OCR-pipeline tests have something to convert to an
    image without needing to commit a binary fixture file to the repo.
    """
    from io import BytesIO

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (600, 200), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 80), "RBL Bank - Sample Scanned Document", fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PDF")
    return buffer.getvalue()


@pytest.fixture
def pdf_attachment_fixture_factory(sample_pdf_bytes):
    """
    Return a factory function that builds a raw Microsoft Graph API
    attachment object (as returned by GraphClient.get_attachment_content)
    wrapping the sample single-page PDF, base64-encoded as contentBytes.
    """
    import base64

    def _build(attachment_id: str = "att-001", name: str = "branch_audit_report.pdf"):
        return {
            "id": attachment_id,
            "name": name,
            "contentType": "application/pdf",
            "contentBytes": base64.b64encode(sample_pdf_bytes).decode("utf-8"),
        }

    return _build
