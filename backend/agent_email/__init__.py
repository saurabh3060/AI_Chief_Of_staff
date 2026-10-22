"""
agent_email package
=====================
Email & Summary Agent.

Responsibility: fetch the most recent email conversation (thread) from
the configured mailbox, OCR any attachments (PDF pages and images) via
the local vision model, and produce a concise summary ahead of a
meeting.

This agent OWNS the OCR pipeline (`ocr.py`) for mail attachments - the
Action & Follow-up Tracker agent must consume this agent's persisted
summaries via `agent_email.service.get_latest_summaries()` rather than
re-fetching or re-parsing mail itself, per the project's "no duplicate
mail-fetch logic" requirement.
"""
