"""
agent_action_tracker package
==============================
Action & Follow-up Tracker Agent.

Responsibility: classify tracked action items derived from email
conversations into overdue / in-progress / on-track buckets.

This agent does NOT fetch or parse mail itself. It reads action-item
candidates from the Email & Summary agent's already-persisted summaries
via `agent_email.service.get_latest_summaries()` - never via
`GraphClient` or `agent_email.ocr` directly - per the project's "chained
on Email agent, no duplicate mail-fetch logic" requirement.
"""
