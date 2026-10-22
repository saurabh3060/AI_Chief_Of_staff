"""
core/storage package
======================
Repository-pattern storage abstraction shared by every agent.

Agents never talk to a database (or an in-memory dict) directly - they
call `core.storage.factory.get_action_repository()` /
`get_brief_repository()` / `get_summary_repository()` and work only
against the abstract interfaces defined in `base.py`. Swapping the
backend from in-memory to Postgres later (via the `STORAGE_BACKEND`
env var) requires zero changes in agent code.
"""
