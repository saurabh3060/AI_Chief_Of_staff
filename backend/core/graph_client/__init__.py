"""
core/graph_client package
===========================
Microsoft Graph API integration used by the Calendar & Meeting agent
and the Email & Summary agent (the only two agents that talk to Graph
directly - Action Tracker chains off Email agent's output instead of
calling Graph itself, per project requirements).

Exposes a single `GraphClient` class (see `client.py`) that transparently
supports three authentication modes, controlled by `GRAPH_AUTH_MODE`:
- "delegated": reuses a bearer token obtained via the browser OAuth flow
  (`/login` + `/callback`, see the `auth/` package) and passed in on
  each request.
- "app_only": the client acquires and caches its own token via MSAL's
  client-credentials flow, requiring Application-level Graph
  permissions to be granted in Azure AD.
- "local": no Microsoft calls; mail is served from `.eml` files under
  `EML_STORAGE_DIR` (see `local_mailbox.py`) and calendar events from
  `LOCAL_CALENDAR_PATH` (see `local_calendar.py`).

Agents call `GraphClient` methods without needing to know which mode is
active.
"""

from core.graph_client.client import GraphClient

__all__ = ["GraphClient"]
