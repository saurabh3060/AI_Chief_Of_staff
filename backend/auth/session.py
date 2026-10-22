"""
auth/session.py
=================
Server-side session store for the delegated Microsoft OAuth flow.

The browser never sees the Graph API access/id tokens. Instead, after a
successful `/auth/callback`, the tokens are stored here keyed by an
opaque, cryptographically-random session id, and only that id is handed
to the browser inside an HTTP-only cookie. Every subsequent agent
request presents the cookie; the backend looks the session up here and
uses the stored access token to call Microsoft Graph API.

This is an in-process, memory-backed store (a plain dict guarded by a
lock), which matches the project's default `STORAGE_BACKEND=memory`
posture and needs no external dependency. It is intentionally small and
self-contained so it can later be swapped for a shared store (Redis /
Postgres) without touching the routes or the auth dependency - only
`get_session_store()` would change.

NOTE: because sessions live in process memory, they are lost on restart
and are not shared across multiple worker processes. For a multi-worker
or multi-instance deployment, back this with Redis (see the TODO in
`get_session_store`).

IMPORTANT - this is a real, currently-live constraint, not a hypothetical:
if uvicorn is ever started with `--workers N` (N > 1) for this app, each
worker gets its own process and therefore its own empty `SessionStore`.
A browser's requests are load-balanced across workers, so `/callback` may
create a session in worker A while a later `/auth/me` or agent call lands
on worker B, which has never seen that session id - the cookie is valid
and present, but `SessionStore.get()` returns `None`, producing a 401
that looks identical to "cookie didn't survive the round trip" and can
retrigger the login redirect. The same happens across a process restart
(deploy, crash, autoscale) even with a single worker. Do not scale workers
or run multiple instances of this backend until sessions are moved to a
shared store (Redis) - `get_session_store()` is the only call site that
would need to change.
"""

import secrets
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class Session:
    """
    A single authenticated user session.

    - `access_token` / `id_token`: the delegated Graph tokens obtained at
      `/auth/callback`. `access_token` is what agents send to Graph API.
    - `expires_at`: monotonic-independent wall-clock epoch (seconds) after
      which the session is considered stale and treated as absent.
    - `data`: free-form metadata (e.g. the signed-in user's name/email)
      that routes may populate for display; never required for auth.
    """

    access_token: str
    id_token: str | None
    expires_at: float
    data: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at


class SessionStore:
    """
    Thread-safe in-memory session store.

    Session ids are generated with `secrets.token_urlsafe`, giving 256
    bits of entropy by default - not guessable, safe to place in a cookie.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(
        self,
        access_token: str,
        id_token: str | None,
        ttl_seconds: int,
        data: dict | None = None,
    ) -> str:
        """Create a new session and return its opaque id (to be set as a cookie)."""
        session_id = secrets.token_urlsafe(32)
        session = Session(
            access_token=access_token,
            id_token=id_token,
            expires_at=time.time() + ttl_seconds,
            data=data or {},
        )
        with self._lock:
            self._sessions[session_id] = session
        return session_id

    def get(self, session_id: str | None) -> Session | None:
        """
        Return the session for `session_id`, or None if it is missing or
        expired. Expired sessions are evicted on access (lazy cleanup).
        """
        if not session_id:
            return None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired:
                del self._sessions[session_id]
                return None
            return session

    def delete(self, session_id: str | None) -> None:
        """Remove a session (used by `/auth/logout`). No-op if it doesn't exist."""
        if not session_id:
            return
        with self._lock:
            self._sessions.pop(session_id, None)


@lru_cache
def get_session_store() -> SessionStore:
    """
    Return the process-wide singleton session store.

    TODO: to support multiple workers/instances, replace the returned
    object with a Redis-backed implementation exposing the same
    create/get/delete interface. Nothing else in the auth layer needs to
    change.
    """
    return SessionStore()
