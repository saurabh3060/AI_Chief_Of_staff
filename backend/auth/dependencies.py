"""
auth/dependencies.py
=======================
FastAPI dependency that resolves the Graph API bearer token an agent
should use for an incoming request, based on the configured auth mode.

This is the single place agent routes obtain their token, replacing the
previous per-route `Authorization: Bearer <token>` header parsing. The
token now comes from the server-side session (see `auth/session.py`),
looked up via the HTTP-only session cookie set at `/auth/callback`.

Behaviour by `GRAPH_AUTH_MODE`:
- "delegated": read the session cookie, look up the session, and return
  its `access_token`. If there is no valid session, raise 401 so the
  frontend can redirect the user to `/auth/login`.
- "app_only": return `None`. `GraphClient` ignores the per-request token
  in this mode and acquires its own app token internally, so no session
  is required.
- "local": return `None`. Mail is served from local `.eml` files and no
  Microsoft session exists.
"""

from fastapi import HTTPException, Request

from auth.session import Session, get_session_store
from core.config import get_settings


def get_session_id(request: Request) -> str | None:
    """Read the opaque session id out of the configured HTTP-only cookie, if present."""
    settings = get_settings()
    return request.cookies.get(settings.session_cookie_name)


async def require_session(request: Request) -> Session | None:
    """
    Enforce that the caller presents a valid server-side session, BEFORE
    any route handler body (and therefore any DB/business-logic lookup)
    runs.

    This is the single shared authentication gate applied at the router
    level (see `main.py`'s `include_router(..., dependencies=[...])`), so
    every current and future endpoint under an authenticated router is
    protected by default - no per-endpoint copy-paste check, and no way to
    add a new route that silently skips auth.

    Behaviour mirrors `get_bearer_token` so the API's auth posture is
    identical to today for already-working deployments:

    - "delegated" (the production/UAT mode, and the one the VAPT retest
      ran against): read the `rbl_session` cookie, look it up in the
      server-side `SessionStore` (the SAME store `/callback` populates),
      and raise `HTTPException(401)` immediately if it is missing,
      unknown, or expired. The expired case is handled by
      `SessionStore.get`, which evicts and returns `None`.
    - "app_only" / "local": there is no browser session by design
      (`/auth/me` itself reports authenticated with a null user in these
      modes, and `GraphClient` manages its own token), so this is a no-op
      that returns `None` - exactly matching the pre-existing behaviour of
      `get_bearer_token` and keeping local/app-only flows unchanged.

    Returns the resolved `Session` (so handlers that want the signed-in
    user's identity for scoping can `Depends(require_session)` directly),
    or `None` in app-only/local mode.
    """
    settings = get_settings()

    if settings.graph_auth_mode in ("app_only", "local"):
        return None

    session = get_session_store().get(get_session_id(request))
    if session is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Sign in via /auth/login first.",
        )
    return session


async def get_bearer_token(request: Request) -> str | None:
    """
    Resolve the Graph API bearer token for this request.

    Use as a FastAPI dependency in agent routes:

        token = Depends(get_bearer_token)

    In delegated mode this returns the session's access token (or raises
    401 when the caller has no valid session). In app-only mode it
    returns None, since `GraphClient` manages its own token there.
    """
    settings = get_settings()

    if settings.graph_auth_mode in ("app_only", "local"):
        return None

    session = get_session_store().get(get_session_id(request))
    if session is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Sign in via /auth/login first.",
        )
    return session.access_token
