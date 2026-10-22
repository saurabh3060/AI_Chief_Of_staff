"""
auth/routes.py
================
Delegated Microsoft OAuth flow with a server-side session.

Flow (frontend and backend on different origins):
1. Browser hits the app entry point, which is proxied to this backend
   on port 8516.
2. `/auth/login` redirects the browser to Microsoft's sign-in page.
3. Microsoft redirects back to `/callback` (the redirect URI registered
   in Azure) with an authorization code.
4. `/callback` exchanges the code for Graph tokens **server-to-
   server**, stores them in the server-side session store keyed by an
   opaque session id, sets that id as an HTTP-only cookie, and then
   redirects the browser to the frontend SPA (`FRONTEND_URL`).
5. From then on the SPA calls agent endpoints with `credentials:
   "include"`; the cookie rides along, the backend resolves the session
   to the stored access token (see `auth/dependencies.py`), and the
   agents can reach Graph API.

The Graph access/id tokens are therefore never exposed to the browser
or placed in a URL - only the opaque session id is, inside an HTTP-only
cookie.

These routes are only meaningful when `GRAPH_AUTH_MODE=delegated`. They
remain registered even in `app_only` mode so the API surface doesn't
change based on configuration.
"""

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import asyncio

from auth.dependencies import get_session_id
from auth.session import get_session_store
from auth.token_store import save_tokens
from core.config import Settings, get_settings
from core.graph_client.auth import build_authorization_url, exchange_code_for_token
from core.logging_config import get_agent_logger

logger = get_agent_logger("auth")

router = APIRouter(prefix="/auth", tags=["auth"])

# The OAuth redirect URI registered in Azure is `{APP_BASE_URL}/callback`
# (no `/auth` prefix), so the callback handler is mounted on this separate
# prefix-less router while login/me/logout stay under `/auth`.
callback_router = APIRouter(tags=["auth"])


def _redirect_uri() -> str:
    """
    Build the redirect URI registered on the Azure app registration.

    This MUST match the value configured in Azure exactly. The app
    registration uses `{APP_BASE_URL}/callback` (not `/auth/callback`),
    so the callback route below is also mounted at `/callback`.
    """
    settings = get_settings()
    return f"{settings.app_base_url}/callback"


def _set_session_cookie(response: Response, settings: Settings, session_id: str) -> None:
    """
    Attach the HTTP-only session cookie to a response.

    `httponly=True` keeps the id out of reach of page JavaScript (XSS
    hardening). `secure`/`samesite` come from config so a cross-origin
    HTTPS deployment can use `samesite=none; secure` while local http
    development uses `samesite=lax`.
    """
    response.set_cookie(
        key=settings.session_cookie_name,
        value=session_id,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite=settings.session_cookie_samesite,
        path="/",
    )


# OAuth `prompt` values we're willing to forward to Microsoft. Anything
# else (including an empty string or junk) is ignored so we fall back to
# the default silent-SSO behaviour. Crucially, `prompt=login` is NOT
# included: forcing a full re-authentication on every app load makes the
# pre-login wait longer (it defeats existing SSO and can re-trigger MFA),
# which is exactly the latency we're trying to remove. It stays opt-in.
_ALLOWED_PROMPTS = {"login", "select_account", "consent", "none"}


@router.get("/login")
def login(prompt: str | None = None):
    """
    Redirect the browser to Microsoft's authorize endpoint to begin the
    delegated sign-in flow.

    By default NO `prompt` parameter is sent, so Microsoft silently
    reuses an existing SSO session when one is available - the fastest
    path to the authorize/redirect. A `prompt` is only forwarded when the
    caller explicitly passes a recognised value (`?prompt=select_account`,
    etc.); an unrecognised or empty value is dropped rather than blindly
    forwarded. In particular the app never *defaults* to `prompt=login`,
    which would force a slow full re-auth on every load.
    """
    settings = get_settings()

    if settings.is_local_mail_mode:
        logger.info("Local mail mode: skipping Microsoft login and sending the browser to the SPA")
        return RedirectResponse(settings.frontend_url)

    normalized_prompt = (prompt or "").strip().lower() or None
    if normalized_prompt is not None and normalized_prompt not in _ALLOWED_PROMPTS:
        # Strip CR/LF from the user-controlled prompt before logging to
        # prevent log injection / forged log lines (CWE-117).
        sanitized_item_id = str(prompt).replace("\r\n", "").replace("\r", "").replace("\n", "")
        logger.warning("Ignoring unrecognised prompt value: %r", sanitized_item_id)
        normalized_prompt = None

    auth_url = build_authorization_url(
        settings, redirect_uri=_redirect_uri(), prompt=normalized_prompt
    )

    # Strip CR/LF from the prompt value before logging to prevent log
    # injection / forged log lines (CWE-117).
    item_normalized_prompt = str(normalized_prompt).replace("\r\n", "").replace("\r", "").replace("\n", "")

    logger.info("Redirecting to Microsoft authorize endpoint (prompt=%s)", item_normalized_prompt)
    return RedirectResponse(auth_url)


@callback_router.get("/callback")
async def callback(request: Request):
    """
    Handle Microsoft's redirect back to this app after sign-in.

    Exchanges the authorization code for Graph tokens via a direct
    server-to-server call, stores them in a new server-side session,
    sets the session-id cookie, and redirects the browser back to the
    frontend SPA. On error, redirects to the frontend with an
    `?auth_error=...` query param so the SPA can surface it.

    This handler is `async def` on purpose. The delegated token persist
    below (`save_tokens`) uses the shared asyncpg pool, whose connections
    are bound to the event loop that created them (uvicorn's main loop in
    `init_db`). Running the handler on that same loop lets the DB write use
    the pool directly. A previous sync implementation drove the write with
    `asyncio.run(...)`, which spun up a *second* event loop in a worker
    thread and used a pool connection across loops - the source of the
    "another operation is in progress" / "connection was closed in the
    middle of operation" errors that left the pipeline runner with no
    stored token.
    """
    settings = get_settings()

    code = request.query_params.get("code")
    if not code:
        error = request.query_params.get("error_description", "login_failed")
        # Strip CR/LF before logging to prevent log injection (CWE-117).
        sanitized_error = str(error).replace("\r\n", "").replace("\r", "").replace("\n", "")
        logger.warning(
            "OAuth callback received without an authorization code: %s",
            sanitized_error,
        )
        return RedirectResponse(f"{settings.frontend_url}?auth_error={error}")

    # `exchange_code_for_token` is a blocking MSAL/HTTP call; run it in a
    # worker thread so it doesn't stall the event loop that serves every
    # other request (and that the pool write below depends on).
    result = await asyncio.to_thread(
        exchange_code_for_token, settings, code, redirect_uri=_redirect_uri()
    )

    if "error" in result:
        error = result.get("error_description", "token_error")
        # Strip CR/LF before logging to prevent log injection (CWE-117).
        sanitized_error = str(error).replace("\r\n", "").replace("\r", "").replace("\n", "")
        logger.warning("Token exchange failed: %s", sanitized_error)
        return RedirectResponse(f"{settings.frontend_url}?auth_error={error}")

    # Prefer Microsoft's own expires_in for the session TTL so the session
    # never outlives the access token, falling back to the configured TTL.
    ttl = int(result.get("expires_in") or settings.session_ttl_seconds)
    claims = result.get("id_token_claims") or {}
    session_id = get_session_store().create(
        access_token=result["access_token"],
        id_token=result.get("id_token"),
        ttl_seconds=ttl,
        data={
            "name": claims.get("name"),
            "email": claims.get("preferred_username") or claims.get("email"),
        },
    )

    # IMPORTANT: set the cookie on a 200 HTML response, NOT on a 3xx
    # redirect. Load balancers commonly strip `Set-Cookie` from redirect
    # responses (and browsers are inconsistent about storing cookies from
    # redirects), which is exactly what caused the "redirected too many
    # times" loop: the cookie set on the /callback 307 never made it back
    # to the app. A 200 response keeps the header intact; the browser
    # stores the cookie, THEN this page navigates to the SPA client-side,
    # so the session cookie is already present on the SPA's /auth/me call.
    #
    # Destination carries `post_login=1` so that if the cookie still
    # doesn't survive the round trip (e.g. a browser privacy setting, or
    # a proxy in front of nginx stripping Set-Cookie), `root()`'s
    # loop-breaker in main.py can detect it and surface a diagnostic page
    # instead of silently redirecting back into another login attempt.
    #
    # Destination is the SPA (FRONTEND_URL). In cross-origin deployments
    # this is a different host than the backend; the SPA then verifies
    # the session via /auth/me with credentials: "include".

    # Durably persist the delegated credentials so the EXTERNAL clustering
    # pipeline runner (backend/run_pipeline.py) can authenticate to Graph
    # long after this request - and across backend restarts - by refreshing
    # the access token from the stored refresh token. The refresh token is
    # written to AWS Secrets Manager (the single source of truth); the
    # short-lived access token is cached in PostgreSQL. This is the single
    # sign-in that bootstraps every later unattended pipeline run.
    # Best-effort: a Secrets Manager / Postgres hiccup here must not break
    # the user's login, so failures are logged (never the token) and
    # swallowed. The handler runs on the main event loop (async def), so we
    # await the write directly - the asyncpg pool connection is used on the
    # loop that owns it, avoiding the cross-loop corruption `asyncio.run`
    # caused.
    try:
        await save_tokens(
            access_token=result["access_token"],
            refresh_token=result.get("refresh_token"),
            expires_in=ttl,
            user_email=(claims.get("preferred_username") or claims.get("email")),
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist delegated tokens for the pipeline runner: %s", exc)

    # NOTE: signing in never triggers the BU email pipeline. The backend no
    # longer runs the pipeline at all - it is produced out-of-process by the
    # external runner `backend/run_pipeline.py` (run on demand or by an
    # external scheduler), which uses the tokens persisted just above. This
    # login's only pipeline-related job is to capture/refresh those tokens.
    # The SPA's read of GET /agents/email/bu/records simply returns whatever
    # cluster records already exist in PostgreSQL.

    logger.info("Delegated session created; returning cookie-setting landing page")
    sep = "&" if "?" in settings.frontend_url else "?"
    destination = f"{settings.frontend_url}{sep}post_login=1"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Signing you in...</title>
<meta name="referrer" content="no-referrer">
<script>window.location.replace({destination!r});</script>
</head>
<body style="font-family:system-ui;padding:2rem">
Signing you in&hellip; If you are not redirected,
<a href="{destination}">click here</a>.
</body></html>"""
    response = HTMLResponse(content=html)
    _set_session_cookie(response, settings, session_id)
    return response


@router.get("/me")
def me(request: Request):
    """
    Report the current authentication state for the calling browser.

    The SPA calls this on load (with `credentials: "include"`) to decide
    whether to show the app or bounce the user to `/auth/login`. In
    app-only mode there is no user session, so it reports authenticated
    with a null user (agents work without a browser sign-in).
    """
    settings = get_settings()

    if settings.graph_auth_mode == "app_only":
        return {"authenticated": True, "mode": "app_only", "user": None}

    if settings.is_local_mail_mode:
        return {
            "authenticated": True,
            "mode": "local",
            "user": {"name": "Local Dev", "email": "local@localhost"},
        }

    session = get_session_store().get(get_session_id(request))
    if session is None:
        return JSONResponse(
            {"authenticated": False, "mode": "delegated", "user": None},
            status_code=401,
        )
    return {"authenticated": True, "mode": "delegated", "user": session.data}


@router.post("/logout")
def logout(request: Request):
    """Invalidate the current session and clear the session cookie."""
    settings = get_settings()
    get_session_store().delete(get_session_id(request))
    response = JSONResponse({"ok": True})
    response.delete_cookie(key=settings.session_cookie_name, path="/")
    return response


@router.get("/config")
def config():
    """
    Expose frontend configuration (polling intervals, etc.) to the SPA.
    
    Returns non-sensitive configuration values that the frontend needs
    to function correctly, such as polling intervals for real-time updates.
    """
    settings = get_settings()
    return {
        "calendar_polling_interval_seconds": settings.calendar_polling_interval_seconds,
    }
