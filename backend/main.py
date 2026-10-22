"""
main.py
=======
Application entrypoint for the RBL Bank Digital Office backend.

Responsible for:
- Creating the FastAPI app instance.
- Wiring up startup/shutdown hooks (logging setup).
- Registering every agent's router plus the auth and chatbot routers.
- Exposing a lightweight `/health` endpoint used for smoke checks.

Run locally with:
    uvicorn main:app --host 0.0.0.0 --port 8516
"""
# Test change: offline Git sync workflow

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from auth.dependencies import require_session
from auth.session import get_session_store

from agent_action_tracker.routes import router as action_tracker_router
from agent_open_tracker.routes import router as open_tracker_router
from agent_calendar.routes import router as calendar_router
from agent_calendar.routes import notes_router as meeting_notes_router
from agent_email.routes import premeeting_brief_router
from agent_email.routes import router as email_router
from agent_mis.routes import router as mis_router
from auth.routes import callback_router as auth_callback_router
from auth.routes import router as auth_router
from chatbot.routes import router as chatbot_router
from documents.sync_routes import router as documents_sync_router
from eml_summarizer.routes import router as eml_router
from core.config import get_settings
from core.logging_config import get_agent_logger, setup_logging

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan hook. Runs once at startup (before requests are
    accepted) and once at shutdown. Used to initialise the centralized
    logging system and build/load the chatbot Weaviate index.
    """
    setup_logging()
    logger = get_agent_logger("app")
    logger.info("Application startup at: %s (env=%s)", settings.app_name, settings.app_env)

    # Initialize PostgreSQL connection pool (no-op if POSTGRES_DSN is empty).
    from core.db import init_db, close_db
    await init_db()

    # Three-layer storage self-healing: when the email backend is PostgreSQL,
    # Weaviate is a rebuildable copy of PostgreSQL. At startup ensure every
    # owned collection exists (create-if-missing, never delete), log the
    # REST/gRPC hosts + node count + per-collection object counts, and
    # rebuild from PostgreSQL's STORED embeddings any collection that is
    # missing/empty while PostgreSQL has rows (no embedding/LLM calls).
    if getattr(settings, "email_backend_is_postgres", False):
        try:
            from agent_email import vector_store_backend as wvs
            from agent_email import postgres_store
            # Pre-start readiness gate: wait until every owned collection is
            # created AND actually queryable over gRPC, retrying through any
            # transient REST/gRPC schema desync, BEFORE serving reads. This
            # is what prevents the "could not find class" thrash at startup.
            readiness = await asyncio.to_thread(wvs.ensure_schema_ready, settings)
            not_ready = [n for n, r in readiness.items() if not r.get("ready")]
            if not_ready:
                logger.warning(
                    "[WEAVIATE] startup: collection(s) not yet queryable %s "
                    "(continuing; background monitor will retry)", not_ready
                )
            wvs.startup_report(settings)
            # Rebuild from PostgreSQL only what is genuinely missing/empty
            # (a transient desync is skipped, not rebuilt — see weaviate_sync).
            restored = await postgres_store.ensure_weaviate_consistent(settings)
            if restored:
                logger.info("[RESTORE] startup rebuilt namespaces: %s", restored)
        except Exception as exc:  # noqa: BLE001
            # Weaviate being unreachable must not crash startup: PostgreSQL
            # (the system of record) still serves reads. Log and continue;
            # the before-write / background checks will retry.
            logger.error("Weaviate startup check failed (continuing): %s", exc)

        # Redis is a disposable read cache. Ping it (log connected true/
        # false) and refill every key from PostgreSQL/Weaviate. No LLM or
        # embedding calls happen here — refill only copies persisted
        # records. A Redis outage never blocks startup.
        try:
            from core.redis.client import ping as redis_ping
            from agent_email import redis_cache
            connected = await redis_ping(settings)
            logger.info("[REDIS] startup ping connected=%s", connected)
            if connected:
                # Refill every cache key from the system of record so the
                # cache starts consistent (no stale/half-populated keys).
                # warm_derived defaults to True here, so a cold/cleared Redis
                # at server startup also repopulates the derived pre-meeting
                # brief cache (meeting_clusters:*) — not just the input
                # snapshots — so the UI serves briefs from cache immediately
                # instead of recomputing against Weaviate on the first click.
                await redis_cache.refill_from_source(settings)
            else:
                # Redis is a disposable cache: a miss falls through to
                # PostgreSQL/Weaviate, so an outage degrades to slower reads,
                # never wrong reads. Do not block startup on it.
                logger.warning(
                    "[REDIS] startup: cache unavailable — reads will fall through "
                    "to the system of record (continuing)"
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Redis startup check failed (continuing): %s", exc)

    # Initialize the chatbot Weaviate index
    from chatbot.routes import initialize_index
    await initialize_index()

    if settings.graph_auth_mode == "local":
        from core.graph_client.local_calendar import ensure_local_calendar_exists, load_local_calendar_events
        calendar_path = ensure_local_calendar_exists(settings)
        logger.info("Local calendar ready at %s", calendar_path)
        from agent_email.service import precompute_meeting_embeddings
        await precompute_meeting_embeddings(load_local_calendar_events(settings), settings)

    # The email cluster pipeline is INTENTIONALLY NOT started here. The
    # backend is now read-only with respect to the pipeline: it only serves
    # whatever cluster/topic records already exist in PostgreSQL (via the
    # read routes). Producing those records is the job of a separate,
    # explicitly-invoked process - `backend/run_pipeline.py` - run on demand
    # or by an external scheduler (cron/systemd timer/CI). This guarantees a
    # server restart (or an accidental page refresh/login) never kicks off a
    # mailbox fetch + LLM clustering pass. See run_pipeline.py for how it
    # authenticates to Graph using the delegated tokens persisted at
    # /auth/callback (auth/token_store).

    # Self-heal the derived pre-meeting brief cache on startup. The
    # expensive work (clustering + summarization + embeddings) is already
    # durable in PostgreSQL; the only thing an accidental restart loses is
    # the derived meeting_clusters:* cache (it is invalidated when meeting
    # embeddings are restored just above, and is never refilled by
    # refill_from_source). Rebuilding it is pure MATCHING over existing data
    # — NO re-clustering, NO re-summarization, NO mailbox fetch — so the
    # briefs are ready before the first request instead of coming back empty.
    # Non-fatal and only meaningful on the PostgreSQL email backend.
    if getattr(settings, "email_backend_is_postgres", False):
        try:
            from agent_email.service import prepare_stored_meeting_matches
            prepared = await prepare_stored_meeting_matches(settings)
            logger.info("[STARTUP] rebuilt pre-meeting brief cache for %d meeting(s)", prepared)
        except Exception as exc:  # noqa: BLE001
            # A failure here only means briefs recompute lazily on first
            # click (the read path self-heals on a cache miss), so never
            # block startup on it.
            logger.error("[STARTUP] pre-meeting brief cache rebuild failed (continuing): %s", exc)

    # Optional light storage consistency monitor (every 5 min). Only runs
    # on the PostgreSQL email backend; the single extra timer besides the
    # pipeline cron above.
    from agent_email.health_monitor import start_storage_health_monitor, stop_storage_health_monitor
    start_storage_health_monitor()

    # Start the daily OneDrive -> document library sync scheduler
    # (no-op if ONEDRIVE_SYNC_ENABLED=false in .env).
    from documents.sync_scheduler import start_sync_scheduler, stop_sync_scheduler
    start_sync_scheduler()

    yield
    await stop_sync_scheduler()
    await stop_storage_health_monitor()
    await close_db()

    # Close the shared Weaviate connection, if one was opened (only
    # happens when a PostgreSQL DSN is configured). No-op otherwise.
    from core.weaviate.client import close_client
    close_client()

    # Close the Redis cache connection, if one was opened. No-op otherwise.
    from core.redis.client import close_redis
    await close_redis()

    # Release any Ollama HTTP resources (per-call ChatOllama sockets are
    # already closed after each call; this is the final sweep). Keeps
    # shutdown free of the ":11434" unclosed-socket ResourceWarning.
    from core.ollama_client.client import close_ollama
    await close_ollama()

    logger.info("Application shutdown")


app = FastAPI(
    title=settings.app_name,
    description=(
        "Multi-agent backend powering the CEO's Digital Office.\n\n"
        "**Agents:**\n"
        "- **Calendar** — current/next meeting from Outlook via Microsoft Graph API\n"
        "- **Email** — summarise inbox threads, extract action candidates\n"
        "- **Action Tracker** — classify & track action items from email summaries\n"
        "- **MIS** — MIS insights (stub, not yet implemented)\n"
        "- **Chatbot** — natural-language Q&A grounded in agent data\n\n"
        "**Auth:** Delegated Microsoft OAuth or app-only Graph API access."
    ),
    version="1.0.0",
    docs_url=None,  # Disable default docs, we'll add custom one
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Trust nginx's X-Forwarded-Proto/X-Forwarded-For/Host headers so
# `request.base_url` and `request.url.scheme` reflect the public-facing
# `https://digitaoffice-dev.rbl.bank.in` origin instead of uvicorn's own
# `http://<internal-ip>:8516` bind address. Without this, the same-origin
# check in `_spa_or_landing()` below always sees `frontend != request_origin`
# (scheme AND host both differ) and redirects `/` -> FRONTEND_URL -> `/`
# forever - this was the actual cause of the reported redirect loop.
#
# `trusted_hosts="*"` is safe here because nginx is the sole process that
# talks to this backend directly (it's not exposed publicly on its own
# port); if that topology changes, tighten this to nginx's specific
# address instead of the wildcard.
#
# Placement relative to CORSMiddleware doesn't matter for correctness -
# every `add_middleware` call wraps the router, so the corrected ASGI
# scope (scheme/host/client) reaches route handlers regardless of add
# order between the two.
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# CORS: same-origin in this deployment (APP_BASE_URL == FRONTEND_URL), but
# kept as an explicit allow-list (not "*") so credentialed requests still
# work if a future deployment splits the SPA onto its own origin. The
# primary allowed origin is FRONTEND_URL; common local dev origins are
# added for convenience.
_allowed_origins = list(
    dict.fromkeys(
        [
            settings.frontend_url,
            "http://localhost:8518",
            "http://127.0.0.1:8518",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://10.240.87.68:3000",
            "http://10.240.87.68:5173",
        ]
    )
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health_check() -> dict:
    """
    Simple liveness endpoint. Returns process status only - intentionally
    does not touch Graph API, Ollama, or storage so it stays fast and
    dependency-free for uptime probes.
    """
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


@app.get("/health/storage", tags=["health"])
async def storage_health() -> JSONResponse:
    """
    Per-store health for the three-layer storage (PostgreSQL system of
    record, Weaviate searchable copy, Redis disposable cache).

    Reports connection status, collection existence, and object counts.
    Deliberately exposes NO host names, ports, DSNs, or credentials — only
    booleans and counts — so it is safe to surface publicly.

    When the email backend is not PostgreSQL, the three-layer stores are
    not in use and this reports ``mode="csv"`` with the stores marked not
    applicable. When ALL stores are down while PostgreSQL is expected,
    responds 503 (never pretend the data is available).
    """
    report: dict = {
        "mode": "postgres" if settings.email_backend_is_postgres else "disabled",
        "postgres": {"configured": bool((settings.postgres_dsn or "").strip()), "connected": False},
        "weaviate": {"connected": False, "collections": {}},
        "redis": {"enabled": settings.redis_enabled, "connected": False},
    }

    if not settings.email_backend_is_postgres:
        report["status"] = "ok"
        report["detail"] = "three-layer storage inactive (POSTGRES_DSN not set)"
        return JSONResponse(report)

    # --- PostgreSQL (system of record) ---
    pg_ok = False
    try:
        from core.db import get_pool
        pool = get_pool()
        if pool is not None:
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1")
                report["postgres"]["rows"] = {
                    "email_clusters": int(await conn.fetchval("SELECT count(*) FROM email_clusters") or 0),
                    "meeting_embeddings": int(await conn.fetchval("SELECT count(*) FROM meeting_embeddings") or 0),
                    "meeting_history": int(await conn.fetchval("SELECT count(*) FROM meeting_history") or 0),
                }
            pg_ok = True
    except Exception as exc:  # noqa: BLE001
        report["postgres"]["error"] = type(exc).__name__
    report["postgres"]["connected"] = pg_ok

    # --- Weaviate (searchable copy) ---
    wv_ok = False
    try:
        from agent_email import vector_store_backend as wvs
        rep = await asyncio.to_thread(wvs.startup_report, settings)
        report["weaviate"]["nodes"] = rep.get("nodes", 0)
        for namespace, info in rep.get("collections", {}).items():
            # Do not leak the actual collection name/prefix either; key by
            # the logical namespace only.
            report["weaviate"]["collections"][namespace] = {
                "exists": bool(info.get("exists")),
                "objects": int(info.get("objects", 0)),
            }
        wv_ok = True
    except Exception as exc:  # noqa: BLE001
        report["weaviate"]["error"] = type(exc).__name__
    report["weaviate"]["connected"] = wv_ok

    # --- Redis (disposable cache) ---
    redis_ok = False
    try:
        from core.redis.client import ping as redis_ping
        redis_ok = await redis_ping(settings)
    except Exception as exc:  # noqa: BLE001
        report["redis"]["error"] = type(exc).__name__
    report["redis"]["connected"] = redis_ok

    # All stores down while PostgreSQL is expected => explicit failure, never
    # a misleading "ok". Redis is disposable so it does not count against
    # availability; PostgreSQL or Weaviate being up is enough to serve reads.
    if not pg_ok and not wv_ok:
        report["status"] = "unavailable"
        return JSONResponse(report, status_code=503)

    report["status"] = "ok" if (pg_ok and wv_ok) else "degraded"
    return JSONResponse(report)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    Custom Swagger UI endpoint that doesn't rely on CDN.
    Uses swagger-ui-dist package bundled with FastAPI.
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_favicon_url="/static/favicon.ico" if hasattr(app, 'mount') else None,
    )


@app.get("/", include_in_schema=False)
def root(request: Request):
    """
    Entry point for the public domain (the load balancer routes the whole
    domain here).

    This must NOT be a blind server-side redirect: because FRONTEND_URL
    lives on this same domain, a `/` -> `/auth/login` -> `/callback` ->
    `/` redirect chain loops forever whenever the session cookie isn't
    read back (e.g. a proxy dropping Set-Cookie on a 307). Instead we
    return a tiny HTML bootstrap that checks the session *client-side*
    via `/auth/me` and then either sends the user into the SPA or into
    the sign-in flow - exactly once, with no server-side loop.
    """
    if settings.graph_auth_mode in ("app_only", "local"):
        return _spa_or_landing(request)

    # Already authenticated (cookie present and valid) -> into the app.
    session = get_session_store().get(request.cookies.get(settings.session_cookie_name))
    if session is not None:
        return _spa_or_landing(request)

    # Not authenticated -> start sign-in. Guard against a redirect loop:
    # if we *just* came back from the callback (the browser followed the
    # post-login redirect here) but still have no session, the cookie is
    # not surviving the round trip - surface that instead of looping.
    if request.query_params.get("post_login") == "1":
        return HTMLResponse(
            "<h3>Sign-in succeeded but the session cookie did not persist.</h3>"
            "<p>This usually means a proxy/load balancer is stripping the "
            "<code>Set-Cookie</code> header, or the cookie's Secure/SameSite "
            "flags don't match the browser-facing scheme. Check the backend "
            "SESSION_COOKIE_* settings and the LB configuration.</p>",
            status_code=500,
        )
    return RedirectResponse("/auth/login")


def _spa_or_landing(request: Request):
    """
    Hand an authenticated browser off to the SPA - without looping.

    If FRONTEND_URL points at a *different* origin than this request, we
    redirect there. If it points back at this same origin's root (the
    common single-domain setup where the LB fronts everything), a
    redirect would loop `/` -> `/` forever, so we serve a minimal landing
    page instead. Replace this page's body with the real SPA once its
    serving location is finalized.
    """
    frontend = settings.frontend_url.rstrip("/")
    request_origin = str(request.base_url).rstrip("/")

    # Different origin -> safe to redirect to the SPA.
    if frontend and frontend != request_origin:
        return RedirectResponse(settings.frontend_url)

    # Same origin as `/` -> don't redirect (would loop). Serve a landing.
    return HTMLResponse(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>RBL Digital Office</title></head>"
        "<body style='font-family:system-ui;padding:2rem'>"
        "<h2>Signed in.</h2>"
        "<p>You are authenticated. The application UI loads here.</p>"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# Router registration and the single authentication gate.
#
# Every authenticated router below is mounted with
# `dependencies=[Depends(require_session)]`. FastAPI evaluates a router's
# dependencies BEFORE the matched endpoint's own parameters/body and before
# its handler body runs, so this one line protects every current AND future
# endpoint on that router: a missing/invalid/expired `rbl_session` cookie is
# rejected with 401 before any DB lookup or business logic executes (this is
# what fixes the "fake itemId returns 404 instead of 401" finding - the auth
# dependency now runs ahead of the itemId lookup). Adding a new route to any
# of these routers inherits the gate automatically; there is no per-endpoint
# check to forget.
#
# `require_session` is a no-op in app_only/local mode (see auth/dependencies),
# so this does NOT change behaviour for app-only deployments or the smoke
# suite - it only enforces in delegated mode, which is the deployed posture
# the VAPT retest targeted.
#
# PUBLIC routes are deliberately NOT gated and are listed explicitly:
#   - GET /health, GET /health/storage      (app-level, defined above)
#   - GET /, GET /docs, GET /redoc, /openapi (app-level, defined above)
#   - the entire auth router (/auth/login, /auth/me, /auth/logout,
#     /auth/config) and the prefix-less /callback - these MUST stay reachable
#     pre-login or the sign-in flow itself breaks.
# ---------------------------------------------------------------------------

# Delegated OAuth routes (/auth/login, /auth/me, /auth/logout) plus the
# prefix-less /callback the Azure app registration redirects to. Always
# registered regardless of GRAPH_AUTH_MODE so the API surface stays
# stable, but only functionally required when GRAPH_AUTH_MODE=delegated.
# PUBLIC by design - must be reachable before a session exists.
app.include_router(auth_router)
app.include_router(auth_callback_router)

# Authenticated routers: a valid server-side session is required (delegated
# mode). The gate is applied here, at include time, not inside each handler.
_authenticated = [Depends(require_session)]

# Calendar & Meeting agent routes.
app.include_router(calendar_router, dependencies=_authenticated)

# Meeting notes routes (POST /agents/meeting-notes, GET /agents/meeting-notes/{id}).
# Persists CEO MoM notes to Postgres so cluster synthesis can read them back
# by Graph meeting_id.
app.include_router(meeting_notes_router, dependencies=_authenticated)

# Email & Summary agent routes.
app.include_router(email_router, dependencies=_authenticated)

# Action & Follow-up Tracker agent routes.
app.include_router(action_tracker_router, dependencies=_authenticated)

# Open Actions Status Tracker routes (Launchpad card — reads cluster records from the database).
app.include_router(open_tracker_router, dependencies=_authenticated)

# MIS Insight agent routes (stub - returns 501, not yet implemented).
app.include_router(mis_router, dependencies=_authenticated)

# Pre-Meeting Brief routes (served by the Email agent; folded in from
# the former standalone agent_premeeting_brief package).
app.include_router(premeeting_brief_router, dependencies=_authenticated)

# Chatbot agent routes (vector-search RAG over indexed .eml files).
app.include_router(chatbot_router, dependencies=_authenticated)

# EML file summariser routes (list files, summarise by name).
app.include_router(eml_router, dependencies=_authenticated)
