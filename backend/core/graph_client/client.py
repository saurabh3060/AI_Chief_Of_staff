"""
core/graph_client/client.py
==============================
`GraphClient` - the single entry point agents use to talk to Microsoft
Graph API. Wraps authentication (delegated or app-only, chosen via
`GRAPH_AUTH_MODE`), retry/backoff, concurrency limiting, and logging
behind a small set of async methods:

- `get_current_user(bearer_token)` - delegated-mode only.
- `get_messages(bearer_token=None, top=10, skip=0)`
- `get_message_detail(message_id, bearer_token=None)`
- `get_attachments(message_id, bearer_token=None)`
- `get_calendar_events(bearer_token=None, top=50)`

In "delegated" mode, callers must supply the bearer token obtained via
the `/login`/`/callback` flow (see `auth/routes.py`). In "app_only"
mode, the client acquires and manages its own token internally and the
`bearer_token` argument is ignored. In "local" mode, mail is read from
`.eml` files under `EML_STORAGE_DIR` and calendar events come from
`LOCAL_CALENDAR_PATH` (`backend/data/local_calendar.json`).

All outbound HTTP calls go through `httpx.AsyncClient` (chosen over
`requests` for native async support, since agents run inside an async
LangGraph/FastAPI process), wrapped with `core.retry.with_retry` for
exponential backoff and limited by `core.concurrency.get_graph_semaphore`
so we never exceed `MAX_CONCURRENT_GRAPH_CALLS` simultaneous requests.

**SSRF mitigation (CWE-918)**

All Graph URL-safety logic lives in ONE module,
``core.graph_client.url_safety`` (the ``graph_url`` module is a thin
backwards-compatible shim over it). Defence is layered:

1. Origin is config-pinned. The scheme/host/port every request can reach
   comes solely from ``settings.graph_base_url`` (env ``GRAPH_BASE_URL``),
   which is structurally validated at startup via
   ``url_safety.validate_graph_base_url`` (HTTPS only, exactly
   ``graph.microsoft.com``, default port, no credentials/query/fragment).

2. Dynamic path segments (message IDs, drive IDs, attachment IDs, …) are
   sanitised at URL-construction time in ``endpoints.py`` via
   ``url_safety.safe_path_segment`` (``graph_url.safe_segment``), which
   treats each value as a single opaque segment - rejecting empty values,
   path separators, encoded separators, traversal sequences, malformed
   percent-escapes, and control characters, then percent-encodes it.

3. Last-line egress guard. EVERY fully-assembled URL about to be handed to
   ``httpx`` - the main ``_get``/``_request_once`` path, the SharePoint
   ``/content`` fallback, and any server-returned ``@odata.nextLink`` - is
   re-parsed and re-validated via ``url_safety.validate_graph_request_url``
   before the request is sent. This is independent of how the URL was built,
   so even a construction bug cannot send a request to a non-Graph origin.

The ONLY deliberate exception is the pre-authenticated
``@microsoft.graph.downloadUrl`` CDN link returned by a validated Graph
API response (see ``download_sharepoint_file``): it points at a Microsoft
CDN host (not ``graph.microsoft.com``), is server-provided rather than
user-controlled, and is documented inline at its call site.
"""

import time
from typing import Awaitable, Callable, Protocol

import httpx

from core.concurrency import get_graph_semaphore
from core.config import Settings, get_settings
from core.exceptions import (
    GraphApiError,
    GraphAuthError,
    GraphNotFoundError,
    GraphThrottledError,
)
from core.graph_client import endpoints
from core.graph_client.auth import acquire_app_only_token
from core.graph_client.url_safety import (
    UnsafeGraphUrlError,
    validate_graph_request_url,
)
from core.graph_client.local_calendar import list_local_calendar_events, list_local_calendar_view
from core.graph_client.local_mailbox import load_local_mailbox
from core.logging_config import get_agent_logger, log_request_outcome
from core.retry import with_retry
from core.utils import sanitize_log_input

logger = get_agent_logger("graph_client")

# Every outbound request this client makes is funnelled through
# `core.graph_client.url_safety.validate_graph_request_url` - the SINGLE,
# canonical, structured (urlsplit-based) egress guard that pins the
# scheme/host/port to an approved Graph origin. It is used at the last-line
# HTTP egress point in `_request_once` (and on the SharePoint `/content`
# fallback path), and for server-returned `@odata.nextLink` URLs, so no URL
# can leave this process for a non-Graph origin regardless of how it was
# built. This replaces the previous duplicated allow-list check.


class TokenProvider(Protocol):
    """
    Async callable that returns a currently-valid delegated access token.

    Called with ``force_refresh=False`` to get the token to USE for a
    request (reusing a cached one when still valid), and with
    ``force_refresh=True`` after a 401 to mint a brand-new access token
    from the stored refresh token. Implementations should return ``None``
    when no usable credential can be produced (e.g. the refresh token was
    revoked), in which case the original 401 is surfaced unchanged.

    The reference implementation is `auth.token_store.refresh_access_token`,
    adapted to this shape by `_make_delegated_token_provider` in
    `backend/run_pipeline.py`.
    """

    def __call__(self, *, force_refresh: bool = False) -> Awaitable[str | None]:
        ...


# Convenience alias for annotating optional providers without importing the
# Protocol everywhere a GraphClient is constructed.
TokenProviderCallable = Callable[..., Awaitable["str | None"]]


class GraphClient:
    """
    Thin async wrapper around Microsoft Graph API's REST endpoints.
    Agents should construct this via dependency injection or a simple
    `GraphClient(get_settings())` call - it holds no per-request state,
    so a single instance can be reused across requests.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        token_provider: "TokenProvider | None" = None,
    ) -> None:
        """
        `token_provider`, when given, is an async callable that returns a
        currently-valid delegated access token (refreshing it from the
        stored refresh token when needed). It is the hook that enables
        TRUE in-flight re-authentication: if a delegated call comes back
        401 mid-run (the access token expired or was revoked while a long
        pipeline was executing), `_get` calls this provider once with
        `force_refresh=True` to mint a brand-new access token and retries
        the request with it, instead of failing the whole run.

        It is only consulted in delegated mode. In app-only mode the client
        mints its own token via MSAL (`acquire_app_only_token`) and in local
        mode no token is used, so the provider is irrelevant there and left
        unused. When no provider is supplied the client behaves exactly as
        before: a 401 propagates as a `GraphApiError`.
        """
        self._settings = settings or get_settings()
        self._token_provider = token_provider

    def _is_local(self) -> bool:
        return self._settings.is_local_mail_mode

    def _headers(self, bearer_token: str | None) -> dict[str, str]:
        """
        Build the Authorization header for an outbound Graph API call.
        In app-only mode, acquires (or reuses MSAL's cached) app token
        and ignores whatever `bearer_token` the caller passed in - the
        caller doesn't need to know which mode is active.

        Also sets `Prefer: outlook.timezone="UTC"`. Without this header,
        Graph returns event `start`/`end` dateTime values in whatever
        time zone the mailbox's default is set to (e.g. India Standard
        Time), but the `dateTime` string itself carries no offset info -
        it just looks like a naive local time. `agent_calendar.service`
        treats every `dateTime` value as UTC, so without this header a
        12:30 PM IST meeting comes back as `"12:30:00"` with no offset,
        gets stamped as `12:30 UTC`, and then the frontend adds the
        IST offset again on top when displaying it (12:30 + 5:30 =
        18:00) - exactly the "Test 1" bug reported. Forcing UTC here
        makes the raw `dateTime` string actually be UTC, matching what
        `_parse_graph_datetime` assumes.
        """
        if self._settings.graph_auth_mode == "app_only":
            token = acquire_app_only_token(self._settings)
        else:
            if not bearer_token:
                raise GraphApiError(
                    "Delegated Graph auth mode requires a bearer token; none was provided."
                )
            token = bearer_token
        return {
            "Authorization": f"Bearer {token}",
            "Prefer": 'outlook.timezone="UTC"',
        }

    @staticmethod
    def _assert_safe_graph_url(url: str) -> None:
        """
        Last-line SSRF egress guard (CWE-918). Runs the single canonical
        structured validator over a fully-assembled URL right before it is
        handed to ``httpx`` and raises ``GraphApiError`` if it does not
        resolve to an approved Graph origin.

        Every outbound GET in this client calls this exactly once at its
        egress point (the main ``_request_once`` path and the SharePoint
        ``/content`` fallback), so there is a single, uniform choke point
        no request can skip. The ``UnsafeGraphUrlError`` raised by the
        validator is translated to a ``GraphApiError`` (the client's public
        error type) whose message contains "validation failed".
        """
        try:
            validate_graph_request_url(url)
        except UnsafeGraphUrlError as exc:
            raise GraphApiError(
                f"Outbound Graph request URL SSRF validation failed: {exc}"
            ) from exc

    async def _get(self, url: str, bearer_token: str | None, params: dict | None = None) -> dict:
        """
        Perform a concurrency-limited GET against Graph API and return the
        parsed JSON body, with TWO layers of recovery wrapped around it:

        1. Transient-error backoff (connection/timeout/429-503 throttle) is
           handled inside `_request_once` by `core.retry.with_retry`, exactly
           as before.
        2. TRUE in-flight re-authentication: if the request comes back 401
           (`GraphAuthError`) in delegated mode AND this client was given a
           `token_provider`, mint a fresh access token from the stored
           refresh token and retry the request ONCE with the new token. This
           is what lets a long-running pipeline survive its access token
           expiring part-way through instead of failing outright.

        If no provider is configured, or the single refresh+retry still 401s,
        the `GraphAuthError` propagates like any other `GraphApiError`.
        """
        try:
            return await self._request_once(url, bearer_token, params=params)
        except GraphAuthError as auth_exc:
            # App-only/local never carry a caller-supplied delegated token to
            # refresh, and a 401 there is a configuration problem (bad client
            # secret / missing app permission), not an expired user token.
            if self._settings.graph_auth_mode != "delegated" or self._token_provider is None:
                raise
            logger.warning(
                "[GRAPH-AUTH] GET %s returned 401 mid-run; refreshing delegated "
                "access token and retrying once", sanitize_log_input(str(url)),
            )
            try:
                new_token = await self._token_provider(force_refresh=True)
            except Exception as provider_exc:  # noqa: BLE001
                logger.error(
                    "[GRAPH-AUTH] token refresh after 401 failed: %s", provider_exc
                )
                raise auth_exc
            if not new_token:
                logger.error(
                    "[GRAPH-AUTH] token refresh after 401 produced no token "
                    "(refresh token revoked/expired?); surfacing original 401"
                )
                raise auth_exc
            # Retry exactly once with the freshly minted token. A second 401
            # (e.g. the new token is itself rejected) is NOT refreshed again -
            # we let it propagate to avoid an unbounded refresh loop.
            return await self._request_once(url, new_token, params=params)

    @with_retry(
        max_attempts=5,
        base_delay=2.0,
        # `GraphThrottledError` (429/503, incl. the shared mailbox's own
        # "ApplicationThrottled"/MailboxConcurrency limit) is now retried
        # like a transient network error - `_resolve_delay` in
        # `core.retry` honors its `retry_after_seconds` (from Graph's own
        # `Retry-After` header) when present, otherwise falls back to the
        # same exponential backoff used for connection/timeout errors.
        # Previously a 429 immediately raised a plain `GraphApiError`,
        # which is NOT in this retry set, so every throttled call failed
        # outright with no retry at all.
        #
        # NOTE: `GraphAuthError` (401) is intentionally NOT in this set. A
        # 401 is not a transient blip, so blind backoff would just resend
        # the same dead token. It is handled one level up in `_get`, which
        # swaps in a freshly refreshed token and retries once.
        retry_exceptions=(httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout, GraphThrottledError),
        connection_exceptions=(httpx.ConnectError,),
        timeout_exceptions=(httpx.TimeoutException, httpx.ReadTimeout),
        logger=logger,
    )
    async def _request_once(self, url: str, bearer_token: str | None, params: dict | None = None) -> dict:
        """
        Perform a single retried, concurrency-limited GET request against
        Graph API and return the parsed JSON body. Raises `GraphApiError`
        on any non-2xx response (after retries for transient errors are
        exhausted) - the response body is included for debuggability. A 401
        is raised as `GraphAuthError` so the caller (`_get`) can refresh the
        token and retry.
        """
        # SSRF guard (defence-in-depth, CWE-918). The URL is already built
        # from the trusted, structurally-validated base URL plus opaque,
        # validated+encoded path segments (see `endpoints.py` /
        # `core.graph_client.url_safety`), so by construction it points at
        # the Graph host. This is the LAST-LINE check at the single HTTP
        # egress point: funnel the fully-assembled URL through the ONE
        # canonical structured validator (`validate_graph_request_url`,
        # which uses `urlsplit` and validates the parsed scheme/hostname/
        # port, not a substring/netloc compare) and refuse to send it
        # anywhere other than an approved Graph origin, regardless of how it
        # was produced. `hostname` (unlike `netloc`) excludes any
        # `user:pass@` prefix and `:port` suffix, so
        # `https://graph.microsoft.com@evil.com/` resolves to hostname
        # `evil.com` and is rejected; pinning the scheme to https blocks
        # `file://`, `gopher://`, etc.
        self._assert_safe_graph_url(url)

        headers = self._headers(bearer_token)
        async with get_graph_semaphore():
            started_at = time.monotonic()
            async with httpx.AsyncClient(
                timeout=self._settings.graph_request_timeout_seconds,
                follow_redirects=False,
            ) as http_client:
                response = await http_client.get(url, headers=headers, params=params)
            duration_ms = (time.monotonic() - started_at) * 1000
            log_request_outcome(logger, "GET", url, response.status_code, duration_ms)

            if response.status_code in (429, 503):
                retry_after_header = response.headers.get("Retry-After")
                retry_after_seconds = float(retry_after_header) if retry_after_header else None
                raise GraphThrottledError(
                    f"Graph API GET {url} was throttled with status {response.status_code}: {response.text}",
                    retry_after_seconds=retry_after_seconds,
                    status_code=response.status_code,
                )

            # A verified 404 is the ONLY status that may be read as "the
            # resource is absent from the queried scope" (a deletion
            # candidate). Raise the dedicated type so callers can catch it
            # explicitly; it is deliberately NOT in `_get`'s retry set (a
            # 404 is a definitive answer, not a transient error). Every
            # other >= 400 status (403 permission, 5xx, etc.) stays a plain
            # `GraphApiError` - inconclusive, never a deletion signal - and
            # still carries its status_code for structural inspection.
            if response.status_code == 404:
                raise GraphNotFoundError(
                    f"Graph API GET {url} returned 404 (resource not found): {response.text}",
                    status_code=404,
                )

            # A 401 means the delegated access token is expired/invalid/
            # revoked. Raise the dedicated `GraphAuthError` so `_get` can
            # mint a fresh token from the stored refresh token and retry
            # once. It is deliberately NOT in the `with_retry` set above (a
            # 401 is not transient; blind backoff would resend a dead token).
            if response.status_code == 401:
                raise GraphAuthError(
                    f"Graph API GET {url} returned 401 (unauthorized): {response.text}",
                    status_code=401,
                )

            if response.status_code >= 400:
                raise GraphApiError(
                    f"Graph API GET {url} failed with status {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )
            return response.json()

    async def get_current_user(self, bearer_token: str) -> dict:
        """
        Fetch the signed-in user's profile. Delegated mode only (there is
        no "current user" concept for an app-only token).
        """
        url = endpoints.me_url(self._settings)
        return await self._get(url, bearer_token)

    async def get_messages(
        self,
        bearer_token: str | None = None,
        top: int = 10,
        skip: int = 0,
        received_after: str | None = None,
    ) -> dict:
        """
        List messages from the configured mailbox's inbox (or the
        signed-in user's inbox in delegated mode without a shared
        mailbox configured), most recent first.

        `received_after` (ISO 8601 UTC, e.g. "2026-09-09T00:00:00Z"), if
        given, is applied server-side as `$filter=receivedDateTime ge
        <value>` so Graph never returns messages older than that
        timestamp in the first place, rather than fetching them and
        filtering them out afterwards.
        """
        url = endpoints.messages_url(self._settings)
        params = {
            "$top": top,
            "$skip": skip,
            "$select": "id,conversationId,subject,from,toRecipients,ccRecipients,receivedDateTime,hasAttachments,bodyPreview,body",
            "$orderby": "receivedDateTime desc",
        }
        if received_after:
            params["$filter"] = f"receivedDateTime ge {received_after}"
        if self._is_local():
            logger.info("Local mailbox: listing messages top=%s skip=%s", top, skip)
            return load_local_mailbox(self._settings).list_messages(
                top=top, skip=skip, received_after=received_after
            )
        return await self._get(url, bearer_token, params=params)

    async def get_message_detail(self, message_id: str, bearer_token: str | None = None) -> dict:
        """Fetch the full detail (including body) of a single message."""
        url = endpoints.message_detail_url(self._settings, message_id)
        params = {
            "$select": "id,conversationId,subject,from,toRecipients,ccRecipients,receivedDateTime,body,hasAttachments,isRead"
        }
        if self._is_local():
            try:
                return load_local_mailbox(self._settings).get_message(message_id)
            except KeyError as exc:
                raise GraphApiError(f"Local mailbox has no message id {message_id}") from exc
        return await self._get(url, bearer_token, params=params)

    async def get_attachments(self, message_id: str, bearer_token: str | None = None) -> dict:
        """List attachment metadata for a given message id."""
        url = endpoints.message_attachments_url(self._settings, message_id)
        if self._is_local():
            return load_local_mailbox(self._settings).list_attachments(message_id)
        return await self._get(url, bearer_token)

    async def get_attachment_content(self, message_id: str, attachment_id: str, bearer_token: str | None = None) -> dict:
        """Fetch a single attachment's metadata plus base64-encoded contentBytes."""
        url = endpoints.attachment_detail_url(self._settings, message_id, attachment_id)
        if self._is_local():
            try:
                return load_local_mailbox(self._settings).get_attachment(message_id, attachment_id)
            except KeyError as exc:
                raise GraphApiError(
                    f"Local mailbox has no attachment {attachment_id} on message {message_id}"
                ) from exc
        return await self._get(url, bearer_token)

    async def get_calendar_events(self, bearer_token: str | None = None, top: int = 50) -> dict:
        """
        List upcoming and recent calendar events for the configured
        mailbox, ordered by start time - used by the Calendar & Meeting
        agent to detect the current/next meeting.
        """
        url = endpoints.calendar_events_url(self._settings)
        params = {
            "$top": top,
            "$select": "id,subject,start,end,organizer,attendees,body,bodyPreview",
            "$orderby": "start/dateTime asc",
        }
        if self._is_local():
            logger.info("Local mailbox: serving dummy calendar events from disk")
            return list_local_calendar_events(top=top, settings=self._settings)
        return await self._get(url, bearer_token, params=params)

    async def get_calendar_view(
        self,
        start_date_time: str,
        end_date_time: str,
        bearer_token: str | None = None,
        top: int = 50,
    ) -> dict:
        """
        List every event (including expanded recurring-meeting instances)
        that falls within `[start_date_time, end_date_time)` via Graph's
        `calendarView` endpoint. Both bounds are ISO 8601 strings (e.g.
        "2026-08-31T00:00:00Z").

        Used for "everything scheduled on day X" queries - `/events`
        (plain `get_calendar_events`) has no date bound, so ordering it
        by start time ascending surfaces the oldest events in the whole
        mailbox rather than anything happening today.
        """
        url = endpoints.calendar_view_url(self._settings)
        params = {
            "startDateTime": start_date_time,
            "endDateTime": end_date_time,
            "$top": top,
            "$select": "id,subject,start,end,organizer,attendees,body,bodyPreview",
            "$orderby": "start/dateTime asc",
        }
        if self._is_local():
            logger.info("Local mailbox: serving dummy calendar view from disk")
            return list_local_calendar_view(
                start_date_time=start_date_time,
                end_date_time=end_date_time,
                top=top,
                settings=self._settings,
            )
        return await self._get(url, bearer_token, params=params)

    # ── SharePoint / OneDrive sharing URL methods ────────────────
    #
    # Uses Graph's /shares API to resolve a sharing URL into a driveItem,
    # then navigates via driveId + itemId. Works with Sites.Selected
    # permission — no Files.Read.All needed.

    async def resolve_sharing_url(
        self,
        sharing_url: str,
        bearer_token: str | None = None,
        expand_children: bool = True,
    ) -> dict:
        """Resolve a SharePoint/OneDrive sharing URL to a driveItem.

        Uses Graph's ``/shares/{encoded}/driveItem`` endpoint. Returns
        the driveItem dict which includes ``id``, ``name``,
        ``parentReference.driveId``, and optionally ``children``.

        In local mode, returns an empty dict.
        """
        if self._is_local():
            logger.info("Local mode: resolve_sharing_url returning empty dict")
            return {}

        url = endpoints.shares_driveitem_url(sharing_url)
        params = {"$expand": "children"} if expand_children else None
        return await self._get(url, bearer_token, params=params)

    async def list_drive_children(
        self,
        drive_id: str,
        item_id: str,
        bearer_token: str | None = None,
    ) -> list[dict]:
        """List children of a drive item by driveId + itemId.

        Handles ``@odata.nextLink`` pagination and returns the full list.

        **SSRF note**: ``@odata.nextLink`` is a server-returned URL that
        could theoretically point anywhere. It is validated via
        ``url_safety.validate_graph_request_url`` both here (fail-fast) and
        again at the ``_get`` -> ``_request_once`` egress point before any
        request is made, ensuring it targets ``https://graph.microsoft.com``
        only.
        """
        if self._is_local():
            return []

        url = endpoints.drive_item_children_url(drive_id, item_id)
        params: dict | None = {
            "$select": "id,name,size,file,folder,lastModifiedDateTime,parentReference",
            "$top": 200,
        }

        items: list[dict] = []
        while url:
            data = await self._get(url, bearer_token, params=params)
            items.extend(data.get("value", []))
            next_link = data.get("@odata.nextLink")
            if next_link:
                # Validate the server-returned pagination URL before using
                # it. ``_get()`` -> ``_request_once`` validates again at the
                # egress point, but failing early here gives a clearer error
                # and never even queues an unsafe URL. Uses the SAME single
                # canonical structured validator as the egress guard.
                try:
                    validate_graph_request_url(next_link)
                except UnsafeGraphUrlError as exc:
                    raise GraphApiError(
                        f"@odata.nextLink URL failed SSRF validation: {exc}"
                    ) from exc
                url = next_link
                params = None  # nextLink includes query params already
            else:
                url = None  # type: ignore[assignment]
        return items

    async def list_sharepoint_files_recursive(
        self,
        sharing_url: str,
        bearer_token: str | None = None,
        file_extensions: tuple[str, ...] = (".pdf",),
    ) -> tuple[list[dict], str, str]:
        """Recursively list all matching files under a shared folder.

        Resolves the sharing URL first to get the root driveId + itemId,
        then walks subfolders recursively via ``/drives/{driveId}/items/{itemId}/children``.

        Returns ``(files, drive_id, root_item_id)`` where each file dict
        is a Graph drive item with an extra ``_relative_path`` key and
        a ``_drive_id`` key.
        """
        if self._is_local():
            return [], "", ""

        # Step 1: Resolve sharing URL to root driveItem
        root = await self.resolve_sharing_url(sharing_url, bearer_token, expand_children=True)
        drive_id = (root.get("parentReference") or {}).get("driveId", "")
        root_item_id = root.get("id", "")

        if not drive_id:
            raise GraphApiError(
                "Could not resolve driveId from sharing URL. "
                "Check that SHAREPOINT_SITE is correct and the app has Sites.Selected access."
            )

        # Step 2: Process children from the resolved root
        children = root.get("children", [])
        ext_lower = tuple(ext.lower() for ext in file_extensions)

        files = await self._walk_children_recursive(
            drive_id, children, bearer_token, ext_lower, _current_path="",
        )
        return files, drive_id, root_item_id

    async def _walk_children_recursive(
        self,
        drive_id: str,
        children: list[dict],
        bearer_token: str | None,
        ext_lower: tuple[str, ...],
        _current_path: str = "",
    ) -> list[dict]:
        """Walk children list recursively, collecting matching files."""
        files: list[dict] = []

        for item in children:
            name = item.get("name", "")

            if item.get("folder") is not None:
                # Subfolder — fetch its children and recurse
                sub_item_id = item.get("id", "")
                sub_relative = f"{_current_path}/{name}".strip("/")
                try:
                    sub_children = await self.list_drive_children(
                        drive_id, sub_item_id, bearer_token,
                    )
                    sub_files = await self._walk_children_recursive(
                        drive_id, sub_children, bearer_token, ext_lower,
                        _current_path=sub_relative,
                    )
                    files.extend(sub_files)
                except Exception as exc:
                    logger.warning(
                        "Could not list subfolder '%s' (id=%s): %s",
                        name, sub_item_id, exc,
                    )

            elif item.get("file") is not None:
                # File — check extension
                if any(name.lower().endswith(ext) for ext in ext_lower):
                    item["_relative_path"] = _current_path
                    item["_drive_id"] = drive_id
                    files.append(item)

        return files

    async def download_sharepoint_file(
        self,
        drive_id: str,
        item_id: str,
        dest_path: str,
        bearer_token: str | None = None,
    ) -> str:
        """Download a file from SharePoint/OneDrive by driveId + itemId.

        First fetches item metadata to get the ``@microsoft.graph.downloadUrl``
        (a pre-authenticated CDN URL), then downloads the content. Falls
        back to the ``/content`` endpoint if the download URL isn't present.

        Returns the absolute path of the written file.
        """
        import aiofiles

        # Step 1: Get item metadata for the pre-authenticated download URL.
        # ``endpoints.drive_item_by_id_url`` validates drive_id and item_id
        # via ``safe_segment()``.
        meta_url = endpoints.drive_item_by_id_url(drive_id, item_id)
        meta = await self._get(meta_url, bearer_token)
        download_url = meta.get("@microsoft.graph.downloadUrl", "")

        if download_url:
            # Pre-authenticated CDN URL — no bearer token needed.
            # This URL is returned by Graph and points to a Microsoft CDN,
            # NOT graph.microsoft.com. It bypasses _get() intentionally
            # since the CDN host varies. The URL is server-provided (not
            # user-controlled) and obtained from a validated Graph API call.
            async with httpx.AsyncClient(
                timeout=120,
                follow_redirects=True,
            ) as http_client:
                response = await http_client.get(download_url)
                response.raise_for_status()
        else:
            # Fallback: /content endpoint (Graph 302 → CDN).
            # ``drive_item_content_by_id_url`` validates IDs via safe_segment().
            content_url = endpoints.drive_item_content_by_id_url(drive_id, item_id)
            # Last-line SSRF egress guard (CWE-918): this request does NOT go
            # through `_get`/`_request_once`, so re-validate the fully-
            # assembled URL through the SAME single canonical structured
            # validator before sending. `follow_redirects=True` is required
            # because Graph's `/content` replies with a 302 to a per-request
            # CDN URL; the guard pins the INITIAL hop to graph.microsoft.com
            # (the only host we control), and the subsequent redirect target
            # is a Microsoft-issued, server-chosen CDN location.
            self._assert_safe_graph_url(content_url)
            headers = self._headers(bearer_token)
            async with httpx.AsyncClient(
                timeout=120,
                follow_redirects=True,
            ) as http_client:
                response = await http_client.get(content_url, headers=headers)
                response.raise_for_status()

        from pathlib import Path
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(dest_path, "wb") as f:
            await f.write(response.content)

        logger.info(
            "Downloaded SharePoint file %s/%s -> %s (%d bytes)",
            sanitize_log_input(str(drive_id)), sanitize_log_input(str(item_id)),
            sanitize_log_input(str(dest_path)), len(response.content),
        )
        return str(dest_path)
