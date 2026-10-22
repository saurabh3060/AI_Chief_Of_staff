"""
core/graph_client/auth.py
============================
MSAL-based authentication helpers for Microsoft Graph API, supporting
both authentication modes the backend can run in:

- Delegated (browser OAuth, "on behalf of the signed-in user"): builds
  the authorization URL and exchanges the authorization code for tokens.
  This is the flow ported from the user's reference code and is used by
  `auth/routes.py` (`/login`, `/callback`).
- App-only (client credentials, "as the application itself"): acquires
  and caches an access token scoped to `https://graph.microsoft.com/.default`,
  used once Application-level Graph permissions are granted for the
  shared mailbox.

Both flows share the same `ConfidentialClientApplication` builder so
tenant/client id/secret are only ever read from `core.config` in one
place.
"""

import msal

from core.config import Settings

GRAPH_DEFAULT_SCOPE = ["https://graph.microsoft.com/.default"]
# NOTE on refresh tokens / `offline_access`:
# MSAL *always* adds the reserved OIDC scopes (`offline_access`, `openid`,
# `profile`) to the authorization request itself - see
# `ClientApplication._decorate_scope`. That means a *refresh* token is
# returned for the delegated authorization-code flow without us listing
# `offline_access` explicitly. In fact, newer MSAL versions raise
# `ValueError: You cannot use any scope value that is reserved.` if any of
# those reserved scopes are passed in the `scopes` argument. So we must
# request only the *resource* permissions and let MSAL inject the reserved
# scopes (and thus the refresh token) for us.
#
# `DELEGATED_SCOPES` therefore contains resource scopes only and is shared by
# both the authorize URL and the code->token exchange. The separate
# `DELEGATED_RESOURCE_SCOPES` alias is kept for the refresh-token call, which
# has the same requirement.
DELEGATED_SCOPES = [
    "User.Read",
    "Mail.Read",
    "Calendars.Read",
    "Calendars.Read.Shared",
]
# The resource scopes only (no reserved OIDC scopes). MSAL's
# `acquire_token_by_refresh_token` rejects the reserved scopes
# (`offline_access`/`openid`/`profile`) in its `scopes` argument, so the
# refresh call must request just the resource permissions.
DELEGATED_RESOURCE_SCOPES = [
    "User.Read",
    "Mail.Read",
    "Calendars.Read",
    "Calendars.Read.Shared",
]


def build_msal_app(
    settings: Settings,
    token_cache: msal.TokenCache | None = None,
) -> msal.ConfidentialClientApplication:
    """
    Construct the MSAL confidential client application used for both
    delegated and app-only flows, authorised against the tenant's
    authority endpoint.

    `token_cache` lets the caller attach a cache so the refresh token MSAL
    captures during a code exchange can be read back out (see
    `exchange_code_for_token`). When None, MSAL uses an in-memory cache
    that is discarded with the app instance (the previous behaviour).
    """
    authority = f"https://login.microsoftonline.com/{settings.tenant_id}"
    return msal.ConfidentialClientApplication(
        settings.client_id,
        authority=authority,
        client_credential=settings.client_secret,
        token_cache=token_cache,
    )


def _extract_refresh_token(cache: msal.SerializableTokenCache) -> str | None:
    """
    Pull the raw refresh-token string out of an MSAL token cache.

    MSAL for Python does NOT return `refresh_token` in the result of
    `acquire_token_by_authorization_code` - it stashes it inside its token
    cache and expects callers to renew via `acquire_token_silent`. Because
    this app persists the raw refresh token to AWS Secrets Manager (and
    renews with `acquire_token_by_refresh_token`), we read the token back
    out of the cache here. Returns None when the cache holds no refresh
    token. The value is never logged.
    """
    entries = cache.find(msal.TokenCache.CredentialType.REFRESH_TOKEN)
    for entry in entries:
        secret = entry.get("secret")
        if secret:
            return secret
    return None


def build_authorization_url(settings: Settings, redirect_uri: str, prompt: str | None = None) -> str:
    """
    Build the Microsoft "authorize" URL the browser should be redirected
    to for the delegated OAuth flow.

    When `prompt` is None/empty (the default) NO `prompt` query parameter
    is added, so Microsoft silently reuses an existing SSO session where
    possible - the quickest path to the authorize page. A `prompt` value
    is only included when a caller explicitly supplies one; notably the
    caller never defaults to `prompt=login`, which would force a slower
    full re-authentication (and possibly re-run MFA) on every sign-in.

    `redirect_uri` is passed through to MSAL verbatim and MUST exactly
    match the value registered on the Azure app registration (and the one
    later used in the token exchange at `/callback`).
    """
    app = build_msal_app(settings)
    kwargs = {"prompt": prompt} if prompt else {}
    return app.get_authorization_request_url(DELEGATED_SCOPES, redirect_uri=redirect_uri, **kwargs)


def exchange_code_for_token(settings: Settings, code: str, redirect_uri: str) -> dict:
    """
    Exchange an authorization code (received at the `/callback` redirect)
    for an access token + id token via a direct server-to-server call to
    Microsoft's token endpoint. Returns the raw MSAL result dict, which
    contains either the tokens or an "error"/"error_description" pair.

    MSAL does not place `refresh_token` in this result (it keeps it in its
    token cache). We attach a `SerializableTokenCache`, then - on success -
    read the raw refresh token out of it and inject it into the returned
    dict under `refresh_token`, so callers (auth/routes.py -> save_tokens)
    can persist it to AWS Secrets Manager. The token value is never logged.
    """
    cache = msal.SerializableTokenCache()
    app = build_msal_app(settings, token_cache=cache)
    result = app.acquire_token_by_authorization_code(
        code, scopes=DELEGATED_SCOPES, redirect_uri=redirect_uri
    )
    if "access_token" in result and not result.get("refresh_token"):
        extracted = _extract_refresh_token(cache)
        if extracted:
            result["refresh_token"] = extracted
    return result


def acquire_app_only_token(settings: Settings) -> str:
    """
    Acquire (or return a cached) access token for the app-only
    client-credentials flow, scoped to the default Graph API permission
    set granted to this app registration.

    Raises `RuntimeError` if MSAL cannot obtain a token (e.g. missing
    application permissions, invalid client secret) - the error detail
    from MSAL is included so misconfiguration is easy to diagnose.
    """
    app = build_msal_app(settings)
    result = app.acquire_token_for_client(scopes=GRAPH_DEFAULT_SCOPE)
    if "access_token" not in result:
        error_description = result.get("error_description", "unknown error acquiring app-only token")
        raise RuntimeError(f"Failed to acquire app-only Graph API token: {error_description}")
    return result["access_token"]


def acquire_token_by_refresh_token(settings: Settings, refresh_token: str) -> dict:
    """
    Exchange a stored delegated *refresh* token for a fresh access token
    (and usually a rotated refresh token) via MSAL. Returns the raw MSAL
    result dict, which contains either the new tokens or an
    "error"/"error_description" pair.

    This is the unattended renewal path the external pipeline runner uses:
    a user signs in once (so a refresh token is captured and persisted),
    and every later pipeline run mints a short-lived access token from that
    stored refresh token without any browser interaction.

    Microsoft rotates refresh tokens on use, so callers MUST persist the
    `refresh_token` returned here (when present) so the next run can renew
    again. The scopes requested here are the resource scopes only - the
    reserved `offline_access` scope must not be passed to this call.
    """
    app = build_msal_app(settings)
    return app.acquire_token_by_refresh_token(refresh_token, scopes=DELEGATED_RESOURCE_SCOPES)