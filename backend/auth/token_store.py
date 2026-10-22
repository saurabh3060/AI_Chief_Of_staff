"""
auth/token_store.py
=====================
Durable store for delegated Microsoft Graph OAuth tokens.

Single source of truth: AWS Secrets Manager
--------------------------------------------
BOTH delegated tokens now live ONLY in the JSON secret
``rbl-digitalofficeofceo-prod-parameterstore`` (region ``ap-south-1``),
via `core.secrets_manager`:

* **refresh_token** — the long-lived credential that lets the unattended
  pipeline runner (`backend/run_pipeline.py`) authenticate to Graph across
  restarts. Stored under the ``refresh_token`` key.
* **access_token** (+ ``access_token_expires_at``) — the short-lived (~1h)
  token, stored under the ``access_token`` / ``access_token_expires_at``
  keys so a still-valid token is reused across processes without a refresh
  round-trip.

There is NO LONGER a PostgreSQL ``graph_oauth_tokens`` table. The old split
(refresh token in the vault, access token in Postgres) is gone — the secret
holds everything, which removes the confusing two-place architecture.

Why this exists
---------------
The delegated sign-in in `auth/routes.py` yields a short-lived access token
and a long-lived refresh token. On EVERY sign-in both are written to the
secret (a fresh secret version), so the latest tokens are always available
to later unattended runs. The in-memory session store (`auth/session.py`)
still keeps the access token for the lifetime of the server process for the
browser session; the pipeline runner, which has no browser session, reads
the tokens straight from the secret.

Tokens are secrets. Neither the access token nor the refresh token is ever
logged — only lengths, SHA-256 fingerprints, and the new secret version id.
"""

from __future__ import annotations

import hashlib
import time

from core.config import Settings, get_settings
from core.graph_client.auth import acquire_token_by_refresh_token
from core.logging_config import get_agent_logger
from core.secrets_manager import SecretsManagerClient, SecretsManagerError

logger = get_agent_logger("auth")

# A single, well-known key used when the id token carries no user email
# (e.g. some app configurations). Keeps the "one logical delegated
# identity" model simple.
_FALLBACK_USER_KEY = "default"


def _secrets_client(settings: Settings) -> SecretsManagerClient:
    """Build a Secrets Manager client from the active settings."""
    return SecretsManagerClient(
        secret_name=settings.aws_secret_name,
        region_name=settings.aws_region,
    )


def _fingerprint(token: str | None) -> str:
    """
    SHA-256 hex (first 12 chars) of a token for safe before/after
    comparison in logs, or ``<none>`` when absent. Never the token itself.
    """
    if not (token or "").strip():
        return "<none>"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


async def save_tokens(
    access_token: str,
    refresh_token: str | None,
    expires_in: int,
    user_email: str | None = None,
    settings: Settings | None = None,
) -> bool:
    """
    Persist the delegated tokens from a sign-in / refresh into AWS Secrets
    Manager as a SINGLE new secret version, preserving every other key
    (``POSTGRES_*`` etc.).

    * The **access token** + its absolute expiry overwrite the
      ``access_token`` / ``access_token_expires_at`` keys so the latest
      token is always published.
    * The **refresh token** overwrites the ``refresh_token`` key only when
      non-empty; an empty/missing refresh token leaves the previously
      stored one untouched (Microsoft does not always return a rotated one).

    To prove the replacement worked, this logs the BEFORE and AFTER
    fingerprints of both tokens (SHA-256 prefix, never the value) plus the
    new secret version id, and whether each token actually changed.

    Returns True when a new secret version was written, False when there was
    nothing to persist. Never logs token values.
    """
    active = settings or get_settings()
    client = _secrets_client(active)

    key = (user_email or "").strip().lower() or _FALLBACK_USER_KEY
    expires_at = time.time() + max(int(expires_in or 0), 0)

    # Read the current tokens first so we can prove (via fingerprints) that
    # the write actually replaced them. A read failure is non-fatal — we
    # still attempt the write.
    before_access = before_refresh = None
    try:
        before = client.get_tokens()
        before_access = before.get("access_token")
        before_refresh = before.get("refresh_token")
    except SecretsManagerError as exc:
        logger.warning(
            "[TOKEN-STORE] could not read tokens before write (continuing): %s",
            exc,
        )

    try:
        version_id = client.put_tokens(
            access_token=access_token,
            refresh_token=refresh_token,
            access_token_expires_at=expires_at,
            user_email=key if key != _FALLBACK_USER_KEY else (user_email or None),
        )
    except SecretsManagerError as exc:
        # Swallow so a Secrets Manager hiccup cannot break sign-in; the next
        # successful rotation re-persists. Never log the token values.
        logger.error("[TOKEN-STORE] could not store delegated tokens: %s", exc)
        return False

    if version_id is None:
        logger.info(
            "[TOKEN-STORE] no tokens to persist (both empty); "
            "Secrets Manager left unchanged"
        )
        return False

    after_access = (access_token or "").strip() or before_access
    after_refresh = (refresh_token or "").strip() or before_refresh

    logger.info(
        "[TOKEN-STORE] delegated tokens written to Secrets Manager "
        "(version=%s, user=%s)",
        (version_id[:8] + "…") if version_id else "?",
        "key",
    )
    logger.info(
        "[TOKEN-STORE] access_token: before=%s after=%s -> %s",
        _fingerprint(before_access),
        _fingerprint(after_access),
        "REPLACED" if _fingerprint(before_access) != _fingerprint(after_access)
        else "unchanged",
    )
    logger.info(
        "[TOKEN-STORE] refresh_token: before=%s after=%s -> %s",
        _fingerprint(before_refresh),
        _fingerprint(after_refresh),
        "REPLACED" if _fingerprint(before_refresh) != _fingerprint(after_refresh)
        else "unchanged (Microsoft returned no rotated refresh token)",
    )
    return True


async def get_latest_tokens(settings: Settings | None = None) -> dict | None:
    """
    Return the latest delegated credential as a dict with keys:
    ``user_email``, ``access_token``, ``refresh_token``,
    ``access_token_expires_at`` — all read from AWS Secrets Manager (the
    single source of truth).

    Returns ``None`` only when there is no credential at all (neither a
    stored refresh token nor a cached access token). Token values are not
    logged.
    """
    active = settings or get_settings()
    try:
        tokens = _secrets_client(active).get_tokens()
    except SecretsManagerError as exc:
        logger.error("[TOKEN-STORE] could not read delegated tokens: %s", exc)
        return None

    if not tokens.get("access_token") and not tokens.get("refresh_token"):
        return None

    return {
        "user_email": tokens.get("user_email"),
        "access_token": tokens.get("access_token"),
        "access_token_expires_at": tokens.get("access_token_expires_at") or 0.0,
        "refresh_token": tokens.get("refresh_token"),
    }


async def get_tokens_for_user(
    user_email: str,
    settings: Settings | None = None,
) -> dict | None:
    """
    Return the stored delegated credential for ``user_email``.

    The store is a single delegated identity (one set of tokens in the
    secret), so this returns the same credential as ``get_latest_tokens``.
    Kept as a named entry point for callers (e.g. ``scripts/run_sync.py``)
    that pass an explicit user.
    """
    return await get_latest_tokens(settings)


async def refresh_access_token(
    row: dict | None = None,
    skew_seconds: int = 300,
    settings: Settings | None = None,
    force_refresh: bool = False,
) -> str | None:
    """
    Return a usable delegated access token, refreshing it via the stored
    refresh token (from Secrets Manager) when the cached one is missing or
    within ``skew_seconds`` of expiry. On a successful refresh both the new
    access token and the (possibly rotated) refresh token are written back
    to Secrets Manager so the next run renews cleanly.

    ``force_refresh=True`` bypasses the "cached token still valid" shortcut
    and always performs the refresh-token exchange (the path taken when a
    Graph call returned 401 mid-run and the cached token is known-dead).

    Returns ``None`` when no refresh token exists, or the refresh fails
    (e.g. the refresh token was revoked / expired). Callers treat ``None``
    as "no valid delegated credential available — a human must sign in
    again". Token values are never logged.
    """
    active = settings or get_settings()
    data = row if row is not None else await get_latest_tokens(active)
    if data is None:
        logger.warning(
            "[TOKEN-STORE] no stored delegated credential; cannot provide an access token"
        )
        return None

    access_token = data.get("access_token")
    expires_at = float(data.get("access_token_expires_at") or 0.0)
    refresh_token = data.get("refresh_token")

    # Still comfortably valid -> reuse without a network call. Skipped when
    # `force_refresh` is set (a 401 proved the "valid" token is actually dead).
    if not force_refresh and access_token and time.time() < (expires_at - skew_seconds):
        return access_token

    if not refresh_token:
        # Access token stale/absent and nothing to renew with.
        logger.warning(
            "[TOKEN-STORE] stored access token is stale and no refresh token is "
            "available; a fresh sign-in is required"
        )
        if force_refresh:
            return None
        return access_token or None

    logger.info("[TOKEN-STORE] access token stale; refreshing via stored refresh token")
    result = acquire_token_by_refresh_token(active, refresh_token)
    if "access_token" not in result:
        error = result.get("error_description", result.get("error", "unknown error"))
        logger.error("[TOKEN-STORE] refresh failed: %s", error)
        return None

    new_access = result["access_token"]
    # Microsoft rotates refresh tokens; persist the new one (Secrets Manager)
    # so the chain continues. If none is returned, the existing stored token
    # is kept (save_tokens -> put_tokens is a no-op on the empty refresh key).
    await save_tokens(
        access_token=new_access,
        refresh_token=result.get("refresh_token"),
        expires_in=int(result.get("expires_in") or 0),
        user_email=data.get("user_email"),
        settings=active,
    )
    logger.info("[TOKEN-STORE] refreshed delegated access token")
    return new_access
