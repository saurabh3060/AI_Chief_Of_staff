"""
core/secrets_manager.py
========================
Thin, auditable wrapper around AWS Secrets Manager for the two secrets
this backend must NOT keep in `.env` or the application database:

1. **PostgreSQL credentials** — ``POSTGRES_USER``, ``POSTGRES_PASSWORD``
   and ``POSTGRES_DB`` are read from the JSON secret and used by
   ``core/config.py`` to build the asyncpg DSN at startup. They are no
   longer read from ``.env``.
2. **The delegated Microsoft Graph OAuth tokens** — BOTH the long-lived
   ``refresh_token`` AND the short-lived ``access_token`` (plus its
   ``access_token_expires_at`` absolute epoch) are stored as keys of the
   SAME secret (see ``auth/token_store.py``). There is no longer any
   PostgreSQL ``graph_oauth_tokens`` table: the AWS secret is the single
   source of truth for every delegated token. Every sign-in overwrites both
   tokens with the latest values so later unattended pipeline runs always
   pick up fresh credentials.

Design rules enforced here
--------------------------
* **One secret, JSON-shaped.** The secret
  ``rbl-digitalofficeofceo-prod-parameterstore`` (region ``ap-south-1``)
  holds a single JSON object. We only ever read/merge keys — writing the
  refresh token retrieves the current JSON, updates ONLY ``refresh_token``,
  and preserves ``POSTGRES_DB`` / ``POSTGRES_USER`` / ``POSTGRES_PASSWORD``
  and every other existing key, then calls ``PutSecretValue`` to create a
  new version.
* **No hardcoded AWS credentials.** boto3 is created with no explicit keys
  so it uses the standard credential chain (on EC2: the attached instance
  IAM role).
* **Secrets never hit the logs.** Error paths log the AWS error *type* and
  the secret *name*/*key names* only — never the secret values, the
  password, or the refresh token.
* **Clear, typed failures.** AWS access errors (``AccessDeniedException``),
  a missing secret, invalid JSON, and missing/empty required keys each
  raise ``SecretsManagerError`` with an actionable (secret-free) message.

The caller supplies the secret name and region (wired from ``core/config.py``
settings), so nothing deployment-specific is hardcoded except the AWS SDK
defaults.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from core.logging_config import get_agent_logger

logger = get_agent_logger("secrets")

# Keys that live inside the single JSON secret.
POSTGRES_USER_KEY = "POSTGRES_USER"
POSTGRES_PASSWORD_KEY = "POSTGRES_PASSWORD"
POSTGRES_DB_KEY = "POSTGRES_DB"
REFRESH_TOKEN_KEY = "refresh_token"
ACCESS_TOKEN_KEY = "access_token"
ACCESS_TOKEN_EXPIRES_AT_KEY = "access_token_expires_at"
# The user the stored delegated tokens belong to (informational — the
# model is a single delegated identity). Never a secret.
TOKEN_USER_EMAIL_KEY = "token_user_email"


class SecretsManagerError(RuntimeError):
    """
    Raised for every failure sourcing or updating the AWS secret: AWS
    access/permission errors, a missing secret, malformed (non-JSON)
    secret payloads, and missing/empty required credential keys.

    The message is always safe to log and surface — it never contains a
    secret value.
    """


class SecretsManagerClient:
    """
    Minimal Secrets Manager accessor for ONE JSON secret.

    Instances are cheap and thread-safe; a single boto3 client is created
    lazily on first use and reused (boto3 clients are thread-safe for
    calls). The client holds NO AWS credentials of its own — boto3's
    default credential chain (EC2 instance IAM role in production) is used.
    """

    def __init__(self, secret_name: str, region_name: str) -> None:
        self._secret_name = secret_name
        self._region_name = region_name
        self._client: Any = None
        self._lock = threading.Lock()

    # -- boto3 client (lazy, no hardcoded credentials) ---------------------

    def _get_client(self) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    # No aws_access_key_id / aws_secret_access_key passed:
                    # boto3 resolves credentials via its standard chain,
                    # which on EC2 is the attached instance IAM role.
                    self._client = boto3.client(
                        "secretsmanager", region_name=self._region_name
                    )
        return self._client

    # -- read --------------------------------------------------------------

    def get_secret_dict(self) -> dict[str, Any]:
        """
        Fetch the secret and return it parsed as a JSON object.

        Raises ``SecretsManagerError`` on an AWS access/permission error,
        a missing secret, or a payload that is not a JSON object. The
        returned dict's values are NEVER logged.
        """
        client = self._get_client()
        try:
            response = client.get_secret_value(SecretId=self._secret_name)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "ClientError")
            if code == "AccessDeniedException":
                raise SecretsManagerError(
                    "Access denied reading secret "
                    f"{self._secret_name!r} in {self._region_name!r}. The EC2 "
                    "instance IAM role needs secretsmanager:GetSecretValue on "
                    "this secret."
                ) from exc
            if code == "ResourceNotFoundException":
                raise SecretsManagerError(
                    f"Secret {self._secret_name!r} was not found in region "
                    f"{self._region_name!r}."
                ) from exc
            # DecryptionFailure, InvalidRequestException, throttling, etc.
            raise SecretsManagerError(
                f"AWS error ({code}) reading secret {self._secret_name!r}."
            ) from exc
        except BotoCoreError as exc:
            # Networking / credential-chain resolution failures. str(exc)
            # here is a botocore message about transport/credentials, not
            # the secret contents.
            raise SecretsManagerError(
                f"Could not reach AWS Secrets Manager for secret "
                f"{self._secret_name!r}: {type(exc).__name__}."
            ) from exc

        raw = response.get("SecretString")
        if raw is None:
            # Binary secrets are not used by this app.
            raise SecretsManagerError(
                f"Secret {self._secret_name!r} has no SecretString (binary "
                "secrets are not supported)."
            )

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            # Report WHERE it failed, never WHAT the content was.
            raise SecretsManagerError(
                f"Secret {self._secret_name!r} is not valid JSON "
                f"(parse error at position {exc.pos})."
            ) from exc

        if not isinstance(parsed, dict):
            raise SecretsManagerError(
                f"Secret {self._secret_name!r} must be a JSON object, got "
                f"{type(parsed).__name__}."
            )
        return parsed

    # -- postgres credentials ---------------------------------------------

    def get_postgres_credentials(self) -> dict[str, str]:
        """
        Return ``{"user": ..., "password": ..., "db": ...}`` sourced from
        the secret.

        Raises ``SecretsManagerError`` if any of ``POSTGRES_USER``,
        ``POSTGRES_PASSWORD`` or ``POSTGRES_DB`` is missing or empty — an
        empty password is treated as a hard error so we never silently
        connect with a blank credential.
        """
        data = self.get_secret_dict()

        required = (POSTGRES_USER_KEY, POSTGRES_PASSWORD_KEY, POSTGRES_DB_KEY)
        missing = [
            key
            for key in required
            if not str(data.get(key) or "").strip()
        ]
        if missing:
            raise SecretsManagerError(
                "Secret "
                f"{self._secret_name!r} is missing or has empty value(s) for "
                f"required PostgreSQL key(s): {', '.join(missing)}."
            )

        return {
            "user": str(data[POSTGRES_USER_KEY]),
            "password": str(data[POSTGRES_PASSWORD_KEY]),
            "db": str(data[POSTGRES_DB_KEY]),
        }

    # -- refresh token -----------------------------------------------------

    def get_refresh_token(self) -> str | None:
        """
        Return the stored ``refresh_token`` (latest secret version), or
        ``None`` when the key is absent/empty (i.e. nobody has signed in
        yet). The token value is never logged.
        """
        data = self.get_secret_dict()
        token = data.get(REFRESH_TOKEN_KEY)
        token = str(token).strip() if token is not None else ""
        return token or None

    def get_tokens(self) -> dict[str, Any]:
        """
        Return ALL stored delegated tokens from the latest secret version:
        ``{"access_token": str|None, "refresh_token": str|None,
        "access_token_expires_at": float, "user_email": str|None}``.

        Any absent/empty token comes back as ``None`` (expiry as ``0.0``).
        This is the single read used by ``auth/token_store.get_latest_tokens``
        now that neither token lives in PostgreSQL. Token values are never
        logged.
        """
        data = self.get_secret_dict()

        access = data.get(ACCESS_TOKEN_KEY)
        access = str(access).strip() if access is not None else ""
        refresh = data.get(REFRESH_TOKEN_KEY)
        refresh = str(refresh).strip() if refresh is not None else ""
        user_email = data.get(TOKEN_USER_EMAIL_KEY)
        user_email = str(user_email).strip() if user_email is not None else ""

        try:
            expires_at = float(data.get(ACCESS_TOKEN_EXPIRES_AT_KEY) or 0.0)
        except (TypeError, ValueError):
            expires_at = 0.0

        return {
            "access_token": access or None,
            "refresh_token": refresh or None,
            "access_token_expires_at": expires_at,
            "user_email": user_email or None,
        }

    def get_secret_dict_for_stage(self, version_stage: str) -> dict[str, Any]:
        """
        Fetch and parse a specific VERSION STAGE of the secret.

        AWS Secrets Manager labels the latest version ``AWSCURRENT`` and the
        immediately-previous one ``AWSPREVIOUS`` automatically. This lets a
        caller read the prior version to compare it against the current one
        (e.g. "did the refresh token actually change on the last write?").

        Raises ``SecretsManagerError`` with the same, secret-free messages as
        ``get_secret_dict``. Returns an empty dict when the requested stage
        does not exist yet (e.g. ``AWSPREVIOUS`` before the second write).
        """
        client = self._get_client()
        try:
            response = client.get_secret_value(
                SecretId=self._secret_name, VersionStage=version_stage
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "ClientError")
            if code == "ResourceNotFoundException":
                # The stage/version simply doesn't exist yet.
                return {}
            if code == "AccessDeniedException":
                raise SecretsManagerError(
                    "Access denied reading secret "
                    f"{self._secret_name!r} stage {version_stage!r}."
                ) from exc
            raise SecretsManagerError(
                f"AWS error ({code}) reading secret {self._secret_name!r} "
                f"stage {version_stage!r}."
            ) from exc
        except BotoCoreError as exc:
            raise SecretsManagerError(
                f"Could not reach AWS Secrets Manager for secret "
                f"{self._secret_name!r} stage {version_stage!r}: "
                f"{type(exc).__name__}."
            ) from exc

        raw = response.get("SecretString")
        if raw is None:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def put_refresh_token(self, refresh_token: str) -> None:
        """
        Persist ``refresh_token`` into the secret, preserving every other
        key (``POSTGRES_DB``/``POSTGRES_USER``/``POSTGRES_PASSWORD`` and
        anything else). Reads the current JSON, updates ONLY the
        ``refresh_token`` key, and writes a new version via
        ``PutSecretValue``.

        A falsy/empty token is rejected rather than written, so a missing
        rotated token can never clobber a good stored one with a blank.
        """
        if not (refresh_token or "").strip():
            raise SecretsManagerError(
                "Refusing to store an empty refresh_token into secret "
                f"{self._secret_name!r}."
            )

        current = self.get_secret_dict()
        # Merge: keep all existing keys, update only refresh_token.
        current[REFRESH_TOKEN_KEY] = refresh_token

        client = self._get_client()
        try:
            client.put_secret_value(
                SecretId=self._secret_name,
                SecretString=json.dumps(current),
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "ClientError")
            if code == "AccessDeniedException":
                raise SecretsManagerError(
                    "Access denied writing secret "
                    f"{self._secret_name!r}. The EC2 instance IAM role needs "
                    "secretsmanager:PutSecretValue on this secret."
                ) from exc
            if code == "ResourceNotFoundException":
                raise SecretsManagerError(
                    f"Secret {self._secret_name!r} was not found when writing "
                    f"in region {self._region_name!r}."
                ) from exc
            raise SecretsManagerError(
                f"AWS error ({code}) writing secret {self._secret_name!r}."
            ) from exc
        except BotoCoreError as exc:
            raise SecretsManagerError(
                f"Could not reach AWS Secrets Manager to write secret "
                f"{self._secret_name!r}: {type(exc).__name__}."
            ) from exc

        # Length only — never the token itself.
        logger.info(
            "[SECRETS] refresh_token updated in secret (new version written)"
        )

    def put_tokens(
        self,
        access_token: str | None,
        refresh_token: str | None,
        access_token_expires_at: float,
        user_email: str | None = None,
    ) -> str | None:
        """
        Persist BOTH delegated tokens into the secret in a SINGLE new
        version, preserving every other key (``POSTGRES_*`` etc.). Reads the
        current JSON, overwrites ``access_token`` /
        ``access_token_expires_at`` / ``refresh_token`` (and optionally
        ``token_user_email``), then calls ``PutSecretValue`` once.

        Semantics matching the old split store:

        * ``access_token`` is always written when non-empty (the whole point
          of a sign-in / refresh is to publish the latest access token). An
          empty access token is a no-op for that key.
        * ``refresh_token`` is only overwritten when non-empty — Microsoft
          does not always return a rotated refresh token, and a good stored
          one must never be clobbered with a blank.

        Returns the new ``VersionId`` on success (useful for log/audit), or
        ``None`` when there was nothing to write (both tokens empty). Raises
        ``SecretsManagerError`` on an AWS failure. Token values are never
        logged.
        """
        access_clean = (access_token or "").strip()
        refresh_clean = (refresh_token or "").strip()

        if not access_clean and not refresh_clean:
            # Nothing to publish — don't spend a secret version on a no-op.
            logger.info(
                "[SECRETS] put_tokens called with no access/refresh token; "
                "secret left unchanged"
            )
            return None

        current = self.get_secret_dict()

        # Merge: keep all existing keys, overwrite only the token keys that
        # actually carry a value.
        if access_clean:
            current[ACCESS_TOKEN_KEY] = access_clean
            current[ACCESS_TOKEN_EXPIRES_AT_KEY] = float(access_token_expires_at or 0.0)
        if refresh_clean:
            current[REFRESH_TOKEN_KEY] = refresh_clean
        if (user_email or "").strip():
            current[TOKEN_USER_EMAIL_KEY] = user_email.strip()

        client = self._get_client()
        try:
            response = client.put_secret_value(
                SecretId=self._secret_name,
                SecretString=json.dumps(current),
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "ClientError")
            if code == "AccessDeniedException":
                raise SecretsManagerError(
                    "Access denied writing secret "
                    f"{self._secret_name!r}. The EC2 instance IAM role needs "
                    "secretsmanager:PutSecretValue on this secret."
                ) from exc
            if code == "ResourceNotFoundException":
                raise SecretsManagerError(
                    f"Secret {self._secret_name!r} was not found when writing "
                    f"in region {self._region_name!r}."
                ) from exc
            raise SecretsManagerError(
                f"AWS error ({code}) writing secret {self._secret_name!r}."
            ) from exc
        except BotoCoreError as exc:
            raise SecretsManagerError(
                f"Could not reach AWS Secrets Manager to write secret "
                f"{self._secret_name!r}: {type(exc).__name__}."
            ) from exc

        version_id = response.get("VersionId")
        # Lengths + which keys changed only — never the token values.
        logger.info(
            "[SECRETS] delegated tokens updated in secret (new version=%s): "
            "access_token=%s refresh_token=%s",
            (version_id[:8] + "…") if version_id else "?",
            f"written(len={len(access_clean)})" if access_clean else "unchanged",
            f"written(len={len(refresh_clean)})" if refresh_clean else "unchanged",
        )
        return version_id
