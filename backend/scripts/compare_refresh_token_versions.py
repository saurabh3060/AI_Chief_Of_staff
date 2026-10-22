"""
scripts/compare_refresh_token_versions.py
==========================================
Read-only proof that NEW delegated tokens were written to AWS Secrets
Manager on the last sign-in, and how they compare to the previous version —
WITHOUT ever printing a token.

Both the ``access_token`` AND the ``refresh_token`` now live in the secret
(there is no ``graph_oauth_tokens`` table anymore), so this report covers
BOTH keys. It is the quickest way to confirm "the login actually replaced
the old tokens".

How it works
------------
Every ``PutSecretValue`` creates a new secret version. AWS automatically
labels the latest version ``AWSCURRENT`` and the one before it
``AWSPREVIOUS``. This script reads both stages, then for EACH of the
``access_token`` and ``refresh_token`` keys it reports only:

  * length
  * a SHA-256 fingerprint (hex) of the token

and whether CURRENT differs from PREVIOUS. The raw token is never shown or
logged — a fingerprint lets you confirm "it changed / it's the same" safely.

Also lists the secret's version metadata (version id + stage labels + created
date) so you can see the write history.

Usage (from the backend/ directory, venv active)
-------------------------------------------------
    python scripts/compare_refresh_token_versions.py

Exit code: 0 always (this is an informational, read-only report).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from core.config import get_settings  # noqa: E402
from core.secrets_manager import (  # noqa: E402
    ACCESS_TOKEN_KEY,
    REFRESH_TOKEN_KEY,
    SecretsManagerClient,
    SecretsManagerError,
)


def _fingerprint(token: str | None) -> str:
    """SHA-256 hex of a token, or '<none>' when absent. Never the token."""
    if not token:
        return "<none>"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _describe_versions(client: SecretsManagerClient, secret_name: str) -> None:
    """Print version id + stage labels + created date for each version."""
    boto = client._get_client()  # internal accessor; read-only use
    try:
        resp = boto.list_secret_version_ids(SecretId=secret_name, IncludeDeprecated=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not list versions: {type(exc).__name__})")
        return
    versions = resp.get("Versions", [])
    # Most-recent first by created date when present.
    versions.sort(key=lambda v: v.get("CreatedDate") or 0, reverse=True)
    for v in versions:
        stages = ",".join(v.get("VersionStages", [])) or "-"
        created = v.get("CreatedDate")
        vid = v.get("VersionId", "")
        short = f"{vid[:8]}…" if vid else "?"
        print(f"  version {short}  stages=[{stages}]  created={created}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reveal",
        action="store_true",
        help="DANGER: also print the RAW refresh token values (CURRENT and "
        "PREVIOUS). Off by default. The raw token is a long-lived credential "
        "— printing it exposes it in your terminal and shell history.",
    )
    args = parser.parse_args()

    settings = get_settings()
    client = SecretsManagerClient(
        secret_name=settings.aws_secret_name,
        region_name=settings.aws_region,
    )

    # print(f"Secret: {settings.aws_secret_name} @ {settings.aws_region}")
    print()

    print("[Version history]")
    _describe_versions(client, settings.aws_secret_name)
    print()

    try:
        current = client.get_secret_dict_for_stage("AWSCURRENT")
        previous = client.get_secret_dict_for_stage("AWSPREVIOUS")
    except SecretsManagerError as exc:
        print(f"ERROR: {exc}")
        return 0

    for key in (ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY):
        _report_key(key, current, previous)
        print()

    return 0


def _report_key(key: str, current: dict, previous: dict) -> None:
    """Print the CURRENT vs PREVIOUS comparison for a single token key."""
    cur_token = str(current.get(key) or "").strip() or None
    prev_token = str(previous.get(key) or "").strip() or None

    cur_fp = _fingerprint(cur_token)
    prev_fp = _fingerprint(prev_token)

    print(f"[{key}: CURRENT vs PREVIOUS]")
    print(f"  CURRENT  : len={len(cur_token) if cur_token else 0:<5} sha256={cur_fp}")
    print(f"  PREVIOUS : len={len(prev_token) if prev_token else 0:<5} sha256={prev_fp}")

    if cur_token is None:
        print(f"  RESULT: no {key} in the current version yet.")
    elif prev_token is None:
        print(
            f"  RESULT: a {key} exists in CURRENT and there is no previous "
            "version to compare — this looks like the FIRST write."
        )
    elif cur_fp != prev_fp:
        print(
            f"  RESULT: {key} CHANGED between the previous and current version "
            "— a new token was inserted (the login replaced the old one)."
        )
    else:
        print(
            f"  RESULT: {key} is IDENTICAL to the previous version — the last "
            "write re-stored the same token (e.g. a non-destructive "
            "write-probe, or Microsoft returned no rotated token)."
        )


if __name__ == "__main__":
    raise SystemExit(main())
