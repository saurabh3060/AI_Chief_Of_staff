"""
scripts/validate_secrets_manager.py
===================================
Read-only validation that the AWS Secrets Manager wiring is correct, using
the SAME config and clients the app uses. Nothing secret is ever printed.

What it checks
--------------
1. Secrets Manager access: the EC2 instance IAM role can read the secret
   (``secretsmanager:GetSecretValue``) and the payload is valid JSON.
2. PostgreSQL credentials: ``POSTGRES_USER`` / ``POSTGRES_PASSWORD`` /
   ``POSTGRES_DB`` are present and non-empty (values are NOT printed — only
   "present/length" is reported).
3. Database connectivity: opens ONE asyncpg connection with the DSN built
   from those credentials and runs ``SELECT 1`` (no writes).
4. Delegated token readback: reports whether BOTH the ``access_token`` and
   the ``refresh_token`` are stored, each token's length, and the access
   token's expiry (absolute epoch + seconds remaining) — never the token
   values themselves. This reflects the new model where BOTH delegated
   tokens live in Secrets Manager (there is no ``graph_oauth_tokens`` table).

Optional (explicit) write probe
--------------------------------
Pass ``--write-probe`` to additionally verify ``PutSecretValue`` works: it
reads the current access + refresh tokens, writes them straight back
unchanged (so the stored values are preserved — this is NOT a destructive
test), then reads them back and asserts they match. Skipped by default so a
normal run performs no writes at all.

Usage (from the backend/ directory, venv active)
------------------------------------------------
    python scripts/validate_secrets_manager.py [--write-probe]

Exit code: 0 = all checks passed, 1 = a check failed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import asyncpg  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.secrets_manager import SecretsManagerClient, SecretsManagerError  # noqa: E402

CONNECT_TIMEOUT_SECONDS = 15


def _ok(msg: str) -> None:
    print(f"  OK   : {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL : {msg}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-probe",
        action="store_true",
        help="Also verify PutSecretValue by rewriting the current refresh "
        "token unchanged (non-destructive).",
    )
    args = parser.parse_args()

    settings = get_settings()
    client = SecretsManagerClient(
        secret_name=settings.aws_secret_name,
        region_name=settings.aws_region,
    )

    failures = 0

    # 1) Secrets Manager access + valid JSON.
    # print(f"[1] Secrets Manager access ({settings.aws_secret_name} @ {settings.aws_region})")
    try:
        data = client.get_secret_dict()
        _ok(f"secret read; {len(data)} key(s) present")
    except SecretsManagerError as exc:
        _fail(str(exc))
        # Can't continue without the secret.
        print("\nRESULT: FAILED")
        return 1

    # 2) PostgreSQL credentials present (values never printed).
    print("[2] PostgreSQL credentials")
    try:
        creds = client.get_postgres_credentials()
        _ok(
            "POSTGRES_USER/POSTGRES_PASSWORD/POSTGRES_DB present "
            f"(user len={len(creds['user'])}, password len={len(creds['password'])}, "
            f"db len={len(creds['db'])})"
        )
    except SecretsManagerError as exc:
        _fail(str(exc))
        failures += 1
        creds = None

    # 3) Database connectivity (SELECT 1), only if a host is configured and
    #    we have creds (the DSN is already built on settings).
    print("[3] PostgreSQL connectivity")
    dsn = (settings.postgres_dsn or "").strip()
    if not dsn:
        print("  SKIP : POSTGRES_HOST not set — Postgres disabled (JSON-file fallback).")
    elif creds is None:
        print("  SKIP : credentials missing (see check 2).")
    else:
        conn = None
        try:
            conn = await asyncpg.connect(dsn=dsn, timeout=CONNECT_TIMEOUT_SECONDS)
            one = await conn.fetchval("SELECT 1")
            if one == 1:
                db = await conn.fetchval("SELECT current_database()")
                user = await conn.fetchval("SELECT current_user")
                _ok(f"connected (database={db}, user={user})")
            else:
                _fail(f"unexpected SELECT 1 result: {one!r}")
                failures += 1
        except (OSError, asyncpg.PostgresError, asyncio.TimeoutError) as exc:
            _fail(f"could not connect: {type(exc).__name__}: {exc}")
            failures += 1
        finally:
            if conn is not None:
                await conn.close()

    # 4) Delegated token readback (presence + length + expiry only).
    print("[4] Delegated token readback (both tokens)")
    try:
        tokens = client.get_tokens()
        access = tokens.get("access_token")
        refresh = tokens.get("refresh_token")
        expires_at = float(tokens.get("access_token_expires_at") or 0.0)

        if access:
            remaining = int(expires_at - time.time()) if expires_at else None
            if remaining is None:
                _ok(f"access_token present (length={len(access)}, no expiry recorded)")
            elif remaining > 0:
                _ok(
                    f"access_token present (length={len(access)}, "
                    f"expires in {remaining}s)"
                )
            else:
                _ok(
                    f"access_token present (length={len(access)}) but EXPIRED "
                    f"{abs(remaining)}s ago — a refresh will mint a new one"
                )
        else:
            print("  INFO : no access_token stored yet (nobody has signed in).")

        if refresh:
            _ok(f"refresh_token present (length={len(refresh)})")
        else:
            print("  INFO : no refresh_token stored yet (nobody has signed in).")
    except SecretsManagerError as exc:
        _fail(str(exc))
        failures += 1

    # 5) Optional non-destructive write probe (rewrites BOTH tokens unchanged).
    if args.write_probe:
        print("[5] PutSecretValue write probe (non-destructive, both tokens)")
        try:
            before = client.get_tokens()
            before_access = before.get("access_token")
            before_refresh = before.get("refresh_token")
            if not before_access and not before_refresh:
                print("  SKIP : no stored tokens to rewrite; sign in first.")
            else:
                client.put_tokens(
                    access_token=before_access,
                    refresh_token=before_refresh,
                    access_token_expires_at=float(
                        before.get("access_token_expires_at") or 0.0
                    ),
                    user_email=before.get("user_email"),
                )
                after = client.get_tokens()
                if (
                    after.get("access_token") == before_access
                    and after.get("refresh_token") == before_refresh
                ):
                    _ok("write succeeded and readback matches (values unchanged)")
                else:
                    _fail("readback did not match the written values")
                    failures += 1
        except SecretsManagerError as exc:
            _fail(str(exc))
            failures += 1

    print()
    if failures:
        print(f"RESULT: FAILED ({failures} check(s))")
        return 1
    print("RESULT: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
