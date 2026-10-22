"""
test_weaviate_connection.py
===========================
Standalone Weaviate connectivity + schema check for the backend, modelled on
``test_db_connection.py``.

Why this exists
---------------
Weaviate holds the backend's embedding vectors (the email agent's cluster +
meeting vectors, and the documents bot's block/series vectors). It is a
*rebuildable* copy — PostgreSQL is the master — but the app still needs it
reachable to serve vector search. The Weaviate v4 client talks over TWO
independent endpoints (REST over TLS + gRPC, often on different hosts/ports),
so "it half-connects" is a common failure: REST is reachable but gRPC is
blocked, which only shows up on the first vector query. This script isolates
*just* the Weaviate connection using the SAME config (the ``WEAVIATE_*``
settings from ``backend/.env`` via ``core.config``) and the SAME client call
(``weaviate.connect_to_custom``) the app uses (see ``core/weaviate/client.py``).

Run it (from the backend/ folder, with the venv active):

    python test_weaviate_connection.py

It will:
  1. Load the WEAVIATE_* settings from .env (via core.config, same as the app).
  2. Connect over REST + gRPC with connect_to_custom (same as the app).
  3. Check is_ready() (liveness) and is_live().
  4. Force a schema load (collections.list_all) — this is what proves the
     gRPC path actually works, not just REST. This is the exact cold-start
     check core/weaviate/client.py warms up on connect.
  5. List collections and show each one's approximate object count.

Nothing here writes to Weaviate.
"""

import sys

from core.config import get_settings


def main() -> int:
    settings = get_settings()

    try:
        import weaviate
    except ImportError:
        print("ERROR: the 'weaviate-client' package is not installed. Activate the venv first:")
        print(r"    .\venv\Scripts\activate")
        return 1

    print(
        "Using Weaviate: "
        f"REST {settings.weaviate_http_host}:{settings.weaviate_http_port} "
        f"(secure={settings.weaviate_http_secure}) | "
        f"gRPC {settings.weaviate_grpc_host}:{settings.weaviate_grpc_port} "
        f"(secure={settings.weaviate_grpc_secure})"
    )
    print(f"Auth          : {'API key' if settings.weaviate_api_key else 'anonymous'}")
    print("Connecting (connect_to_custom, same as the app)...")

    auth = None
    if settings.weaviate_api_key:
        from weaviate.classes.init import Auth
        auth = Auth.api_key(settings.weaviate_api_key)

    # --- 1. Connect over REST + gRPC (identical to core/weaviate/client.py) ---
    try:
        client = weaviate.connect_to_custom(
            http_host=settings.weaviate_http_host,
            http_port=settings.weaviate_http_port,
            http_secure=settings.weaviate_http_secure,
            grpc_host=settings.weaviate_grpc_host,
            grpc_port=settings.weaviate_grpc_port,
            grpc_secure=settings.weaviate_grpc_secure,
            auth_credentials=auth,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED to connect: {type(exc).__name__}: {exc}")
        print("Usually this is the gRPC endpoint: the REST host can be reachable")
        print("while gRPC (a different host/port) is blocked by a firewall or")
        print("wrong WEAVIATE_GRPC_HOST/PORT. Verify both endpoints in .env.")
        return 2

    try:
        # --- 2/3. Readiness + liveness ---
        try:
            ready = client.is_ready()
        except Exception as exc:  # noqa: BLE001
            print(f"\nFAILED: is_ready() raised: {type(exc).__name__}: {exc}")
            return 2
        if not ready:
            print("\nFAILED: connected but the deployment reports NOT ready.")
            print("REST reached the server but it is not serving yet. Retry shortly.")
            return 2
        print("is_ready() -> True  (REST connectivity OK)")

        try:
            live = client.is_live()
            print(f"is_live()  -> {live}")
        except Exception as exc:  # noqa: BLE001
            print(f"is_live() raised (non-fatal): {type(exc).__name__}: {exc}")

        try:
            meta = client.get_meta()
            print(f"Weaviate version: {meta.get('version')}")
        except Exception as exc:  # noqa: BLE001
            print(f"(could not read get_meta(): {type(exc).__name__}: {exc})")

        # --- 4. Force a schema load — proves the gRPC path works, not just REST.
        # This is the exact cold-start warm-up core/weaviate/client.py does;
        # a failure here (while is_ready() was True) is the classic
        # REST-OK-but-gRPC-broken case.
        print("\nLoading schema (collections.list_all) — this exercises gRPC...")
        try:
            collections = client.collections.list_all()
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            print("  REST is reachable but the gRPC path is not. Vector queries")
            print("  would fail with 'could not find class ... in schema'. Check")
            print("  WEAVIATE_GRPC_HOST/WEAVIATE_GRPC_PORT/WEAVIATE_GRPC_SECURE.")
            return 3

        names = sorted(collections.keys())
        print(f"  OK — schema loaded. Collections ({len(names)}): "
              f"{', '.join(names) if names else '(none)'}")

        # --- 5. Per-collection object counts ---
        for name in names:
            try:
                coll = client.collections.get(name)
                agg = coll.aggregate.over_all(total_count=True)
                print(f'  collection "{name}": {agg.total_count} object(s)')
            except Exception as exc:  # noqa: BLE001
                print(f'  collection "{name}": count unavailable '
                      f"({type(exc).__name__}: {exc})")

        print("\nWeaviate connectivity (REST + gRPC) and schema read both work.")
        return 0

    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
