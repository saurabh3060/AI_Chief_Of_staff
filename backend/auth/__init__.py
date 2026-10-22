"""
auth package
=============
Delegated Microsoft OAuth routes (`/login`, `/callback`), ported from
the reference implementation the user provided. These routes are only
meaningful when `GRAPH_AUTH_MODE=delegated` - they let a user sign in
via the browser and obtain a bearer token that agents/routes can then
pass to `core.graph_client.GraphClient` methods.

When `GRAPH_AUTH_MODE=app_only` these routes are still registered
(so the API surface doesn't change based on config) but are not
required for agents to function, since the app acquires its own token.
"""
