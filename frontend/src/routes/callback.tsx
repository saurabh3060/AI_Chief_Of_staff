import { createFileRoute, redirect } from "@tanstack/react-router";

/**
 * /callback
 * =========
 * OAuth landing route. nginx proxies `/callback` to the frontend (TanStack
 * Start SSR server on port 8518), so when Microsoft redirects the browser
 * to `https://digitaoffice-dev.rbl.bank.in/callback?code=...` after
 * sign-in, this route handles it.
 *
 * The actual token exchange must happen on the backend (auth/routes.py
 * `GET /callback`), which is reachable through nginx at `/api/callback`
 * (the `/api/` prefix is stripped by nginx before it hits the backend).
 *
 * Strategy: use `beforeLoad` to immediately redirect to `/api/callback`
 * with the original query string. `beforeLoad` runs before any component
 * rendering on **both server and client**:
 *   - SSR (server): returns a 302 Response → browser follows it as a
 *     full-page navigation to the backend endpoint.
 *   - CSR (hydration): `reloadDocument: true` triggers `window.location`
 *     navigation instead of client-side routing.
 *
 * This avoids the "stuck on loading" problem that `useEffect`-based
 * redirects cause in SSR apps (useEffect never fires server-side, and
 * hydration can fail or be delayed).
 *
 * The backend's `/callback` handler exchanges the code for Graph tokens,
 * creates a server-side session, sets an HTTP-only cookie on a 200
 * response, and that response's inline `<script>` navigates the browser
 * to FRONTEND_URL — completing the sign-in loop.
 *
 * If there is no `code` param (direct navigation / stale bookmark), we
 * redirect to `/` so the normal auth-check in AppState.tsx takes over.
 */
export const Route = createFileRoute("/callback")({
  // Redirect before any rendering happens — works in SSR and CSR.
  beforeLoad: ({ location }) => {
    // Extract the query string from the location. TanStack Router provides
    // several representations; we try them in order of reliability:
    //   1. location.href — full path like "/callback?code=...&state=..."
    //   2. location.searchStr — the "?code=...&state=..." portion
    //   3. location.search — parsed object (may be stripped by validators)
    let qs = "";

    // Prefer extracting from href (always has the raw path + query).
    const href = location.href || "";
    const qIdx = href.indexOf("?");
    if (qIdx !== -1) {
      const rawQs = href.slice(qIdx);
      if (rawQs.includes("code=")) {
        qs = rawQs;
      }
    }

    // Fallback: check searchStr directly.
    if (!qs) {
      const searchStr = location.searchStr || "";
      if (searchStr.includes("code=")) {
        qs = searchStr.startsWith("?") ? searchStr : `?${searchStr}`;
      }
    }

    // Last resort: check the parsed search object.
    if (!qs && typeof location.search === "object" && location.search !== null) {
      const s = location.search as Record<string, string>;
      if (s.code) {
        qs = "?" + new URLSearchParams(s as Record<string, string>).toString();
      }
    }

    if (qs) {
      // Forward the OAuth code to the backend's /callback handler via the
      // /api/ prefix that nginx routes to the backend (with prefix strip).
      // reloadDocument: true ensures a full-page navigation (not SPA routing)
      // so the browser processes the Set-Cookie header from the response.
      throw redirect({
        href: `/api/callback${qs}`,
        statusCode: 302,
        reloadDocument: true,
      });
    }

    // No authorization code — nothing to exchange. Go home.
    throw redirect({ to: "/", replace: true });
  },
  // Component should never render (beforeLoad always redirects), but
  // provide a minimal fallback just in case.
  component: () => null,
});
