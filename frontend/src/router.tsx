import { QueryClient } from "@tanstack/react-query";
import { createRouter } from "@tanstack/react-router";
import { routeTree } from "./routeTree.gen";

export const getRouter = () => {
  const queryClient = new QueryClient();

  const router = createRouter({
    routeTree,
    context: { queryClient },
    scrollRestoration: true,
    defaultPreloadStaleTime: 0,
    // This is a client-only SPA: the auth check (`/auth/me`) and every
    // data fetch in AppState run in a `useEffect` (client-only), and the
    // UI is gated behind `AuthGate` until that resolves. Server-rendering
    // therefore only ever produces the loading shell, but the SSR render
    // pass still reaches the backend over the network during the request
    // - and when that fetch throws (backend unreachable / connection
    // reset mid-render), h3 turns it into a catastrophic 500
    // ("h3 swallowed SSR error: HTTPError") and the aborted in-flight
    // request surfaces as an unhandled ECONNRESET on the Node server.
    // Disabling SSR makes the server ship only the shell and moves all
    // rendering + fetching to the browser (the model the app already
    // assumes), eliminating both the swallowed-500 and the abort noise.
    defaultSsr: false,
  });

  return router;
};
