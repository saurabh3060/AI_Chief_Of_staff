import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Outlet,
  Link,
  createRootRouteWithContext,
  useRouter,
  HeadContent,
  Scripts,
} from "@tanstack/react-router";

import appCss from "../styles.css?url";
import { AppStateProvider, useAppState } from "@/context/AppState";
import { Sidebar } from "@/components/Sidebar";
import { TopHeader } from "@/components/TopHeader";
import { Toast } from "@/components/Toast";

function NotFoundComponent() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-7xl font-bold text-foreground">404</h1>
        <h2 className="mt-4 text-xl font-semibold text-foreground">Page not found</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <div className="mt-6">
          <Link
            to="/"
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Go home
          </Link>
        </div>
      </div>
    </div>
  );
}

function ErrorComponent({ error, reset }: { error: Error; reset: () => void }) {
  console.error(error);
  const router = useRouter();

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-xl font-semibold tracking-tight text-foreground">
          This page didn't load
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Something went wrong on our end. You can try refreshing or head back home.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <button
            onClick={() => {
              router.invalidate();
              reset();
            }}
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Try again
          </button>
          <a
            href="/"
            className="inline-flex items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent"
          >
            Go home
          </a>
        </div>
      </div>
    </div>
  );
}

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      { title: "RBL - CEO's Digital Office Remix" },
      { name: "description", content: "AI Agent to enable CEO of a large retail bank in India" },
      { name: "author", content: "Lovable" },
      { property: "og:title", content: "RBL - CEO's Digital Office Remix" },
      { property: "og:description", content: "AI Agent to enable CEO of a large retail bank in India" },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
      { name: "twitter:site", content: "@Lovable" },
      { name: "twitter:title", content: "RBL - CEO's Digital Office Remix" },
      { name: "twitter:description", content: "AI Agent to enable CEO of a large retail bank in India" },
      { property: "og:image", content: "https://pub-bb2e103a32db4e198524a2e9ed8f35b4.r2.dev/95181638-509e-4e0d-b3e9-098fe3403ae7/id-preview-825b9184--65ecdc1e-465f-4138-9d01-a478f90ffe6d.lovable.app-1779786282059.png" },
      { name: "twitter:image", content: "https://pub-bb2e103a32db4e198524a2e9ed8f35b4.r2.dev/95181638-509e-4e0d-b3e9-098fe3403ae7/id-preview-825b9184--65ecdc1e-465f-4138-9d01-a478f90ffe6d.lovable.app-1779786282059.png" },
    ],
    links: [
      { rel: "preconnect", href: "https://fonts.googleapis.com" },
      { rel: "preconnect", href: "https://fonts.gstatic.com", crossOrigin: "anonymous" },
      { rel: "stylesheet", href: "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Serif+Display&display=swap" },
      { rel: "stylesheet", href: appCss },
    ],
  }),
  shellComponent: RootShell,
  component: RootComponent,
  notFoundComponent: NotFoundComponent,
  errorComponent: ErrorComponent,
});

function RootShell({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body>
        {children}
        <Scripts />
      </body>
    </html>
  );
}

// ── Auth loading gate ──────────────────────────────────────────────────────
// Blocks rendering of the app shell until the /auth/me check resolves.
function AuthGate({ children }: { children: ReactNode }) {
  const { authChecked, authState, authError, retryAuthCheck, loginRedirectExhausted, signIn } =
    useAppState();

  if (authError) {
    return <AuthErrorScreen message={authError} onRetry={retryAuthCheck} />;
  }

  if (loginRedirectExhausted) {
    return <SignInRetryScreen onRetry={signIn} />;
  }

  if (!authChecked || !authState?.authenticated) {
    return <AuthLoadingScreen />;
  }

  return <>{children}</>;
}

function AuthLoadingScreen() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-[#F4F6FA]">
      <div className="flex flex-col items-center gap-3">
        <div
          className="h-8 w-8 rounded-full border-2 border-[#E2E8F0] border-t-[#0A2540] animate-spin"
          role="status"
          aria-label="Checking sign-in status"
        />
        <div className="text-[12.5px] text-[#475569]">Checking sign-in status…</div>
      </div>
    </div>
  );
}

function SignInRetryScreen({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="flex min-h-screen items-center justify-center bg-[#F4F6FA]">
      <div className="flex flex-col items-center gap-3 text-center max-w-[360px]">
        <div className="text-[15px] font-semibold text-[#0A2540]">Sign-in required</div>
        <div className="text-[12.5px] text-[#475569]">
          We couldn't confirm your session. This can happen if sign-in didn't complete
          or the session expired.
        </div>
        <button
          onClick={onRetry}
          className="mt-1 text-[12.5px] font-semibold px-4 py-2 rounded-[6px] bg-[#0A2540] text-white hover:opacity-90 transition"
        >
          Sign in
        </button>
      </div>
    </div>
  );
}

function AuthErrorScreen({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex min-h-screen items-center justify-center bg-[#F4F6FA]">
      <div className="flex flex-col items-center gap-3 text-center max-w-[360px]">
        <div className="text-[15px] font-semibold text-[#0A2540]">
          Couldn't check sign-in status
        </div>
        <div className="text-[12.5px] text-[#475569]">
          We couldn't reach the sign-in service in time. This is usually a
          temporary network or server issue, not a problem with your account.
        </div>
        <div className="text-[11px] text-[#94A3B8] break-words">{message}</div>
        <button
          onClick={onRetry}
          className="mt-1 text-[12.5px] font-semibold px-4 py-2 rounded-[6px] bg-[#0A2540] text-white hover:opacity-90 transition"
        >
          Try again
        </button>
      </div>
    </div>
  );
}

// ── App shell: sidebar + header + page content ─────────────────────────────
function AppShell() {
  return (
    <div className="flex min-w-[1280px] bg-[#F4F6FA] min-h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <TopHeader />
        <main className="flex-1">
          <Outlet />
        </main>
      </div>
      <Toast />
    </div>
  );
}

function RootComponent() {
  const { queryClient } = Route.useRouteContext();

  return (
    <QueryClientProvider client={queryClient}>
      <AppStateProvider>
        <AuthGate>
          <AppShell />
        </AuthGate>
      </AppStateProvider>
    </QueryClientProvider>
  );
}
