// @lovable.dev/vite-tanstack-config already includes the following — do NOT add them manually
// or the app will break with duplicate plugins:
//   - tanstackStart, viteReact, tailwindcss, tsConfigPaths, cloudflare (build-only),
//     componentTagger (dev-only), VITE_* env injection, @ path alias, React/TanStack dedupe,
//     error logger plugins, and sandbox detection (port/host/strictPort).
// You can pass additional config via defineConfig({ vite: { ... } }) if needed.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";

// Redirect TanStack Start's bundled server entry to src/server.ts (our SSR error wrapper).
// @cloudflare/vite-plugin builds from this — wrangler.jsonc main alone is insufficient.
export default defineConfig({
  tanstackStart: {
    server: { entry: "server" },
  },
  vite: {
    server: {
      port: 8518,
      host: true,
      strictPort: true,
      // nginx proxies requests with the original public Host header
      // (digitaoffice-dev.rbl.bank.in), which Vite's dev server otherwise
      // rejects as an untrusted host. Allow it explicitly, plus localhost
      // for direct local access to the dev server.
      allowedHosts: ["digitaloffice-uat.rbl.bank.in","digitaoffice-dev.rbl.bank.in", "digitaloffice.rbl.bank.in", "localhost", "127.0.0.1", "10.240.87.68"],
      proxy: {
        "/api": {
          target: "http://localhost:8516",
          changeOrigin: true,
          rewrite: (path: string) => path.replace(/^\/api/, ""),
          timeout: 600000,
          proxyTimeout: 600000,
        },
      },
    },
  },
});
