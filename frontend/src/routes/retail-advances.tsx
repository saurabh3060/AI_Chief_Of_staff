import { createFileRoute } from "@tanstack/react-router";
import { RetailAssetsWorkspace } from "@/views/RetailAssetsWorkspace";

export const Route = createFileRoute("/retail-advances")({
  component: RetailAssetsPage,
});

function RetailAssetsPage() {
  return <RetailAssetsWorkspace />;
}
