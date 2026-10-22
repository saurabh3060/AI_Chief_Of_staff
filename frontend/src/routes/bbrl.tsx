import { createFileRoute } from "@tanstack/react-router";
import { RetailWorkspace } from "@/views/RetailWorkspace";

export const Route = createFileRoute("/bbrl")({
  component: RetailPage,
});

function RetailPage() {
  return <RetailWorkspace />;
}
