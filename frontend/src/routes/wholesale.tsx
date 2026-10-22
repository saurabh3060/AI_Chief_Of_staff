import { createFileRoute } from "@tanstack/react-router";
import { WholesaleWorkspace } from "@/views/WholesaleWorkspace";

export const Route = createFileRoute("/wholesale")({
  component: WholesalePage,
});

function WholesalePage() {
  return <WholesaleWorkspace />;
}
