import { createFileRoute } from "@tanstack/react-router";
import { CROWorkspace } from "@/views/CROWorkspace";

export const Route = createFileRoute("/credit")({
  component: CROPage,
});

function CROPage() {
  return <CROWorkspace />;
}
