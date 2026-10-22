import { createFileRoute } from "@tanstack/react-router";
import { Launchpad } from "@/views/Launchpad";

export const Route = createFileRoute("/")({
  component: LaunchpadPage,
});

function LaunchpadPage() {
  return <Launchpad />;
}
