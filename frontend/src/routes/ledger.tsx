import { createFileRoute } from "@tanstack/react-router";
import { Ledger } from "@/views/Ledger";

export const Route = createFileRoute("/ledger")({
  component: LedgerPage,
});

function LedgerPage() {
  return <Ledger />;
}
