import { useNavigate } from "@tanstack/react-router";
import type { ViewName } from "@/context/AppState";

/** Map ViewName → URL path */
export const VIEW_PATH: Record<ViewName, string> = {
  Launchpad:    "/",
  CRO:          "/credit",
  Retail:       "/bbrl",
  Wholesale:    "/wholesale",
  RetailAssets: "/retail-advances",
  Ledger:       "/ledger",
};

/**
 * Drop-in replacement for the old `navigate(view, phase?)` from AppState.
 * Phase is intentionally ignored — it was never consumed by view components
 * (PhaseTabs manages its own local state). If phase-in-URL is needed later,
 * add it here as a search param.
 */
export function useAppNavigate() {
  const navigate = useNavigate();
  return (view: ViewName, _phase?: string) => {
    navigate({ to: VIEW_PATH[view] });
  };
}
