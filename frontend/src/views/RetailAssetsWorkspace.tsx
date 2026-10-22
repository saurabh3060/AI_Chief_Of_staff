import { PreMeetingBrief } from "@/components/PreMeetingBrief";
import { RA_CHIPS, retailAssetsResponse } from "@/data/mockData";
import { useAppState } from "@/context/AppState";
import { useMemo } from "react";
import type { DrMetric } from "@/components/DRWorkspace";
import type { OpenActionItem } from "@/components/PreMeetingBrief";

/* ── Retail Assets & Collections metrics · Latest MIS ─────────── */
const RA_METRICS: DrMetric[] = [
  { label: "HL-LAP Book (Mortgages)", value: "₹21,775 Cr", delta: "↑ MoM +1.76%", tone: "emerald" },
  { label: "LAGO Book", value: "₹4,320 Cr", delta: "↑ MoM +3.35%", tone: "emerald" },
  { label: "RVF Book", value: "₹3,830 Cr", delta: "↑ MoM +0.24%", tone: "emerald" },
  { label: "Business Banking Book", value: "₹2,642 Cr", delta: "↑ MoM +6.36%", tone: "emerald" },
  { label: "MFI Book", value: "₹2,180 Cr", delta: "↑ MoM +1.87%", tone: "emerald" },
  { label: "MFI Collection Efficiency", value: "97.8%", delta: "↓ MoM -30bps", tone: "crimson" },
  { label: "PBT Overall", value: "₹42 Cr", delta: "↑ MoM +20%", tone: "emerald" },
];

export function RetailAssetsWorkspace() {
  const { todayMeetings, actions } = useAppState();

  const openActionItems = useMemo<OpenActionItem[]>(() =>
    actions
      .filter((a) => a.bu === "Retail Advances - Excluding Credit Cards" && !a.completed)
      .map((a) => ({
        text: a.title,
        due: a.due,
        status:
          a.status === "Overdue" ? "Overdue"
          : a.status === "In Progress" ? "In Progress"
          : "On Track",
        actionId: a.id,
      })),
    [actions]
  );
  
  return (
    <PreMeetingBrief
      cfg={{
        headerText: "Head, Retail Assets & Collections - Pre-meeting brief",
        department: "Retail Advances - Excluding Credit Cards",
        metrics: RA_METRICS,
        metricsTitle: "Retail Assets metrics · Latest MIS",
        availableMeetings: todayMeetings,
        pastMeetings: [],
        openActionItems: openActionItems,
        decisions: [],
        questions: [],
        chatChips: RA_CHIPS,
        chatResponder: retailAssetsResponse,
      }}
    />
  );
}
