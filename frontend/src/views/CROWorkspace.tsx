import { PreMeetingBrief } from "@/components/PreMeetingBrief";
import { CRO_CHIPS, croResponse } from "@/data/mockData";
import { useAppState } from "@/context/AppState";
import { useMemo } from "react";
import type { DrMetric } from "@/components/DRWorkspace";
import type { OpenActionItem } from "@/components/PreMeetingBrief";

/* ── Credit Cards metrics · Latest MIS ────────────────────────────────────── */
const RISK_METRICS: DrMetric[] = [
  { label: "Overall Outstanding (ENR)", value: "₹20,150 Cr", delta: "↑ MoM +1.31%", tone: "emerald" },
  { label: "Credit Cost", value: "190bps", delta: "↓ MoM -20bps", tone: "emerald" },
  { label: "Collection / Agency Cost", value: "₹112.5 Cr", delta: "↓ MoM -1.4%", tone: "emerald" },
  { label: "New Card Acquisitions", value: "1,18,238", delta: "↑ MoM +11%", tone: "emerald" },
  { label: "PBT%", value: "0.3%", delta: "↑ MoM +50bps", tone: "emerald" },
  { label: "Spends", value: "₹7,597 Cr", delta: "↑ MoM +17.4%", tone: "emerald" },
];

export function CROWorkspace() {
  const { todayMeetings, actions } = useAppState();

  const openActionItems = useMemo<OpenActionItem[]>(() =>
    actions
      .filter((a) => a.bu === "Credit Cards" && !a.completed)
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
        headerText: "Credit Cards - Pre-meeting brief",
        department: "Credit Cards",
        metrics: RISK_METRICS,
        metricsTitle: "Credit Cards metrics · Latest MIS",
        availableMeetings: todayMeetings,
        pastMeetings: [],
        openActionItems: openActionItems,
        decisions: [],
        questions: [],
        chatChips: CRO_CHIPS,
        chatResponder: croResponse,
      }}
    />
  );
}
