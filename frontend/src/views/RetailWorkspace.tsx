import { PreMeetingBrief } from "@/components/PreMeetingBrief";
import { RETAIL_CHIPS, retailResponse } from "@/data/mockData";
import { useAppState } from "@/context/AppState";
import { useMemo } from "react";
import type { DrMetric } from "@/components/DRWorkspace";
import type { OpenActionItem } from "@/components/PreMeetingBrief";

/* ── Monthly metrics (BBRL deposits, Feb'26 vs Jan'26) ────────── */
const MONTHLY_METRICS: DrMetric[] = [
  { label: "CA", value: "₹270.0 Cr", delta: "↑ MoM +0.04%", tone: "emerald" },
  { label: "SA", value: "₹68.6 Cr", delta: "↑ MoM +60.7%", tone: "emerald" },
  { label: "TD", value: "₹1,239.9 Cr", delta: "↑ MoM +13.7%", tone: "emerald" },
  { label: "Retail Deposits <₹3Cr", value: "₹890 Cr", delta: "↑ MoM +5.3%", tone: "emerald" },
  { label: "Avg Deposits/Branch (86 branches)", value: "₹18.4 Cr", delta: "↑ MoM +2.8%", tone: "emerald" },
];

export function RetailWorkspace() {
  const { todayMeetings, actions } = useAppState();

  const openActionItems = useMemo<OpenActionItem[]>(() =>
    actions
      .filter((a) => a.bu === "BBRL" && !a.completed)
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
        headerText: "Head, BBRL - Pre-meeting brief",
        department: "BB&RL",
        metrics: MONTHLY_METRICS,
        availableMeetings: todayMeetings,
        pastMeetings: [],
        openActionItems: openActionItems,
        decisions: [],
        questions: [],
        chatChips: RETAIL_CHIPS,
        chatResponder: retailResponse,
      }}
    />
  );
}
