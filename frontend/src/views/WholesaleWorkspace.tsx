import { PreMeetingBrief } from "@/components/PreMeetingBrief";
import { WHOLESALE_CHIPS, wholesaleResponse } from "@/data/mockData";
import { useAppState } from "@/context/AppState";
import { useMemo } from "react";
import type { DrMetric } from "@/components/DRWorkspace";
import type { OpenActionItem } from "@/components/PreMeetingBrief";

/* ── Wholesale Banking metrics (Jul'26 vs Jun'26) ─────────────── */
const WHOLESALE_METRICS: DrMetric[] = [
  { label: "Advances Booked", value: "₹52,289 Cr", delta: "↑ MoM +1.88%", tone: "emerald" },
  { label: "Deposits Booked", value: "₹38,083 Cr", delta: "↓ MoM -0.77%", tone: "crimson" },
  { label: "CA – INR Deposits", value: "₹6,112 Cr", delta: "↑ MoM +8.19%", tone: "emerald" },
  { label: "Fee Income", value: "₹68.4 Cr", delta: "↑ MoM +6.5%", tone: "emerald" },
  { label: "Non-Fund Based Book", value: "₹28,702 Cr", delta: "↑ MoM +4.85%", tone: "emerald" },
];

export function WholesaleWorkspace() {
  const { todayMeetings, actions } = useAppState();

  const openActionItems = useMemo<OpenActionItem[]>(() =>
    actions
      .filter((a) => a.bu === "Wholesale Banking" && !a.completed)
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
        headerText: "Head, Wholesale Banking - Pre-meeting brief",
        department: "Wholesale Banking",
        metrics: WHOLESALE_METRICS,
        metricsTitle: "Wholesale metrics · Latest MIS",
        availableMeetings: todayMeetings,
        pastMeetings: [],
        openActionItems: openActionItems,
        decisions: [],
        questions: [],
        chatChips: WHOLESALE_CHIPS,
        chatResponder: wholesaleResponse,
      }}
    />
  );
}
