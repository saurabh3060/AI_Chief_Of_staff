import { useState } from "react";
import { useAppState, type DrKey, type BU, type Status } from "@/context/AppState";
import { useAppNavigate } from "@/hooks/useAppNavigate";
import { resolveLedgerActionId } from "@/lib/ledgerLink";
import type { DrConfig } from "@/components/DRWorkspace";
import { DECISION_CONTEXT } from "@/components/DRWorkspace";
import { RightRail } from "@/components/RightRail";
import { ChevronDown } from "lucide-react";

const C = {
  danger: "#DC2626",
  warning: "#D97706",
  success: "#16A34A",
  info: "#1D4ED8",
  navy: "#0A2540",
  body: "#0A2540",
};

const STATUS_TONE: Record<string, { bg: string; text: string }> = {
  Reviewed:      { bg: "#DCFCE7", text: "#166534" },
  "In Progress": { bg: "#FEF3C7", text: "#92400E" },
  "On Track":    { bg: "#DCFCE7", text: "#166534" },
  Overdue:       { bg: "#FEE2E2", text: "#991B1B" },
  Ahead:         { bg: "#DCFCE7", text: "#166534" },
  New:           { bg: "#DBEAFE", text: "#1E40AF" },
};

const MOTION_TONE: Record<string, { bg: string; text: string }> = {
  Overdue:       { bg: "#FEE2E2", text: "#991B1B" },
  "In Progress": { bg: "#FEF3C7", text: "#92400E" },
  Ahead:         { bg: "#DCFCE7", text: "#166534" },
};

const TEAM_COLOR: Record<DrKey, string> = {
  Risk: C.danger, "BB&RL": C.info, Wholesale: C.success, "Retail Assets": C.warning,
};

// Map DrKey to full business unit name
const DR_TO_BU: Record<DrKey, string> = {
  Risk: "Credit Cards",
  "BB&RL": "BBRL",
  Wholesale: "Wholesale Banking",
  "Retail Assets": "Retail Advances - Excluding Credit Cards",
};

const URG_COLOR = { overdue: C.danger, today: C.warning, this_week: C.info } as const;

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`bg-white border border-[#E2E8F0] rounded-[14px] ${className}`}>{children}</div>;
}
function SectionTitle({ children }: { children: React.ReactNode }) {
  return <div className="text-[12px] uppercase tracking-[0.08em] font-bold text-[#0A2540] mb-3">{children}</div>;
}

function DRHeader({ cfg }: { cfg: DrConfig }) {
  const text = `${cfg.fullName} · ${cfg.title}`;
  return (
    <div
      className="w-full"
      style={{
        background: "#0A2540", color: "#FFFFFF", fontSize: 13, fontWeight: 600,
        letterSpacing: "0.08em", textTransform: "uppercase",
        padding: "12px 20px", borderRadius: 14,
      }}
    >
      {text}
    </div>
  );
}

export function OverviewWorkspace({ cfg }: { cfg: DrConfig }) {
  return (
    <div className="grid gap-5 p-5 bg-[#F4F6FA] min-h-[calc(100vh-64px)]" style={{ gridTemplateColumns: "60% 40%" }}>
      <div className="space-y-5 min-w-0">
        <DRHeader cfg={cfg} />
        <AwaitingDecisionCard cfg={cfg} />
        <OpenActionsTrackerCard cfg={cfg} />
        <PastMeetingsCard cfg={cfg} />
        <CrossFunctionalCard cfg={cfg} />
      </div>
      <div className="min-w-0">
        <RightRail
          metrics={cfg.metrics}
          metricsTitle={`${cfg.drKey} metrics · Latest MIS`}
          cosPreloaded={cfg.cosPreloaded}
          cosChips={cfg.cosChips}
          cosResponder={cfg.cosResponder}
          sourcePage={`${cfg.drKey} workspace`}
        />
      </div>
    </div>
  );
}

function AwaitingDecisionCard({ cfg }: { cfg: DrConfig }) {
  const { pending } = useAppState();
  const items = pending.filter((p) => p.team === cfg.drKey && p.status === "open");
  const isStatic = cfg.drKey === "Wholesale" || cfg.drKey === "Retail Assets";
  return (
    <Card className="px-5 py-4 lov-urgent">
      <SectionTitle>Awaiting your decision</SectionTitle>
      {items.length === 0 && <div className="text-[11.5px] text-[#94A3B8]">No items awaiting your direction.</div>}
      <div className="space-y-1.5">
        {items.map((p) => (
          <div
            key={p.id}
            className={`flex items-start gap-2.5 py-1 -mx-1 px-1 rounded-[6px] ${
              isStatic ? "" : "cursor-pointer hover:bg-[#F1F5F9]"
            }`}
          >
            <span className="mt-1.5 inline-block w-2 h-2 rounded-full shrink-0" style={{ background: URG_COLOR[p.urgency] }} />
            <div className="flex-1 min-w-0 text-[12.5px]" style={{ color: C.body }}>
              <div>
                {p.text}
                <span className="ml-2 text-[10px] uppercase tracking-[0.04em] text-[#94A3B8]">{p.business_unit || p.team}</span>
              </div>
              {DECISION_CONTEXT[p.text] && (
                <div className="text-[12px] text-[#475569] mt-0.5 leading-snug normal-case">{DECISION_CONTEXT[p.text]}</div>
              )}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

function OpenActionsTrackerCard({ cfg }: { cfg: DrConfig }) {
  const { motion, setLedgerTeamFilter, setFocusActionId, actions } = useAppState();
  const navigate = useAppNavigate();
  const items = motion.filter((m) => m.team === cfg.drKey);
  function openLedger(id?: string, text?: string) {
    const buMap: Record<DrKey, BU> = {
      Risk: "Credit Cards", "BB&RL": "BBRL", Wholesale: "Wholesale Banking", "Retail Assets": "Retail Advances - Excluding Credit Cards",
    };
    setLedgerTeamFilter(buMap[cfg.drKey]);
    const fid = id ? resolveLedgerActionId(id, text ?? "", actions) : null;
    if (fid) setFocusActionId(fid);
    navigate("Ledger");
  }
  return (
    <Card className="px-5 py-4 lov-urgent">
      <SectionTitle>Open Actions Status Tracker</SectionTitle>
      {items.length === 0 && <div className="text-[11.5px] text-[#94A3B8]">No open actions tracked here.</div>}
      <div className="space-y-1">
        {items.map((m) => {
          const tone = MOTION_TONE[m.status];
          return (
            <div
              key={m.id}
              onClick={() => openLedger(m.id, m.text)}
              className="flex items-center gap-2 text-[12px] py-1 cursor-pointer hover:bg-[#F1F5F9] -mx-1 px-1 rounded-[6px]"
            >
              <div className="flex-1 truncate" style={{ color: C.body }}>{m.text}</div>
              <span className="text-[10px] text-[#94A3B8] shrink-0">{m.team}</span>
              <span className="text-[10px] font-semibold px-2 py-[2px] rounded-[6px] shrink-0" style={{ background: tone.bg, color: tone.text }}>{m.status}</span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function PastMeetingsCard({ cfg }: { cfg: DrConfig }) {
  const [openIdx, setOpenIdx] = useState<number | null>(0);
  return (
    <Card className="px-5 py-4">
      <SectionTitle>Recent meeting summaries</SectionTitle>
      <div className="space-y-2.5">
        {cfg.past.map((p, i) => {
          const open = openIdx === i;
          return (
            <div
              key={i}
              onClick={() => setOpenIdx(open ? null : i)}
              className="border border-[#E2E8F0] rounded-[10px] p-3 cursor-pointer hover:border-[#94A3B8] transition"
            >
              <div className="flex items-center justify-between gap-2">
                <div className="text-[12.5px] font-semibold flex items-center gap-2" style={{ color: C.body }}>
                  {p.date} · {p.type}
                  {!open && p.openItems.length > 0 && (
                    <span className="inline-flex items-center justify-center text-[10px] font-bold text-white rounded-full" style={{ background: C.danger, minWidth: 16, height: 16, padding: "0 5px" }}>
                      {p.openItems.length}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <ChevronDown size={13} className={`text-[#94A3B8] transition ${open ? "rotate-180" : ""}`} />
                </div>
              </div>
              {open && (
                <>
                  {/* Outcomes/Summary */}
                  {p.outcomes.length > 0 && (
                    <>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-bold text-[#0A2540] mt-2 mb-1">Summary</div>
                      <ul className="space-y-0.5">
                        {p.outcomes.map((o, j) => (
                          <li key={j} className="text-[11.5px] text-[#475569] flex gap-2"><span className="text-[#94A3B8]">·</span>{o}</li>
                        ))}
                      </ul>
                    </>
                  )}

                  {/* Action Items (from email) */}
                  {p.actionItems && p.actionItems.length > 0 && (
                    <>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-bold text-[#0A2540] mt-2 mb-1">Action Items</div>
                      <ul className="space-y-0.5">
                        {p.actionItems.map((item, j) => (
                          <li key={j} className="text-[11.5px] text-[#475569] flex gap-2"><span className="text-[#94A3B8]">·</span>{item}</li>
                        ))}
                      </ul>
                    </>
                  )}

                  {/* Top Questions (from email) */}
                  {p.topQuestions && p.topQuestions.length > 0 && (
                    <>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-bold text-[#0A2540] mt-2 mb-1">Top Questions</div>
                      <ul className="space-y-0.5">
                        {p.topQuestions.map((q, j) => (
                          <li key={j} className="text-[11.5px] text-[#475569] flex gap-2"><span className="text-[#94A3B8]">·</span>{q}</li>
                        ))}
                      </ul>
                    </>
                  )}

                  {/* Open Items */}
                  {p.openItems.length > 0 && (
                    <>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-bold text-[#0A2540] mt-2 mb-1">Open items</div>
                      <ul className="space-y-0.5">
                        {p.openItems.map((o, j) => (
                          <li key={j} className="text-[11.5px] flex gap-2" style={{ color: C.danger }}><span>·</span>{o}</li>
                        ))}
                      </ul>
                    </>
                  )}
                </>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function CrossFunctionalCard({ cfg }: { cfg: DrConfig }) {
  const { crossTags, updateCrossTag, setFocusActionId } = useAppState();
  const navigate = useAppNavigate();
  const items = crossTags.filter((c) => c.team === cfg.drKey);
  return (
    <Card className="px-5 py-4 lov-urgent">
      <SectionTitle>Cross-functional items tagged here</SectionTitle>
      {items.length === 0 ? (
        <div className="text-[11.5px] text-[#94A3B8]">No cross-functional items</div>
      ) : (
        <div className="space-y-2">
          {items.map((c) => {
            const tone = STATUS_TONE[c.status];
            return (
              <div
                key={c.id}
                onClick={() => { setFocusActionId(c.id); navigate("Ledger"); }}
                title="Open in Action & Accountability Ledger"
                className="border border-[#E2E8F0] rounded-[10px] p-2.5 cursor-pointer hover:border-[#94A3B8] hover:bg-[#F1F5F9] transition"
              >
                <div className="flex items-start gap-2">
                  <select
                    value={c.status}
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => updateCrossTag(c.id, { status: e.target.value as Status })}
                    className="text-[10px] font-semibold px-1.5 py-[2px] rounded-[6px] border-0 focus:outline-none"
                    style={{ background: tone.bg, color: tone.text }}
                  >
                    <option value="Overdue">Overdue</option>
                    <option value="In Progress">In Progress</option>
                    <option value="On Track">On Track</option>
                  </select>
                  <div className="flex-1 text-[12px]" style={{ color: C.body }}>{c.text}</div>
                  <div className="text-[10.5px] text-[#475569] tabular-nums"><span className="text-[#94A3B8]">Due:</span> {c.dueDate}</div>
                </div>
                <div className="mt-1 flex items-center gap-2 text-[10px] text-[#94A3B8] uppercase tracking-[0.06em]">
                  <span>From: {c.sourceDR} · {c.sourceMeeting}</span>
                  <span
                    className="px-1.5 py-[1px] rounded-[4px] normal-case font-semibold"
                    style={{ background: `${TEAM_COLOR[c.team]}1A`, color: TEAM_COLOR[c.team] }}
                  >
                    {c.team}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}
