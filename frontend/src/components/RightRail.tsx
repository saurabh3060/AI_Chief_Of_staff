import { useState } from "react";
import { ChatPanel, type ChatMessage } from "@/components/ChatPanel";
import { useAppState, type DrKey, type ActionItem, type Status, type MotionStatus, isDuePastOrToday } from "@/context/AppState";
import type { ChatResponse } from "@/data/mockData";

export interface RailMetric {
  label: string;
  value: string;
  delta: string;
  tone: "emerald" | "amber" | "crimson" | "info" | "grey";
}

const TONE: Record<string, string> = {
  emerald: "#16A34A", amber: "#D97706", crimson: "#DC2626", info: "#1D4ED8", grey: "#64748B",
};

const TEAMS: DrKey[] = ["Risk", "BB&RL", "Wholesale", "Retail Assets"];
const BU_MAP: Record<DrKey, ActionItem["bu"]> = {
  Risk: "Credit Cards", "BB&RL": "BBRL", Wholesale: "Wholesale Banking", "Retail Assets": "Retail Advances - Excluding Credit Cards",
};

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`bg-white border border-[#E2E8F0] rounded-[14px] ${className}`}>{children}</div>;
}

export function RightRail({
  metrics,
  metricsTitle = "Metrics · Latest MIS",
  cosPreloaded = [],
  cosChips = [],
  cosResponder,
  sourcePage,
  sourceDR,
}: {
  metrics: RailMetric[];
  metricsTitle?: string;
  cosPreloaded?: ChatMessage[];
  cosChips?: string[];
  cosResponder: (q: string) => ChatResponse;
  sourcePage: string;
  sourceDR?: DrKey;
}) {
  return (
    <div className="flex flex-col gap-5 min-w-0">
      <Card className="px-5 py-4">
        <div className="text-[12px] uppercase tracking-[0.08em] font-bold text-[#0A2540] mb-2.5">{metricsTitle}</div>
        <div className="grid grid-cols-2 gap-3">
          {metrics.map((m) => (
            <div key={m.label} className="border border-[#E2E8F0] rounded-[10px] px-3 py-3 overflow-hidden">
              <div className="text-[11px] uppercase tracking-[0.05em] font-semibold text-[#0A2540] line-clamp-2 break-words">{m.label}</div>
              <div className="text-[22px] font-semibold leading-tight mt-1 tabular-nums" style={{ color: "#0A2540" }}>{m.value}</div>
              <div className="text-[10px] font-semibold mt-1" style={{ color: TONE[m.tone] }}>{m.delta}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card className="px-4 py-4">
        <div className="text-[12px] uppercase tracking-[0.08em] font-bold text-[#0A2540] mb-2.5">Ask your Chief of Staff</div>
        <ChatPanel
          preloaded={cosPreloaded}
          chips={cosChips}
          getResponse={cosResponder}
          placeholder="Ask anything across all data sources…"
          compact
          loadHistory
        />
      </Card>

      <TagItemCard sourcePage={sourcePage} sourceDR={sourceDR} />
    </div>
  );
}

function TagItemCard({ sourcePage, sourceDR }: { sourcePage: string; sourceDR?: DrKey }) {
  const { addCrossTag, addActions, addMotion, showToast } = useAppState();
  const [desc, setDesc] = useState("");
  const [team, setTeam] = useState<DrKey | "">("");
  const [due, setDue] = useState("");
  const [dueErr, setDueErr] = useState(false);
  const [teamErr, setTeamErr] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);

  function submit() {
    if (!desc.trim()) return;
    let bad = false;
    if (!team) { setTeamErr(true); bad = true; }
    if (!due.trim()) { setDueErr(true); bad = true; }
    if (bad) return;
    const formattedDue = due.match(/^\d{4}-\d{2}-\d{2}$/)
      ? (() => { const [y, mo, d] = due.split("-").map(Number); return new Date(y, mo - 1, d).toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" }); })()
      : due;
    const today = new Date().toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" });
    const id = `TG${Date.now()}`;
    const teamKey = team as DrKey;
    const overdue = isDuePastOrToday(formattedDue);
    const status: Status = overdue ? "Overdue" : "In Progress";
    // Any tag from a DR workspace → cross-functional (appears in target team's Cross-functional items).
    // Only Executive Launchpad tags (no sourceDR) are non-cross-functional and go to Open Actions Status Tracker.
    const isCrossFunctional = !!sourceDR;

    if (isCrossFunctional) {
      addCrossTag({
        id, text: desc.trim(), team: teamKey, sourceDR: sourceDR!,
        sourceMeeting: `Tagged from ${sourcePage} · ${today}`,
        date: today, dueDate: formattedDue, status,
      });
    }
    addActions([{
      id, title: desc.trim(), owner: `${teamKey} Head`, bu: BU_MAP[teamKey],
      sourceMeeting: `Tagged from ${sourcePage} · ${today}`,
      due: formattedDue, status, origin: "seed",
      crossFunctional: isCrossFunctional,
      involvedBUs: isCrossFunctional && sourceDR ? [BU_MAP[teamKey], BU_MAP[sourceDR]] : undefined,
      flag: "New", createdToday: true,
    }]);
    // Launchpad-tagged items also surface in the Open Actions Status Tracker.
    if (!sourceDR) {
      const motionStatus: MotionStatus = overdue ? "Overdue" : "In Progress";
      addMotion({ id, text: desc.trim(), team: teamKey, status: motionStatus });
    }
    showToast(`Item tagged to ${team}`);
    setSuccess(`Item tagged to ${team} · Due ${formattedDue}`);
    setDesc(""); setTeam(""); setDue(""); setDueErr(false); setTeamErr(false);
    setTimeout(() => setSuccess(null), 3000);
  }

  return (
    <div className="bg-white border border-[#E2E8F0] rounded-[14px] px-5 py-4">
      <div className="text-[13px] uppercase tracking-[0.08em] font-semibold text-[#0A2540] mb-2.5">Tag item to a team</div>
      <div className="space-y-2">
        <input
          value={desc}
          onChange={(e) => setDesc(e.target.value)}
          placeholder="Describe the item to tag"
          className="w-full text-[12.5px] px-2.5 py-2 border border-[#E2E8F0] rounded-[8px] focus:outline-none focus:border-[#0A2540]"
        />
        <select
          value={team}
          onChange={(e) => { setTeam(e.target.value as DrKey); if (e.target.value) setTeamErr(false); }}
          className={`w-full text-[12.5px] px-2.5 py-2 rounded-[8px] focus:outline-none bg-white ${teamErr ? "border-2 border-[#F59E0B] bg-[#FFFBEB]" : "border border-[#E2E8F0]"}`}
        >
          <option value="">Select team…</option>
          {TEAMS.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <div>
          <label className="text-[11px] uppercase tracking-[0.06em] font-semibold text-[#475569] block mb-1">Due date</label>
          <input
            type="date" value={due}
            onChange={(e) => { setDue(e.target.value); if (e.target.value) setDueErr(false); }}
            className={`w-full text-[12.5px] px-2.5 py-2 rounded-[8px] focus:outline-none ${dueErr ? "border-2 border-[#F59E0B] bg-[#FFFBEB]" : "border border-[#E2E8F0]"}`}
          />
          {dueErr && <div className="mt-1 text-[11px] text-[#B45309]">Due date required</div>}
        </div>
        <button
          onClick={submit}
          className="w-full text-[13px] font-semibold px-3 py-2 rounded-[8px] text-white"
          style={{ background: "#E8192C" }}
        >
          Tag item
        </button>
        {success && (
          <div className="text-[12px]" style={{ color: "#166534" }}>{success}</div>
        )}
      </div>
    </div>
  );
}
