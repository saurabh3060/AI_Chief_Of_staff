import { useAppState, type BU } from "@/context/AppState";
import { useAppNavigate } from "@/hooks/useAppNavigate";
import { BU_COLORS } from "@/data/mockData";

const statusTone = {
  Overdue:      { background: "#FEE2E2", color: "#991B1B" },
  "In Progress":{ background: "#FEF3C7", color: "#92400E" },
  "On Track":   { background: "#DCFCE7", color: "#166534" },
  Ahead:        { background: "#DCFCE7", color: "#166534" },
} as const;

export function CrossFunctionalPanel({ bu }: { bu: BU }) {
  const { actions, setFocusActionId } = useAppState();
  const navigate = useAppNavigate();
  const items = actions.filter((a) => a.bu === bu && a.crossFunctional);
  return (
    <div className="bg-white rounded-[14px] border border-[#E2E8F0] p-4 shadow-[0_1px_3px_rgba(10,37,64,0.06)]">
      <div className="flex items-center justify-between mb-3">
        <div className="text-[11px] uppercase tracking-[0.08em] font-semibold text-[#475569]">Cross-Functional Tasks - {bu}</div>
        <span className="text-[10px] text-[#94A3B8]">{items.length} item{items.length === 1 ? "" : "s"}</span>
      </div>
      {items.length === 0 ? (
        <div className="text-[12px] text-[#94A3B8] py-4 text-center">No cross-functional tasks yet.</div>
      ) : (
        <ul className="space-y-2">
          {items.map((a) => (
            <li
              key={a.id}
              onClick={() => { setFocusActionId(a.id); navigate("Ledger"); }}
              className={`flex items-start gap-2 p-2 rounded-[8px] border fade-in cursor-pointer hover:bg-[#F1F5F9] ${a.status === "Overdue" ? "border-l-[3px] border-l-[#FEE2E2] border-[#FEE2E2] bg-[#FFF7F7]" : "border-[#EEF2F7]"}`}
              title="Open in Action & Accountability Ledger"
            >
              <div className="flex-1">
                <div className="text-[12.5px] text-[#0A2540] font-medium flex items-center gap-2 flex-wrap">
                  <span>{a.title}</span>
                  {a.flag && <span className={`text-[10px] font-bold px-1.5 py-[1px] rounded-[4px] uppercase text-white ${a.flag === "New" ? "bg-[#DCFCE7]" : "bg-[#1E40AF]"}`}>{a.flag}</span>}
                </div>
                <div className="text-[11px] text-[#475569] mt-0.5">{a.owner} · Due {a.due} · <span className="italic">{a.sourceMeeting}</span></div>
              </div>
              <span className="text-[10px] font-semibold px-2 py-[2px] rounded-[6px] whitespace-nowrap" style={statusTone[a.status]}>{a.status}</span>
              <span className="text-[10px] font-semibold px-2 py-[2px] rounded-[6px]" style={{ background: BU_COLORS[a.bu].bg, color: BU_COLORS[a.bu].text }}>{a.bu}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}