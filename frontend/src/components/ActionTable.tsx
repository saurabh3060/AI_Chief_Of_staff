import { useAppState, type BU, type Status } from "@/context/AppState";
import { BU_COLORS } from "@/data/mockData";
import { Mail, Calendar, Check } from "lucide-react";

const statusTone: Record<Status, React.CSSProperties> = {
  Overdue:      { background: "#FEE2E2", color: "#991B1B" },
  "In Progress":{ background: "#FEF3C7", color: "#92400E" },
  "On Track":   { background: "#DCFCE7", color: "#166534" },
  Ahead:        { background: "#DCFCE7", color: "#166534" },
};

interface Props { filterBU?: BU | "All"; filterStatus?: Status | "All"; showSource?: boolean; }

export function ActionTable({ filterBU = "All", filterStatus = "All", showSource = true }: Props) {
  const { actions, updateAction, showToast } = useAppState();
  const rows = actions.filter((a) =>
    (filterBU === "All" || a.bu === filterBU) &&
    (filterStatus === "All" || a.status === filterStatus)
  );
  return (
    <div className="bg-white rounded-[14px] border border-[#E2E8F0] overflow-hidden">
      <table className="w-full text-[12.5px]">
        <thead>
          <tr className="bg-[#F1F5F9] text-[10px] uppercase tracking-[0.08em] text-[#475569]">
            <th className="text-left font-semibold px-3 py-2.5 w-10">#</th>
            <th className="text-left font-semibold px-3 py-2.5">Action Item</th>
            <th className="text-left font-semibold px-3 py-2.5">Owner</th>
            <th className="text-left font-semibold px-3 py-2.5">BU</th>
            {showSource && <th className="text-left font-semibold px-3 py-2.5">Source</th>}
            <th className="text-left font-semibold px-3 py-2.5">Due</th>
            <th className="text-left font-semibold px-3 py-2.5">Status</th>
            <th className="text-left font-semibold px-3 py-2.5 w-24">Actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((a, i) => (
            <tr key={a.id} className={`border-t border-[#EEF2F7] hover:bg-[#F1F5F9] fade-in ${a.status === "Overdue" ? "border-l-[3px] border-l-[#FEE2E2]" : ""}`}>
              <td className="px-3 py-3 text-[#94A3B8]">{i + 1}</td>
              <td className="px-3 py-3 text-[#0A2540] font-medium">
                <div className="flex items-center gap-2 flex-wrap">
                  <span>{a.title}</span>
                  {a.flag && <span className={`text-[10px] font-bold px-1.5 py-[1px] rounded-[4px] uppercase text-white ${a.flag === "New" ? "bg-[#DCFCE7]" : "bg-[#1E40AF]"}`}>{a.flag}</span>}
                </div>
              </td>
              <td className="px-3 py-3 text-[#475569]">{a.owner}</td>
              <td className="px-3 py-3"><span className="text-[10px] font-semibold px-2 py-[2px] rounded-[6px]" style={{ background: BU_COLORS[a.bu].bg, color: BU_COLORS[a.bu].text }}>{a.bu}</span></td>
              {showSource && <td className="px-3 py-3 text-[#475569] italic text-[11.5px]">{a.sourceMeeting}</td>}
              <td className="px-3 py-3 text-[#475569]">{a.due}</td>
              <td className="px-3 py-3"><span className="text-[10px] font-semibold px-2 py-[2px] rounded-[6px]" style={statusTone[a.status]}>{a.status}</span></td>
              <td className="px-3 py-3">
                <div className="flex gap-1.5 text-[#475569]">
                  <button onClick={() => showToast(`Reminder queued to ${a.owner}`)} className="p-1 hover:text-[#0A2540] hover:bg-[#EEF2F7] rounded" title="Send reminder"><Mail size={14} /></button>
                  <button onClick={() => showToast(`Follow-up invite sent to ${a.owner}`)} className="p-1 hover:text-[#0A2540] hover:bg-[#EEF2F7] rounded" title="Schedule follow-up"><Calendar size={14} /></button>
                  <button onClick={() => updateAction(a.id, { status: "On Track" })} className="p-1 hover:text-[#166534] hover:bg-[#DCFCE7] rounded" title="Mark on track"><Check size={14} /></button>
                </div>
              </td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr><td colSpan={showSource ? 8 : 7} className="text-center text-[#94A3B8] py-6 text-[12px]">No actions match these filters.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}