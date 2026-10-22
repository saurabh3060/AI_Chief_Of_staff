import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { useAppState } from "@/context/AppState";
import { useAppNavigate } from "@/hooks/useAppNavigate";
import type { ViewName } from "@/context/AppState";
import { TODAY_CALENDAR } from "@/data/mockData";

export function CalendarTracker() {
  const { currentMeeting } = useAppState();
  const navigate = useAppNavigate();
  const [open, setOpen] = useState(false);

  // Format time from ISO string
  const formatTime = (iso: string) => {
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "Asia/Kolkata" });
    } catch {
      return "";
    }
  };

  const hasMeeting = !!currentMeeting;
  const meetingLabel = hasMeeting
    ? `Now: ${currentMeeting.subject}`
    : TODAY_CALENDAR.length > 0
      ? `Next: ${TODAY_CALENDAR[0].title}`
      : "No meetings scheduled";

  return (
    <div className="relative">
      <div className="flex items-center bg-[#EEF2F7] hover:bg-[#E2E8F0] transition rounded-[8px] overflow-hidden">
        <button
          onClick={() => hasMeeting ? navigate("CRO", "Pre-Meeting") : undefined}
          className="flex items-center gap-2 px-3 py-1.5"
          title={hasMeeting ? "Open workspace" : undefined}
        >
          <span className={`w-2 h-2 rounded-full ${hasMeeting ? "bg-[#16A34A] pulse-dot" : "bg-[#94A3B8]"}`} />
          <span className="text-[12px] text-[#0A2540] font-medium">{meetingLabel}</span>
        </button>
        <button onClick={() => setOpen((o) => !o)} className="px-2 py-1.5 border-l border-[#E2E8F0]" title="Today's schedule">
          <ChevronDown size={14} className="text-[#475569]" />
        </button>
      </div>
      {open && (
        <div className="absolute right-0 top-full mt-2 w-[360px] bg-white rounded-[14px] border border-[#E2E8F0] shadow-lg z-50 fade-in">
          <div className="px-4 py-3 border-b border-[#EEF2F7] text-[11px] uppercase tracking-[0.08em] font-semibold text-[#475569]">Today</div>
          {currentMeeting && (
            <div className="w-full text-left flex items-start gap-3 px-4 py-3 bg-[#EAF2FB] border-b border-[#EEF2F7]">
              <div className="text-[12px] font-semibold text-[#0A2540] w-12">{formatTime(currentMeeting.start)}</div>
              <div className="flex-1">
                <div className="text-[12.5px] text-[#0A2540] font-semibold">{currentMeeting.subject}</div>
                <div className="text-[10px] text-[#16A34A] mt-0.5 font-semibold">In progress</div>
              </div>
            </div>
          )}
          {TODAY_CALENDAR.map((m) => (
            <button key={m.time} onClick={() => { if (m.view) { navigate(m.view as ViewName, "Pre-Meeting"); setOpen(false); } }} className={`w-full text-left flex items-start gap-3 px-4 py-3 hover:bg-[#F1F5F9] border-b border-[#EEF2F7] last:border-0 ${m.view ? "cursor-pointer" : "cursor-default opacity-70"}`}>
              <div className="text-[12px] font-semibold text-[#0A2540] w-12">{m.time}</div>
              <div className="flex-1">
                <div className="text-[12.5px] text-[#0A2540]">{m.title}</div>
                {m.view && <div className="text-[10px] text-[#1E40AF] mt-0.5 font-semibold">Open pre-meeting brief →</div>}
              </div>
            </button>
          ))}
          {!currentMeeting && TODAY_CALENDAR.length === 0 && (
            <div className="px-4 py-4 text-[12px] text-[#94A3B8] text-center">No meetings scheduled for today</div>
          )}
        </div>
      )}
    </div>
  );
}
