import { useState } from "react";
import { Bell } from "lucide-react";
import { useAppState } from "@/context/AppState";
import { useAppNavigate } from "@/hooks/useAppNavigate";
import type { ViewName } from "@/context/AppState";

const toneStyle = {
  info:    { dot: "#1E40AF", bg: "#DBEAFE" },
  amber:   { dot: "#D97706", bg: "#FEF3C7" },
  crimson: { dot: "#DC2626", bg: "#FEE2E2" },
} as const;

export function NotificationBell() {
  const { notifications, readNotifIds, markNotifsRead } = useAppState();
  const navigate = useAppNavigate();
  const [open, setOpen] = useState(false);
  const unread = notifications.filter((n) => !readNotifIds.has(n.id)).length;

  return (
    <div className="relative">
      <button onClick={() => { setOpen((o) => !o); if (!open) markNotifsRead(); }} className="relative p-2 hover:bg-[#EEF2F7] rounded-[8px] transition">
        <Bell size={18} className="text-[#0A2540]" />
        {unread > 0 && (
          <span className="absolute -top-0.5 -right-0.5 min-w-[18px] h-[18px] px-1 rounded-full bg-[#E8192C] text-white text-[10px] font-bold flex items-center justify-center pulse-dot">{unread}</span>
        )}
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-2 w-[360px] max-h-[480px] overflow-y-auto bg-white rounded-[14px] border border-[#E2E8F0] shadow-lg z-50 fade-in">
          <div className="px-4 py-3 border-b border-[#EEF2F7] flex items-center justify-between">
            <div className="text-[11px] uppercase tracking-[0.08em] font-semibold text-[#475569]">Notifications</div>
            <span className="text-[10px] text-[#94A3B8]">{notifications.length} total</span>
          </div>
          {notifications.map((n) => {
            const t = toneStyle[n.tone];
            return (
              <button key={n.id} onClick={() => { navigate(n.target.view as ViewName, n.target.phase); setOpen(false); }} className="w-full text-left flex items-start gap-3 px-4 py-3 hover:bg-[#F1F5F9] border-b border-[#EEF2F7] last:border-0 fade-in">
                <div className="w-2 h-2 rounded-full mt-1.5 shrink-0" style={{ background: t.dot }} />
                <div className="flex-1">
                  <div className="text-[12.5px] font-semibold text-[#0A2540]">{n.title}</div>
                  <div className="text-[11.5px] text-[#475569] mt-0.5">{n.context}</div>
                  <div className="text-[10px] text-[#94A3B8] mt-1">{n.time}</div>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}