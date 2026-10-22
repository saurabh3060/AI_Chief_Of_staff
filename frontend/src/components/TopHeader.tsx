import { useState, useEffect } from "react";
import { CalendarTracker } from "./CalendarTracker";
import { NotificationBell } from "./NotificationBell";

function LiveClock() {
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const dateStr = now.toLocaleDateString("en-IN", {
    weekday: "short",
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
  const timeStr = now.toLocaleTimeString("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });

  return <span>{dateStr} · {timeStr}</span>;
}

export function TopHeader() {
  return (
    <header className="sticky top-0 z-40 h-[60px] bg-white border-b border-[#E2E8F0] flex items-center px-6 gap-6">
      <div className="flex-1" />
      <CalendarTracker />
      <NotificationBell />
      <div className="text-[12px] text-[#475569] whitespace-nowrap">
        <LiveClock />
      </div>
    </header>
  );
}
