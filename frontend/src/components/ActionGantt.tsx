import { useMemo } from "react";
import { useAppState, type ActionItem, type BU, type Status } from "@/context/AppState";

const TODAY = new Date(2026, 5, 25); // 25 Jun 2026

const statusBar: Record<Status, string> = {
  Overdue:       "#FEE2E2",
  "In Progress": "#FEF3C7",
  "On Track":    "#DCFCE7",
  Ahead:         "#DCFCE7",
};

const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

function parseDate(s: string): Date | null {
  if (!s) return null;
  if (/today|just now/i.test(s)) return TODAY;
  const m = s.match(/(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})/);
  if (!m) return null;
  const day = parseInt(m[1], 10);
  const mon = MONTHS.findIndex((x) => x.toLowerCase() === m[2].slice(0, 3).toLowerCase());
  if (mon < 0) return null;
  return new Date(parseInt(m[3], 10), mon, day);
}

function fmt(d: Date) {
  return `${d.getDate()} ${MONTHS[d.getMonth()]}`;
}

// retained for back-compat with other modules
export function relativeTime(s?: string, today: Date = TODAY): string {
  if (!s) return "-";
  if (/today|just now/i.test(s)) return "just now";
  const d = parseDate(s);
  if (!d) return s;
  const diffDays = Math.round((today.getTime() - d.getTime()) / 86400000);
  if (diffDays <= 0) return "today";
  if (diffDays === 1) return "1 day ago";
  if (diffDays < 7) return `${diffDays} days ago`;
  if (diffDays < 30) { const w = Math.round(diffDays / 7); return `${w} week${w === 1 ? "" : "s"} ago`; }
  const m = Math.round(diffDays / 30);
  return `${m} month${m === 1 ? "" : "s"} ago`;
}

interface Props {
  filterBU?: BU | "All";
  filterStatus?: Status | "All";
  compact?: boolean;
  filterFn?: (a: ActionItem) => boolean;
}

const ROW_MIN_H = 48;
const LABEL_COL = 220;
const MAX_HEIGHT = 360;

export function ActionGantt({ filterBU = "All", filterStatus = "All", filterFn }: Props) {
  const { actions } = useAppState();
  const rows = actions.filter((a) => {
    if (filterFn && !filterFn(a)) return false;
    if (filterBU !== "All" && a.bu !== filterBU) return false;
    if (filterStatus !== "All" && a.status !== filterStatus) return false;
    return true;
  });

  const items = useMemo(() => {
    return rows.map((a) => {
      const startFromSource = parseDate(a.sourceMeeting);
      const end = parseDate(a.due) ?? new Date(TODAY.getTime() + 14 * 86400000);
      const start = startFromSource ?? new Date(end.getTime() - 14 * 86400000);
      return { a, start, end: end < start ? new Date(start.getTime() + 7 * 86400000) : end };
    });
  }, [rows]);

  if (items.length === 0) {
    return <div className="text-[12px] text-[#94A3B8] py-6 text-center bg-white border border-[#E2E8F0] rounded-[14px]">No actions match these filters.</div>;
  }

  const min = new Date(Math.min(...items.map((x) => x.start.getTime())));
  const max = new Date(Math.max(...items.map((x) => x.end.getTime())));
  const startMs = Math.min(min.getTime(), new Date(2026, 4, 1).getTime());
  const endMs = Math.max(max.getTime(), new Date(2026, 6, 31).getTime());

  // Build month list spanning [startMs..endMs]
  const months: { y: number; m: number }[] = [];
  {
    const c = new Date(startMs);
    c.setDate(1);
    const last = new Date(endMs);
    while (c.getFullYear() < last.getFullYear() || (c.getFullYear() === last.getFullYear() && c.getMonth() <= last.getMonth())) {
      months.push({ y: c.getFullYear(), m: c.getMonth() });
      c.setMonth(c.getMonth() + 1);
    }
  }

  // Build column stops: month start for each month + TODAY inserted in the
  // today-month + a final end stop. Each adjacent pair becomes one equal-
  // width column in the header (and one equal-width segment for the body).
  type Stop = { t: number };
  const stops: Stop[] = [];
  const labels: string[] = [];
  const todayMonth = months.findIndex((mo) => mo.y === TODAY.getFullYear() && mo.m === TODAY.getMonth());
  months.forEach((mo, i) => {
    stops.push({ t: new Date(mo.y, mo.m, 1).getTime() });
    labels.push(`${MONTHS[mo.m]} ${String(mo.y).slice(2)}`);
    if (i === todayMonth) {
      stops.push({ t: TODAY.getTime() });
      labels.push("TODAY");
    }
  });
  const lastMo = months[months.length - 1];
  stops.push({ t: new Date(lastMo.y, lastMo.m + 1, 1).getTime() });
  const segCount = Math.max(1, stops.length - 1);

  function pctOf(d: Date) {
    const tt = d.getTime();
    if (tt <= stops[0].t) return 0;
    if (tt >= stops[stops.length - 1].t) return 100;
    for (let i = 0; i < stops.length - 1; i++) {
      if (tt <= stops[i + 1].t) {
        const frac = (tt - stops[i].t) / Math.max(1, stops[i + 1].t - stops[i].t);
        return ((i + frac) / segCount) * 100;
      }
    }
    return 100;
  }

  const todayPct = pctOf(TODAY);
  const scrollBodyMax = MAX_HEIGHT;

  return (
    <div className="bg-white rounded-[14px] border border-[#E2E8F0] overflow-hidden">
      {/* Header */}
      <div className="flex border-b border-[#E2E8F0]" style={{ background: "#F8FAFC" }}>
        <div
          className="px-3 py-2 shrink-0 border-r border-[#E2E8F0] flex items-center"
          style={{ width: LABEL_COL, fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "#475569" }}
        >
          Action
        </div>
        <div className="flex flex-1">
          {labels.map((lbl, i) => (
            <div
              key={i}
              className="flex-1 border-l border-[#E2E8F0] px-2 py-2 text-center truncate first:border-l-0"
              style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: lbl === "TODAY" ? "#0A2540" : "#475569" }}
            >
              {lbl}
            </div>
          ))}
        </div>
      </div>

      {/* Body */}
      <div className="relative overflow-y-auto" style={{ maxHeight: scrollBodyMax }}>
        <div>
          {items.map(({ a, start, end }) => {
            const left = pctOf(start);
            const right = pctOf(end);
            const width = Math.max(2, right - left);
            const bg = statusBar[a.status];
            return (
              <div key={a.id} className="flex items-stretch border-b border-[#EEF2F7] hover:bg-[#F1F5F9]" style={{ minHeight: ROW_MIN_H }}>
                <div className="px-3 py-2 shrink-0 border-r border-[#EEF2F7] flex items-center" style={{ width: LABEL_COL }}>
                  <div className="text-[13px] text-[#0A2540] font-medium leading-snug break-words" title={a.title}>{a.title}</div>
                </div>
                <div className="relative flex-1 self-stretch">
                  {/* gridlines aligned with header columns */}
                  {labels.map((_, i) => (
                    i === 0 ? null : <div key={i} className="absolute top-0 bottom-0 border-l border-[#EEF2F7]" style={{ left: `${(i / segCount) * 100}%` }} />
                  ))}
                  {/* today line */}
                  {todayPct >= 0 && todayPct <= 100 && (
                    <div className="absolute top-0 bottom-0 border-l border-dotted z-10 pointer-events-none" style={{ left: `${todayPct}%`, borderColor: "#475569" }} />
                  )}
                  {/* bar */}
                  <div
                    className="absolute top-1/2 -translate-y-1/2 min-h-5 rounded-[4px] flex items-center px-2 overflow-hidden"
                    style={{ left: `${left}%`, width: `${width}%`, background: bg, maxWidth: `calc(100% - ${left}%)` }}
                    title={`${a.title} · ${fmt(start)} → ${fmt(end)} · ${a.status}`}
                  >
                    {width >= 8 && (
                      <span className="text-[11px] font-semibold whitespace-normal break-words leading-tight" style={{ color: a.status === "Overdue" ? "#991B1B" : a.status === "In Progress" ? "#92400E" : "#166534" }}>{fmt(end)} · {a.status === "Ahead" ? "On Track" : a.status}</span>
                    )}
                  </div>
                  {width < 8 && (
                    <span
                      className="absolute top-1/2 -translate-y-1/2 text-[11px] font-medium pointer-events-none text-[#0A2540] whitespace-normal break-words leading-tight pr-2"
                      style={{ left: `calc(${left}% + ${width}% + 6px)`, maxWidth: `calc(100% - ${left}% - ${width}% - 8px)` }}
                    >
                      {fmt(end)} · {a.status === "Ahead" ? "On Track" : a.status}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 px-3 py-2 border-t border-[#EEF2F7] bg-[#F1F5F9] text-[11px] text-[#475569]">
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-[3px]" style={{ background: statusBar.Overdue }} /> Overdue</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-[3px]" style={{ background: statusBar["In Progress"] }} /> In Progress</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-[3px]" style={{ background: statusBar["On Track"] }} /> On Track</span>
      </div>
    </div>
  );
}
