import { useEffect, useState } from "react";
import { useAppState, parseDueDate, type ViewName, type Urgency, type DrKey, type BU, type Status, type PendingDecision, DR_VIEW } from "@/context/AppState";
import { useAppNavigate } from "@/hooks/useAppNavigate";

function daysUntil(dateStr: string): number | null {
  const d = parseDueDate(dateStr);
  if (!d) return null;
  const today = new Date(); today.setHours(0, 0, 0, 0);
  return Math.round((d.getTime() - today.getTime()) / 86_400_000);
}
import { TODAY_CALENDAR } from "@/data/mockData";
import { Pencil, X, MessageSquare, ChevronDown, RefreshCw, MessageCircle, Trash2 } from "lucide-react";
import { DECISION_CONTEXT } from "@/components/DRWorkspace";
import { resolveLedgerActionId } from "@/lib/ledgerLink";
import { fetchTodayEvents, fetchAuthState, fetchAppConfig } from "@/lib/api";
import { ChatPanel } from "@/components/ChatPanel";
import type { ChatResponse } from "@/data/mockData";
import { useOpenTracker, useCompleteTrackerItem, useUncompleteTrackerItem, useEditTrackerItem, useDeleteTrackerItem } from "@/hooks/useBackend";
import type { OpenTrackerItem, PendingDecisionItem } from "@/lib/api";
import { EditActionModal, type EditActionData } from "@/components/EditActionModal";

interface MetricTile {
  label: string;
  value: string;
  delta: string;
  tone: "emerald" | "amber" | "crimson" | "info" | "grey";
}

const LAUNCHPAD_METRICS: MetricTile[] = [
  { label: "Deposit Growth",          value: "₹1,18,900 Cr", delta: "↑ MoM +1.88%", tone: "emerald" },
  { label: "Advances Growth",         value: "₹1,04,320 Cr", delta: "↑ MoM +1.74%", tone: "emerald" },
  { label: "NIM",                     value: "5.4%",         delta: "↓ MoM -10bps",  tone: "crimson" },
  { label: "Credit Cost",             value: "210bps",       delta: "↓ MoM -20bps",  tone: "emerald" },
  { label: "Cost to Income",          value: "58.2%",        delta: "↓ MoM -70bps",  tone: "emerald" },
  { label: "OP Profit / Avg Advances", value: "3.55%",       delta: "↑ MoM +4bps",   tone: "emerald" },
  { label: "CASA",                    value: "₹42,150 Cr · 35.5%", delta: "↑ MoM +6bps", tone: "emerald" },
  { label: "PBT",                     value: "₹185 Cr",      delta: "↑ MoM +15.6%",  tone: "emerald" },
  { label: "Secured Retail Advances", value: "₹29,805 Cr",   delta: "↑ MoM +2%",     tone: "emerald" },
  { label: "Retail Deposits (BBRL)",  value: "₹1,578.4 Cr",  delta: "↑ MoM +12.5%",  tone: "emerald" },
];

const LAUNCHPAD_CHIPS: string[] = [
  // "What's my biggest priority today?",
  // "Show me overdue action items",
  // "What meetings need my attention?",
  // "Summarize pending decisions",
  // "How are we performing this month?"
];

function launchpadResponse(q: string): ChatResponse {
  const query = q.toLowerCase();
  
  if (query.includes("priority") || query.includes("important")) {
    return {
      text: "Based on your dashboard, your biggest priorities today are:\n\n1. **3 overdue decisions** requiring immediate attention\n2. **High-stakes meetings** with pre-meeting briefs available\n3. **Monthly metrics review** - overall performance is positive with +20% advances growth\n\nI recommend starting with the overdue decisions in your 'Awaiting Your Decision' section."
    };
  }
  
  if (query.includes("overdue") || query.includes("action")) {
    return {
      text: "You currently have **3 overdue items** that need immediate attention:\n\n• Risk-related decisions requiring your input\n• Follow-ups from recent MoM sessions\n• Strategic approvals pending since last week\n\nClick on any item in the 'Awaiting Your Decision' section to open it in the Action & Accountability Ledger for detailed tracking."
    };
  }
  
  if (query.includes("meeting") || query.includes("schedule")) {
    return {
      text: "Your schedule shows several **high-stakes touchpoints** today. Meetings with available pre-meeting briefs will show 'Open Pre Meeting Brief' - click these to access:\n\n• Key discussion points\n• Background context\n• Recommended questions to ask\n\nThis ensures you're fully prepared for each strategic conversation."
    };
  }
  
  if (query.includes("decision") || query.includes("pending")) {
    return {
      text: "You have **pending decisions** across multiple areas:\n\n• **Risk**: Credit policy approvals\n• **BBRL**: Product strategy decisions  \n• **Wholesale**: Client relationship matters\n• **Retail Assets**: Portfolio management calls\n\nEach decision includes context, urgency level, and the requesting team. Use the Action Ledger to track resolution progress."
    };
  }
  
  if (query.includes("performance") || query.includes("metrics") || query.includes("month")) {
    return {
      text: "**Strong monthly performance** across key metrics:\n\n✅ **Advances**: ₹3,800 Cr (+20% MoM)\n✅ **Deposits**: ₹3,200 Cr (+19% MoM) \n✅ **CASA**: ₹2,900 Cr (+16% MoM)\n✅ **PBT**: ₹1,600 Cr (+20% MoM)\n\n⚠️ **Watch areas**: Cost to Income at 15% (+3% MoM), Credit Cost at ₹1,500 Cr (+10% MoM)\n\nOverall trajectory remains positive with strong growth momentum."
    };
  }
  
  return {
    text: "Not found in the provided emails"
  };
}


const C = {
  danger: "#DC2626",
  warning: "#D97706",
  success: "#16A34A",
  info: "#1D4ED8",
  navy: "#0A2540",
  body: "#0A2540",
  muted: "#475569",
  hairline: "#E2E8F0",
  meetingTint: "#EAF2FB",
};

const TONE: Record<string, string> = {
  emerald: "#16A34A",
  amber: "#D97706",
  crimson: "#DC2626",
  info: "#1D4ED8",
  grey: "#64748B",
};
const URG_LABEL: Record<Urgency, string> = {
  overdue: "Overdue", today: "Today", this_week: "This week",
};

/* Bold the heading portion before a hyphen / en-dash / em-dash separator */
function renderDecisionText(text: string) {
  const m = text.match(/^(.*?)\s+[-–—]\s+(.*)$/);
  if (!m) return <span className="font-semibold">{text}</span>;
  return (<><span className="font-semibold">{m[1]}</span>{` — ${m[2]}`}</>);
}

/* Card wrapper - white background, visible border, internal padding */
function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-white border border-[#E2E8F0] rounded-[14px] ${className}`}>
      {children}
    </div>
  );
}

function SectionHeader({ children, right }: { children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between mb-3">
      <div className="text-[13px] uppercase tracking-[0.08em] font-bold text-[#0A2540]">{children}</div>
      {right}
    </div>
  );
}

export function Launchpad() {
  const touchpoints = TODAY_CALENDAR.length;
  const [userName, setUserName] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const authState = await fetchAuthState();
        if (cancelled) return;
        if (authState.authenticated && authState.user?.name) {
          setUserName(authState.user.name);
        }
      } catch {
        // Silently fail - will fall back to default greeting
        if (cancelled) return;
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const greeting = userName 
    ? `Good morning, ${userName}` 
    : "Good morning";

  return (
    <div className="p-7">
      <div>
        <h1 className="serif text-[32px] leading-tight font-normal" style={{ color: C.navy }}>
          {greeting}
        </h1>
        <div className="text-[13px] text-[#475569] mt-1"></div>
      </div>

      <div className="grid gap-7 mt-6 items-start" style={{ gridTemplateColumns: "58% 42%" }}>
        {/* LEFT COLUMN - 58% */}
        <div className="space-y-5 min-w-0">
          <ScheduleCard />
          <AwaitingDecisionCard />
          <DecisionsInMotionCard />
        </div>

        {/* RIGHT RAIL - 42% - Metrics Only */}
        <div className="min-w-0">
          <Card className="px-5 py-4">
            <div className="text-[12px] uppercase tracking-[0.08em] font-bold text-[#0A2540] mb-2.5">
              Monthly Metrics
            </div>
            <div className="grid grid-cols-2 gap-3">
              {LAUNCHPAD_METRICS.map((m) => (
                <div key={m.label} className="border border-[#E2E8F0] rounded-[10px] px-3 py-3 overflow-hidden">
                  <div className="text-[11px] uppercase tracking-[0.05em] font-semibold text-[#0A2540] line-clamp-2 break-words">
                    {m.label}
                  </div>
                  <div className="text-[22px] font-semibold leading-tight mt-1 tabular-nums" style={{ color: "#0A2540" }}>
                    {m.value}
                  </div>
                  <div className="text-[11px] font-semibold mt-1" style={{ color: TONE[m.tone] }}>
                    {m.delta}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
      
      <ChatBubble chips={LAUNCHPAD_CHIPS} responder={launchpadResponse} />
    </div>
  );
}


/* =====================================================
   Card 1 - Today's Schedule (local edit + add)
   ===================================================== */
interface CalEvent { id?: string; time: string; title: string; view: ViewName | null; start?: string; end?: string; }

function ScheduleCard() {
  const { loadMeetingBrief, loadPremeetingBrief, clearPremeetingBrief } = useAppState();
  const navigate = useAppNavigate();
  const [events, setEvents] = useState<CalEvent[]>(
    TODAY_CALENDAR.map((m) => ({ time: m.time, title: m.title, view: (m.view as ViewName | null) ?? null }))
  );
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const [loadingOutlook, setLoadingOutlook] = useState(true);
  const [outlookError, setOutlookError] = useState(false);
  const [usingOutlookData, setUsingOutlookData] = useState(false);
  const [pollingInterval, setPollingInterval] = useState<number>(300); // default 5 min
  const [refreshing, setRefreshing] = useState(false);

  // Fetch app config to get polling interval
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const config = await fetchAppConfig();
        if (cancelled) return;
        setPollingInterval(config.calendar_polling_interval_seconds);
      } catch {
        if (cancelled) return;
        // Use default polling interval if config fetch fails
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Pull the signed-in user's FULL current-day schedule from Outlook via
  // the Calendar agent (GET /agents/calendar/today, which fetches every
  // event on today's calendar through Microsoft Graph and returns them
  // sorted by start time). Falls back to the local/mock rows above if
  // the call fails or the signed-in user has no events today.
  const fetchSchedule = async () => {
    try {
      const res = await fetchTodayEvents();
      if (res.events.length > 0) {
        setEvents(
          res.events.map((e) => ({
            id: e.id,
            // Always render in Asia/Kolkata (IST) explicitly, regardless of
            // the browser/OS's own local timezone - relying on the client
            // device being set to IST caused meeting times to render wrong
            // when it wasn't (see calendar time fix, logged in
            // backend/logs/calendar_agent.log for verification).
            time: new Date(e.start).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "Asia/Kolkata" }),
            title: e.subject,
            view: (e.matched_view as ViewName | null) ?? null,
            start: e.start,
            end: e.end,
          }))
        );
        setUsingOutlookData(true);
      }
      setOutlookError(false);
    } catch {
      setOutlookError(true);
    }
  };

  // Initial load on component mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      await fetchSchedule();
      if (!cancelled) setLoadingOutlook(false);
    })();
    return () => { cancelled = true; };
  }, []);

  // Set up polling interval to refresh calendar in real-time
  useEffect(() => {
    // Don't set up polling if still loading or if we have an error
    if (loadingOutlook || outlookError) return;

    const pollIntervalMs = pollingInterval * 1000; // convert seconds to milliseconds
    const intervalId = setInterval(async () => {
      await fetchSchedule();
    }, pollIntervalMs);

    return () => clearInterval(intervalId);
  }, [loadingOutlook, outlookError, pollingInterval]);

  // Highlight the meeting currently in progress, or otherwise the next
  // upcoming one. For live Outlook data this is derived from actual
  // start/end timestamps; for the local/mock rows it falls back to
  // whichever row has a linked workspace (`view`), preserving the
  // original behavior.
  const nextIdx = usingOutlookData
    ? (() => {
        const now = Date.now();
        const inProgress = events.findIndex(
          (e) => e.start && e.end && new Date(e.start).getTime() <= now && now < new Date(e.end).getTime()
        );
        if (inProgress !== -1) return inProgress;
        let bestIdx = -1;
        let bestStart = Infinity;
        events.forEach((e, i) => {
          if (!e.start) return;
          const t = new Date(e.start).getTime();
          if (t > now && t < bestStart) { bestStart = t; bestIdx = i; }
        });
        return bestIdx;
      })()
    : events.findIndex((m) => m.view);

  function openRow(e: CalEvent) {
    if (e.id) {
      loadMeetingBrief(e.id);
      // Record the clicked meeting + fetch its meeting-scoped email match
      // so the destination workspace shows only THIS meeting's threads
      // instead of the whole department mailbox. Content-driven on the
      // backend, so it generalizes to every meeting.
      loadPremeetingBrief({ id: e.id, title: e.title, view: e.view });
    } else {
      // No Graph id (e.g. a locally-added row): can't scope to a meeting,
      // so clear any stale selection and let the workspace show the
      // department-wide view.
      clearPremeetingBrief();
    }
    if (e.view) navigate(e.view, "Pre-Meeting");
  }
  function saveEdit(i: number, time: string, title: string) {
    setEvents((evs) => evs.map((e, j) => (j === i ? { ...e, time: time || e.time, title: title || e.title } : e)));
    setEditingIdx(null);
  }

  async function handleRefresh() {
    setRefreshing(true);
    await fetchSchedule();
    setRefreshing(false);
  }

  return (
    <Card className="px-5 py-4">
      <SectionHeader
        right={
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            title="Refresh calendar"
            className="flex items-center gap-1.5 text-[11.5px] font-semibold text-[#475569] hover:text-[#0A2540] disabled:opacity-50 transition"
          >
            <RefreshCw size={13} className={refreshing ? "animate-spin" : ""} />
            Refresh
          </button>
        }
      >
        Today's schedule · Outlook
      </SectionHeader>
      <div className="text-[12px] text-[#94A3B8] mb-2 -mt-1">Click any meeting with a brief to open its pre-meeting workspace.</div>
      {loadingOutlook && (
        <div className="text-[12px] text-[#94A3B8] py-2">Loading today's schedule from Outlook…</div>
      )}
      {!loadingOutlook && outlookError && (
        <div className="text-[12px] text-[#94A3B8] mb-2">Couldn't reach Outlook calendar — showing local schedule.</div>
      )}
      {!loadingOutlook && events.length === 0 && (
        <div className="text-[12px] text-[#94A3B8] py-2">No meetings scheduled for today.</div>
      )}
      <div className="space-y-2">
        {!loadingOutlook && events.map((m, i) => {
          const isNext = i === nextIdx;
          const isEditing = editingIdx === i;
          return (
            <div
              key={i}
              className={`group flex items-center gap-3 px-3 py-2.5 rounded-[10px] border cursor-pointer ${isNext ? "border-[#C9DCF2]" : "border-[#E2E8F0] hover:bg-[#F1F5F9]"}`}
              style={{ background: isNext ? C.meetingTint : "transparent" }}
              onClick={() => !isEditing && openRow(m)}
            >
              {isEditing ? (
                <EditRow
                  initialTime={m.time}
                  initialTitle={m.title}
                  onSave={(t, ti) => saveEdit(i, t, ti)}
                  onCancel={() => setEditingIdx(null)}
                />
              ) : (
                <>
                  <div className="text-[12.5px] font-semibold tabular-nums w-12" style={{ color: C.navy }}>{m.time}</div>
                  <div className="flex-1 text-[12.5px]" style={{ color: C.body }}>
                    {m.title}
                    {m.view && <span className="ml-2 text-[10px] uppercase tracking-[0.06em] font-semibold" style={{ color: C.info }}>· Open Pre Meeting Brief</span>}
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); setEditingIdx(i); }}
                    className="text-[#94A3B8] hover:text-[#0A2540] opacity-0 group-hover:opacity-100 transition"
                    title="Edit"
                  >
                    <Pencil size={12} />
                  </button>
                </>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function EditRow({ initialTime, initialTitle, onSave, onCancel }: { initialTime: string; initialTitle: string; onSave: (t: string, ti: string) => void; onCancel: () => void }) {
  const [time, setTime] = useState(initialTime);
  const [title, setTitle] = useState(initialTitle);
  return (
    <>
      <input
        autoFocus
        value={time}
        onChange={(e) => setTime(e.target.value)}
        placeholder="09:30"
        onKeyDown={(e) => { if (e.key === "Escape") onCancel(); }}
        className="w-16 text-[12px] tabular-nums border border-[#E2E8F0] rounded-[6px] px-2 py-1 focus:outline-none focus:border-[#0A2540]"
        onClick={(e) => e.stopPropagation()}
      />
      <input
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Meeting title"
        onKeyDown={(e) => { if (e.key === "Enter") onSave(time, title); if (e.key === "Escape") onCancel(); }}
        className="flex-1 text-[12.5px] border border-[#E2E8F0] rounded-[6px] px-2 py-1 focus:outline-none focus:border-[#0A2540]"
        onClick={(e) => e.stopPropagation()}
      />
      <button
        onClick={(e) => { e.stopPropagation(); onSave(time, title); }}
        className="text-[11px] font-semibold px-2 py-1 rounded-[6px]"
        style={{ background: C.navy, color: "white" }}
      >
        Save
      </button>
      <button onClick={(e) => { e.stopPropagation(); onCancel(); }} className="text-[#94A3B8] hover:text-[#0A2540]">
        <X size={13} />
      </button>
    </>
  );
}

/* =====================================================
   Card 2 - Awaiting Your Decision (collapsible)
   ===================================================== */
function AwaitingDecisionCard() {
  const { pending } = useAppState();
  const [open, setOpen] = useState(false);
  const { data: trackerData } = useOpenTracker();
  const openItems = pending.filter((p) => p.status === "open");

  // Apply 7-day upcoming window to pending decisions for Launchpad display.
  const filteredPending = filterUpcoming7Days(
    (trackerData?.pending_decisions ?? []).map((d) => ({ ...d, status: d.status }))
  );
  const overdue = filteredPending.filter((p) => p.status === "Overdue").length;
  const today = filteredPending.filter((p) => p.status === "Due Today").length;
  const thisWeek = filteredPending.filter((p) => p.status === "Upcoming").length;

  return (
    <Card className="lov-urgent">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-[#F8FAFC] transition rounded-[14px]"
      >
        <div className="flex items-center gap-3">
          <div className="text-[13px] uppercase tracking-[0.08em] font-bold text-[#0A2540]">Awaiting your decision</div>
          <div className="inline-flex items-center justify-center w-6 h-6 rounded-full text-white text-[13px] font-bold" style={{ backgroundColor: C.danger }}>
            {openItems.length}
          </div>
          <div className="text-[12px] text-[#64748B]">
            <span className="tabular-nums">{overdue}</span> overdue · <span className="tabular-nums">{today}</span> due today · <span className="tabular-nums">{thisWeek}</span> upcoming
          </div>
        </div>
        <ChevronDown size={16} className={`text-[#64748B] transition ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="px-5 pb-4 border-t border-[#E2E8F0] pt-3">
          <PendingList />
        </div>
      )}
    </Card>
  );
}

/* ---- Undo Toast ---- */
interface UndoToast {
  id: string;
  text: string;
  onUndo: () => void;
}

function UndoToastBanner({ toast, onDismiss }: { toast: UndoToast; onDismiss: () => void }) {
  useEffect(() => {
    const t = setTimeout(onDismiss, 5000);
    return () => clearTimeout(t);
  }, [onDismiss]);

  return (
    <div className="fixed bottom-6 right-6 z-[100] w-[360px] slide-in-right flex items-center justify-between gap-3 bg-[#0A2540] text-white text-[12px] px-4 py-2.5 rounded-[10px] shadow-lg">
      <span className="truncate">
        <span className="font-semibold text-[#4ADE80] mr-1.5">✓ Done.</span>
        {toast.text.length > 60 ? toast.text.slice(0, 60) + "…" : toast.text}
      </span>
      <button
        onClick={toast.onUndo}
        className="shrink-0 text-[11.5px] font-semibold text-[#93C5FD] hover:text-white underline transition"
      >
        Undo
      </button>
    </div>
  );
}

function PendingList() {
  const { pending, updatePending, addPending, actions, setFocusActionId, completePendingDecision, showToast } = useAppState();
  const navigate = useAppNavigate();
  const [adding, setAdding] = useState(false);
  const [noteFor, setNoteFor] = useState<string | null>(null);
  const [editingPending, setEditingPending] = useState<PendingDecision | null>(null);
  const editItem = useEditTrackerItem();
  const completeItem = useCompleteTrackerItem();
  const open = pending.filter((p) => p.status === "open");

  function openLedger(id: string, text: string) {
    const fid = resolveLedgerActionId(id, text, actions);
    if (fid) setFocusActionId(fid);
    navigate("Ledger");
  }

  function add(val: string) {
    if (!val.trim()) { setAdding(false); return; }
    addPending({
      id: `PD${Date.now()}`, text: val.trim(), urgency: "this_week",
      source: "MoM", team: "Risk", dueDate: "This week", status: "open",
    });
    setAdding(false);
  }

  function handleComplete(e: React.SyntheticEvent, p: PendingDecision) {
    e.stopPropagation();
    // Immediately remove from UI
    updatePending(p.id, { status: "resolved" });
    // Inject into Ledger Completed tab
    completePendingDecision(p);
    // For backend items, also call the API
    const backendId = p.id.startsWith("tracker-pd-") ? p.id.replace("tracker-pd-", "") : null;
    if (backendId) {
      completeItem.mutate(backendId);
    }
    showToast("Marked complete — visible in Action & Accountability Ledger");
  }

  return (
    <div>
      <div className="text-[12px] text-[#94A3B8] mb-2">These decisions await your input — click any row to open it in the Ledger →</div>
      <div className="space-y-2">
        {open.length === 0 && <div className="text-[11.5px] text-[#94A3B8] py-2">No decisions awaiting input.</div>}
        {open.map((p) => {
          const isToday = p.urgency === "today";
          const isOverdue = p.urgency === "overdue";
          const cardBg = isOverdue ? "#FEE2E2" : isToday ? "#FEF3C7" : "#F1F5F9";
          const cardBorder = isOverdue ? "#FECACA" : isToday ? "#F5DCB1" : "#E2E8F0";
          return (
          <div key={p.id} onClick={() => { if (noteFor !== p.id) openLedger(p.id, p.text); }} title="Open in Action & Accountability Ledger" className="group rounded-[10px] border p-2.5 flex items-start gap-2.5 cursor-pointer" style={{ background: cardBg, borderColor: cardBorder }}>
            {/* Checkbox on the left — click to complete */}
            <input
              type="checkbox"
              checked={false}
              onChange={(e) => handleComplete(e, p)}
              onClick={(e) => e.stopPropagation()}
              title="Mark as complete"
              className="mt-1 w-3.5 h-3.5 accent-[#16A34A] shrink-0 cursor-pointer"
            />
            <div className="flex-1 min-w-0">
              <div className="text-[12.5px] break-words" style={{ color: C.body }}>{renderDecisionText(p.text)}</div>
              {DECISION_CONTEXT[p.text] && (
                <div className="text-[12px] text-[#475569] break-words mt-1 leading-snug">{DECISION_CONTEXT[p.text]}</div>
              )}
              <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                <Tag color="#94A3B8">{p.dueDate || URG_LABEL[p.urgency]}</Tag>
                <button
                  onClick={(e) => { e.stopPropagation(); navigate(DR_VIEW[p.team]); }}
                  className="text-[10px] font-semibold px-1.5 py-[1px] rounded-[4px] uppercase tracking-[0.04em] bg-white/70 text-[#0A2540] hover:bg-white transition"
                  title={`Open ${p.team} workspace`}
                >
                  {p.business_unit || p.team}
                </button>
              </div>
              {noteFor === p.id && (
                <input
                  autoFocus
                  placeholder="Add a note, Enter to save"
                  defaultValue={p.note ?? ""}
                  onClick={(e) => e.stopPropagation()}
                  onBlur={(e) => { updatePending(p.id, { note: e.target.value }); setNoteFor(null); }}
                  onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); if (e.key === "Escape") setNoteFor(null); }}
                  className="mt-1 w-full text-[11.5px] bg-white border border-[#E2E8F0] rounded-[6px] px-2 py-1 focus:outline-none focus:border-[#0A2540]"
                />
              )}
            </div>
            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition shrink-0" onClick={(e) => e.stopPropagation()}>
              {/* Edit — only for backend-sourced pending decisions (tracker-pd-*) */}
              {p.id.startsWith("tracker-pd-") && (
                <IconBtn title="Edit action item" onClick={() => setEditingPending(p)}><Pencil size={12} /></IconBtn>
              )}
              <IconBtn title="Add note" onClick={() => setNoteFor(p.id)}><MessageSquare size={12} /></IconBtn>
              <IconBtn title="Dismiss" onClick={() => updatePending(p.id, { status: "dismissed" })}><X size={12} /></IconBtn>
            </div>
          </div>
          );
        })}
      </div>
      {/* Edit Action Item modal for pending decisions */}
      <EditActionModal
        open={editingPending !== null}
        initial={{
          text: editingPending?.text ?? "",
          owner: editingPending?.text.split(":")[0]?.trim() ?? "",
          business_unit: editingPending?.business_unit ?? "Wholesale Banking",
          due_date: editingPending?.dueRaw ?? "",
        }}
        onSave={(data: EditActionData) => {
          if (!editingPending) return;
          // Strip "tracker-pd-" prefix to get the backend item id.
          const backendId = editingPending.id.replace("tracker-pd-", "");
          editItem.mutate(
            {
              itemId: backendId,
              body: {
                owner: data.owner,
                due_date: data.due_date || undefined,
              },
            },
            {
              onSuccess: (res) => {
                // Update local pending state so UI reflects immediately.
                const item = res?.item;
                if (item) {
                  const newUrgency: Urgency = item.status === "Overdue" ? "overdue" : item.status === "Due Today" ? "today" : "this_week";
                  updatePending(editingPending.id, {
                    dueRaw: item.due_date || undefined,
                    dueDate: item.due_label || undefined,
                    urgency: newUrgency,
                  });
                }
                setEditingPending(null);
              },
              onError: () => {
                // keep modal open so user can retry
              },
            },
          );
        }}
        onClose={() => setEditingPending(null)}
        saving={editItem.isPending}
      />
    </div>
  );
}

/* =====================================================
   Card 3 - Open Actions Status Tracker (read-only, collapsible)
   Feeds from GET /agents/open-tracker — email_clusters.csv data,
   classified by due date. Auto-refreshes every 60 s.
   ===================================================== */

/**
 * 7-day look-ahead window filter for the Launchpad cards.
 * Keeps Overdue, Due Today, Upcoming with no due date, and
 * Upcoming items whose due_date is within 7 days of today.
 * The Ledger uses the unfiltered backend response directly.
 */
function filterUpcoming7Days<T extends { status: string; due_date?: string | null }>(items: T[]): T[] {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() + 7);
  cutoff.setHours(23, 59, 59, 999);
  return items.filter((i) => {
    if (i.status !== "Upcoming") return true;
    if (!i.due_date) return true; // no due date → always show
    return new Date(i.due_date) <= cutoff;
  });
}

function DecisionsInMotionCard() {
  const [open, setOpen] = useState(false);
  const { data, isLoading: isQueryLoading, isError } = useOpenTracker();

  // Treat "pipeline hasn't completed a run yet this backend session" the
  // same as a loading state, so a backend restart never flashes stale
  // leftover data from before the restart (see pipeline_fresh on the
  // response — false means the backend deliberately withheld its data).
  const isWarmingUp = !isQueryLoading && data?.pipeline_fresh === false;
  const isLoading = isQueryLoading || isWarmingUp;

  // Apply 7-day upcoming window for Launchpad display only.
  const filteredItems = filterUpcoming7Days(data?.items ?? []);

  const overdue   = filteredItems.filter((i) => i.status === "Overdue").length;
  const dueToday  = filteredItems.filter((i) => i.status === "Due Today").length;
  const upcoming  = filteredItems.filter((i) => i.status === "Upcoming").length;
  const total     = filteredItems.length;

  return (
    <Card className="lov-urgent">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-[#F8FAFC] transition rounded-[14px]"
      >
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <div className="text-[13px] uppercase tracking-[0.08em] font-bold text-[#0A2540] shrink-0">
            Open Actions Status Tracker
          </div>
          {/* Count badge */}
          <div
            className="inline-flex items-center justify-center w-6 h-6 rounded-full text-white text-[12px] font-bold shrink-0"
            style={{ backgroundColor: overdue > 0 ? C.danger : C.info }}
          >
            {isLoading ? "…" : total}
          </div>
          {/* Summary chips */}
          {!isLoading && !isError && (
            <div className="text-[12px] text-[#64748B] truncate">
              <span className="tabular-nums font-semibold" style={{ color: overdue > 0 ? C.danger : undefined }}>
                {overdue}
              </span>{" "}
              overdue ·{" "}
              <span className="tabular-nums font-semibold" style={{ color: dueToday > 0 ? C.warning : undefined }}>
                {dueToday}
              </span>{" "}
              due today ·{" "}
              <span className="tabular-nums">{upcoming}</span> upcoming
            </div>
          )}
          {isWarmingUp && !isQueryLoading && (
            <div className="text-[12px] text-[#94A3B8]">Pipeline warming up…</div>
          )}
          {isQueryLoading && <div className="text-[12px] text-[#94A3B8]">Loading…</div>}
          {isError  && <div className="text-[12px] text-[#94A3B8]">Couldn't load — will retry</div>}
        </div>
        <ChevronDown size={16} className={`text-[#64748B] transition shrink-0 ml-2 ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <div className="border-t border-[#E2E8F0]">
          <OpenTrackerList items={filteredItems} isLoading={isLoading} isError={isError} />
        </div>
      )}
    </Card>
  );
}

/* Colour tokens per status */
const OT_STYLE: Record<string, { cardBg: string; cardBorder: string; dotColor: string; labelColor: string }> = {
  Overdue:    { cardBg: "#FEE2E2", cardBorder: "#FECACA", dotColor: C.danger,  labelColor: "#991B1B" },
  "Due Today":{ cardBg: "#FEF3C7", cardBorder: "#F5DCB1", dotColor: C.warning, labelColor: "#92400E" },
  Upcoming:   { cardBg: "#DCFCE7", cardBorder: "#BBF7D0", dotColor: C.success, labelColor: "#166534" },
};

/**
 * Map the full department name used in email_clusters.csv to the BU
 * value expected by the Action & Accountability Ledger (AppState BU type).
 */
const BU_MAP: Record<string, BU> = {
  "BB&RL":                                     "BBRL",
  "Wholesale Banking":                         "Wholesale Banking",
  "Retail Advances - Excluding Credit Cards":  "Retail Advances - Excluding Credit Cards",
  "Credit Cards":                              "Credit Cards",
};

function mapToBU(department: string): BU {
  return BU_MAP[department] ?? "BBRL";
}

function OpenTrackerList({
  items,
  isLoading,
  isError,
}: {
  items: OpenTrackerItem[];
  isLoading: boolean;
  isError: boolean;
}) {
  const { updateAction, removeAction, showToast, refreshOpenTracker } = useAppState();
  const navigate = useAppNavigate();
  const completeItem = useCompleteTrackerItem();
  const editItem = useEditTrackerItem();
  const deleteItem = useDeleteTrackerItem();

  // Optimistic local set — IDs we've already clicked ✓ on this session.
  // The item disappears instantly; the server write happens in background.
  const [localCompleted, setLocalCompleted] = useState<Set<string>>(new Set());
  const [localDeleted, setLocalDeleted] = useState<Set<string>>(new Set());

  // Edit modal state.
  const [editingItem, setEditingItem] = useState<OpenTrackerItem | null>(null);

  // Clear local completed items that are no longer in the fetched items list
  // This ensures that if the backend has processed the completion and removed
  // the item from the list, we don't keep it in our optimistic set
  useEffect(() => {
    if (!isLoading && !isError) {
      const currentItemIds = new Set(items.map(item => item.id));
      setLocalCompleted(prev => {
        const updated = new Set<string>();
        prev.forEach(id => {
          if (currentItemIds.has(id)) {
            updated.add(id);
          }
        });
        if (updated.size !== prev.size) {
          return updated;
        }
        return prev;
      });
    }
  }, [items, isLoading, isError]);

  function handleComplete(item: OpenTrackerItem) {
    if (!item.id || localCompleted.has(item.id)) return;

    // 1. Optimistic: hide immediately
    setLocalCompleted((prev) => new Set(prev).add(item.id));

    // 2. Call the backend
    completeItem.mutate(item.id, {
      onSuccess: (res) => {
        // 3. Mark the existing Ledger item as completed. The open item
        //    was already injected by the sync useEffect above with id
        //    `tracker-${item.id}`, so we update it in place.
        const completedDate = res.item.completed_at
          ? new Date(res.item.completed_at).toLocaleDateString("en-IN", {
              day: "2-digit", month: "short", year: "numeric",
            })
          : "today";

        updateAction(`tracker-${item.id}`, {
          status: "Completed",
          completed: true,
          lastUpdated: `${completedDate} · just now`,
        });

        showToast("Marked complete — visible in Action & Accountability Ledger");
      },
      onError: () => {
        // Roll back optimistic update if server write failed
        setLocalCompleted((prev) => {
          const next = new Set(prev);
          next.delete(item.id);
          return next;
        });
        showToast("Couldn't save — please try again");
      },
    });
  }

  function handleEditItem(item: OpenTrackerItem) {
    setEditingItem(item);
  }

  function handleSaveEdit(data: EditActionData) {
    if (!editingItem) return;
    editItem.mutate(
      {
        itemId: editingItem.id,
        body: {
          text: data.text,
          owner: data.owner,
          business_unit: data.business_unit,
          due_date: data.due_date || undefined,
        },
      },
      {
        onSuccess: () => {
          const nd = data.due_date
            ? (() => { const [y, mo, d] = data.due_date.split("-").map(Number); return new Date(y, mo - 1, d); })()
            : null;
          const days = data.due_date ? daysUntil(data.due_date) : null;
          const newStatus: Status =
            days === null || days >= 1 ? "On Track"
            : days === 0              ? "In Progress"
            :                           "Overdue";
          // Also update the Ledger's local copy (which BU views also read from).
          updateAction(`tracker-${editingItem.id}`, {
            title: data.owner && data.owner !== "Unassigned"
              ? `${data.owner}: ${data.text}`
              : data.text,
            owner: data.owner,
            bu: data.business_unit as BU,
            due: nd
              ? nd.toLocaleDateString("en-IN", {
                  day: "2-digit", month: "short", year: "numeric",
                })
              : "No due date",
            dueRaw: data.due_date || undefined,
            status: newStatus,
          });
          showToast("Action item updated");
          setEditingItem(null);
        },
        onError: () => {
          showToast("Couldn't save — please try again");
        },
      },
    );
  }

  function handleDeleteItem(item: OpenTrackerItem) {
    if (!item.id || localDeleted.has(item.id)) return;
    if (!confirm("Delete this action item permanently?")) return;
    // Optimistic: hide immediately
    setLocalDeleted((prev) => new Set(prev).add(item.id));
    removeAction(`tracker-${item.id}`);
    deleteItem.mutate(item.id, {
      onSuccess: () => refreshOpenTracker(),
      onError: () => {
        setLocalDeleted((prev) => { const next = new Set(prev); next.delete(item.id); return next; });
        showToast("Couldn't delete — please try again");
      },
    });
  }

  if (isLoading) {
    return <div className="px-5 py-4 text-[12px] text-[#94A3B8]">Fetching open action items…</div>;
  }
  if (isError) {
    return <div className="px-5 py-4 text-[12px] text-[#94A3B8]">Could not load action items. Will retry automatically.</div>;
  }

  const visibleItems = items.filter((i) => !localCompleted.has(i.id) && !localDeleted.has(i.id));

  if (visibleItems.length === 0) {
    return <div className="px-5 py-4 text-[12px] text-[#94A3B8]">No open action items found.</div>;
  }

  return (
    <div className="px-5 py-4 space-y-2 overflow-y-auto" style={{ maxHeight: "420px" }}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-[12px] text-[#94A3B8]">
          Action items across all business units · auto-refreshes every 60 s
        </span>
        <button
          onClick={() => navigate("Ledger")}
          className="text-[11px] font-semibold text-[#1D4ED8] hover:underline shrink-0 ml-3"
        >
          View Ledger →
        </button>
      </div>

      {visibleItems.map((item) => {
        const s = OT_STYLE[item.status] ?? OT_STYLE["Upcoming"];
        const completing = completeItem.isPending && !localCompleted.has(item.id);

        return (
          <div
            key={item.id || item.text}
            className="group rounded-[10px] border p-2.5 flex items-start gap-2.5"
            style={{ background: s.cardBg, borderColor: s.cardBorder }}
          >
            {/* Mark complete checkbox — always visible, no hover needed */}
            <input
              type="checkbox"
              checked={false}
              onChange={() => handleComplete(item)}
              disabled={completing || localCompleted.has(item.id)}
              title="Mark as complete"
              className="mt-1 w-3.5 h-3.5 accent-[#16A34A] shrink-0 cursor-pointer disabled:opacity-40 disabled:cursor-default"
            />

            <div className="flex-1 min-w-0">
              {/* Action text */}
              <div className="text-[12.5px] break-words leading-snug" style={{ color: C.body }}>
                {item.owner && item.owner !== "Unassigned" && (
                  <span className="font-semibold">{item.owner}: </span>
                )}
                {item.text}
              </div>

              {/* Tags row */}
              <div className="flex items-center gap-1.5 mt-1.5 flex-wrap">
                <span
                  className="text-[10px] font-semibold px-1.5 py-[2px] rounded-[4px] uppercase tracking-[0.04em]"
                  style={{ background: "rgba(255,255,255,0.6)", color: s.labelColor }}
                >
                  {item.due_label || item.status}
                </span>
                {item.business_unit && (
                  <span className="text-[10px] font-semibold px-1.5 py-[1px] rounded-[4px] uppercase tracking-[0.04em] bg-white/70 text-[#0A2540]">
                    {item.business_unit}
                  </span>
                )}
              </div>
            </div>

            {/* Action buttons — visible on hover */}
            <div className="shrink-0 mt-0.5 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              {/* Edit action item */}
              <button
                onClick={() => handleEditItem(item)}
                title="Edit action item"
                className="w-7 h-7 inline-flex items-center justify-center rounded-full"
                style={{ background: "rgba(255,255,255,0.7)", color: "#1D4ED8", border: "1.5px solid #1D4ED8" }}
              >
                <Pencil size={12} strokeWidth={2} />
              </button>
              {/* Delete action item */}
              <button
                onClick={() => handleDeleteItem(item)}
                title="Delete action item"
                className="w-7 h-7 inline-flex items-center justify-center rounded-full"
                style={{ background: "rgba(255,255,255,0.7)", color: "#DC2626", border: "1.5px solid #DC2626" }}
              >
                <Trash2 size={12} strokeWidth={2} />
              </button>
            </div>
          </div>
        );
      })}

      {/* Edit Action Item modal */}
      <EditActionModal
        open={editingItem !== null}
        initial={{
          text: editingItem?.text ?? "",
          owner: editingItem?.owner ?? "",
          business_unit: editingItem?.business_unit ?? "BBRL",
          due_date: editingItem?.due_date ?? "",
        }}
        onSave={handleSaveEdit}
        onClose={() => setEditingItem(null)}
        saving={editItem.isPending}
      />
    </div>
  );
}

/* =====================================================
   MIS Metrics card - 5 tiles in a row
   ===================================================== */
function LiveMetricsCard() {
  const metrics: { label: string; value: string; delta: string; tone: string; data: number[] }[] = [];
  const tone: Record<string, string> = { emerald: C.success, amber: C.warning, crimson: C.danger, info: C.info };
  const toneTint: Record<string, string> = { emerald: "#DCFCE7", amber: "#FEF3C7", crimson: "#FEE2E2", info: "#E8EFFA" };

  if (metrics.length === 0) return null;


  return (
    <Card className="px-5 py-5">
      <SectionHeader>MIS metrics · Last 5 business days</SectionHeader>
      <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(3, minmax(110px, 1fr))" }}>
        {metrics.map((m) => (
          <div key={m.label} className="border border-[#E2E8F0] rounded-[10px] px-3 py-3 overflow-hidden min-w-[110px]">
            <div className="flex items-start justify-between gap-2">
              <div className="text-[11px] uppercase tracking-[0.05em] font-semibold text-[#0A2540] line-clamp-2 break-words">{m.label}</div>
              <span className="mt-1 inline-block w-[6px] h-[6px] rounded-full shrink-0" style={{ background: tone[m.tone] }} />
            </div>
            <div className="text-[22px] font-semibold leading-tight mt-1 tabular-nums whitespace-nowrap" style={{ color: C.body }}>{m.value}</div>
            <span className="inline-block mt-1 text-[10px] font-semibold px-1.5 py-[2px] rounded-[4px] whitespace-nowrap" style={{ background: toneTint[m.tone], color: tone[m.tone] }}>{m.delta}</span>
          </div>
        ))}
      </div>
      <div className="text-[10px] uppercase tracking-[0.08em] text-[#94A3B8] mt-3">
        Last data sync · --
      </div>
    </Card>
  );
}

/* ---- Micro components ---- */
function Tag({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <span className="text-[10px] font-semibold px-1.5 py-[1px] rounded-[4px] uppercase tracking-[0.04em]" style={{ background: "#EEF2F7", color }}>
      {children}
    </span>
  );
}

function IconBtn({ children, title, onClick }: { children: React.ReactNode; title: string; onClick: () => void }) {
  return (
    <button title={title} onClick={onClick} className="w-6 h-6 inline-flex items-center justify-center rounded-[6px] hover:bg-[#EEF2F7] text-[#475569] hover:text-[#0A2540]">
      {children}
    </button>
  );
}

/* ── Floating "Ask your Chief of Staff" chat bubble ─────────────── */
function ChatBubble({ chips, responder }: { chips: string[]; responder: (q: string) => ChatResponse }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="fixed bottom-6 right-6 z-50">
      {open && (
        <div className="mb-3 w-[360px] shadow-xl fade-in">
          <ChatPanel
            title="Ask your Chief of Staff"
            chips={chips}
            getResponse={responder}
            placeholder="Ask anything across all data sources…"
            loadHistory
          />
        </div>
      )}
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-12 h-12 rounded-full flex items-center justify-center shadow-lg"
        style={{ background: "#0A2540" }}
        title="Ask your Chief of Staff"
      >
        <MessageCircle size={20} className="text-white" />
      </button>
    </div>
  );
}