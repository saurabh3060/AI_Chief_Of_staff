import { useState, useMemo, useEffect, useRef } from "react";
import { useAppState, parseDueDate, type Status, type ActionItem, type BU, BU_VIEW } from "@/context/AppState";
import { BU_COLORS, BU_ORDER } from "@/data/mockData";
import { ITEM_CONTEXT } from "@/components/DRWorkspace";
import { ChevronDown, Pencil, Trash2 } from "lucide-react";
import { useEditTrackerItem, useCompleteTrackerItem, useUncompleteTrackerItem, useDeleteTrackerItem } from "@/hooks/useBackend";
import { EditActionModal, type EditActionData } from "@/components/EditActionModal";

const C = {
  navy: "#0A2540",
  danger: "#DC2626",
  info: "#1D4ED8",
  success: "#16A34A",
  warning: "#D97706",
};

// Display labels for BUs — exact nomenclature as defined.
const BU_LABEL: Record<BU, string> = {
  "BBRL": "BBRL",
  "Credit Cards": "Credit Cards",
  "Retail Advances - Excluding Credit Cards": "Retail Advances - Excluding Credit Cards",
  "Wholesale Banking": "Wholesale Banking",
};

const DEPT_OPTIONS: { label: string; bu: BU }[] = [
  { label: "BBRL", bu: "BBRL" },
  { label: "Credit Cards", bu: "Credit Cards" },
  { label: "Retail Advances - Excluding Credit Cards", bu: "Retail Advances - Excluding Credit Cards" },
  { label: "Wholesale Banking", bu: "Wholesale Banking" },
];

/* ── Due-date bucket helpers ──────────────────────────────────── */
type Bucket = "overdue" | "today" | "upcoming" | "completed";

// Dropdown options for the "Status" filter — All / Overdue / Due today / Upcoming.
const BUCKET_FILTER_OPTIONS: (Bucket | "All")[] = ["All", "overdue", "today", "upcoming"];
const BUCKET_LABEL: Record<Bucket | "All", string> = {
  All: "Status: All",
  overdue: "Overdue",
  today: "Due today",
  upcoming: "Upcoming",
  completed: "Completed",
};

function startOfToday(): Date {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d;
}

function daysUntil(due: string): number | null {
  const d = parseDueDate(due);
  if (!d) return null;
  const today = startOfToday();
  return Math.round((d.getTime() - today.getTime()) / 86_400_000);
}

// Bucket + label are both derived PURELY from the due date, so they
// can never disagree. `a.status` is ignored as a classification source
// because it drifts out of sync after reopen/complete flows (e.g.
// reopenStatus falls back to "On Track" when parsing fails).
function bucketOf(a: ActionItem): Bucket {
  const n = daysUntil(a.due);
  if (n === null) return "upcoming";
  if (n < 0) return "overdue";
  if (n === 0) return "today";
  return "upcoming";
}

function dueLabel(a: ActionItem): { text: string; tone: "danger" | "warning" | "neutral" } {
  if (a.status === "Completed") return { text: "Completed", tone: "neutral" };
  const n = daysUntil(a.due);
  if (n === null) return { text: "No due date", tone: "neutral" };
  if (n < 0) return { text: "Overdue", tone: "danger" };
  if (n === 0) return { text: "Due today", tone: "warning" };
  if (n === 1) return { text: "Due tomorrow", tone: "warning" };
  if (n <= 7) return { text: `Due in ${n} days`, tone: "warning" };
  return { text: `Due in ${n} days`, tone: "neutral" };
}

const DUE_TONE: Record<"danger" | "warning" | "neutral", { bg: string; text: string }> = {
  danger: { bg: "#FEE2E2", text: "#991B1B" },
  warning: { bg: "#FEF3C7", text: "#92400E" },
  neutral: { bg: "#F1F5F9", text: "#475569" },
};

export function Ledger() {
  const {
    actions, updateAction, removeAction, addActions, navigate,
    focusActionId, setFocusActionId,
    ledgerTeamFilter, setLedgerTeamFilter,
    showToast, refreshOpenTracker,
    clearPremeetingBrief,
  } = useAppState();

  const [tab, setTab] = useState<"open" | "completed">("open");
  const [team, setTeam] = useState<BU | "All">("All");
  const completeTrackerItem = useCompleteTrackerItem();
  const uncompleteTrackerItem = useUncompleteTrackerItem();
  const deleteTrackerItem = useDeleteTrackerItem();

  // Deletes an action item. Optimistically removes it from the UI,
  // calls the backend, and restores on error.
  function handleDelete(a: ActionItem) {
    if (!confirm("Delete this action item permanently?")) return;
    removeAction(a.id);
    const backendId = a.id.startsWith("tracker-") ? a.id.replace("tracker-", "") : null;
    if (!backendId) return;
    deleteTrackerItem.mutate(backendId, {
      onSuccess: () => refreshOpenTracker(),
      onError: () => {
        addActions([a]);
        showToast("Couldn't delete — please try again");
      },
    });
  }

  // Toggles an action item's completed state. For backend-sourced tracker
  // items (id "tracker-{backendId}"), this also calls the real backend
  // complete/uncomplete endpoints so the change persists in Postgres/JSON
  // and the item correctly reappears in (or disappears from) the Open
  // Actions Status Tracker / Awaiting Your Decision queue.
  function handleToggleComplete(a: ActionItem, checked: boolean) {
    // Prevent rapid double-clicks from firing multiple backend calls.
    if (completeTrackerItem.isPending || uncompleteTrackerItem.isPending) return;
    // When unchecking, derive the correct status from the due date so the
    // "Completed" label doesn't linger on the item in the Open tab.
    function reopenStatus(): Status {
      const n = daysUntil(a.due);
      if (n === null || n >= 1) return "On Track";   // upcoming
      if (n === 0) return "In Progress";             // due today
      return "Overdue";                              // past due
    }
    updateAction(a.id, { completed: checked, status: checked ? "Completed" : reopenStatus() });
    const backendId = a.id.startsWith("tracker-") ? a.id.replace("tracker-", "") : null;
    if (!backendId) return;
    if (checked) {
      completeTrackerItem.mutate(backendId, {
        onSuccess: () => refreshOpenTracker(),
        onError: () => {
          updateAction(a.id, { completed: false });
          showToast("Couldn't save — please try again");
        },
      });
    } else {
      uncompleteTrackerItem.mutate(backendId, {
        onSuccess: () => refreshOpenTracker(),
        onError: () => {
          updateAction(a.id, { completed: true, status: "Completed" });
          showToast("Couldn't save — please try again");
        },
      });
    }
  }
  const [bucketFilter, setBucketFilter] = useState<Bucket | "All">("All");

  // Honour cross-page filter (set from DR/Overview pages)
  useEffect(() => {
    if (ledgerTeamFilter) {
      setTeam(ledgerTeamFilter);
      setLedgerTeamFilter(null);
    }
  }, [ledgerTeamFilter, setLedgerTeamFilter]);

  const scopeBase = useMemo(() => actions.filter((a) => (tab === "completed" ? !!a.completed : !a.completed)), [actions, tab]);

  /* ---- Summary-by-department table (always over the full open/completed scope, ignoring st/team/bucket filters) ---- */
  const deptGroups: { bu: BU; label: string; color: { bg: string; text: string }; items: ActionItem[] }[] = BU_ORDER
    .map((bu) => ({
      bu,
      label: BU_LABEL[bu],
      color: BU_COLORS[bu],
      items: scopeBase.filter((a) => a.bu === bu && !a.crossFunctional),
    }))
    .filter((g) => g.items.length > 0);
  const crossFunctionalItems = scopeBase.filter((a) => a.crossFunctional === true);

  const summaryRows = [
    ...deptGroups.map((g) => ({
      key: g.bu as BU | "Cross-functional",
      label: g.label,
      color: g.color,
      overdue: g.items.filter((a) => bucketOf(a) === "overdue").length,
      today: g.items.filter((a) => bucketOf(a) === "today").length,
      upcoming: g.items.filter((a) => bucketOf(a) === "upcoming").length,
      total: g.items.length,
    })),
    ...(crossFunctionalItems.length > 0
      ? [{
          key: "Cross-functional" as const,
          label: "Cross-functional",
          color: { bg: "#E0E7FF", text: "#3730A3" },
          overdue: crossFunctionalItems.filter((a) => bucketOf(a) === "overdue").length,
          today: crossFunctionalItems.filter((a) => bucketOf(a) === "today").length,
          upcoming: crossFunctionalItems.filter((a) => bucketOf(a) === "upcoming").length,
          total: crossFunctionalItems.length,
        }]
      : []),
  ];
  const grandTotal = {
    overdue: summaryRows.reduce((s, r) => s + r.overdue, 0),
    today: summaryRows.reduce((s, r) => s + r.today, 0),
    upcoming: summaryRows.reduce((s, r) => s + r.upcoming, 0),
    total: summaryRows.reduce((s, r) => s + r.total, 0),
  };

  function filterFromSummary(deptKey: BU | "Cross-functional", bucket: Bucket | "All") {
    setBucketFilter(bucket);
    setTeam(deptKey === "Cross-functional" ? "All" : deptKey);
  }

  /* ---- Filtered list rendered below (grouped by department) ---- */
  const filtered = useMemo(() => {
    return scopeBase.filter((a) => {
      if (bucketFilter !== "All" && bucketOf(a) !== bucketFilter) return false;
      if (team !== "All" && a.bu !== team) return false;
      return true;
    });
  }, [scopeBase, bucketFilter, team]);

  const showCrossFunctional = team === "All";
  const sections: { key: string; label: string; color: { bg: string; text: string }; items: ActionItem[] }[] = [
    ...BU_ORDER.map((bu) => ({
      key: bu,
      label: BU_LABEL[bu],
      color: BU_COLORS[bu],
      items: filtered.filter((a) => a.bu === bu && !a.crossFunctional),
    })),
    ...(showCrossFunctional
      ? [{
          key: "Cross-functional",
          label: "Cross-functional",
          color: { bg: "#E0E7FF", text: "#3730A3" },
          items: filtered.filter((a) => a.crossFunctional === true),
        }]
      : []),
  ].filter((s) => s.items.length > 0);

  // Expanded rows
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggleRow = (id: string) =>
    setExpanded((p) => { const n = new Set(p); n.has(id) ? n.delete(id) : n.add(id); return n; });

  // Edit modal state
  const [editingAction, setEditingAction] = useState<ActionItem | null>(null);
  const editTrackerItem = useEditTrackerItem();

  // Focused row: ensure filters clear so its section is visible, expand it, scroll, then clear focus
  const rowRefs = useRef<Record<string, HTMLDivElement | null>>({});
  useEffect(() => {
    if (!focusActionId) return;
    const item = actions.find((a) => a.id === focusActionId);
    if (item) {
      setTab(item.completed ? "completed" : "open");
      setBucketFilter("All");
      setTeam(item.crossFunctional ? "All" : item.bu);
      setExpanded((p) => new Set(p).add(focusActionId));
    }
    const t = setTimeout(() => {
      const el = rowRefs.current[focusActionId];
      if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
      setTimeout(() => setFocusActionId(null), 2400);
    }, 60);
    return () => clearTimeout(t);
  }, [focusActionId, actions, setFocusActionId]);

  function goToBU(bu: BU) {
    const v = BU_VIEW[bu];
    if (v) {
      // Same reasoning as the sidebar: this is a direct nav to the full
      // department view, so drop any meeting-scoped selection left over
      // from a previous "Open Pre Meeting Brief" click.
      clearPremeetingBrief();
      navigate(v);
    }
  }

  const openCount = actions.filter((a) => !a.completed).length;
  const completedCount = actions.filter((a) => a.completed).length;

  return (
    <div className="p-6 space-y-5">
      <div>
        <h1 className="serif text-[28px] text-[#0A2540] leading-tight">Action &amp; Accountability Ledger</h1>
        <div className="text-[12.5px] text-[#475569] mt-1">Cross-functional · single source of truth</div>
        <div className="text-[12px] text-[#94A3B8] mt-1">Single source of truth across all workspaces — all action items flow here automatically.</div>
      </div>

      {/* ── Summary by department ─────────────────────────────── */}
      <div className="bg-white border border-[#E2E8F0] rounded-[14px] p-5">
        <div
          className="inline-block text-[12px] uppercase tracking-[0.08em] font-bold pb-1 mb-1"
          style={{ color: C.navy, borderBottom: `2px solid ${C.danger}` }}
        >
          Summary by department
        </div>
        <div className="text-[11px] text-[#94A3B8] mb-3">Click any number to filter.</div>

        <div className="overflow-x-auto">
          <table className="w-full text-[12.5px]">
            <thead>
              <tr>
                <th className="text-left font-semibold text-[10.5px] uppercase tracking-[0.06em] text-[#94A3B8] pb-2"></th>
                <th className="text-right font-semibold text-[10.5px] uppercase tracking-[0.06em] text-[#94A3B8] pb-2 px-3">Overdue</th>
                <th className="text-right font-semibold text-[10.5px] uppercase tracking-[0.06em] text-[#94A3B8] pb-2 px-3">Due today</th>
                <th className="text-right font-semibold text-[10.5px] uppercase tracking-[0.06em] text-[#94A3B8] pb-2 px-3">Upcoming</th>
                <th className="text-right font-semibold text-[10.5px] uppercase tracking-[0.06em] text-[#94A3B8] pb-2 px-3">Total</th>
              </tr>
            </thead>
            <tbody>
              {summaryRows.length === 0 && (
                <tr>
                  <td colSpan={5} className="py-4 text-[12px] text-[#94A3B8]">No {tab} action items yet.</td>
                </tr>
              )}
              {summaryRows.map((r) => (
                <tr key={r.key} className="border-t border-[#EEF2F7]">
                  <td className="py-2">
                    <span className="inline-flex items-center gap-2">
                      <span className="w-[3px] h-3.5 rounded inline-block" style={{ background: r.color.text }} />
                      <span className="font-medium" style={{ color: C.navy }}>{r.label}</span>
                    </span>
                  </td>
                  <SummaryCell value={r.overdue} tone={C.danger} onClick={() => filterFromSummary(r.key, "overdue")} />
                  <SummaryCell value={r.today} tone={C.warning} onClick={() => filterFromSummary(r.key, "today")} />
                  <SummaryCell value={r.upcoming} tone={C.info} onClick={() => filterFromSummary(r.key, "upcoming")} />
                  <SummaryCell value={r.total} tone={C.navy} onClick={() => filterFromSummary(r.key, "All")} bold />
                </tr>
              ))}
              {summaryRows.length > 0 && (
                <tr className="border-t-2 border-[#E2E8F0]">
                  <td className="py-2 font-bold" style={{ color: C.navy }}>All departments</td>
                  <SummaryCell value={grandTotal.overdue} tone={C.danger} onClick={() => { setBucketFilter("overdue"); setTeam("All"); }} bold />
                  <SummaryCell value={grandTotal.today} tone={C.warning} onClick={() => { setBucketFilter("today"); setTeam("All"); }} bold />
                  <SummaryCell value={grandTotal.upcoming} tone={C.info} onClick={() => { setBucketFilter("upcoming"); setTeam("All"); }} bold />
                  <SummaryCell value={grandTotal.total} tone={C.navy} onClick={() => { setBucketFilter("All"); setTeam("All"); }} bold highlight />
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Open / Completed toggle + filters ────────────────────── */}
      <div className="flex items-center gap-2 flex-wrap">
        <button
          onClick={() => setTab("open")}
          className={`text-[12.5px] font-semibold px-4 py-1.5 rounded-[8px] transition ${tab === "open" ? "text-white" : ""}`}
          style={tab === "open" ? { background: C.navy } : { background: "#EEF2F7", color: C.navy }}
        >
          Open ({openCount})
        </button>
        <button
          onClick={() => setTab("completed")}
          className={`text-[12.5px] font-semibold px-4 py-1.5 rounded-[8px] transition ${tab === "completed" ? "text-white" : ""}`}
          style={tab === "completed" ? { background: C.success } : { background: "#DCFCE7", color: "#166534" }}
        >
          Completed ({completedCount})
        </button>
        <select
          value={bucketFilter}
          onChange={(e) => setBucketFilter(e.target.value as Bucket | "All")}
          className="text-[12px] px-2.5 py-1.5 rounded-[8px] border border-[#E2E8F0] bg-white focus:outline-none focus:border-[#0A2540]"
        >
          {BUCKET_FILTER_OPTIONS.map((b) => <option key={b} value={b}>{BUCKET_LABEL[b]}</option>)}
        </select>
        <select
          value={team}
          onChange={(e) => setTeam(e.target.value as BU | "All")}
          className="text-[12px] px-2.5 py-1.5 rounded-[8px] border border-[#E2E8F0] bg-white focus:outline-none focus:border-[#0A2540]"
        >
          <option value="All">Team: All</option>
          {DEPT_OPTIONS.map((d) => <option key={d.bu} value={d.bu}>{d.label}</option>)}
        </select>
        <div className="ml-auto text-[11.5px] text-[#94A3B8]">{filtered.length} item{filtered.length === 1 ? "" : "s"}</div>
      </div>

      {/* ── Grouped item list ─────────────────────────────────── */}
      {sections.length === 0 && (
        <div className="bg-white border border-[#E2E8F0] rounded-[14px] p-6 text-[12.5px] text-[#94A3B8] text-center">
          No {tab} action items match the current filters.
        </div>
      )}

      {sections.map((sec) => {
        const overdueN = sec.items.filter((a) => bucketOf(a) === "overdue").length;
        const todayN = sec.items.filter((a) => bucketOf(a) === "today").length;
        const upcomingN = sec.items.filter((a) => bucketOf(a) === "upcoming").length;
        return (
          <div key={sec.key} className="bg-white border border-[#E2E8F0] rounded-[14px] p-4 space-y-1">
            <div className="flex items-center gap-3 pb-2">
              <span className="w-[3px] h-5 rounded" style={{ background: sec.color.text }} />
              <span className="text-[14px] font-bold" style={{ color: C.navy }}>{sec.label}</span>
              <span className="text-[11px] font-semibold px-2 py-[2px] rounded-[6px]" style={{ background: sec.color.bg, color: sec.color.text }}>{sec.items.length} item{sec.items.length === 1 ? "" : "s"}</span>
              <div className="ml-auto flex items-center gap-3 text-[11px]">
                <span className="text-[#DC2626] font-semibold">{overdueN} Overdue</span>
                <span className="text-[#D97706] font-semibold">{todayN} Due today</span>
                <span className="text-[#1D4ED8] font-semibold">{upcomingN} Upcoming</span>
              </div>
            </div>
            <div className="divide-y divide-[#EEF2F7]">
              {sec.items.map((a) => {
                const isOpen = expanded.has(a.id);
                const isFocus = focusActionId === a.id;
                const buView = BU_VIEW[a.bu];
                const due = dueLabel(a);
                const dueTone = DUE_TONE[due.tone];
                return (
                  <div
                    key={a.id}
                    ref={(el) => { rowRefs.current[a.id] = el; }}
                    className={`py-2.5 transition rounded-[6px] ${isFocus ? "ring-2 ring-[#1D4ED8] bg-[#EAF2FB]" : ""}`}
                  >
                    <div className="flex items-start gap-3 hover:bg-[#F1F5F9] -mx-2 px-2 py-1 rounded-[6px]">
                      <input
                        type="checkbox"
                        checked={!!a.completed}
                        onChange={(e) => handleToggleComplete(a, e.target.checked)}
                        onClick={(e) => e.stopPropagation()}
                        className="mt-1 w-3.5 h-3.5 accent-[#16A34A] shrink-0 cursor-pointer"
                        title="Mark complete"
                      />
                      <span className="text-[10px] font-semibold px-2 py-[3px] rounded-[6px] shrink-0 mt-0.5" style={{ background: dueTone.bg, color: dueTone.text }}>{due.text}</span>
                      <div className="flex-1 min-w-0 cursor-pointer" onClick={() => toggleRow(a.id)}>
                        <div className="text-[12.5px] font-medium break-words" style={{ color: "#0A2540" }}>{a.title}</div>
                        {ITEM_CONTEXT[a.title] && (
                          <div className="text-[12px] font-normal text-[#94A3B8] break-words whitespace-normal mt-0.5">{ITEM_CONTEXT[a.title]}</div>
                        )}
                        <div className="text-[11px] text-[#475569] mt-0.5">
                          {a.owner}
                          {a.lastUpdated && <span className="text-[#94A3B8]"> · updated {a.lastUpdated}</span>}
                        </div>
                      </div>
                      <button
                        onClick={(e) => { e.stopPropagation(); if (buView) goToBU(a.bu); }}
                        disabled={!buView}
                        className="text-[10px] font-semibold px-2 py-[2px] rounded-[6px] shrink-0 mt-0.5 hover:opacity-80 disabled:cursor-default disabled:opacity-60"
                        style={{ background: BU_COLORS[a.bu]?.bg ?? "#EEF2F7", color: BU_COLORS[a.bu]?.text ?? "#0A2540" }}
                        title={buView ? `Open ${a.bu} workspace` : a.bu}
                      >
                        {BU_LABEL[a.bu]}
                      </button>
                      <div className="text-[11px] text-[#475569] tabular-nums shrink-0 mt-0.5">Due {a.due}</div>
                      <button
                        onClick={(e) => { e.stopPropagation(); setEditingAction(a); }}
                        className="text-[#1D4ED8] hover:bg-[#EEF2F7] shrink-0 mt-0.5 w-5 h-5 inline-flex items-center justify-center rounded-full"
                        title="Edit action item"
                      >
                        <Pencil size={11} strokeWidth={2} />
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDelete(a); }}
                        className="text-[#DC2626] hover:bg-[#FEE2E2] shrink-0 mt-0.5 w-5 h-5 inline-flex items-center justify-center rounded-full"
                        title="Delete action item"
                      >
                        <Trash2 size={11} strokeWidth={2} />
                      </button>
                      <ChevronDown
                        size={13}
                        className={`text-[#94A3B8] mt-1 transition cursor-pointer shrink-0 ${isOpen ? "rotate-180" : ""}`}
                        onClick={(e) => { e.stopPropagation(); toggleRow(a.id); }}
                      />
                    </div>
                    {isOpen && (
                      <div className="mt-2 ml-1 pl-3 border-l-2 border-[#E2E8F0] text-[11.5px] text-[#475569] space-y-1">
                        <div><span className="text-[#94A3B8]">Owner:</span> {a.owner}</div>
                        <div><span className="text-[#94A3B8]">Due:</span> {a.due || "No due date"}</div>
                        {a.lastUpdated && <div><span className="text-[#94A3B8]">Last updated:</span> {a.lastUpdated}</div>}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

      {/* Edit Action Item modal */}
      <EditActionModal
        open={editingAction !== null}
        initial={{
          text: editingAction?.title ?? "",
          owner: editingAction?.owner ?? "",
          business_unit: editingAction?.bu ?? "BBRL",
          due_date: editingAction?.dueRaw ?? "",
        }}
        onSave={(data: EditActionData) => {
          if (!editingAction) return;
          const trackerId = editingAction.id.startsWith("tracker-") ? editingAction.id.replace("tracker-", "") : null;
          if (trackerId) {
            editTrackerItem.mutate(
              {
                itemId: trackerId,
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
                  updateAction(editingAction.id, {
                    title: data.text,
                    owner: data.owner,
                    due: nd
                      ? nd.toLocaleDateString("en-IN", {
                          day: "2-digit", month: "short", year: "numeric",
                        })
                      : "No due date",
                    dueRaw: data.due_date || undefined,
                    status: newStatus,
                  });
                  showToast("Action item updated");
                  setEditingAction(null);
                },
                onError: () => showToast("Couldn't save — please try again"),
              },
            );
          } else {
            // Non-tracker (seed) items — local-only update.
            const nd = data.due_date
              ? (() => { const [y, mo, d] = data.due_date.split("-").map(Number); return new Date(y, mo - 1, d); })()
              : null;
            const days = data.due_date ? daysUntil(data.due_date) : null;
            const newStatus: Status =
              days === null || days >= 1 ? "On Track"
              : days === 0              ? "In Progress"
              :                           "Overdue";
            updateAction(editingAction.id, {
              title: data.text,
              owner: data.owner,
              due: nd
                ? nd.toLocaleDateString("en-IN", {
                    day: "2-digit", month: "short", year: "numeric",
                  })
                : "No due date",
              dueRaw: data.due_date || undefined,
              status: newStatus,
            });
            showToast("Action item updated");
            setEditingAction(null);
          }
        }}
        onClose={() => setEditingAction(null)}
        saving={editTrackerItem.isPending}
      />
    </div>
  );
}

function SummaryCell({
  value, tone, onClick, bold = false, highlight = false,
}: { value: number; tone: string; onClick: () => void; bold?: boolean; highlight?: boolean }) {
  const isZero = value === 0;
  return (
    <td className="text-right px-3 py-2">
      <button
        onClick={onClick}
        disabled={isZero}
        className={`tabular-nums px-1.5 py-[2px] rounded-[6px] transition ${isZero ? "cursor-default" : "hover:bg-[#F1F5F9] cursor-pointer"} ${bold ? "font-bold" : "font-medium"}`}
        style={{ color: isZero ? "#CBD5E1" : tone, background: highlight ? "#EAF2FB" : undefined }}
      >
        {isZero ? "–" : value}
      </button>
    </td>
  );
}
