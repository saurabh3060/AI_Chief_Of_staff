import { useMemo, useState } from "react";
import { ChevronDown, Pencil, MessageCircle, Trash2 } from "lucide-react";
import { ChatPanel } from "@/components/ChatPanel";
import { LogMeetingNotesPanel } from "@/components/LogMeetingNotesPanel";
import { ActionItemsView, type ActionItem as ActionItemType } from "@/components/ActionItemsView";
import type { ChatResponse } from "@/data/mockData";
import type { DrMetric } from "@/components/DRWorkspace";
import { useAppState } from "@/context/AppState";
import { useEmailClusters, useCompleteTrackerItem, useDeleteTrackerItem } from "@/hooks/useBackend";
import { parseClusterEmailRefs, splitBuField, type BackendEmailClusterRecord } from "@/lib/api";

/* ── Feature toggles ──────────────────────────────────────────── */
// Set to true to show source email references inside past meeting cards.
// Kept false for demo; flip to true to re-enable.
const SHOW_SOURCE_EMAILS = false;

/* ── Colour tokens (shared spec palette) ──────────────────────── */
const C = {
  navy: "#0A2540",
  danger: "#DC2626",
  success: "#16A34A",
  muted: "#475569",
  faint: "#94A3B8",
  hairline: "#E2E8F0",
  info: "#1D4ED8",
  tint: "#F4F6FA",
};

export interface PastMeetingSource { date: string; subject: string; }
export interface PastMeetingItem {
  date: string;
  type: string;
  outcomes?: string[];
  sources?: PastMeetingSource[];
}
export interface OpenActionItem { text: string; due: string; status: "In Progress" | "Overdue" | "On Track"; actionId?: string; }
export interface DecisionItem { text: string; tags: string[]; context: string; }
export interface QuestionItem { text: string; context?: string; }
export interface MeetingOption { time: string; title: string; }

export interface PreMeetingBriefConfig {
  headerText: string;
  metrics: DrMetric[];
  metricsTitle?: string;
  pastMeetings: PastMeetingItem[];
  openActionItems: OpenActionItem[];
  decisions: DecisionItem[];
  questionsArisingFrom?: string;
  questions: QuestionItem[];
  chatChips: string[];
  chatResponder: (q: string) => ChatResponse;
  availableMeetings?: MeetingOption[]; // NEW: Optional list of meetings for logging notes
  /**
   * The cluster department this tab represents (one of the fixed BU
   * departments, e.g. "BB&RL", "Wholesale Banking"). "Past meeting
   * summaries" / "Open action items" / "Top questions" are sourced from
   * that DEPARTMENT's clustered rows in the BU email CSV (via
   * `GET /agents/email/bu/records?department=...` and
   * `/bu/department-views?department=...`) - the clustering step assigns
   * each email to a department, so this shows exactly the emails
   * clustered to this tab. This is now the ONLY sourcing key (the older
   * BU-head-mailbox `buId`/`buIds` sourcing has been removed).
   */
  department?: string;
}

const STATUS_TONE: Record<OpenActionItem["status"], { bg: string; text: string }> = {
  Overdue: { bg: "#FEE2E2", text: "#991B1B" },
  "In Progress": { bg: "#FEF3C7", text: "#92400E" },
  "On Track": { bg: "#DCFCE7", text: "#166534" },
};

const TONE_COLOR: Record<string, string> = { emerald: C.success, amber: "#D97706", crimson: C.danger, info: C.info };

/* ── Universal date extraction helper ──────────────────────────── */
/**
 * Month name/abbreviation -> 0-based month index. Covers 3-letter
 * abbreviations, the "sept" variant, and full month names, in any case
 * (JAN / Jan / january all resolve).
 */
const MONTH_INDEX: Record<string, number> = {
  jan: 0, feb: 1, mar: 2, apr: 3, may: 4, jun: 5,
  jul: 6, aug: 7, sep: 8, sept: 8, oct: 9, nov: 10, dec: 11,
};

function monthIndexFromName(name: string): number | null {
  const key = name.slice(0, 4).toLowerCase();
  if (key === "sept") return 8;
  const idx = MONTH_INDEX[name.slice(0, 3).toLowerCase()];
  return idx === undefined ? null : idx;
}

/** Normalise a 2- or 4-digit year to a full year (2-digit -> 20YY). */
function normaliseYear(y: string): number {
  const n = Number(y);
  return y.length === 2 ? 2000 + n : n;
}

/**
 * Parse the FIRST recognisable date token found anywhere in a free-text
 * string into a Date at local midnight. Deliberately format-agnostic
 * rather than a growing list of position-anchored patterns - it accepts
 * (in priority order):
 *   - ISO:            YYYY-MM-DD            (2026-04-18)
 *   - alpha month:    DD-Mon-YYYY / DD Mon YYYY / DD-Mon-YY, and the
 *                     Mon-DD-YYYY variant  (18-APR-2026, 18 Apr 26, Apr 18 2026)
 *   - all-numeric:    DD-MM-YYYY / DD/MM/YYYY  (20-09-2026, 20/09/2026)
 * Month names are case-insensitive (APR == Apr == april). Returns null
 * when no date is found. This is the single source of truth for turning
 * action-item text into a real Date, so status ("Overdue"/"Due"/…) and
 * the displayed due date always agree.
 */
function parseDateFromText(text: string): Date | null {
  if (!text) return null;

  // 1) ISO YYYY-MM-DD anywhere.
  let m = text.match(/(\d{4})-(\d{1,2})-(\d{1,2})/);
  if (m) {
    const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
    if (!isNaN(d.getTime())) return d;
  }

  // 2) DD<sep>Mon<sep>YY(YY) - separators may be '-', '/' or space, month
  //    is any alpha name/abbreviation (this is what "18-APR-2026" hits).
  m = text.match(/(\d{1,2})[-/\s]([A-Za-z]{3,9})[-/\s](\d{2,4})/);
  if (m) {
    const mo = monthIndexFromName(m[2]);
    if (mo !== null) {
      const d = new Date(normaliseYear(m[3]), mo, Number(m[1]));
      if (!isNaN(d.getTime())) return d;
    }
  }

  // 3) Mon<sep>DD<sep>YY(YY) - e.g. "Apr 18 2026" / "April 18, 2026".
  m = text.match(/([A-Za-z]{3,9})[-/\s]+(\d{1,2}),?[-/\s]+(\d{2,4})/);
  if (m) {
    const mo = monthIndexFromName(m[1]);
    if (mo !== null) {
      const d = new Date(normaliseYear(m[3]), mo, Number(m[2]));
      if (!isNaN(d.getTime())) return d;
    }
  }

  // 4) All-numeric DD-MM-YYYY or DD/MM/YYYY (day-first, matching the
  //    existing app convention).
  m = text.match(/(\d{1,2})[-/](\d{1,2})[-/](\d{4})/);
  if (m) {
    const d = new Date(Number(m[3]), Number(m[2]) - 1, Number(m[1]));
    if (!isNaN(d.getTime())) return d;
  }

  return null;
}

/**
 * Extract a date from action-item text and return it as a consistent
 * display string ("18 Apr 2026"), or "TBD" when no date is present.
 * Built on `parseDateFromText` so it accepts every format that one does
 * - in particular the previously-failing "18-APR-2026".
 */
function extractDateFromText(text: string): string {
  const d = parseDateFromText(text);
  if (!d) return "TBD";
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

/**
 * Produce the DISPLAY-ONLY form of an action-item string by stripping a
 * trailing due-date tail, e.g.
 *   "Kavya Iyer: finalise ORION pilot scope … - due 30 Jun - 2026-06-30"
 *   -> "Kavya Iyer: finalise ORION pilot scope …"
 *
 * IMPORTANT: this is purely cosmetic. The FULL untrimmed string is still
 * what gets passed to `parseDateFromText`/`extractDateFromText` for the
 * due-date column and the derived status badge, so date picking is
 * unaffected - callers must derive the date from the original text
 * BEFORE trimming for display.
 *
 * Handles the shapes actually present in the data:
 *   "… - due 30 Jun - 2026-06-30"   (human due + ISO tail)
 *   "… - 2026-09-30"                 (bare ISO tail, no "due")
 *   "… - 18-APR-2026" / "… - 20-May-2026" (DD-Mon-YYYY tail)
 * It only removes trailing " - <date-ish>" segments so mid-text content
 * (like "branches and success metrics") is preserved.
 */
function stripDueSuffixForDisplay(text: string): string {
  let out = text.trim();

  // Strip trailing "(by <anything>)" patterns, e.g. "(by 2026-02-10)",
  // "(by 20-Feb-2026)", "(by 15 Aug)", "(by Aug 2026)".
  const parenByRe = /\s*\(\s*by\s+[^)]+\)\s*$/i;
  let prev: string;
  do {
    prev = out;
    out = out.replace(parenByRe, "").trim();
  } while (out !== prev && out.length > 0);

  // Repeatedly peel a trailing " - <segment>" when that segment is a
  // date token (ISO / DD-Mon-YYYY / DD-MM-YYYY) or a "due …" phrase.
  // Looping handles the "- due 30 Jun - 2026-06-30" case (two tails).
  const tailRe =
    /\s*[-–—]\s*(?:due\s+.*|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[-/\s][A-Za-z]{3,9}[-/\s]\d{2,4}|[A-Za-z]{3,9}[-/\s]+\d{1,2},?[-/\s]+\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\s*$/i;
  do {
    prev = out;
    out = out.replace(tailRe, "").trim();
  } while (out !== prev && out.length > 0);

  return out.length > 0 ? out : text.trim();
}

/* ── Due-date -> status derivation ─────────────────────────────── */
export type DerivedStatus = "Overdue" | "Due" | "In Progress";

/**
 * Derive an action item's status purely from its due date, matching the
 * "status is derived from the due date" note shown in the UI:
 *   - due date in the past           -> "Overdue"
 *   - due today                      -> "Due"
 *   - due in the future / no date    -> "In Progress" (upcoming)
 * `dueDate` may be any string `parseDateFromText` understands, or "TBD".
 */
function deriveStatusFromDue(dueDate: string): DerivedStatus {
  const d = parseDateFromText(dueDate);
  if (!d) return "In Progress";
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const diffDays = Math.round((d.getTime() - today.getTime()) / 86_400_000);
  if (diffDays < 0) return "Overdue";
  if (diffDays === 0) return "Due";
  return "In Progress";
}

/* ── Shared building blocks ────────────────────────────────────── */
function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`bg-white border border-[#E2E8F0] rounded-[14px] ${className}`}>{children}</div>;
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="inline-block text-[12px] uppercase tracking-[0.08em] font-bold pb-1 mb-3"
      style={{ color: C.navy, borderBottom: `2px solid ${C.danger}` }}
    >
      {children}
    </div>
  );
}

function CountBadge({ count, tone = "navy" }: { count: number; tone?: "navy" | "danger" }) {
  return (
    <span
      className="inline-flex items-center justify-center text-[10px] font-bold text-white rounded-full px-[6px]"
      style={{ background: tone === "danger" ? C.danger : C.navy, minWidth: 18, height: 18 }}
    >
      {count}
    </span>
  );
}

/* ── Full-width header bar ─────────────────────────────────────── */
function BriefHeader({ text }: { text: string }) {
  return (
    <div
      className="w-full"
      style={{
        background: C.navy,
        color: "#FFFFFF",
        fontSize: 13,
        fontWeight: 600,
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        padding: "12px 20px",
        borderRadius: 14,
      }}
    >
      {text}
    </div>
  );
}

/**
 * Deduplicate a list of free-text strings for display, case/whitespace
 * -insensitively, keeping the first occurrence's original casing. The
 * BU CSV can contain the same action item or question more than once
 * (the same underlying request forwarded/re-sent as separate mail, or
 * a genuinely repeated instruction) - this is a display-side safety
 * net on top of the backend's own content-based dedup (see
 * `agent_email.service._make_bu_unique_id`), so even rows the backend
 * dedup doesn't catch (e.g. same fact phrased slightly differently by
 * the LLM across two different source emails) don't show as visible
 * duplicates in the UI.
 */
/**
 * Dedup action-item strings, keeping only the latest entry per unique
 * stripped text (date suffix ignored for identity comparison).
 * "Latest" = the entry whose extracted due date is most recent.
 * If no date can be extracted, the first occurrence wins.
 */
function dedupeText(items: string[]): string[] {
  // Map: normalised-stripped-text → { raw: string, date: number }
  const best = new Map<string, { raw: string; dateMs: number }>();

  for (const item of items) {
    const raw = item.trim();
    if (!raw) continue;

    // Strip the date tail just for the identity key.
    let stripped = raw.toLowerCase().replace(/\s+/g, " ");
    // Remove "(by …)" suffix
    stripped = stripped.replace(/\s*\(\s*by\s+[^)]+\)\s*$/i, "").trim();
    // Remove "- due …" / bare ISO tail
    stripped = stripped
      .replace(/\s*[-–—]\s*(?:due\s+.*|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[-/\s][A-Za-z]{3,9}[-/\s]\d{2,4})\s*$/i, "")
      .trim();

    if (!stripped) continue;

    // Extract the due date from the raw string for "latest" comparison.
    const extracted = extractDateFromText(raw);
    let dateMs = 0;
    if (extracted && extracted !== "TBD") {
      const d = parseDateFromText(extracted);
      if (d) dateMs = d.getTime();
    }

    const existing = best.get(stripped);
    if (!existing || dateMs > existing.dateMs) {
      best.set(stripped, { raw, dateMs });
    }
  }

  return Array.from(best.values()).map((v) => v.raw);
}

function splitSummaryBullets(summary: string): string[] {
  return summary
    .split("\n")
    .map((line) => line.replace(/^[•\-\u2022]\s*/, "").trim())
    .filter(Boolean);
}

function parseBriefOutcomes(briefText: string): string[] {
  return briefText
    .split("\n")
    .map((line) => line.replace(/^[•\-\u2022]\s*/, "").trim())
    .filter(Boolean);
}

/* ── Status badge styling + human label ────────────────────────── */
const EXPANDED_STATUS_STYLES: Record<DerivedStatus, { bg: string; text: string }> = {
  "Overdue": { bg: "#FEE2E2", text: "#991B1B" },
  "Due": { bg: "#FEF3C7", text: "#92400E" },
  "In Progress": { bg: "#DCFCE7", text: "#166534" },
};

/**
 * Human-readable badge text for an action item, computed from its
 * (already date-derived) status plus its due date - e.g.
 * "Overdue by 7 days", "Due today", "Due 20 Jul". Falls back to the
 * bare status when there is no parseable due date.
 */
function statusBadgeLabel(status: DerivedStatus, dueDate: string): string {
  const d = parseDateFromText(dueDate);
  if (!d) return status === "In Progress" ? "Upcoming" : status;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const diffDays = Math.round((d.getTime() - today.getTime()) / 86_400_000);
  if (diffDays < 0) {
    const n = Math.abs(diffDays);
    return `Overdue by ${n} day${n === 1 ? "" : "s"}`;
  }
  if (diffDays === 0) return "Due today";
  const shortDate = d.toLocaleDateString("en-GB", { day: "numeric", month: "short" });
  return `Due ${shortDate}`;
}

/* ── Expanded Action Items View (inline) ───────────────────────── */
function ActionItemsExpandedView({ actionItems, onComplete, onDelete }: {
  actionItems: ActionItemType[];
  onComplete?: (actionId: string) => void;
  onDelete?: (actionId: string) => void;
}) {
  const [statusFilter, setStatusFilter] = useState<"All" | DerivedStatus>("All");
  const [teamFilter, setTeamFilter] = useState<string>("All");
  // Track locally completed ids so they disappear immediately on click.
  const [completedIds, setCompletedIds] = useState<Set<string>>(new Set());

  // Unique team values actually present in the data, for the Team filter.
  const teams = useMemo(() => {
    return Array.from(new Set(actionItems.map(item => item.team).filter(Boolean) as string[]));
  }, [actionItems]);

  // Filter items by the selected status and team. Status values come
  // straight from the item's (date-derived) status, so the dropdown
  // actually narrows the list instead of being decorative.
  const filteredItems = useMemo(() => {
    return actionItems.filter(item => {
      if (completedIds.has(item.id)) return false;
      if (statusFilter !== "All" && item.status !== statusFilter) return false;
      if (teamFilter !== "All" && item.team !== teamFilter) return false;
      return true;
    });
  }, [actionItems, statusFilter, teamFilter, completedIds]);

  const handleToggleItem = (itemId: string) => {
    if (completedIds.has(itemId)) return; // already done
    setCompletedIds(prev => new Set(prev).add(itemId));
    onComplete?.(itemId);
  };

  return (
    <div className="space-y-3">
      {/* Description */}
      <div className="text-[11.5px] text-[#94A3B8]">
        Extracted from MoM's and manually added — status is derived from the due date.
      </div>
      
      {/* Filters and item count */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {/* Status Filter */}
          <div className="relative">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as "All" | DerivedStatus)}
              className="appearance-none bg-white border border-[#E2E8F0] rounded px-3 py-1.5 pr-8 text-[11px] focus:outline-none focus:border-blue-500"
              style={{ color: C.navy }}
            >
              <option value="All">Status: All</option>
              <option value="Overdue">Status: Overdue</option>
              <option value="Due">Status: Due today</option>
              <option value="In Progress">Status: Upcoming</option>
            </select>
            <ChevronDown size={12} className="absolute right-2 top-1/2 transform -translate-y-1/2 pointer-events-none text-[#94A3B8]" />
          </div>

          {/* Team Filter */}
          <div className="relative">
            <select
              value={teamFilter}
              onChange={(e) => setTeamFilter(e.target.value)}
              className="appearance-none bg-white border border-[#E2E8F0] rounded px-3 py-1.5 pr-8 text-[11px] focus:outline-none focus:border-blue-500"
              style={{ color: C.navy }}
            >
              <option value="All">Team: All</option>
              {teams.map((t) => (
                <option key={t} value={t}>Team: {t}</option>
              ))}
            </select>
            <ChevronDown size={12} className="absolute right-2 top-1/2 transform -translate-y-1/2 pointer-events-none text-[#94A3B8]" />
          </div>
        </div>
        
        <div className="text-[11px] text-[#94A3B8]">
          {filteredItems.length} items
        </div>
      </div>

      {/* Action items list */}
      <div className="space-y-2">
        {filteredItems.length === 0 && (
          <div className="text-[11.5px] text-[#94A3B8] py-2">No action items match the current filters.</div>
        )}
        {filteredItems.map(item => {
          const derived = (item.status as DerivedStatus) in EXPANDED_STATUS_STYLES
            ? (item.status as DerivedStatus)
            : "In Progress";
          const statusStyle = EXPANDED_STATUS_STYLES[derived];
          const badgeText = statusBadgeLabel(derived, item.dueDate);
          const isSelected = completedIds.has(item.id);
          
          return (
            <div key={item.id} className="group flex items-start gap-3 py-2 border-b border-[#F1F5F9] last:border-b-0">
              {/* Checkbox */}
              <input
                type="checkbox"
                checked={isSelected}
                onChange={() => handleToggleItem(item.id)}
                className="mt-1 w-3.5 h-3.5 rounded border-[#E2E8F0] focus:ring-2 focus:ring-blue-500"
              />

              {/* Status badge (derived from the due date) */}
              <span
                className="px-2 py-1 rounded text-[10px] font-medium whitespace-nowrap mt-0.5"
                style={{ backgroundColor: statusStyle.bg, color: statusStyle.text }}
              >
                {badgeText}
              </span>
              
              {/* Team badges */}
              <div className="flex gap-1 mt-0.5">
                {item.team && (
                  <span 
                    className="px-2 py-1 rounded text-[10px] font-medium"
                    style={{ 
                      backgroundColor: C.navy, 
                      color: "white" 
                    }}
                  >
                    {item.team}
                  </span>
                )}
                {item.category && (
                  <span 
                    className="px-2 py-1 rounded text-[10px] font-medium"
                    style={{ 
                      backgroundColor: C.danger, 
                      color: "white" 
                    }}
                  >
                    {item.category}
                  </span>
                )}
              </div>
              
              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="text-[12px] font-medium leading-snug" style={{ color: C.navy }}>
                  {item.text}
                </div>
                {item.description && (
                  <div className="text-[11px] mt-1 leading-relaxed" style={{ color: C.muted }}>
                    {item.description}
                  </div>
                )}
              </div>
              
              {/* Due date */}
              <div className="text-[11px] font-medium whitespace-nowrap mt-0.5" style={{ color: C.muted }}>
                {item.dueDate}
              </div>

              {/* Delete button */}
              {onDelete && (
                <button
                  onClick={() => { if (confirm("Delete this action item permanently?")) onDelete(item.id); }}
                  className="text-[#DC2626] hover:bg-[#FEE2E2] shrink-0 mt-0.5 w-5 h-5 inline-flex items-center justify-center rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                  title="Delete action item"
                >
                  <Trash2 size={11} strokeWidth={2} />
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Past meeting summaries / Open action items tab card ───────── */
/*
 * Data source priority:
 *  1. Meeting-scoped: one card for the matched cluster (grouped mode).
 *  2. Standalone: one card per cluster in this department.
 *  3. Static `pastMeetings`/`openActionItems` defaults.
 */
function SummaryTabsCard({
  pastMeetings,
  openActionItems,
  clusterRecords,
  onEditMeeting,
  onSelectRecord,
  actionItemsForView,
  masterActionItems = [],
  pending = false,
  grouped = false,
  meetingTitle,
  groupedClusterSummary = [],
  onCompleteAction,
  onDeleteAction,
}: {
  pastMeetings: PastMeetingItem[];
  openActionItems: OpenActionItem[];
  clusterRecords: BackendEmailClusterRecord[];
  onEditMeeting: (meetingTitle: string) => void;
  onSelectRecord: (record: BackendEmailClusterRecord | null) => void;
  actionItemsForView: ActionItemType[];
  masterActionItems?: string[];
  pending?: boolean;
  grouped?: boolean;
  meetingTitle?: string;
  /** Cluster synthesis bullets for meeting-scoped combined summary (from premeeting brief). */
  groupedClusterSummary?: string[];
  onCompleteAction?: (actionId: string) => void;
  onDeleteAction?: (actionId: string) => void;
}) {
  const { momEntries } = useAppState();
  const [tab, setTab] = useState<"past" | "open">("past");
  const [openIdx, setOpenIdx] = useState<number | null>(0);
  const [showActionItemsExpanded, setShowActionItemsExpanded] = useState(false);

  // Notify parent which buRecord is selected (null = deselected / static entry)
  function handleSelectIdx(idx: number | null) {
    setOpenIdx(idx);
    if (idx === null || idx >= sortedRecords.length) {
      onSelectRecord(null);
      return;
    }
    onSelectRecord(sortedRecords[idx]);
  }

  const usingClusterRecords = clusterRecords.length > 0;

  const sortedRecords = useMemo(() => {
    const sorted = [...clusterRecords].sort(
      (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
    );
    const seen = new Set<string>();
    const unique: BackendEmailClusterRecord[] = [];
    for (const r of sorted) {
      const key = `${(r.topic || "").trim().toLowerCase()}|${(r.summary || "").trim().toLowerCase()}`;
      if (seen.has(key)) continue;
      seen.add(key);
      unique.push(r);
    }
    return unique;
  }, [clusterRecords]);

  const formatSourceDate = (raw: string) => {
    const parsed = raw ? new Date(raw) : null;
    if (parsed && !isNaN(parsed.getTime())) {
      return parsed.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
    }
    return raw || "";
  };

  // Used only to gate the "loading combined summary…" placeholder while
  // the meeting-scoped brief is still being fetched.
  const clusterSummaryBullets = dedupeText(groupedClusterSummary);

  // One card PER cluster record - both in meeting-scoped (grouped) mode
  // and standalone mode. A recurring meeting series returns several
  // clusters (one occurrence each, e.g. April..August), so each must
  // render as its own card with its own summary and its own source
  // emails, rather than being flattened into a single combined card.
  const displayPastMeetings: PastMeetingItem[] = usingClusterRecords
    ? [
        ...sortedRecords.map((r) => {
          const refs = parseClusterEmailRefs(r.email_refs);
          return {
            date: formatSourceDate(r.updated_at),
            type: r.topic || refs[0]?.subject || meetingTitle || "Email cluster",
            outcomes: splitBuField(r.summary || ""),
            sources: refs.map((ref) => ({
              date: formatSourceDate(ref.date),
              subject: ref.subject || "Email",
            })),
          };
        }),
        // In meeting-scoped mode show only the matched clusters; the
        // department's static past meetings are appended only standalone.
        ...(grouped ? [] : pastMeetings),
      ]
    : pastMeetings;

  // For CSV-derived items, derive the due date from the FULL string
  // first, then show a display-only trimmed text (no trailing
  // "- due … - <ISO>" tail). Static config items are left untouched.
  const toDisplayItem = (text: string): { text: string; due?: string; status?: OpenActionItem["status"] } => {
    const extractedDate = extractDateFromText(text);
    return {
      text: stripDueSuffixForDisplay(text),
      due: extractedDate !== "TBD" ? extractedDate : undefined,
    };
  };

  const displayOpenActionItems: { text: string; due?: string; status?: OpenActionItem["status"] }[] =
    openActionItems.length > 0
      ? openActionItems
      : masterActionItems.length > 0
        ? masterActionItems.map((text) => toDisplayItem(text))
        : usingClusterRecords
          ? dedupeText(sortedRecords.flatMap((r) => splitBuField(r.action_items))).map((text) => toDisplayItem(text))
          : openActionItems;

  return (
    <Card className="px-5 py-4">
      <div className="flex items-center gap-6 mb-3">
        <button onClick={() => setTab("past")} className="flex items-center gap-2 pb-1">
          <span className="text-[12px] uppercase tracking-[0.08em] font-bold" style={{ color: C.navy }}>Past meeting summaries</span>
          <CountBadge count={displayPastMeetings.length} tone="navy" />
        </button>
        <button 
          onClick={() => {
            setTab("open");
            setShowActionItemsExpanded(true);
          }} 
          className="flex items-center gap-2 pb-1"
        >
          <span className="text-[12px] uppercase tracking-[0.08em] font-bold" style={{ color: C.navy }}>Open action items</span>
          <CountBadge count={actionItemsForView.length} tone="danger" />
        </button>
      </div>

      {pending ? (
        <div className="py-8 text-center text-[12.5px]" style={{ color: C.faint }}>
          Loading meeting-relevant emails…
        </div>
      ) : tab === "past" ? (
        <div className="space-y-2.5">
          {displayPastMeetings.map((m, i) => {
            const open = openIdx === i;
            const meetingKey = `${m.date} · ${m.type}`;
            // Find CEO notes for this meeting
            const ceoNotes = momEntries.filter(entry => 
              entry.meeting === meetingKey || 
              entry.meeting.includes(m.type) ||
              meetingKey.includes(entry.meeting)
            );
            
            return (
              <div
                key={i}
                onClick={() => handleSelectIdx(open ? null : i)}
                className="border border-[#E2E8F0] rounded-[12px] p-3 cursor-pointer hover:border-[#94A3B8] transition"
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="text-[12.5px] font-semibold" style={{ color: C.info }}>{m.date} · {m.type}</div>
                  <div className="flex items-center gap-2 shrink-0">
                    <Pencil 
                      size={12} 
                      className="text-[#94A3B8] hover:text-[#0A2540] cursor-pointer" 
                      onClick={(e) => {
                        e.stopPropagation();
                        onEditMeeting(meetingKey);
                      }}
                    />
                    {(m.outcomes?.length || m.sources?.length || ceoNotes.length > 0) && <ChevronDown size={13} className={`text-[#94A3B8] transition ${open ? "rotate-180" : ""}`} />}
                  </div>
                </div>
                {open && (
                  <>
                    {grouped && pending && clusterSummaryBullets.length === 0 && (
                      <div className="text-[11.5px] mt-2" style={{ color: C.faint }}>
                        Loading combined summary from {m.sources?.length ?? 0} related thread(s)…
                      </div>
                    )}
                    {m.outcomes && m.outcomes.length > 0 && (
                      <>
                        <div className="text-[10px] uppercase tracking-[0.08em] font-semibold text-[#94A3B8] mt-2 mb-1">
                          {grouped ? "Summary" : "Key outcomes"}
                        </div>
                        <ul className="space-y-0.5">
                          {m.outcomes.map((o, j) => (
                            <li key={j} className="text-[11.5px] flex gap-2" style={{ color: C.navy }}>
                              <span className="text-[#94A3B8]">·</span>{o}
                            </li>
                          ))}
                        </ul>
                      </>
                    )}
                    {SHOW_SOURCE_EMAILS && m.sources && m.sources.length > 0 && (
                      <div className="mt-3">
                        <div className="text-[10px] uppercase tracking-[0.08em] font-semibold text-[#94A3B8] mb-1">
                          Source emails · {m.sources.length}
                        </div>
                        <ul className="space-y-0.5">
                          {m.sources.map((source, si) => (
                            <li key={`${source.date}-${source.subject}-${si}`} className="text-[11.5px] flex gap-2" style={{ color: C.muted }}>
                              <span className="text-[#94A3B8]">·</span>
                              {source.date} · {source.subject}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {ceoNotes.length > 0 && (
                      <div className="mt-3">
                        <div className="text-[10px] uppercase tracking-[0.08em] font-semibold text-[#94A3B8] mb-1">CEO Notes</div>
                        {ceoNotes.map((note, ni) => (
                          <div key={ni} className="mb-2 last:mb-0">
                            <div className="text-[11px] text-[#94A3B8] mb-0.5">{note.date} · {note.time}</div>
                            <div className="text-[11.5px] leading-relaxed whitespace-pre-wrap" style={{ color: C.navy }}>
                              {note.note}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            );
          })}
        </div>
      ) : showActionItemsExpanded ? (
        <ActionItemsExpandedView actionItems={actionItemsForView} onComplete={onCompleteAction} onDelete={onDeleteAction} />
      ) : (
        <div className="space-y-1.5">
          {displayOpenActionItems.length === 0 && (
            <div className="text-[11.5px] text-[#94A3B8]">No open action items.</div>
          )}
          {displayOpenActionItems.map((a, i) => {
            const tone = a.status ? STATUS_TONE[a.status] : STATUS_TONE["In Progress"];
            return (
              <div key={i} className="flex items-start gap-2 text-[12px] py-1">
                {a.status && (
                  <span className="text-[10px] font-semibold px-2 py-[2px] rounded-[6px] shrink-0 mt-[2px]" style={{ background: tone.bg, color: tone.text }}>{a.status}</span>
                )}
                <div className="flex-1" style={{ color: C.navy }}>{a.text}</div>
                {a.due && <div className="text-[10.5px] text-[#475569] tabular-nums shrink-0">{a.due}</div>}
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}

/* ── Awaiting your decision ────────────────────────────────────── */
function AwaitingDecisionCard({ decisions }: { decisions: DecisionItem[] }) {
  const [checked, setChecked] = useState<Record<number, boolean>>({});
  return (
    <Card className="px-5 py-4">
      <SectionHeading>Awaiting your decision</SectionHeading>
      {decisions.length === 0 ? (
        <div className="text-[11.5px] text-[#94A3B8]">No items awaiting your direction.</div>
      ) : (
        <div className="space-y-2.5">
          {decisions.map((d, i) => (
            <div key={i} className="flex items-start gap-2.5">
              <input
                type="checkbox"
                checked={!!checked[i]}
                onChange={(e) => setChecked((p) => ({ ...p, [i]: e.target.checked }))}
                className="mt-1 w-3.5 h-3.5 accent-[#0A2540] shrink-0"
              />
              <div className="flex-1 min-w-0">
                <div className="text-[12.5px]" style={{ color: C.navy }}>
                  <span className="font-semibold">{d.text}</span>
                  <span className="ml-2 text-[10px] uppercase tracking-[0.04em] text-[#94A3B8]">{d.tags.join(" · ")}</span>
                </div>
                <div className="text-[12px] text-[#475569] mt-0.5 leading-snug">{d.context}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

/* ── Questions to ask ───────────────────────────────────────────── */
/*
 * Question source priority:
 *  1. This BU's own rows in the BU email CSV (`buRecords`, deduplicated
 *     across rows via `dedupeText`) - already-processed, persisted
 *     results.
 *  2. `selectedMeetingBrief.questions` - meeting-specific questions
 *     fetched from GET /agents/calendar/brief/{meeting_id}, populated by
 *     clicking a row in "Today's schedule · Outlook" on the Launchpad.
 *  3. The static default `questions` passed in via cfg.
 */
function QuestionsCard({ arisingFrom, questions, masterQuestions = [], masterLabel = "department master brief", pending = false }: { arisingFrom?: string; questions: QuestionItem[]; masterQuestions?: string[]; masterLabel?: string; pending?: boolean }) {
  const { selectedMeetingBrief, selectedMeeting, loadingMeetingBrief, meetingBriefError } = useAppState();
  const [isAddingQuestion, setIsAddingQuestion] = useState(false);
  const [newQuestionText, setNewQuestionText] = useState("");
  const [customQuestions, setCustomQuestions] = useState<QuestionItem[]>([]);

  // Only use meeting-brief questions when a meeting is actively selected,
  // preventing stale questions from a previously viewed BU leaking across tabs.
  const meetingQuestions = selectedMeeting ? (selectedMeetingBrief?.questions ?? []) : [];

  const usingMaster = masterQuestions.length > 0;
  const usingMeeting = !usingMaster && meetingQuestions.length > 0;

  const displayItems: { text: string; context?: string; custom?: boolean }[] = pending
    ? []
    : [
        ...(usingMaster
          ? masterQuestions.map((text) => ({ text }))
          : usingMeeting
            ? meetingQuestions.map((text) => ({ text }))
            : questions),
        ...customQuestions.map(q => ({ ...q, custom: true })),
      ];

  const arisingLabel = usingMaster
    ? masterLabel
    : usingMeeting
      ? selectedMeetingBrief!.meeting_subject
      : arisingFrom;

  function handleAddQuestion() {
    if (!newQuestionText.trim()) return;
    const newQuestion: QuestionItem = {
      text: newQuestionText.trim(),
    };
    setCustomQuestions((prev) => [...prev, newQuestion]);
    setNewQuestionText("");
    setIsAddingQuestion(false);
  }

  function handleDeleteQuestion(index: number) {
    setCustomQuestions((prev) => prev.filter((_, i) => i !== index - (displayItems.length - customQuestions.length)));
  }

  return (
    <Card className="px-5 py-4">
      <SectionHeading>Questions to ask</SectionHeading>
      {!usingMaster && loadingMeetingBrief && (
        <div className="text-[12px] text-[#94A3B8] mb-2">Generating questions…</div>
      )}
      {!usingMaster && meetingBriefError && !usingMeeting && (
        <div className="text-[12px] text-[#94A3B8] mb-2">Couldn't load live questions.</div>
      )}
      {displayItems.length === 0 && !loadingMeetingBrief && (
        <div className="text-[12px] text-[#94A3B8] mb-2">No questions sourced from the emails yet.</div>
      )}
      <ol className="space-y-2.5">
        {displayItems.map((q, i) => (
          <li key={i} className="flex items-start gap-2.5">
            <span className="text-[12px] font-semibold tabular-nums w-4 mt-0.5" style={{ color: C.info }}>{i + 1}</span>
            <div className="flex-1 min-w-0">
              <div className="text-[12.5px] leading-snug" style={{ color: C.navy }}>{q.text}</div>
              {q.context && <div className="text-[12px] text-[#475569] mt-0.5 leading-snug">{q.context}</div>}
            </div>
            {q.custom && (
              <button
                onClick={() => handleDeleteQuestion(i)}
                className="text-[#94A3B8] hover:text-[#DC2626] transition"
                title="Delete custom question"
              >
                <span className="text-[14px]">×</span>
              </button>
            )}
          </li>
        ))}
      </ol>
      
      {/* Add question form */}
      {isAddingQuestion ? (
        <div className="mt-3 space-y-2 pt-3 border-t border-[#E2E8F0]">
          <input
            autoFocus
            type="text"
            value={newQuestionText}
            onChange={(e) => setNewQuestionText(e.target.value)}
            placeholder="Enter your question"
            className="w-full text-[12px] px-3 py-2 border border-[#E2E8F0] rounded-[8px] focus:outline-none focus:border-[#0A2540]"
            onKeyDown={(e) => {
              if (e.key === "Enter") handleAddQuestion();
              if (e.key === "Escape") {
                setIsAddingQuestion(false);
                setNewQuestionText("");
              }
            }}
          />
          <div className="flex gap-2">
            <button
              onClick={handleAddQuestion}
              className="text-[12px] font-semibold px-3 py-1.5 rounded-[6px] text-white"
              style={{ background: C.navy }}
            >
              Save
            </button>
            <button
              onClick={() => {
                setIsAddingQuestion(false);
                setNewQuestionText("");
              }}
              className="text-[12px] px-3 py-1.5 rounded-[6px] border border-[#E2E8F0] text-[#475569] hover:bg-[#F1F5F9]"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button
          onClick={() => setIsAddingQuestion(true)}
          className="mt-3 pt-3 border-t border-[#E2E8F0] w-full text-left text-[12px] font-semibold flex items-center gap-1.5 hover:text-[#0A2540] transition"
          style={{ color: C.info }}
        >
          <span className="text-[16px]">+</span> Add question
        </button>
      )}
    </Card>
  );
}

/* ── Monthly metrics ────────────────────────────────────────────── */
function MonthlyMetricsCard({ title = "Monthly metrics", metrics }: { title?: string; metrics: DrMetric[] }) {
  return (
    <Card className="px-5 py-4">
      <SectionHeading>{title}</SectionHeading>
      <div className="grid grid-cols-2 gap-3">
        {metrics.map((m, i) => (
          <div
            key={m.label}
            className="border border-[#E2E8F0] rounded-[10px] px-3 py-3 overflow-hidden"
            style={i === metrics.length - 1 && metrics.length % 2 === 1 ? { gridColumn: "1 / -1" } : undefined}
          >
            <div className="text-[10.5px] uppercase tracking-[0.05em] font-semibold text-[#0A2540] line-clamp-2 break-words">{m.label}</div>
            <div className="text-[20px] font-semibold leading-tight mt-1 tabular-nums" style={{ color: C.navy }}>{m.value}</div>
            <div className="text-[10px] font-semibold mt-1" style={{ color: TONE_COLOR[m.tone] }}>{m.delta}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ── Log meeting notes CTA ──────────────────────────────────────── */
function LogMeetingNotesButton({ onOpen }: { onOpen: () => void }) {
  return (
    <div>
      <button
        onClick={onOpen}
        className="w-full flex items-center justify-center gap-2 text-[13px] font-semibold px-4 py-3 rounded-[10px] text-white"
        style={{ background: C.navy }}
      >
        <Pencil size={14} /> Log meeting notes
      </button>
      <div className="text-[11px] text-[#94A3B8] mt-1.5 text-center">Capture MoM notes and tag action items to a team.</div>
    </div>
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
        style={{ background: C.navy }}
        title="Ask your Chief of Staff"
      >
        <MessageCircle size={20} className="text-white" />
      </button>
    </div>
  );
}

/* ── Screen ──────────────────────────────────────────────────────── */
export function PreMeetingBrief({ cfg }: { cfg: PreMeetingBriefConfig }) {
  const [showNotesPanel, setShowNotesPanel] = useState(false);
  const [preSelectedMeeting, setPreSelectedMeeting] = useState<string | undefined>(undefined);
  const [, setSelectedCluster] = useState<BackendEmailClusterRecord | null>(null);

  const { data: departmentClustersData } = useEmailClusters(cfg.department);
  const allDepartmentClusters = departmentClustersData?.records ?? [];

  const { premeetingBrief, selectedMeeting, loadingPremeetingBrief, updateAction, removeAction, showToast, refreshOpenTracker, actions: allActions } = useAppState();
  const completeTrackerItem = useCompleteTrackerItem();
  const deleteTrackerItem = useDeleteTrackerItem();

  // Called when the user checks an item in the BU action items expanded view.
  function handleCompleteAction(actionId: string) {
    // actionId is "tracker-{backendId}" for backend items.
    updateAction(actionId, { completed: true, status: "Completed" });
    const backendId = actionId.startsWith("tracker-") ? actionId.replace("tracker-", "") : null;
    if (!backendId) return;
    completeTrackerItem.mutate(backendId, {
      onSuccess: () => refreshOpenTracker(),
      onError: () => {
        updateAction(actionId, { completed: false });
        showToast("Couldn't save — please try again");
      },
    });
  }

  // Called when the user clicks the trash icon in the BU action items expanded view.
  function handleDeleteAction(actionId: string) {
    removeAction(actionId);
    const backendId = actionId.startsWith("tracker-") ? actionId.replace("tracker-", "") : null;
    if (!backendId) return;
    deleteTrackerItem.mutate(backendId, {
      onSuccess: () => refreshOpenTracker(),
      onError: () => {
        showToast("Couldn't delete — please try again");
      },
    });
  }

  const clustersPending = !!selectedMeeting && loadingPremeetingBrief;

  const { records: clusterRecords, filteredToMeeting } = useMemo(() => {
    if (clustersPending) {
      return { records: [] as BackendEmailClusterRecord[], filteredToMeeting: false };
    }

    // Only use the meeting-scoped brief if it actually belongs to THIS
    // workspace's department. Without this check, a brief fetched for a
    // Wholesale meeting leaks into Credit Cards / BBRL / Retail Assets
    // workspaces because `premeetingBrief` is a single global slot in
    // AppState shared by all workspaces.
    const briefMatchesThisDept =
      !cfg.department ||
      !premeetingBrief?.department ||
      premeetingBrief.department.trim().toLowerCase() === cfg.department.trim().toLowerCase();

    if (selectedMeeting && premeetingBrief?.matched && briefMatchesThisDept) {
      // The backend returns one card per matched cluster in `clusters`
      // (a recurring series surfaces every occurrence; a one-off meeting
      // a single card). Render one row per card. Fall back to the flat
      // single-cluster fields for older backends without `clusters`.
      const cards = premeetingBrief.clusters?.length
        ? premeetingBrief.clusters
        : [{
            cluster_id: "matched",
            department: premeetingBrief.department,
            topic: premeetingBrief.topic,
            meeting_start: "",
            summary: premeetingBrief.summary,
            questions: premeetingBrief.questions,
            action_items: premeetingBrief.action_items,
            email_refs: premeetingBrief.email_refs ?? [],
          }];

      const matchedClusters: BackendEmailClusterRecord[] = cards.map((card, i) => ({
        cluster_id: card.cluster_id || `matched-${i}`,
        department: card.department,
        topic: card.topic,
        summary: card.summary.join("|"),
        top_questions: card.questions.join("|"),
        action_items: card.action_items.join("|"),
        email_refs: JSON.stringify(card.email_refs ?? []),
        email_count: card.email_refs?.length ?? 0,
        // Prefer the occurrence's own meeting_start so cards sort
        // chronologically; fall back to now when it's absent.
        updated_at: card.meeting_start || new Date().toISOString(),
      }));
      console.info(
        "[PMB] meeting=%s (%s): showing %d matched cluster(s): %o",
        selectedMeeting.id, selectedMeeting.title, matchedClusters.length,
        matchedClusters.map((c) => c.topic),
      );
      return { records: matchedClusters, filteredToMeeting: true };
    }

    return { records: allDepartmentClusters, filteredToMeeting: false };
  }, [allDepartmentClusters, premeetingBrief, selectedMeeting, clustersPending]);

  const meetingScoped =
    !!selectedMeeting &&
    !clustersPending &&
    filteredToMeeting &&
    clusterRecords.length > 0;


    // Aggregate questions/action items across every matched cluster card
  // (a recurring series has one card per month), not just the first.
  const masterQuestions = clustersPending
    ? []
    : meetingScoped
      ? dedupeText(clusterRecords.flatMap((r) => splitBuField(r.top_questions))).slice(0, 4)
      : !selectedMeeting
        ? dedupeText(allDepartmentClusters.flatMap((r) => splitBuField(r.top_questions))).slice(0, 4)
        : [];

  // Scope action items strictly to the clusters shown on this card
  // (the matched clusters). When a selected meeting matches no clusters,
  // this is empty — we intentionally do NOT fall back to the whole
  // department, which previously produced counts like 122 against 2
  // matched clusters.
  const masterActionItems = clustersPending
    ? []
    : dedupeText(
        clusterRecords.flatMap((r) => splitBuField(r.action_items)),
      );

  const sortedClusterRecords = useMemo(() => {
    const sorted = [...clusterRecords].sort(
      (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
    );
    const seen = new Set<string>();
    const unique: BackendEmailClusterRecord[] = [];
    for (const r of sorted) {
      const key = `${(r.topic || "").trim().toLowerCase()}|${(r.summary || "").trim().toLowerCase()}`;
      if (seen.has(key)) continue;
      seen.add(key);
      unique.push(r);
    }
    return unique;
  }, [clusterRecords]);

  const actionItemsForView: ActionItemType[] = useMemo(() => {
    // While meeting-specific data is loading, show nothing to avoid
    // flashing department-wide items that get replaced moments later.
    if (clustersPending) return [];

    // When a meeting is selected, action items must be scoped strictly to
    // the matched clusters shown on this card (never the department-wide
    // openActionItems). If no clusters matched, the list is empty (0).
    if (meetingScoped) {
      // Build a set of completed tracker IDs so cluster-derived items that
      // have been checked off disappear from this view immediately (same
      // behaviour as the sidebar path, where cfg.openActionItems already
      // filters out completed items).
      const completedIds = new Set(
        allActions.filter((a) => a.completed).map((a) => a.id),
      );

      // Build a lookup map from ALL actions in AppState (not just the
      // open ones in cfg.openActionItems) so that a just-completed item
      // still resolves to its real tracker ID.  If we only used
      // cfg.openActionItems, a completed item would vanish from the map
      // (because that list filters !completed), and the cluster text
      // would fall back to a synthetic "action-N" ID — which the
      // completedIds filter above wouldn't recognise, causing the item
      // to reappear.
      //
      // mapTrackerItemToAction prefixes titles with "Owner: " when the
      // item has an assigned owner, so we index by BOTH the full title
      // AND the text after the first ": " so cluster-derived text (which
      // never carries the owner prefix) can still match.
      const stripOwnerPrefix = (t: string) => {
        const idx = t.indexOf(": ");
        return idx > 0 ? t.slice(idx + 2) : t;
      };
      const norm = (t: string) => stripDueSuffixForDisplay(t).trim().toLowerCase().replace(/\s+/g, " ");

      // Index ALL actions for this department (both open and completed)
      // so IDs are stable across completion state changes.
      const realItemsByText = new Map<string, { actionId: string; due?: string }>();
      for (const a of allActions) {
        if (!a.id.startsWith("tracker-")) continue;
        const full = norm(a.title);
        const entry = { actionId: a.id, due: a.due };
        if (full) realItemsByText.set(full, entry);
        const stripped = norm(stripOwnerPrefix(a.title));
        if (stripped && stripped !== full) realItemsByText.set(stripped, entry);
      }

      return dedupeText(sortedClusterRecords.flatMap((r) => splitBuField(r.action_items)))
        .map((text, index) => {
          const strippedText = stripDueSuffixForDisplay(text);
          const normalised = strippedText.trim().toLowerCase().replace(/\s+/g, " ");
          // Try exact match first, then without owner prefix
          const realMatch = realItemsByText.get(normalised)
            ?? realItemsByText.get(norm(stripOwnerPrefix(strippedText)));

          const dueDate = realMatch?.due && realMatch.due !== "No due date"
            ? realMatch.due
            : (() => { const e = extractDateFromText(text); return e !== "TBD" ? e : "TBD"; })();

          return {
            id: realMatch?.actionId ?? `action-${index}`,
            text: strippedText,
            dueDate,
            status: deriveStatusFromDue(dueDate),
          };
        })
        // Drop items whose matched tracker ID is already completed — keeps
        // the list and count in sync with the Ledger's completed tab.
        .filter((item) => !completedIds.has(item.id));
    }

    // No meeting selected: use real structured openActionItems (from
    // AppState) directly — don't mix with cluster text which causes index
    // mismatches and wrong dates.
    if (cfg.openActionItems.length > 0) {
      // Dedup by stripped title, keeping the entry with the latest due date.
      const bestByTitle = new Map<string, typeof cfg.openActionItems[number]>();
      for (const item of cfg.openActionItems) {
        const key = stripDueSuffixForDisplay(item.text).trim().toLowerCase().replace(/\s+/g, " ");
        if (!key) continue;
        const existing = bestByTitle.get(key);
        if (!existing) { bestByTitle.set(key, item); continue; }
        // Compare due dates — keep whichever is later.
        const existingD = parseDateFromText(existing.due && existing.due !== "No due date" ? existing.due : "");
        const newD = parseDateFromText(item.due && item.due !== "No due date" ? item.due : "");
        const existingMs = existingD ? existingD.getTime() : 0;
        const newMs = newD ? newD.getTime() : 0;
        if (newMs > existingMs) bestByTitle.set(key, item);
      }
      return Array.from(bestByTitle.values()).map((item, index) => ({
        id: item.actionId ?? `action-${index}`,
        text: stripDueSuffixForDisplay(item.text),
        dueDate: item.due && item.due !== "No due date" ? item.due : "TBD",
        status: deriveStatusFromDue(item.due && item.due !== "No due date" ? item.due : "TBD"),
      }));
    }

    const clusterItems = clusterRecords.length > 0
      ? dedupeText(sortedClusterRecords.flatMap((r) => splitBuField(r.action_items)))
      : [];
    const items = clusterItems.length > 0 ? clusterItems : cfg.openActionItems.map((item) => item.text);

    return items.map((text, index) => {
      // Prefer the explicit due date from openActionItems (structured data).
      // Only fall back to text extraction when no structured due date exists —
      // prevents ticket IDs like REQ-2627-08-14 being misread as year-2627 dates.
      const structuredDue = cfg.openActionItems[index]?.due;
      const dueDate = structuredDue && structuredDue !== "No due date"
        ? structuredDue
        : (() => { const e = extractDateFromText(text); return e !== "TBD" ? e : "TBD"; })();
      const derivedStatus = deriveStatusFromDue(dueDate);
      const configStatus = cfg.openActionItems[index]?.status as ActionItemType["status"] | undefined;
      const status: ActionItemType["status"] =
        dueDate === "TBD" && configStatus ? configStatus : derivedStatus;

      return {
        id: `action-${index}`,
        // Display-only: drop the trailing "- due … - <ISO>" tail. The
        // date above was already derived from the FULL `text`, so the
        // due-date column and status badge are unaffected.
        text: stripDueSuffixForDisplay(text),
        dueDate,
        status,
      };
    });
  }, [meetingScoped, clustersPending, clusterRecords, sortedClusterRecords, cfg.openActionItems, masterActionItems, allActions]);

  const pastMeetingsList: MeetingOption[] = [];

  sortedClusterRecords.forEach((r) => {
    const date = new Date(r.updated_at).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
    pastMeetingsList.push({ time: date, title: r.topic || "Email cluster" });
  });

  // Add static past meetings
  cfg.pastMeetings.forEach(pm => {
    pastMeetingsList.push({ time: pm.date, title: pm.type });
  });
  
  // Use past meetings as available meetings, or fall back to config meetings
  const meetings = pastMeetingsList.length > 0 ? pastMeetingsList : (cfg.availableMeetings || []);
  
  // Only show decisions section if there are decisions to display
  const showDecisions = cfg.decisions && cfg.decisions.length > 0;

  const openNotesPanel = (meetingTitle?: string) => {
    setPreSelectedMeeting(meetingTitle);
    setShowNotesPanel(true);
  };

  const closeNotesPanel = () => {
    setShowNotesPanel(false);
    setPreSelectedMeeting(undefined);
  };

  return (
    <div className="flex flex-col gap-5 p-5 bg-[#F4F6FA] min-h-[calc(100vh-64px)]">
      <BriefHeader text={meetingScoped && selectedMeeting?.title ? selectedMeeting.title : cfg.headerText} />
      <div className="grid gap-5" style={{ gridTemplateColumns: "60% 40%" }}>
        <div className="flex flex-col gap-5 min-w-0">
          <SummaryTabsCard 
            pastMeetings={cfg.pastMeetings} 
            openActionItems={cfg.openActionItems}
            clusterRecords={clusterRecords}
            onEditMeeting={openNotesPanel}
            onSelectRecord={setSelectedCluster}
            actionItemsForView={actionItemsForView}
            masterActionItems={masterActionItems}
            pending={clustersPending}
            grouped={meetingScoped}
            meetingTitle={selectedMeeting?.title}
            groupedClusterSummary={premeetingBrief?.summary ?? []}
            onCompleteAction={handleCompleteAction}
            onDeleteAction={handleDeleteAction}
          />
          {showDecisions && <AwaitingDecisionCard decisions={cfg.decisions} />}
          <QuestionsCard arisingFrom={cfg.questionsArisingFrom} questions={cfg.questions} masterQuestions={masterQuestions} masterLabel={meetingScoped ? (selectedMeeting?.title ?? "this meeting") : "department master brief"} pending={clustersPending} />
        </div>
        <div className="flex flex-col gap-5 min-w-0">
          <MonthlyMetricsCard title={cfg.metricsTitle} metrics={cfg.metrics} />
          <LogMeetingNotesButton onOpen={() => openNotesPanel(undefined)} />
        </div>
      </div>
      <ChatBubble chips={cfg.chatChips} responder={cfg.chatResponder} />
      
      <LogMeetingNotesPanel
        isOpen={showNotesPanel}
        onClose={closeNotesPanel}
        meetings={meetings}
        preSelectedMeeting={preSelectedMeeting}
        meetingId={selectedMeeting?.id}
      />
    </div>
  );
}
