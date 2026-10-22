import { useMemo, useRef, useState } from "react";
import { ChatPanel } from "@/components/ChatPanel";
import { Sparkline } from "@/components/Sparkline";
import { ActionGantt } from "@/components/ActionGantt";
import { RightRail } from "@/components/RightRail";
import { useAppState, type DrKey, type Status, type ActionItem } from "@/context/AppState";
import { useAppNavigate } from "@/hooks/useAppNavigate";
import type { ChatMessage } from "@/components/ChatPanel";
import type { ChatResponse } from "@/data/mockData";
import { ChevronDown, ChevronRight, FileText, Pencil, Plus, Tag } from "lucide-react";

/* ---- Status colour helpers (spec palette) ---- */
const C = {
  danger: "#DC2626",
  warning: "#D97706",
  success: "#16A34A",
  info: "#1D4ED8",
  navy: "#0A2540",
  body: "#0A2540",
  muted: "#475569",
  hairline: "#E2E8F0",
  bgTint: "#F4F6FA",
};

const STATUS_TONE: Record<string, { bg: string; text: string }> = {
  Reviewed:      { bg: "#DCFCE7", text: "#166534" },
  "In Progress": { bg: "#FEF3C7", text: "#92400E" },
  "On Track":    { bg: "#DCFCE7", text: "#166534" },
  Overdue:       { bg: "#FEE2E2", text: "#991B1B" },
  Ahead:         { bg: "#DCFCE7", text: "#166534" },
  New:           { bg: "#DBEAFE", text: "#1E40AF" },
};

export const ITEM_CONTEXT: Record<string, string> = {};

export const QUESTION_CONTEXT: Record<string, string> = {};

export const DECISION_CONTEXT: Record<string, string> = {};

export const MOTION_CONTEXT: Record<string, string> = {};

const TEAM_COLOR: Record<DrKey, string> = {
  Risk:            C.danger,
  "BB&RL":         C.info,
  Wholesale:       C.success,
  "Retail Assets": C.warning,
};

// Map DrKey to full business unit name
const DR_TO_BU: Record<DrKey, string> = {
  Risk: "Credit Cards",
  "BB&RL": "BBRL",
  Wholesale: "Wholesale Banking",
  "Retail Assets": "Retail Advances - Excluding Credit Cards",
};

export interface DrMetric {
  label: string;
  value: string;
  delta: string;
  tone: "emerald" | "amber" | "crimson" | "info";
}

export interface DrPastMeeting {
  date: string;
  type: string;
  status: "Reviewed" | "In Progress" | "On Track" | "Overdue" | "New";
  outcomes: string[];
  openItems: string[];
  actionItems?: string[];  // Email action items
  topQuestions?: string[];  // Email top questions
}

export interface DrConfig {
  drKey: DrKey;
  firstName: string;
  fullName: string;
  title: string;
  meetingName: string;
  metrics: DrMetric[];
  past: DrPastMeeting[];
  cosPreloaded: ChatMessage[];
  cosChips: string[];
  cosResponder: (q: string) => ChatResponse;
}

/* ---- Date helpers ---- */
function parseDue(s: string): Date | null {
  const m = s.match(/(\d{1,2})\s+([A-Za-z]{3,9})\s*(\d{4})?/);
  if (!m) return null;
  const months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"];
  const mo = months.indexOf(m[2].slice(0,3).toLowerCase());
  if (mo < 0) return null;
  const year = m[3] ? parseInt(m[3], 10) : new Date().getFullYear();
  return new Date(year, mo, parseInt(m[1], 10));
}
function isOverdue(s: string) {
  const d = parseDue(s);
  if (!d) return false;
  const today = new Date("2026-06-25");
  return d < today;
}

export function DRWorkspace({ cfg }: { cfg: DrConfig }) {
  return (
    <div className="grid gap-5 min-h-[calc(100vh-64px)] bg-[#F4F6FA] p-5" style={{ gridTemplateColumns: "60% 40%" }}>
      <div className="overflow-auto min-w-0">
        <LeftPanel cfg={cfg} />
      </div>
      <div className="overflow-auto min-w-0">
        <RightRail
          metrics={cfg.metrics}
          metricsTitle={`${cfg.drKey} metrics · Latest MIS`}
          cosPreloaded={cfg.cosPreloaded}
          cosChips={cfg.cosChips}
          cosResponder={cfg.cosResponder}
          sourcePage={`${cfg.drKey} workspace`}
          sourceDR={cfg.drKey}
        />
      </div>
    </div>
  );
}

/* Card wrapper - every section sits in its own white rounded card */
function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-white border border-[#E2E8F0] rounded-[14px] ${className}`}>
      {children}
    </div>
  );
}

/* =========================================================
   LEFT PANEL - CEO workspace (Risk / BB&RL)
   ========================================================= */
function LeftPanel({ cfg }: { cfg: DrConfig }) {
  return (
    <div className="flex flex-col gap-5">
      <DRHeader cfg={cfg} />
      <Card><CollapsiblePastMeetings cfg={cfg} /></Card>
      <Card><QuestionsSection cfg={cfg} /></Card>
      <Card className="lov-urgent"><CollapsibleOpenActions cfg={cfg} /></Card>
      <Card><NotesSection cfg={cfg} /></Card>
      <Card className="lov-urgent"><CollapsibleCrossFunctional cfg={cfg} /></Card>
    </div>
  );
}


function DRHeader({ cfg }: { cfg: DrConfig }) {
  const text =
    cfg.drKey === "Wholesale" || cfg.drKey === "Retail Assets"
      ? `${cfg.fullName} · ${cfg.title}`
      : `${cfg.title} · Pre-meeting briefing & actions`;
  return (
    <div
      className="w-full"
      style={{
        background: "#0A2540",
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


/* ---- Questions ---- */
function QuestionsSection({ cfg }: { cfg: DrConfig }) {
  const { questions, setQuestions } = useAppState();
  const list = questions[cfg.drKey] ?? [];
  const [editing, setEditing] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [adding, setAdding] = useState(false);

  function save(i: number, val: string) {
    const next = [...list];
    if (val.trim()) next[i] = val.trim();
    setQuestions(cfg.drKey, next);
    setEditing(null);
  }
  function add(val: string) {
    if (!val.trim()) { setAdding(false); return; }
    setQuestions(cfg.drKey, [...list, val.trim()]);
    setAdding(false);
  }

  return (
    <div className="px-5 py-4">
      <div className="text-[12px] uppercase tracking-[0.08em] font-bold text-[#0A2540] mb-2">
        Top 3 questions to ask {cfg.firstName}
      </div>
      
      <ol className="space-y-2">
        {list.map((q, i) => (
          <li key={i} className="flex items-start gap-2.5 group">
            <span className="text-[12px] text-[#94A3B8] tabular-nums w-4 mt-0.5">{i + 1}</span>
            {editing === i ? (
              <input
                autoFocus
                defaultValue={q}
                onBlur={(e) => save(i, e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); if (e.key === "Escape") setEditing(null); }}
                className="flex-1 text-[12.5px] text-[color:var(--ink)] bg-[#F1F5F9] border border-[#E2E8F0] rounded-[6px] px-2 py-1 focus:outline-none focus:border-[#0A2540]"
                style={{ color: C.body }}
              />
            ) : (
              <div className="flex-1 min-w-0">
                <div className="flex items-start gap-2">
                  <div className="flex-1 text-[12.5px] leading-snug" style={{ color: C.body }}>{q}</div>
                  <button onClick={() => setEditing(i)} className="text-[#94A3B8] hover:text-[#0A2540] opacity-0 group-hover:opacity-100 transition shrink-0">
                    <Pencil size={12} />
                  </button>
                </div>
              </div>
            )}
          </li>
        ))}
      </ol>
      {adding ? (
        <input
          autoFocus
          placeholder="Type the question, Enter to save"
          onBlur={(e) => add(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); if (e.key === "Escape") setAdding(false); }}
          className="mt-2 w-full text-[12.5px] bg-[#F1F5F9] border border-[#E2E8F0] rounded-[6px] px-2 py-1.5 focus:outline-none focus:border-[#0A2540]"
        />
      ) : (
        <button onClick={() => setAdding(true)} className="mt-2 inline-flex items-center gap-1 text-[11.5px] text-[#1D4ED8] hover:underline">
          <Plus size={12} /> Add question
        </button>
      )}
    </div>
  );
}

/* ---- Ask your CoS ---- */
function CosSection({ cfg }: { cfg: DrConfig }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="px-5 py-3">
      <button onClick={() => setOpen((o) => !o)} className="w-full flex items-center justify-between text-[12.5px] font-semibold" style={{ color: C.navy }}>
        <span>Ask your Chief of Staff</span>
        <ChevronDown size={14} className={`transition ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="mt-3">
          <ChatPanel
            preloaded={cfg.cosPreloaded}
            chips={cfg.cosChips}
            getResponse={cfg.cosResponder}
            placeholder="Ask anything across all data sources…"
            compact
            loadHistory
          />
        </div>
      )}
    </div>
  );
}

/* ---- Notes with text-tagging ---- */
const TEAMS: DrKey[] = ["Risk", "BB&RL", "Wholesale", "Retail Assets"];

/**
 * Escape EVERY regular-expression metacharacter in a string so it can be
 * safely interpolated into a `new RegExp(...)` as a literal. The previous
 * inline `.replace(/[&]/g, "\\&")` only handled `&` and, critically, did
 * not escape the backslash itself first, so any metacharacter (or a `\`)
 * in a team name would have altered the pattern's meaning (CodeQL
 * js/incomplete-sanitization). The backslash MUST be escaped first,
 * otherwise it would double-escape the replacements added afterwards.
 */
function escapeRegExp(input: string): string {
  return input.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&");
}




function NotesSection({ cfg }: { cfg: DrConfig }) {
  const { addCrossTag, addActions, addPending, showToast, addMeetingNote, addMomEntry } = useAppState();
  const taRef = useRef<HTMLTextAreaElement>(null);
  const [notes, setNotes] = useState("");
  const [mention, setMention] = useState<{ pos: number; query: string } | null>(null);
  const [saved, setSaved] = useState(false);
  const [pendingItems, setPendingItems] = useState<{ team: DrKey; text: string; due: string; err: boolean }[] | null>(null);

  function onChange(val: string) {
    setNotes(val);
    const ta = taRef.current;
    if (!ta) return;
    const caret = ta.selectionStart;
    const upto = val.slice(0, caret);
    const at = upto.lastIndexOf("@");
    if (at >= 0) {
      const after = upto.slice(at + 1);
      if (!/\s/.test(after)) { setMention({ pos: at, query: after.toLowerCase() }); return; }
    }
    setMention(null);
  }

  function pickTeam(team: DrKey) {
    if (!mention) return;
    const before = notes.slice(0, mention.pos);
    const afterStart = mention.pos + 1 + mention.query.length;
    const after = notes.slice(afterStart);
    const inserted = `@${team} `;
    const next = before + inserted + after;
    setNotes(next);
    setMention(null);
    requestAnimationFrame(() => {
      const ta = taRef.current;
      if (ta) {
        const newPos = before.length + inserted.length;
        ta.focus();
        ta.setSelectionRange(newPos, newPos);
      }
    });
  }

  const mentionMatches = mention ? TEAMS : [];

  function extractTagged(): { team: DrKey; text: string }[] {
    const sentences = notes.split(/(?<=[.!?\n])\s+/);
    const out: { team: DrKey; text: string }[] = [];
    for (const s of sentences) {
      const matched = TEAMS.filter((t) => new RegExp(`@${escapeRegExp(t)}\\b`, "i").test(s));
      const clean = s.replace(/@(Risk|BB&RL|Wholesale|Retail Assets)/gi, "").replace(/\s+/g, " ").trim();
      if (matched.length && clean) matched.forEach((team) => out.push({ team, text: clean }));
    }
    return out;
  }

  function prepareLog() {
    if (!notes.trim()) return;
    const tagged = extractTagged();
    if (tagged.length === 0) {
      // No tagged items — just save the note
      commit([]);
      return;
    }
    setPendingItems(tagged.map((t) => ({ ...t, due: "", err: false })));
  }

  function commit(items: { team: DrKey; text: string; due: string }[]) {
    const now = "25 Jun 2026";
    addMeetingNote(cfg.meetingName, notes);
    addMomEntry(cfg.meetingName, notes);
    const buMap: Record<DrKey, ActionItem["bu"]> = {
      Risk: "Credit Cards", "BB&RL": "BBRL", Wholesale: "Wholesale Banking", "Retail Assets": "Retail Advances - Excluding Credit Cards",
    };
    items.forEach((t, i) => {
      const id = `CT${Date.now()}-${i}`;
      const formattedDue = t.due.match(/^\d{4}-\d{2}-\d{2}$/)
        ? (() => { const [y, mo, d] = t.due.split("-").map(Number); return new Date(y, mo - 1, d).toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" }); })()
        : t.due;
      if (t.team !== cfg.drKey) {
        addCrossTag({
          id, text: t.text, team: t.team, sourceDR: cfg.drKey,
          sourceMeeting: cfg.meetingName, date: now, dueDate: formattedDue, status: "In Progress",
        });
      }
      addActions([{
        id, title: t.text, owner: `${t.team} Head`, bu: buMap[t.team],
        sourceMeeting: cfg.meetingName, due: formattedDue, status: "In Progress",
        origin: "seed",
        crossFunctional: t.team !== cfg.drKey,
        involvedBUs: t.team !== cfg.drKey ? [buMap[t.team], buMap[cfg.drKey]] : undefined,
        flag: "New", createdToday: true,
      }]);
    });
    if (/decide|approve|sign[- ]off|green[- ]light/i.test(notes)) {
      addPending({
        id: `PD${Date.now()}`,
        text: notes.split(/[.\n]/).find((l) => /decide|approve|sign[- ]off|green[- ]light/i.test(l))?.trim() || "Pending decision from notes",
        urgency: "this_week", source: "MoM", team: cfg.drKey, dueDate: "This week", status: "open",
      });
    }
    showToast(`MoM saved · ${items.length} item${items.length === 1 ? "" : "s"} logged`);
    setNotes("");
    setPendingItems(null);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  function confirmLog() {
    if (!pendingItems) return;
    const next = pendingItems.map((p) => ({ ...p, err: !p.due.trim() }));
    if (next.some((p) => p.err)) { setPendingItems(next); return; }
    commit(next);
  }

  const previewNodes = useMemo(() => {
    if (!notes.trim()) return null;
    const parts: React.ReactNode[] = [];
    const regex = /@(Risk|BB&RL|Wholesale|Retail Assets)/g;
    let last = 0; let m: RegExpExecArray | null; let i = 0;
    while ((m = regex.exec(notes)) !== null) {
      if (m.index > last) parts.push(notes.slice(last, m.index));
      const team = m[1] as DrKey;
      parts.push(
        <span key={i++} className="font-semibold px-1 rounded" style={{ background: `${TEAM_COLOR[team]}22`, color: TEAM_COLOR[team] }}>
          @{team}
        </span>
      );
      last = m.index + m[0].length;
    }
    if (last < notes.length) parts.push(notes.slice(last));
    return parts;
  }, [notes]);

  return (
    <div className="px-5 py-4">
      <div className="text-[12px] uppercase tracking-[0.08em] font-bold text-[#0A2540] mb-2">Log meeting notes</div>
      

      <div className="relative">
        <textarea
          ref={taRef}
          value={notes}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Type your MoM notes. Use @DepartmentName to tag items to another team."
          className="w-full h-40 text-[13px] p-3 border border-[#E2E8F0] rounded-[8px] focus:outline-none focus:border-[#0A2540] resize-none leading-relaxed"
          style={{ color: C.body }}
        />
        {mention && mentionMatches.length > 0 && (
          <div className="absolute z-20 left-3 bottom-2 bg-white border border-[#E2E8F0] rounded-[8px] shadow-md py-1 min-w-[160px]">
            <div className="px-2 py-1 text-[10px] uppercase tracking-[0.06em] text-[#94A3B8]">Tag to team</div>
            {mentionMatches.map((t) => (
              <button
                key={t}
                onMouseDown={(e) => { e.preventDefault(); pickTeam(t); }}
                className="w-full text-left text-[12px] px-2 py-1.5 hover:bg-[#EEF2F7] flex items-center gap-2"
              >
                <span className="w-2 h-2 rounded-full" style={{ background: TEAM_COLOR[t] }} />
                <span style={{ color: C.body }}>{t}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {previewNodes && (
        <div className="mt-2 text-[11.5px] p-2 border border-dashed border-[#E2E8F0] rounded-[6px] leading-relaxed whitespace-pre-wrap" style={{ color: C.body }}>
          {previewNodes}
        </div>
      )}

      <div className="flex items-center justify-between mt-2 gap-3">
        <div className="text-[10.5px] text-[#475569] flex items-center gap-1">
          <Tag size={11} /> Use @DepartmentName to tag items to another team. Type @ to see options.
        </div>
        <button
          onClick={prepareLog}
          className="text-[12px] font-semibold px-3 py-1.5 rounded-[8px]"
          style={{ background: saved ? C.success : C.navy, color: "white" }}
        >
          {saved ? "Saved · items logged" : "Save & log items"}
        </button>
      </div>

      {pendingItems && pendingItems.length > 0 && (
        <div className="mt-3 border border-[#E2E8F0] rounded-[10px] p-3 bg-[#F1F5F9]">
          <div className="text-[11px] uppercase tracking-[0.06em] font-bold text-[#0A2540] mb-2">Confirm action items · set due date</div>
          <div className="space-y-2">
            {pendingItems.map((p, i) => (
              <div key={i} className="bg-white border border-[#E2E8F0] rounded-[8px] p-2 flex items-center gap-2">
                <span className="text-[10px] font-semibold px-1.5 py-[1px] rounded-[5px] shrink-0" style={{ background: `${TEAM_COLOR[p.team]}1A`, color: TEAM_COLOR[p.team] }}>{DR_TO_BU[p.team] || p.team}</span>
                <div className="flex-1 text-[12px] truncate" style={{ color: C.body }} title={p.text}>{p.text}</div>
                <input
                  type="date" value={p.due}
                  onChange={(e) => setPendingItems((prev) => prev?.map((x, j) => j === i ? { ...x, due: e.target.value, err: e.target.value ? false : x.err } : x) ?? null)}
                  className={`text-[11.5px] px-2 py-1 rounded-[6px] focus:outline-none ${p.err ? "border-2 border-[#F59E0B] bg-[#FFFBEB]" : "border border-[#E2E8F0]"}`}
                />
              </div>
            ))}
          </div>
          {pendingItems.some((p) => p.err) && <div className="mt-2 text-[11px] text-[#B45309]">Due date required.</div>}
          <div className="mt-2 flex items-center justify-end gap-2">
            <button onClick={() => setPendingItems(null)} className="text-[11px] text-[#94A3B8]">Cancel</button>
            <button onClick={confirmLog} className="text-[12px] font-semibold px-3 py-1.5 rounded-[8px]" style={{ background: C.navy, color: "white" }}>Confirm & log</button>
          </div>
        </div>
      )}
    </div>
  );
}




/* =========================================================
   Collapsible section wrappers (default closed; show count badge)
   ========================================================= */
function CollapsibleSection({
  title, count, accent = "#0A2540", defaultOpen = false, children,
}: { title: string; count: number; accent?: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-5 py-3"
      >
        <div className="flex items-center gap-2.5">
          <span className="text-[12px] uppercase tracking-[0.08em] font-bold text-[#0A2540]">{title}</span>
          <span
            className="inline-flex items-center justify-center text-[10px] font-bold text-white rounded-full px-[6px]"
            style={{ background: accent, minWidth: 18, height: 18 }}
          >
            {count}
          </span>
        </div>
        {open
          ? <ChevronDown size={14} className="text-[#475569]" />
          : <ChevronRight size={14} className="text-[#475569]" />}
      </button>
      {open && <div className="border-t border-[#E2E8F0]">{children}</div>}
    </div>
  );
}

function CollapsiblePastMeetings({ cfg }: { cfg: DrConfig }) {
  const { momEntries } = useAppState();
  const myMomEntries = momEntries.filter((e) => e.meeting === cfg.meetingName);
  const total = myMomEntries.length + cfg.past.length;
  return (
    <CollapsibleSection title="Past meeting summaries" count={total} accent="#0A2540">
      <PastMeetings cfg={cfg} />
    </CollapsibleSection>
  );
}

function CollapsibleOpenActions({ cfg }: { cfg: DrConfig }) {
  const { actions } = useAppState();
  const drBU: Record<DrKey, ActionItem["bu"]> = { Risk: "Credit Cards", "BB&RL": "BBRL", Wholesale: "Wholesale Banking", "Retail Assets": "Retail Advances - Excluding Credit Cards" };
  const count = actions.filter((a) => a.bu === drBU[cfg.drKey] && !a.crossFunctional).length;
  return (
    <CollapsibleSection title="Open action items · from MoM's" count={count} accent="#E8192C">
      <OpenActions cfg={cfg} />
    </CollapsibleSection>
  );
}

function CollapsibleCrossFunctional({ cfg }: { cfg: DrConfig }) {
  const { crossTags } = useAppState();
  const count = crossTags.filter((c) => c.team === cfg.drKey).length;
  return (
    <CollapsibleSection title="Cross-functional items tagged here" count={count} accent="#E8192C">
      <CrossFunctionalHere cfg={cfg} />
    </CollapsibleSection>
  );
}


function MetricsStrip({ cfg }: { cfg: DrConfig }) {
  const toneColor: Record<string, string> = { emerald: C.success, amber: C.warning, crimson: C.danger, info: C.info };
  return (
    <div className="px-5 py-4">
      <div className="text-[12px] uppercase tracking-[0.08em] font-bold text-[#0A2540] mb-2.5">Metrics · Latest MIS</div>
      <div className="grid grid-cols-4 gap-3">
        {cfg.metrics.map((m) => (
          <div key={m.label} className="border border-[#E2E8F0] rounded-[10px] px-3 py-3 overflow-hidden">
            <div className="text-[11px] uppercase tracking-[0.05em] font-semibold text-[#0A2540] line-clamp-2 break-words">{m.label}</div>
            <div className="text-[22px] font-semibold leading-tight mt-1 tabular-nums" style={{ color: C.body }}>{m.value}</div>
            <div className="text-[10px] font-semibold mt-1" style={{ color: toneColor[m.tone] }}>{m.delta}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PastMeetings({ cfg }: { cfg: DrConfig }) {
  const { meetingNotes, momEntries } = useAppState();
  const [openIdx, setOpenIdx] = useState<number | null>(0);
  const [notesOpen, setNotesOpen] = useState<Record<number, boolean>>({});
  const myMomEntries = momEntries.filter((e) => e.meeting === cfg.meetingName);
  return (
    <div className="px-5 py-4">
      <div className="space-y-2.5">
        {myMomEntries.map((e, ei) => {
          return (
            <div key={`mom-${ei}`} className="border border-[#E2E8F0] rounded-[14px] p-3">
              <div className="flex items-center justify-between gap-2">
                <div className="text-[12.5px] font-semibold" style={{ color: C.body }}>
                  {e.date} · {e.meeting}
                </div>
              </div>
              <div className="mt-2 text-[11.5px] text-[#475569] whitespace-pre-wrap leading-relaxed">{e.note}</div>
            </div>
          );
        })}
        {cfg.past.map((p, i) => {
          const open = openIdx === i;
          // Match MoM notes for this meeting by date prefix
          const noteKey = Object.keys(meetingNotes).find((k) => k.includes(p.date) || k.includes(cfg.meetingName));
          const savedNote = noteKey ? meetingNotes[noteKey] : null;
          const isNotesOpen = !!notesOpen[i];
          return (
            <div
              key={i}
              className="border border-[#E2E8F0] rounded-[14px] p-3 cursor-pointer hover:border-[#94A3B8] transition"
              onClick={() => setOpenIdx(open ? null : i)}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="text-[12.5px] font-semibold flex items-center gap-2" style={{ color: C.body }}>
                  {p.date} · {p.type}
                  {!open && p.openItems.length > 0 && (
                    <span title={`${p.openItems.length} open item${p.openItems.length === 1 ? "" : "s"}`} className="inline-flex items-center justify-center text-[10px] font-bold text-white rounded-full" style={{ background: C.danger, minWidth: 16, height: 16, padding: "0 5px" }}>
                      {p.openItems.length}
                    </span>
                  )}
                  {!open && savedNote && (
                    <span title="MoM note saved" className="inline-flex items-center text-[#94A3B8]"><FileText size={12} /></span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <ChevronDown size={13} className={`text-[#94A3B8] transition ${open ? "rotate-180" : ""}`} />
                </div>
              </div>
              {open && (
                <>
                  <ul className="mt-2 space-y-0.5">
                    {p.outcomes.map((o, j) => (
                      <li key={j} className="text-[11.5px] text-[#475569] flex gap-2"><span className="text-[#94A3B8]">·</span>{o}</li>
                    ))}
                  </ul>
                  {p.openItems.length > 0 && (
                    <>
                      <div className="text-[10px] uppercase tracking-[0.08em] text-[#94A3B8] mt-2">Open items</div>
                      <ul className="mt-0.5 space-y-0.5">
                        {p.openItems.map((o, j) => (
                          <li key={j} className="text-[11.5px] flex gap-2" style={{ color: C.danger }}><span>·</span>{o}</li>
                        ))}
                      </ul>
                    </>
                  )}
                  {savedNote && (
                    <div className="mt-3 border-t border-[#EEF2F7] pt-2">
                      <button
                        onClick={(e) => { e.stopPropagation(); setNotesOpen((p2) => ({ ...p2, [i]: !p2[i] })); }}
                        className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.06em] font-bold text-[#94A3B8] hover:text-[#0A2540]"
                      >
                        <FileText size={11} /> Notes
                        <ChevronDown size={11} className={`transition ${isNotesOpen ? "rotate-180" : ""}`} />
                      </button>
                      {isNotesOpen && (
                        <div className="mt-1.5 text-[11px] text-[#475569] whitespace-pre-wrap leading-relaxed">{savedNote}</div>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function OpenActions({ cfg }: { cfg: DrConfig }) {
  const { actions, addActions, showToast, setFocusActionId, setLedgerTeamFilter } = useAppState();
  const navigate = useAppNavigate();
  const drBU: Record<DrKey, ActionItem["bu"]> = { Risk: "Credit Cards", "BB&RL": "BBRL", Wholesale: "Wholesale Banking", "Retail Assets": "Retail Advances - Excluding Credit Cards" };
  const items = useMemo(
    () => actions.filter((a) => a.bu === drBU[cfg.drKey] && !a.crossFunctional),
    [actions, cfg.drKey]
  );
  const [adding, setAdding] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [newDue, setNewDue] = useState("");
  const [dueErr, setDueErr] = useState(false);

  function add() {
    if (!newTitle.trim()) { setAdding(false); return; }
    if (!newDue.trim()) { setDueErr(true); return; }
    const formatted = newDue.match(/^\d{4}-\d{2}-\d{2}$/)
      ? (() => { const [y, mo, d] = newDue.split("-").map(Number); return new Date(y, mo - 1, d).toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" }); })()
      : newDue;
    addActions([{
      id: `MAN${Date.now()}`, title: newTitle.trim(), owner: cfg.fullName,
      bu: drBU[cfg.drKey] as ActionItem["bu"], sourceMeeting: "Added by CEO · 25 Jun 2026",
      due: formatted, status: "In Progress", origin: "seed", flag: "New", createdToday: true,
    }]);
    showToast("Action item added to Ledger");
    setNewTitle(""); setNewDue(""); setDueErr(false); setAdding(false);
  }

  function openInLedger(id: string) {
    setLedgerTeamFilter(drBU[cfg.drKey]);
    setFocusActionId(id);
    navigate("Ledger");
  }

  return (
    <div className="px-5 py-4">
      <div className="text-[12px] text-[#94A3B8] mb-2.5">Extracted from MoM's and manually added — click any item to update its status.</div>
      <div className="space-y-1.5">
        {items.length === 0 && <div className="text-[11.5px] text-[#94A3B8]">No open actions for this team.</div>}
        {items.map((a) => {
          const tone = STATUS_TONE[a.status];
          const context = ITEM_CONTEXT[a.title];
          return (
            <div
              key={a.id}
              onClick={() => openInLedger(a.id)}
              className="flex items-start gap-2 text-[12px] cursor-pointer hover:bg-[#F1F5F9] -mx-1 px-1 py-1 rounded-[6px]"
              title="Open in Action & Accountability Ledger"
            >
              <span className="text-[10px] font-semibold px-2 py-[2px] rounded-[6px] shrink-0 mt-[2px]" style={{ background: tone.bg, color: tone.text }}>{a.status === "Ahead" ? "On Track" : a.status}</span>
              <div className="flex-1 min-w-0" style={{ color: C.body }}>
                <div className="break-words">{a.title}</div>
                {context && <div className="text-[12px] font-normal text-[#94A3B8] break-words whitespace-normal mt-0.5">{context}</div>}
              </div>
              <div className="text-[10.5px] text-[#475569] tabular-nums mt-[2px]">{a.due}</div>
            </div>
          );
        })}
      </div>
      {adding ? (
        <div className="mt-2 flex items-center gap-2">
          <input
            autoFocus value={newTitle} onChange={(e) => setNewTitle(e.target.value)}
            placeholder="Action item description"
            className="flex-1 text-[12px] bg-white border border-[#E2E8F0] rounded-[6px] px-2 py-1.5 focus:outline-none focus:border-[#0A2540]"
          />
          <input
            type="date" value={newDue} onChange={(e) => { setNewDue(e.target.value); if (e.target.value) setDueErr(false); }}
            className={`text-[12px] px-2 py-1.5 rounded-[6px] focus:outline-none ${dueErr ? "border-2 border-[#F59E0B] bg-[#FFFBEB]" : "border border-[#E2E8F0]"}`}
          />
          <button onClick={add} className="text-[12px] font-semibold px-3 py-1.5 rounded-[6px] bg-[#0A2540] text-white">Save</button>
          <button onClick={() => { setAdding(false); setDueErr(false); setNewTitle(""); setNewDue(""); }} className="text-[11px] text-[#94A3B8]">Cancel</button>
        </div>
      ) : null}
      {dueErr && <div className="mt-1 text-[11px] text-[#B45309]">Due date required.</div>}

      {(() => {
        const ganttCount = actions.filter((a) => a.bu === drBU[cfg.drKey]).length;
        if (ganttCount >= 2) {
          return (
            <div className="mt-4">
              <div className="text-[10px] uppercase tracking-[0.08em] font-bold text-[#94A3B8] mb-1.5">Timeline</div>
              <ActionGantt filterFn={(a) => a.bu === drBU[cfg.drKey]} compact />
            </div>
          );
        }
        if (items.length === 1) {
          return <div className="mt-3 text-[11px] text-[#94A3B8] italic">Add more items to see timeline view.</div>;
        }
        return null;
      })()}
    </div>
  );
}

function CrossFunctionalHere({ cfg }: { cfg: DrConfig }) {
  const { crossTags, updateCrossTag, setFocusActionId } = useAppState();
  const navigate = useAppNavigate();
  const items = crossTags.filter((c) => c.team === cfg.drKey);
  if (items.length === 0) {
    return (
      <div className="px-5 py-4">
        <div className="text-[12px] uppercase tracking-[0.08em] font-bold text-[#0A2540] mb-2">Cross-functional items tagged here</div>
        <div className="text-[12px] text-[#94A3B8] mb-2">Items tagged to {cfg.drKey} from other workspaces — review and update status as needed.</div>
        <div className="text-[11.5px] text-[#94A3B8]">No cross-functional items</div>
      </div>
    );
  }
  return (
    <div className="px-5 py-4">
      <div className="text-[12px] uppercase tracking-[0.08em] font-bold text-[#0A2540] mb-2">Cross-functional items tagged here</div>
      <div className="text-[12px] text-[#94A3B8] mb-2.5">Items tagged to {cfg.drKey} from other workspaces — review and update status as needed.</div>
      <div className="space-y-2">
        {items.map((c) => {
          const tone = STATUS_TONE[c.status];
          return (
            <div
              key={c.id}
              onClick={() => { setFocusActionId(c.id); navigate("Ledger"); }}
              className="border border-[#E2E8F0] rounded-[10px] p-2.5 cursor-pointer hover:border-[#94A3B8] hover:bg-[#F1F5F9] transition"
              title="Open in Action & Accountability Ledger"
            >
              <div className="flex items-start gap-2">
                <select
                  value={c.status}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => updateCrossTag(c.id, { status: e.target.value as Status })}
                  className="text-[10px] font-semibold px-1.5 py-[2px] rounded-[6px] border-0 focus:outline-none"
                  style={{ background: tone.bg, color: tone.text }}
                >
                  <option value="Overdue">Overdue</option>
                  <option value="In Progress">In Progress</option>
                  <option value="On Track">On Track</option>
                </select>
                <div className="flex-1 text-[12px]" style={{ color: C.body }}>{c.text}</div>
                <div className="text-[10.5px] text-[#475569] tabular-nums"><span className="text-[#94A3B8]">Due:</span> {c.dueDate}</div>
              </div>
              <div className="mt-1 flex items-center gap-2 text-[10px] text-[#94A3B8] uppercase tracking-[0.06em]">
                <span>From: {c.sourceDR} · {c.sourceMeeting}</span>
                <span
                  className="px-1.5 py-[1px] rounded-[4px] normal-case"
                  style={{ background: `${TEAM_COLOR[c.team]}1A`, color: TEAM_COLOR[c.team] }}
                >
                  {c.team}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* Helper export so Launchpad can render contained sparkline */
export function MicroSpark({ data, color }: { data: number[]; color: string }) {
  return (
    <div className="overflow-hidden w-full h-6 mt-1">
      <Sparkline data={data} color={color} responsive width={100} height={24} />
    </div>
  );
}
