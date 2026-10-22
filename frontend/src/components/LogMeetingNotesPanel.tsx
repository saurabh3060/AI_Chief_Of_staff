import { useState, useRef, useEffect } from "react";
import { X, Tag } from "lucide-react";
import { useAppState, type DrKey } from "@/context/AppState";
import { createMeetingNote, fetchMeetingNotes } from "@/lib/api";

// The four taggable teams. Kept module-level so both tag extraction and
// the @mention picker share one source of truth.
const TAGGABLE_TEAMS: DrKey[] = ["Risk", "BB&RL", "Wholesale", "Retail Assets"];

/**
 * Extract the distinct @Team tags present in a note's text, preserving the
 * canonical DrKey spelling. "BB&RL" contains "&" so it is matched literally
 * rather than via \w. Order follows TAGGABLE_TEAMS; duplicates are removed.
 */
function extractTeamTags(text: string): DrKey[] {
  return TAGGABLE_TEAMS.filter((team) => text.includes(`@${team}`));
}

const C = {
  navy: "#0A2540",
  body: "#0A2540",
  success: "#16A34A",
};

const TEAM_COLOR: Record<DrKey, string> = {
  Risk: "#DC2626",
  "BB&RL": "#1D4ED8",
  Wholesale: "#16A34A",
  "Retail Assets": "#D97706",
};

// Map DrKey to full business unit name
const DR_TO_BU: Record<DrKey, string> = {
  Risk: "Credit Cards",
  "BB&RL": "BBRL",
  Wholesale: "Wholesale Banking",
  "Retail Assets": "Retail Advances - Excluding Credit Cards",
};

interface PendingItem {
  text: string;
  team: DrKey;
  due: string;
  err?: boolean;
}

interface LogMeetingNotesPanelProps {
  isOpen: boolean;
  onClose: () => void;
  meetings: Array<{ time: string; title: string }>;
  preSelectedMeeting?: string;
  /**
   * The Graph event id of the meeting currently in context. When present,
   * saved notes are persisted to Postgres (POST /agents/meeting-notes) so
   * cluster synthesis can read them back by meeting_id. When absent (e.g.
   * the generic "Log meeting notes" button with no meeting selected), the
   * note is kept in local state only.
   */
  meetingId?: string;
}

export function LogMeetingNotesPanel({
  isOpen,
  onClose,
  meetings,
  preSelectedMeeting,
  meetingId,
}: LogMeetingNotesPanelProps) {
  const [showMeetingSelector, setShowMeetingSelector] = useState(false);
  const [selectedMeeting, setSelectedMeeting] = useState(preSelectedMeeting || "");
  const [notes, setNotes] = useState("");
  const [mention, setMention] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [pendingItems, setPendingItems] = useState<PendingItem[] | null>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const { addCrossTag, addActions, addMotion, addMeetingNote, addMomEntry } = useAppState();

  const TEAMS = TAGGABLE_TEAMS;

  // Reset state when panel opens
  useEffect(() => {
    if (isOpen) {
      if (preSelectedMeeting) {
        setShowMeetingSelector(false);
        setSelectedMeeting(preSelectedMeeting);
      } else {
        setShowMeetingSelector(false);
        setSelectedMeeting("");
      }
      setNotes("");
      setMention(null);
      setSaved(false);
      setPendingItems(null);
    }
  }, [isOpen, preSelectedMeeting]);

  // Fix #1 — hydrate persisted notes from the backend when the panel opens
  // for a meeting that has a Graph event id. Notes live in Postgres but the
  // app's in-memory `meetingNotes`/`momEntries` are NOT otherwise restored
  // after a page refresh, so previously-saved notes appeared lost. Pulling
  // them back here repopulates the MoM list for the selected meeting.
  useEffect(() => {
    if (!isOpen || !meetingId) return;
    let cancelled = false;
    const meetingLabel = preSelectedMeeting || selectedMeeting || meetingId;

    fetchMeetingNotes(meetingId)
      .then((res) => {
        if (cancelled || !res.notes?.length) return;
        // Most recent note seeds the single-value `meetingNotes` map; every
        // note becomes a MoM list entry so nothing silently disappears.
        const latest = res.notes[res.notes.length - 1];
        if (latest?.note_text) addMeetingNote(meetingLabel, latest.note_text);
        for (const n of res.notes) {
          if (n.note_text) addMomEntry(meetingLabel, n.note_text);
        }
      })
      .catch((err) => {
        // Hydration is best-effort; a failure must not break opening the panel.
        console.error("[LogMeetingNotes] failed to hydrate persisted notes:", err);
      });

    return () => {
      cancelled = true;
    };
    // `addMeetingNote`/`addMomEntry` are stable useCallback refs from context.
  }, [isOpen, meetingId, preSelectedMeeting, selectedMeeting, addMeetingNote, addMomEntry]);

  // Detect @ mention
  useEffect(() => {
    if (!taRef.current) return;
    const v = taRef.current.value;
    const pos = taRef.current.selectionStart;
    const before = v.slice(0, pos);
    const m = before.match(/@(\w*)$/);
    setMention(m ? m[1] : null);
  }, [notes]);

  const mentionMatches = mention !== null ? TEAMS.filter((t) => t.toLowerCase().includes(mention.toLowerCase())) : [];

  function pickTeam(team: DrKey) {
    if (!taRef.current) return;
    const v = taRef.current.value;
    const pos = taRef.current.selectionStart;
    const before = v.slice(0, pos);
    const after = v.slice(pos);
    const prefix = before.replace(/@\w*$/, `@${team} `);
    const newVal = prefix + after;
    setNotes(newVal);
    setTimeout(() => {
      taRef.current?.focus();
      taRef.current?.setSelectionRange(prefix.length, prefix.length);
    }, 0);
    setMention(null);
  }

  function onChange(val: string) {
    setNotes(val);
    setSaved(false);
  }

  /**
   * Persist a note. Always updates local state (so the UI/MoM list react
   * immediately, as before). Additionally, when a Graph `meetingId` is in
   * context, POSTs the note to the backend so it lands in Postgres and is
   * readable by cluster synthesis. The backend LLM-extracts action items
   * when we send an empty list, so we leave extraction to it here.
   */
  function persistNote(meetingKey: string, text: string) {
    addMeetingNote(meetingKey, text);
    addMomEntry(meetingKey, text);

    if (!meetingId) {
      // No Graph event id in context (e.g. generic entry point) — keep the
      // note local only; there's no reliable key to store it under in the DB.
      return;
    }

    createMeetingNote({
      meeting_id: meetingId,
      note_text: text,
      // Carry the @Team mentions as structured tags so cluster synthesis
      // (and any future routing) can see which teams the note was aimed at,
      // rather than losing that signal inside the free-text body.
      tags: extractTeamTags(text),
      source: "ceo_note",
    }).catch((err) => {
      // Don't block the UX on a persistence failure; surface it to the console.
      console.error("[LogMeetingNotes] failed to persist note to backend:", err);
    });
  }

  const previewNodes = notes.split(/(@(?:Risk|BB&RL|Wholesale|Retail Assets))/g).map((part, i) => {
    const m = part.match(/^@(.+)$/);
    if (m) {
      const teamKey = m[1] as DrKey;
      return (
        <span key={i} className="font-semibold" style={{ color: TEAM_COLOR[teamKey] }}>
          {part}
        </span>
      );
    }
    return part;
  });

  function prepareLog() {
    // If no meeting is selected yet, show the meeting selector
    if (!selectedMeeting) {
      setShowMeetingSelector(true);
      return;
    }

    const lines = notes.split("\n").filter((l) => l.trim());
    const items: PendingItem[] = [];
    for (const line of lines) {
      const m = line.match(/@(Risk|BB&RL|Wholesale|Retail Assets)/);
      if (m) {
        const team = m[1] as DrKey;
        const text = line.replace(`@${team}`, "").trim();
        if (text) items.push({ text, team, due: "", err: false });
      }
    }
    if (items.length === 0) {
      persistNote(selectedMeeting, notes);
      setSaved(true);
      setTimeout(() => {
        onClose();
      }, 1000);
    } else {
      setPendingItems(items);
    }
  }

  function confirmLog() {
    if (!pendingItems) return;
    let bad = false;
    const updated = pendingItems.map((p) => {
      if (!p.due.trim()) {
        bad = true;
        return { ...p, err: true };
      }
      return { ...p, err: false };
    });
    if (bad) {
      setPendingItems(updated);
      return;
    }
    for (const p of pendingItems) {
      const formattedDue = p.due.match(/^\d{4}-\d{2}-\d{2}$/)
        ? (() => { const [y, mo, d] = p.due.split("-").map(Number); return new Date(y, mo - 1, d).toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" }); })()
        : new Date(p.due).toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" });
      addActions([
        {
          id: `ACT${Date.now()}-${Math.random()}`,
          title: p.text,
          owner: "Unassigned",
          bu: p.team === "Risk" ? "Credit Cards" : p.team === "BB&RL" ? "BBRL" : p.team === "Wholesale" ? "Wholesale Banking" : "Retail Advances - Excluding Credit Cards",
          sourceMeeting: selectedMeeting,
          due: formattedDue,
          status: "In Progress",
          origin: "seed",
          flag: "New",
          createdToday: true,
        },
      ]);
    }
    persistNote(selectedMeeting, notes);
    setSaved(true);
    setPendingItems(null);
    setTimeout(() => {
      onClose();
    }, 1000);
  }

  function handleMeetingSelect(meeting: string) {
    setSelectedMeeting(meeting);
    setShowMeetingSelector(false);
    // Continue with the save flow after meeting selection
    setTimeout(() => {
      const lines = notes.split("\n").filter((l) => l.trim());
      const items: PendingItem[] = [];
      for (const line of lines) {
        const m = line.match(/@(Risk|BB&RL|Wholesale|Retail Assets)/);
        if (m) {
          const team = m[1] as DrKey;
          const text = line.replace(`@${team}`, "").trim();
          if (text) items.push({ text, team, due: "", err: false });
        }
      }
      if (items.length === 0) {
        persistNote(meeting, notes);
        setSaved(true);
        setTimeout(() => {
          onClose();
        }, 1000);
      } else {
        setPendingItems(items);
      }
    }, 100);
  }

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/20 z-40 transition-opacity"
        onClick={onClose}
      />

      {/* Slide-out panel from right */}
      <div
        className="fixed top-0 right-0 h-full w-[480px] bg-white shadow-2xl z-50 
                   transform transition-transform duration-300 ease-out overflow-hidden flex flex-col"
        style={{ transform: isOpen ? "translateX(0)" : "translateX(100%)" }}
      >
        {/* Header */}
        <div className="bg-[#0A2540] text-white px-6 py-4 flex items-center justify-between shrink-0">
          <div>
            <h2 className="text-[16px] font-semibold">Log meeting notes</h2>
            {selectedMeeting && (
              <p className="text-[12px] text-[#94A3B8] mt-1">{selectedMeeting}</p>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-white/70 hover:text-white transition"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="mb-4">
            <label className="text-[12px] uppercase tracking-[0.08em] font-bold text-[#0A2540] mb-2 block">
              Log Meeting Notes
            </label>
            <div className="relative">
              <textarea
                ref={taRef}
                value={notes}
                onChange={(e) => onChange(e.target.value)}
                placeholder="Type your MoM notes. Use @DepartmentName to tag items to another team."
                className="w-full h-48 text-[13px] p-3 border border-[#E2E8F0] rounded-[8px] 
                         focus:outline-none focus:border-[#0A2540] resize-none leading-relaxed"
                style={{ color: C.body }}
              />
              {mention && mentionMatches.length > 0 && (
                <div className="absolute z-20 left-3 bottom-2 bg-white border border-[#E2E8F0] 
                              rounded-[8px] shadow-md py-1 min-w-[160px]">
                  <div className="px-2 py-1 text-[10px] uppercase tracking-[0.06em] text-[#94A3B8]">
                    Tag to team
                  </div>
                  {mentionMatches.map((t) => (
                    <button
                      key={t}
                      onMouseDown={(e) => {
                        e.preventDefault();
                        pickTeam(t);
                      }}
                      className="w-full text-left text-[12px] px-2 py-1.5 hover:bg-[#EEF2F7] 
                               flex items-center gap-2"
                    >
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{ background: TEAM_COLOR[t] }}
                      />
                      <span style={{ color: C.body }}>{t}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center justify-between mb-4 gap-3">
            <div className="text-[10.5px] text-[#475569] flex items-center gap-1">
              <Tag size={11} /> Use @DepartmentName to tag items
            </div>
            <button
              onClick={prepareLog}
              className="text-[12px] font-semibold px-3 py-1.5 rounded-[8px]"
              style={{
                background: saved ? C.success : C.navy,
                color: "white",
              }}
              disabled={!notes.trim()}
            >
              {saved ? "Saved · items logged" : "Save & log items"}
            </button>
          </div>

          {pendingItems && pendingItems.length > 0 && (
            <div className="border border-[#E2E8F0] rounded-[10px] p-3 bg-[#F1F5F9]">
              <div className="text-[11px] uppercase tracking-[0.06em] font-bold text-[#0A2540] mb-2">
                Confirm action items · set due date
              </div>
              <div className="space-y-2">
                {pendingItems.map((p, i) => (
                  <div
                    key={i}
                    className="bg-white border border-[#E2E8F0] rounded-[8px] p-2 flex items-center gap-2"
                  >
                    <span
                      className="text-[10px] font-semibold px-1.5 py-[1px] rounded-[5px] shrink-0"
                      style={{
                        background: `${TEAM_COLOR[p.team]}1A`,
                        color: TEAM_COLOR[p.team],
                      }}
                    >
                      {DR_TO_BU[p.team] || p.team}
                    </span>
                    <div
                      className="flex-1 text-[12px] truncate"
                      style={{ color: C.body }}
                      title={p.text}
                    >
                      {p.text}
                    </div>
                    <input
                      type="date"
                      value={p.due}
                      onChange={(e) =>
                        setPendingItems(
                          (prev) =>
                            prev?.map((x, j) =>
                              j === i
                                ? { ...x, due: e.target.value, err: e.target.value ? false : x.err }
                                : x
                            ) ?? null
                        )
                      }
                      className={`text-[11.5px] px-2 py-1 rounded-[6px] focus:outline-none ${
                        p.err
                          ? "border-2 border-[#F59E0B] bg-[#FFFBEB]"
                          : "border border-[#E2E8F0]"
                      }`}
                    />
                  </div>
                ))}
              </div>
              {pendingItems.some((p) => p.err) && (
                <div className="mt-2 text-[11px] text-[#B45309]">Due date required.</div>
              )}
              <div className="mt-2 flex items-center justify-end gap-2">
                <button
                  onClick={() => setPendingItems(null)}
                  className="text-[11px] text-[#94A3B8]"
                >
                  Cancel
                </button>
                <button
                  onClick={confirmLog}
                  className="text-[12px] font-semibold px-3 py-1.5 rounded-[8px]"
                  style={{ background: C.navy, color: "white" }}
                >
                  Confirm & log
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Meeting selector modal (centered) */}
      {showMeetingSelector && (
        <div className="fixed inset-0 flex items-center justify-center z-[60]">
          <div className="bg-white rounded-[12px] shadow-2xl w-[540px] max-w-[90vw]">
            <div className="px-6 py-4 border-b border-[#E2E8F0] flex items-center justify-between">
              <h3 className="text-[16px] font-semibold text-[#0A2540]">Log Meeting Notes</h3>
              <button
                onClick={onClose}
                className="text-[#94A3B8] hover:text-[#0A2540] transition"
              >
                <X size={18} />
              </button>
            </div>

            <div className="p-6">
              <label className="text-[11px] uppercase tracking-[0.08em] font-bold text-[#64748B] mb-2 block">
                Please select a meeting from the list
              </label>
              <select
                value={selectedMeeting}
                onChange={(e) => e.target.value && handleMeetingSelect(e.target.value)}
                className="w-full text-[13px] px-3 py-2.5 border border-[#E2E8F0] rounded-[8px] 
                         focus:outline-none focus:border-[#0A2540] bg-white"
                style={{ color: C.body }}
              >
                <option value="">Select a meeting...</option>
                {meetings.map((m, i) => (
                  <option key={i} value={`${m.time} · ${m.title}`}>
                    {m.time} · {m.title}
                  </option>
                ))}
              </select>

              <div className="mt-6 flex items-center justify-end gap-3">
                <button
                  onClick={() => {
                    setShowMeetingSelector(false);
                    setNotes("");
                  }}
                  className="text-[13px] text-[#64748B] hover:text-[#0A2540] transition"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
