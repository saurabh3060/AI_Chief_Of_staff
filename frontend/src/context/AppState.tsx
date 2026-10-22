import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { SEED_ACTIONS, SEED_NOTIFICATIONS } from "@/data/mockData";
import {
  fetchAuthState,
  fetchCurrentMeeting,
  fetchEmailSummary,
  fetchMeetingBriefById,
  fetchOpenTracker,
  fetchPremeetingBriefForMeeting,
  fetchTodayEvents,
  redirectToLogin,
  AuthCheckError,
  type AuthState,
  type BackendCalendarEvent,
  type BackendEmailItemSummary,
  type BackendEmailSummary,
  type BackendMeetingBrief,
  type BackendPreMeetingBrief,
  type OpenTrackerItem,
  type PendingDecisionItem,
} from "@/lib/api";
import { sanitizeBlock, sanitizeLines } from "@/lib/sanitize";



/**
 * Sanitize every text field on a raw `BackendEmailSummary` before it
 * ever reaches components. The backend already strips stray control
 * characters (see `agent_email/prompts.py`), but this is a second,
 * narrow pass so a cached (pre-sanitization) sessionStorage value or
 * any future backend change can't leak unescaped characters into the
 * UI. Only control/non-printable characters are touched - no real
 * summary/action-item/question content is ever removed.
 */
function sanitizeEmailItemSummary(item: BackendEmailItemSummary): BackendEmailItemSummary {
  return {
    ...item,
    subject: sanitizeLines([item.subject])[0] ?? item.subject,
    summary_text: sanitizeBlock(item.summary_text),
    action_items: sanitizeLines(item.action_items),
    top_questions: sanitizeLines(item.top_questions).slice(0, 4),
  };
}

function sanitizeEmailSummary(summary: BackendEmailSummary): BackendEmailSummary {
  return {
    ...summary,
    subject: sanitizeLines([summary.subject])[0] ?? summary.subject,
    summary_text: sanitizeBlock(summary.summary_text),
    action_items: sanitizeLines(summary.action_items),
    top_questions: sanitizeLines(summary.top_questions).slice(0, 4),
    email_summaries: (summary.email_summaries ?? []).map(sanitizeEmailItemSummary),
    action_candidates: summary.action_candidates.map((ac) => ({
      ...ac,
      text: sanitizeLines([ac.text])[0] ?? "",
    })),
  };
}

const MONTHS: Record<string, number> = {
  jan: 0, feb: 1, mar: 2, apr: 3, may: 4, jun: 5,
  jul: 6, aug: 7, sep: 8, sept: 8, oct: 9, nov: 10, dec: 11,
};

/** sessionStorage key for caching email summary to avoid redundant LLM calls within a browser session. */
const EMAIL_SUMMARY_SESSION_KEY = "rbl_email_summary";

/** Read the cached email summary from sessionStorage, returning null if missing/invalid. */
function readCachedEmailSummary(): BackendEmailSummary | null {
  try {
    const cached = sessionStorage.getItem(EMAIL_SUMMARY_SESSION_KEY);
    if (!cached) return null;
    return JSON.parse(cached) as BackendEmailSummary;
  } catch {
    return null;
  }
}

/** Write the email summary to sessionStorage (or clear it if null). */
function writeCachedEmailSummary(summary: BackendEmailSummary | null): void {
  try {
    if (summary === null) {
      sessionStorage.removeItem(EMAIL_SUMMARY_SESSION_KEY);
    } else {
      sessionStorage.setItem(EMAIL_SUMMARY_SESSION_KEY, JSON.stringify(summary));
    }
  } catch {
    // Silently fail if sessionStorage is unavailable (e.g., private browsing)
  }
}

/**
 * Parse the various due-date shapes that appear across the app into a
 * Date at local midnight. Accepts:
 *   - ISO:              "2026-06-12"
 *   - Alpha (no comma): "12 Jun 2026", "12 June 2026", "30 Sept 2026"
 *   - Alpha (comma):    "12 Jun, 2026" (en-IN can emit this on some browsers)
 *   - Alpha (hyphens):  "12-Jun-2026", "30-Sept-2026"
 * Robustness matters: `toLocaleDateString("en-IN", …)` output varies by
 * browser/ICU, and a null here means daysUntil() returns null which then
 * mis-buckets items as "upcoming" and lets stale a.status drive the UI.
 */
export function parseDueDate(s: string | undefined | null): Date | null {
  if (!s) return null;
  const str = s.trim();
  let m = str.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  if (m) return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  // Day + alpha month + year, separated by any combination of
  // spaces / commas / hyphens.
  m = str.match(/^(\d{1,2})[\s,\-/]+([A-Za-z]+)[\s,\-/]+(\d{4})$/);
  if (m) {
    const mo = MONTHS[m[2].slice(0, 3).toLowerCase()];
    if (mo == null) return null;
    return new Date(Number(m[3]), mo, Number(m[1]));
  }
  return null;
}

/** True when the due date is today or earlier (local time). */
export function isDuePastOrToday(due: string | undefined | null): boolean {
  const d = parseDueDate(due);
  if (!d) return false;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return d.getTime() <= today.getTime();
}

export type BU = "BBRL" | "Credit Cards" | "Retail Advances - Excluding Credit Cards" | "Wholesale Banking";
export type Status = "Overdue" | "In Progress" | "On Track" | "Ahead" | "Completed";

/** DR keys used across the Launchpad + DR workspaces. */
export type DrKey = "Risk" | "BB&RL" | "Wholesale" | "Retail Assets";

export interface ChangeEvent {
  at: string;
  type: "created" | "status_change" | "due_changed" | "reassigned" | "note";
  from?: string;
  to?: string;
  by?: string;
  note?: string;
}

export interface ActionItem {
  id: string;
  title: string;
  owner: string;
  bu: BU;
  sourceMeeting: string;
  due: string;
  dueRaw?: string; // ISO date string (YYYY-MM-DD) for editing
  status: Status;
  origin: "seed" | "CRO" | "Retail" | "Wholesale" | "Retail Assets";
  flag?: "New" | "Updated";
  createdToday?: boolean;
  crossFunctional?: boolean;
  involvedBUs?: BU[];
  lastUpdated?: string;
  history?: ChangeEvent[];
  completed?: boolean;
}

export interface AppNotification {
  id: string;
  title: string;
  context: string;
  time: string;
  tone: "info" | "amber" | "crimson";
  target: { view: string; phase?: string };
}

/* -------- CEO Decision Desk (Launchpad) -------- */
export type Urgency = "overdue" | "today" | "this_week";
export type DecisionSource = "MoM" | "Email";

export interface PendingDecision {
  id: string;
  text: string;
  urgency: Urgency;
  source: DecisionSource;
  team: DrKey;
  dueDate?: string;        // optional human label
  dueRaw?: string;         // ISO date string (YYYY-MM-DD) for edit modal
  business_unit?: string;  // raw BU from backend for edit modal
  status: "open" | "resolved" | "dismissed";
  note?: string;
}

export type MotionStatus = "Overdue" | "In Progress" | "Ahead";
export interface DecisionInMotion {
  id: string;
  text: string;
  team: DrKey;
  status: MotionStatus;
}

/* -------- Cross-functional tag (created by tagging text in DR notes) -------- */
export interface CrossTag {
  id: string;
  text: string;
  team: DrKey;          // assigned team
  sourceDR: DrKey;      // DR page tagged from
  sourceMeeting: string;
  date: string;
  dueDate: string;
  status: Status;
}

export type ViewName = "Launchpad" | "CRO" | "Retail" | "Wholesale" | "RetailAssets" | "Ledger";

/** Map DR key (used in workspaces) → sidebar/route view name. */
export const DR_VIEW: Record<DrKey, ViewName> = {
  Risk: "CRO",
  "BB&RL": "Retail",
  Wholesale: "Wholesale",
  "Retail Assets": "RetailAssets",
};

/** Map BU (used in actions ledger) → DR view if any. */
export const BU_VIEW: Record<BU, ViewName | null> = {
  "BBRL": "Retail",
  "Credit Cards": "CRO",
  "Retail Advances - Excluding Credit Cards": "RetailAssets",
  "Wholesale Banking": "Wholesale",
};

interface AppStateValue {
  actions: ActionItem[];
  addActions: (items: ActionItem[]) => void;
  updateAction: (id: string, patch: Partial<ActionItem>) => void;
  removeAction: (id: string) => void;
  croGenerated: boolean;
  setCroGenerated: (b: boolean) => void;
  retailGenerated: boolean;
  setRetailGenerated: (b: boolean) => void;
  wholesaleGenerated: boolean;
  setWholesaleGenerated: (b: boolean) => void;
  retailAssetsGenerated: boolean;
  setRetailAssetsGenerated: (b: boolean) => void;
  notifications: AppNotification[];
  addNotification: (n: AppNotification) => void;
  counts: { open: number; overdue: number; inProgress: number; onTrack: number };
  toast: string | null;
  showToast: (msg: string) => void;
  readNotifIds: Set<string>;
  markNotifsRead: () => void;

  /* ---- New ---- */
  pending: PendingDecision[];
  addPending: (p: PendingDecision) => void;
  updatePending: (id: string, patch: Partial<PendingDecision>) => void;
  completePendingDecision: (p: PendingDecision) => void;
  refreshOpenTracker: () => Promise<void>;

  motion: DecisionInMotion[];
  addMotion: (m: DecisionInMotion) => void;
  updateMotion: (id: string, patch: Partial<DecisionInMotion>) => void;

  crossTags: CrossTag[];
  addCrossTag: (c: CrossTag) => void;
  updateCrossTag: (id: string, patch: Partial<CrossTag>) => void;

  questions: Record<DrKey, string[]>;
  setQuestions: (dr: DrKey, items: string[]) => void;

  /* ---- Cross-page focus/filter ---- */
  focusActionId: string | null;
  setFocusActionId: (id: string | null) => void;
  ledgerTeamFilter: BU | null;
  setLedgerTeamFilter: (bu: BU | null) => void;

  /* ---- Meeting MoM notes ---- */
  meetingNotes: Record<string, string>;
  addMeetingNote: (meetingKey: string, text: string) => void;

  /* ---- MoM log entries shown at top of Past Meetings ---- */
  momEntries: { meeting: string; date: string; time: string; note: string; status: "New" }[];
  addMomEntry: (meeting: string, note: string) => void;

  /* ---- Live backend data ---- */
  currentMeeting: BackendCalendarEvent | null;
  backendConnected: boolean;
  todayMeetings: Array<{ time: string; title: string }>; // NEW: Today's meetings for Log Notes

  /* ---- Per-meeting pre-meet brief (selected from Today's schedule) ---- */
  selectedMeetingBrief: BackendMeetingBrief | null;
  loadingMeetingBrief: boolean;
  meetingBriefError: string | null;
  /** Fetch (or reuse the cached) brief for a specific meeting id and store it. */
  loadMeetingBrief: (meetingId: string) => Promise<void>;
  clearMeetingBrief: () => void;

  /* ---- Meeting-scoped pre-meeting brief (which emails belong to the clicked meeting) ---- */
  /**
   * The meeting the user opened a pre-meeting brief for (from Today's
   * schedule). Carries the Graph event id/title so the destination
   * workspace can request the meeting-scoped email match rather than
   * showing the whole department's mailbox. null = no specific meeting
   * (tab opened directly), so the workspace falls back to the
   * department-wide view.
   */
  selectedMeeting: { id: string; title: string; view: string | null } | null;
  /**
   * Result of the backend embedding-based cluster match for
   * `selectedMeeting`. When `matched` is true, the workspace shows that
   * cluster's precomputed summary/questions/action_items directly.
   */
  premeetingBrief: BackendPreMeetingBrief | null;
  loadingPremeetingBrief: boolean;
  premeetingBriefError: string | null;
  /**
   * Record the clicked meeting and fetch its meeting-scoped brief
   * (GET /agents/premeeting-brief/{id}). Content-driven matching, so it
   * works for every meeting - no hardcoded meeting->email mapping.
   */
  loadPremeetingBrief: (meeting: { id: string; title: string; view: string | null }) => Promise<void>;
  clearPremeetingBrief: () => void;

  /**
   * Latest email-derived summary: one LLM call over the 5 most recent
   * emails involving the configured mailbox address (see
   * `HARDCODED_EMAIL_ADDRESS` in backend `.env`) produces a summary,
   * action items, and questions in one shot. Feeds "Questions
   * to ask", "Past meeting summaries", and "Open action
   * items" on the pre-meeting brief screens. Cached in sessionStorage
   * (see `EMAIL_SUMMARY_SESSION_KEY`) so a page refresh within the same
   * browser session doesn't require a fresh LLM call.
   */
  emailSummary: BackendEmailSummary | null;
  loadingEmailSummary: boolean;
  emailSummaryError: string | null;
  /** Force a fresh fetch (bypassing the sessionStorage cache) and re-cache the result. */
  refreshEmailSummary: () => Promise<void>;

  /* ---- Auth (delegated SSO session) ---- */
  authState: AuthState | null;
  authChecked: boolean;
  /**
   * Non-null when the auth check could not be *resolved* (timeout,
   * network failure, or a 5xx from `/auth/me`) - as opposed to a clean
   * "signed out" 401. Distinct so the gate can show an error/retry screen
   * instead of silently redirecting into another slow sign-in attempt.
   */
  authError: string | null;
  /** Re-run the `/auth/me` check (used to retry after an error). */
  retryAuthCheck: () => void;
  /**
   * True once we've determined the user is unauthenticated AND already
   * used up this session's one automatic redirect to Microsoft sign-in
   * (see `maybeRedirectToLogin`). Callers (the auth loading gate) use
   * this to stop showing a loading spinner and instead offer a manual
   * "Sign in" action - otherwise a stuck/failed sign-in would spin
   * forever with no way for the user to retry.
   */
  loginRedirectExhausted: boolean;
  signIn: () => void;
}

const AppStateContext = createContext<AppStateValue | null>(null);

/* -------- Seeds for the new entities -------- */
const SEED_PENDING: PendingDecision[] = [];

const SEED_MOTION: DecisionInMotion[] = [];

const SEED_QUESTIONS: Record<DrKey, string[]> = {
  Risk: [],
  "BB&RL": [],
  Wholesale: [],
  "Retail Assets": [],
};

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [actions, setActions] = useState<ActionItem[]>(SEED_ACTIONS);
  const [notifications, setNotifications] = useState<AppNotification[]>(SEED_NOTIFICATIONS);
  const [croGenerated, setCroGenerated] = useState(false);
  const [retailGenerated, setRetailGenerated] = useState(false);
  const [wholesaleGenerated, setWholesaleGenerated] = useState(false);
  const [retailAssetsGenerated, setRetailAssetsGenerated] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [readNotifIds, setReadNotifIds] = useState<Set<string>>(new Set());

  const [pending, setPending] = useState<PendingDecision[]>(SEED_PENDING);
  const [motion, setMotion] = useState<DecisionInMotion[]>(SEED_MOTION);
  const [crossTags, setCrossTags] = useState<CrossTag[]>([]);
  const [questions, setQuestionsState] = useState<Record<DrKey, string[]>>(SEED_QUESTIONS);
  const [focusActionId, setFocusActionId] = useState<string | null>(null);
  const [ledgerTeamFilter, setLedgerTeamFilter] = useState<BU | null>(null);
  const [meetingNotes, setMeetingNotes] = useState<Record<string, string>>({});
  const addMeetingNote = useCallback((k: string, t: string) => setMeetingNotes((p) => ({ ...p, [k]: t })), []);
  const [momEntries, setMomEntries] = useState<{ meeting: string; date: string; time: string; note: string; status: "New" }[]>([]);
  const addMomEntry = useCallback((meeting: string, note: string) => {
    const now = new Date();
    const date = now.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
    const time = now.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", hour12: false });
    setMomEntries((prev) => {
      // Idempotent: skip an exact (meeting + note) duplicate so re-hydrating
      // persisted notes on panel open doesn't stack repeated MoM entries.
      if (prev.some((e) => e.meeting === meeting && e.note === note)) return prev;
      return [{ meeting, date, time, note, status: "New" }, ...prev];
    });
  }, []);

  /* ---- Live backend data ---- */
  const [currentMeeting, setCurrentMeeting] = useState<BackendCalendarEvent | null>(null);
  const [backendConnected, setBackendConnected] = useState(false);
  const [todayMeetings, setTodayMeetings] = useState<Array<{ time: string; title: string }>>([]);

  /* ---- Per-meeting pre-meet brief ---- */
  const [selectedMeetingBrief, setSelectedMeetingBrief] = useState<BackendMeetingBrief | null>(null);
  const [loadingMeetingBrief, setLoadingMeetingBrief] = useState(false);
  const [meetingBriefError, setMeetingBriefError] = useState<string | null>(null);

  const loadMeetingBrief = useCallback(async (meetingId: string) => {
    setLoadingMeetingBrief(true);
    setMeetingBriefError(null);
    try {
      const res = await fetchMeetingBriefById(meetingId);
      setSelectedMeetingBrief(res.brief);
    } catch (err) {
      setSelectedMeetingBrief(null);
      setMeetingBriefError(err instanceof Error ? err.message : "Couldn't load brief for this meeting.");
    } finally {
      setLoadingMeetingBrief(false);
    }
  }, []);

  const clearMeetingBrief = useCallback(() => {
    setSelectedMeetingBrief(null);
    setMeetingBriefError(null);
  }, []);

  /* ---- Meeting-scoped pre-meeting brief (email match for the clicked meeting) ---- */
  const [selectedMeeting, setSelectedMeeting] = useState<{ id: string; title: string; view: string | null } | null>(null);
  const [premeetingBrief, setPremeetingBrief] = useState<BackendPreMeetingBrief | null>(null);
  const [loadingPremeetingBrief, setLoadingPremeetingBrief] = useState(false);
  const [premeetingBriefError, setPremeetingBriefError] = useState<string | null>(null);

  const loadPremeetingBrief = useCallback(
    async (meeting: { id: string; title: string; view: string | null }) => {
      // TRACE: this is the client-side counterpart to the backend
      // [PMB-MATCH] logs. Together they let you follow one click end to
      // end: which meeting -> which endpoint -> which emails came back.
      console.info(
        "[PMB] loadPremeetingBrief: meeting=%o -> GET /agents/premeeting-brief/%s",
        meeting, meeting.id,
      );
      setSelectedMeeting(meeting);
      setPremeetingBrief(null);
      setLoadingPremeetingBrief(true);
      setPremeetingBriefError(null);
      try {
        const res = await fetchPremeetingBriefForMeeting(meeting.id);
        setPremeetingBrief(res.brief);
        console.info(
          "[PMB] meeting=%s title=%s -> matched=%s dept=%s topic=%s email_refs=%d",
          meeting.id,
          res.brief.meeting_title,
          res.brief.matched,
          res.brief.department,
          res.brief.topic,
          res.brief.email_refs?.length ?? 0,
        );
        (res.brief.email_refs ?? []).forEach((r) =>
          console.info("[PMB]   source -> date=%s subject=%s from=%s", r.date, r.subject, r.from_address),
        );
        if (!res.brief.matched) {
          console.warn(
            "[PMB] meeting=%s produced no cluster match; workspace falls back to department-wide clusters.",
            meeting.id,
          );
        }
      } catch (err) {
        setPremeetingBrief(null);
        setPremeetingBriefError(err instanceof Error ? err.message : "Couldn't load the meeting-scoped brief.");
        console.error("[PMB] loadPremeetingBrief failed for meeting=%s: %o", meeting.id, err);
      } finally {
        setLoadingPremeetingBrief(false);
      }
    },
    [],
  );

  const clearPremeetingBrief = useCallback(() => {
    setSelectedMeeting(null);
    setPremeetingBrief(null);
    setPremeetingBriefError(null);
  }, []);

  /* ---- Email-derived summary (top questions / summary / action items) ---- */
  const [emailSummary, setEmailSummary] = useState<BackendEmailSummary | null>(() => {
    const cached = readCachedEmailSummary();
    return cached ? sanitizeEmailSummary(cached) : null;
  });
  const [loadingEmailSummary, setLoadingEmailSummary] = useState(false);
  const [emailSummaryError, setEmailSummaryError] = useState<string | null>(null);

  const refreshEmailSummary = useCallback(async () => {
    setLoadingEmailSummary(true);
    setEmailSummaryError(null);
    try {
      const res = await fetchEmailSummary();
      if (res.summary) {
        const sanitized = sanitizeEmailSummary(res.summary);
        setEmailSummary(sanitized);
        writeCachedEmailSummary(sanitized);
      } else {
        setEmailSummary(null);
        writeCachedEmailSummary(null);
      }
    } catch (err) {
      setEmailSummaryError(err instanceof Error ? err.message : "Couldn't load the latest email summary.");
    } finally {
      setLoadingEmailSummary(false);
    }
  }, []);

  /* ---- Auth (delegated SSO session) ---- */
  const [authState, setAuthState] = useState<AuthState | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [loginRedirectExhausted, setLoginRedirectExhausted] = useState(false);
  const signIn = useCallback(() => {
    sessionStorage.removeItem("rbl_login_attempted");
    setLoginRedirectExhausted(false);
    redirectToLogin();
  }, []);

  // Auto-redirect to sign-in exactly once per app load. If we still land
  // here unauthenticated *after* having already bounced (a login that
  // didn't stick), stop and let the user retry via signIn() rather than
  // looping between the SPA and Microsoft.
  const maybeRedirectToLogin = useCallback(() => {
    const KEY = "rbl_login_attempted";
    if (sessionStorage.getItem(KEY) === "1") {
      setLoginRedirectExhausted(true);
      return;
    }
    sessionStorage.setItem(KEY, "1");
    redirectToLogin();
  }, []);

  // Check auth state on mount before loading protected data. When the
  // user is not authenticated, bounce them into the Microsoft sign-in
  // flow on the backend (full-page navigation). On success the backend
  // redirects back to this SPA (FRONTEND_URL), we re-check, and load data.
  const runAuthCheck = useCallback(() => {
    let cancelled = false;
    setAuthError(null);
    (async () => {
      try {
        const state = await fetchAuthState();
        if (cancelled) return;
        setAuthState(state);
        setAuthChecked(true);
        if (state.authenticated) {
          // Signed in: clear the one-shot redirect guard for next time.
          sessionStorage.removeItem("rbl_login_attempted");
        } else {
          // Known unauthenticated (a clean 401): bounce to Microsoft once.
          maybeRedirectToLogin();
        }
      } catch (err) {
        if (cancelled) return;
        // We could NOT resolve the auth state (timeout / network / 5xx).
        // This is an error state, NOT "signed out": redirecting to
        // /auth/login here would just stack another slow round trip onto
        // an already-unresponsive backend. Surface an error the user can
        // retry instead.
        const message =
          err instanceof AuthCheckError
            ? err.message
            : err instanceof Error
              ? err.message
              : "Unknown error checking sign-in status";
        setAuthState(null);
        setAuthError(message);
        setAuthChecked(true);
      }
    })();
    return () => { cancelled = true; };
  }, [maybeRedirectToLogin]);

  // Check auth state on mount, before rendering the app or loading any
  // protected data.
  useEffect(() => runAuthCheck(), [runAuthCheck]);

  // Manual retry after an error: fall back to the loading state and
  // re-issue the (timeout-bounded) /auth/me check.
  const retryAuthCheck = useCallback(() => {
    setAuthChecked(false);
    setAuthError(null);
    runAuthCheck();
  }, [runAuthCheck]);

  // ---- Sync open tracker items → actions array ----
  // Maps department names from email_clusters.csv to BU types used by the Ledger.
  const TRACKER_BU_MAP: Record<string, BU> = {
    "BB&RL":                                     "BBRL",
    "Wholesale Banking":                         "Wholesale Banking",
    "Retail Advances - Excluding Credit Cards":  "Retail Advances - Excluding Credit Cards",
    "Credit Cards":                              "Credit Cards",
  };

  // Shared mapper: backend tracker item (OpenTrackerItem or PendingDecisionItem,
  // both share the same shape) → Ledger ActionItem. Used for both the regular
  // Open Actions Status Tracker items and the "Awaiting Your Decision" items,
  // so both kinds of items show up in the Ledger's Open tab consistently.
  const mapTrackerItemToAction = useCallback(
    (item: OpenTrackerItem | PendingDecisionItem): ActionItem => {
      const bu: BU = TRACKER_BU_MAP[item.business_unit] ?? "BBRL";
      const ledgerStatus: Status =
        item.status === "Overdue" ? "Overdue"
          : item.status === "Due Today" ? "In Progress"
          : "On Track";
      return {
        id: `tracker-${item.id}`,
        title: (() => {
          // Strip trailing date suffixes the backend may append to text:
          //   "(by 2026-09-30)"  — canonical format
          //   " - 2026-07-18"    — bare ISO tail
          //   " - 18-Apr-2026"   — bare DMY-hyphen tail
          //   " - 18 Apr 2026"   — bare DMY-space tail
          const rawText = item.text
            .replace(/\s*\(by [^)]+\)\s*$/, "")
            .replace(/\s*[-–]\s*(?:due\s+)?(\d{4}-\d{2}-\d{2}|\d{1,2}-[A-Za-z]+-\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})\s*$/, "")
            .trim();
          return item.owner && item.owner !== "Unassigned"
            ? `${item.owner}: ${rawText}`
            : rawText;
        })(),
        owner: item.owner || "Unassigned",
        bu,
        sourceMeeting: item.source_topic || item.business_unit,
        due: item.due_date
          ? (() => {
              const [y, mo, d] = item.due_date.split("-").map(Number);
              return new Date(y, mo - 1, d).toLocaleDateString("en-IN", {
                day: "2-digit", month: "short", year: "numeric",
              });
            })()
          : "No due date",
        dueRaw: item.due_date || undefined, // Store raw ISO date for editing
        status: ledgerStatus,
        origin: "seed" as const,
        completed: false,
        lastUpdated: item.due_label,
      };
    },
    [],
  );

  const syncTrackerItems = useCallback((trackerItems: OpenTrackerItem[]) => {
    const freshItems: ActionItem[] = trackerItems.map(mapTrackerItemToAction);
    setActions((prev) => {
      const prevById = new Map(prev.map((a) => [a.id, a]));
      let changed = false;

      // Update existing tracker items in-place (pick up due-date /
      // status reclassification from backend) but preserve user-set
      // `completed` flag.
      const updated = prev.map((a) => {
        const fresh = freshItems.find((f) => f.id === a.id);
        if (fresh && !a.completed) {
          changed = true;
          return { ...a, status: fresh.status, due: fresh.due, dueRaw: fresh.dueRaw, lastUpdated: fresh.lastUpdated };
        }
        return a;
      });

      // Dedup `updated` by title — runs on every sync so duplicates that
      // crept in during an earlier load get cleaned out. Non-tracker items
      // (seed/manual) are keyed by id and never deduplicated.
      const titleKey = (title: string) => title.trim().toLowerCase().replace(/\s+/g, " ");
      const bestByTitle = new Map<string, ActionItem>();
      for (const a of updated) {
        if (!a.id.startsWith("tracker-")) { bestByTitle.set(a.id, a); continue; }
        const k = titleKey(a.title);
        const existing = bestByTitle.get(k);
        if (!existing) { bestByTitle.set(k, a); continue; }
        const existingMs = parseDueDate(existing.due)?.getTime() ?? 0;
        const newMs = parseDueDate(a.due)?.getTime() ?? 0;
        if (newMs >= existingMs) bestByTitle.set(k, a);
        changed = true; // a duplicate was dropped
      }
      const dedupedUpdated = Array.from(bestByTitle.values());

      // Add genuinely new items not yet in the array.
      const dedupedIds = new Set(dedupedUpdated.map((a) => a.id));
      const existingTitles = new Map<string, number>();
      for (const a of dedupedUpdated) {
        if (!a.id.startsWith("tracker-")) continue;
        const k = titleKey(a.title);
        const ms = parseDueDate(a.due)?.getTime() ?? 0;
        const ex = existingTitles.get(k);
        if (ex === undefined || ms > ex) existingTitles.set(k, ms);
      }

      let newItems = freshItems.filter((m) => !prevById.has(m.id) && !dedupedIds.has(m.id));
      const deduped: ActionItem[] = [];
      const seenNewTitles = new Map<string, number>();
      for (const item of newItems) {
        const k = titleKey(item.title);
        const ms = parseDueDate(item.due)?.getTime() ?? 0;
        if (existingTitles.has(k)) continue; // title already in state
        const seenMs = seenNewTitles.get(k);
        if (seenMs === undefined || ms > seenMs) {
          seenNewTitles.set(k, ms);
          deduped.push(item);
        } else {
          const idx = deduped.findIndex((x) => titleKey(x.title) === k);
          if (idx !== -1) deduped[idx] = item;
        }
      }
      newItems = deduped;

      if (newItems.length > 0) changed = true;

      return changed ? [...dedupedUpdated, ...newItems] : prev;
    });
  }, [mapTrackerItemToAction]);

  // Map backend PendingDecisionItem → PendingDecision and merge into pending state.
  // Replaces all backend-sourced items (tracker-pd-*) with fresh data and preserves manually-added items.
  const syncPendingDecisions = useCallback((decisionItems: PendingDecisionItem[]) => {
    const BU_TO_DR: Record<string, DrKey> = {
      "Wholesale Banking": "Wholesale",
      "BB&RL": "BB&RL",
      "Retail Advances - Excluding Credit Cards": "Retail Assets",
      "Credit Cards": "Retail Assets",
    };
    setPending((prev) => {
      // Keep only manually-added items (non tracker-pd-*); drop all stale backend items.
      const manual = prev.filter((p) => !p.id.startsWith("tracker-pd-"));

      // Build fresh backend items, preserving status/note from previous state if present.
      const prevById = new Map(prev.map((p) => [p.id, p]));
      const fresh: PendingDecision[] = decisionItems.map((d) => {
        const existingId = `tracker-pd-${d.id}`;
        const existing = prevById.get(existingId);
        return {
          id: existingId,
          text: d.owner && d.owner !== "Unassigned" ? `${d.owner}: ${d.text}` : d.text,
          urgency: (d.status === "Overdue" ? "overdue" : d.status === "Due Today" ? "today" : "this_week") as Urgency,
          source: "MoM" as const,
          team: BU_TO_DR[d.business_unit] ?? "Wholesale",
          dueDate: d.due_label || undefined,
          dueRaw: d.due_date || undefined,
          business_unit: d.business_unit || undefined,
          // Always reset to "open" so backend-sourced items reappear on re-sync.
          status: "open" as const,
          note: existing?.note,
        };
      });

      return [...manual, ...fresh];
    });

    // Also inject these items into the Ledger's Open tab (as `tracker-{id}`,
    // same id convention as syncTrackerItems / completePendingDecision) so
    // "Awaiting Your Decision" items show up there too — while staying out
    // of the Open Actions Status Tracker card, which reads from the tracker
    // `items` prop, not from `actions`.
    const mappedForLedger = decisionItems.map(mapTrackerItemToAction);
    setActions((prev) => {
      const prevById = new Map(prev.map((a) => [a.id, a]));
      let changed = false;

      // Update existing pd items in-place (pick up reclassification).
      const updated = prev.map((a) => {
        const fresh = mappedForLedger.find((f) => f.id === a.id);
        if (fresh && !a.completed) {
          changed = true;
          return { ...a, status: fresh.status, due: fresh.due, dueRaw: fresh.dueRaw, lastUpdated: fresh.lastUpdated };
        }
        return a;
      });

      const updatedIds = new Set(updated.map((a) => a.id));
      let newItems = mappedForLedger.filter((m) => !prevById.has(m.id) && !updatedIds.has(m.id));

      // Dedup `updated` by title on every sync — cleans out duplicates that
      // crept in from earlier loads. Non-tracker items are keyed by id.
      const titleKey = (title: string) => title.trim().toLowerCase().replace(/\s+/g, " ");
      const bestByTitle = new Map<string, ActionItem>();
      for (const a of updated) {
        if (!a.id.startsWith("tracker-")) { bestByTitle.set(a.id, a); continue; }
        const k = titleKey(a.title);
        const existing = bestByTitle.get(k);
        if (!existing) { bestByTitle.set(k, a); continue; }
        const existingMs = parseDueDate(existing.due)?.getTime() ?? 0;
        const newMs = parseDueDate(a.due)?.getTime() ?? 0;
        if (newMs >= existingMs) bestByTitle.set(k, a);
        changed = true;
      }
      const dedupedUpdated = Array.from(bestByTitle.values());
      const dedupedIds = new Set(dedupedUpdated.map((a) => a.id));
      const existingTitles = new Map<string, number>();
      for (const a of dedupedUpdated) {
        if (!a.id.startsWith("tracker-")) continue;
        const k = titleKey(a.title);
        const ms = parseDueDate(a.due)?.getTime() ?? 0;
        const ex = existingTitles.get(k);
        if (ex === undefined || ms > ex) existingTitles.set(k, ms);
      }
      newItems = mappedForLedger.filter((m) => !prevById.has(m.id) && !dedupedIds.has(m.id));
      const deduped: ActionItem[] = [];
      const seenNewTitles = new Map<string, number>();
      for (const item of newItems) {
        const k = titleKey(item.title);
        const ms = parseDueDate(item.due)?.getTime() ?? 0;
        if (existingTitles.has(k)) continue;
        const seenMs = seenNewTitles.get(k);
        if (seenMs === undefined || ms > seenMs) {
          seenNewTitles.set(k, ms);
          deduped.push(item);
        } else {
          const idx = deduped.findIndex((x) => titleKey(x.title) === k);
          if (idx !== -1) deduped[idx] = item;
        }
      }
      newItems = deduped;

      if (newItems.length > 0) changed = true;

      return changed ? [...dedupedUpdated, ...newItems] : prev;
    });
  }, [mapTrackerItemToAction]);

  // Immediately re-fetches the open tracker + pending decisions and syncs
  // them into state. Used to force an instant refresh right after a
  // complete/uncomplete action (e.g. from the Ledger checkbox), instead of
  // waiting for the 60s poll — so "Awaiting Your Decision" on the
  // Executive Launchpad reflects the change right away.
  const refreshOpenTracker = useCallback(async () => {
    try {
      const trackerRes = await fetchOpenTracker();
      syncTrackerItems(trackerRes.items ?? []);
      syncPendingDecisions(trackerRes.pending_decisions ?? []);
    } catch {
      // Backend not available — ignore, the 60s poll will retry.
    }
  }, [syncTrackerItems, syncPendingDecisions]);

  // Fetch data from backend once authenticated.
  useEffect(() => {
    if (!authState?.authenticated) return;
    let cancelled = false;

    async function loadBackendData() {
      // NOTE: fetchActionItems() was removed here to fix a duplicate-items
      // bug.  That endpoint (/agents/actions) returned the same logical
      // items as /agents/open-tracker but with different IDs, causing
      // duplicates in the Ledger.  The open-tracker path (syncTrackerItems)
      // is the single source of truth for action items now.

      try {
        // Fetch current meeting
        const calRes = await fetchCurrentMeeting();
        if (!cancelled) {
          setCurrentMeeting(calRes.current_meeting);
          setBackendConnected(true);
        }
      } catch {
        if (!cancelled) setBackendConnected(false);
      }

      // Fetch today's meetings for Log Notes panel
      try {
        const todayRes = await fetchTodayEvents();
        if (!cancelled && todayRes.events.length > 0) {
          const meetings = todayRes.events.map((e) => ({
            time: new Date(e.start).toLocaleTimeString("en-IN", { 
              hour: "2-digit", 
              minute: "2-digit", 
              hour12: false,
              timeZone: "Asia/Kolkata",
            }),
            title: e.subject,
          }));
          setTodayMeetings(meetings);
        }
      } catch {
        // Silently fail - use empty meetings list
      }

      // Email-derived summary (top questions / action items / summary):
      // only fetch if nothing is cached in sessionStorage yet, so a page
      // refresh within the same session doesn't trigger another LLM call.
      if (!readCachedEmailSummary() && !cancelled) {
        setLoadingEmailSummary(true);
        setEmailSummaryError(null);
        try {
          const emailRes = await fetchEmailSummary();
          if (!cancelled) {
            if (emailRes.summary) {
              const sanitized = sanitizeEmailSummary(emailRes.summary);
              setEmailSummary(sanitized);
              writeCachedEmailSummary(sanitized);
            } else {
              setEmailSummary(null);
            }
          }
        } catch (err) {
          if (!cancelled) {
            setEmailSummaryError(err instanceof Error ? err.message : "Couldn't load the latest email summary.");
          }
        } finally {
          if (!cancelled) setLoadingEmailSummary(false);
        }
      }

      // Fetch open tracker items and inject them into the actions array
      // so the Ledger can display them even when Launchpad is not mounted.
      try {
        const trackerRes = await fetchOpenTracker();
        if (!cancelled && trackerRes.items.length > 0) {
          syncTrackerItems(trackerRes.items);
        }
        if (!cancelled && trackerRes.pending_decisions?.length > 0) {
          syncPendingDecisions(trackerRes.pending_decisions);
        }
      } catch {
        // Backend not available — skip tracker sync
      }
    }

    loadBackendData();
    // Refresh current meeting every 60s
    const interval = setInterval(async () => {
      try {
        const calRes = await fetchCurrentMeeting();
        if (!cancelled) {
          setCurrentMeeting(calRes.current_meeting);
          setBackendConnected(true);
        }
      } catch {
        // ignore
      }
    }, 60_000);

    // Refresh open tracker items every 60s so the Ledger stays current
    const trackerInterval = setInterval(async () => {
      try {
        const trackerRes = await fetchOpenTracker();
        if (!cancelled && trackerRes.items.length > 0) {
          syncTrackerItems(trackerRes.items);
        }
        if (!cancelled && trackerRes.pending_decisions?.length > 0) {
          syncPendingDecisions(trackerRes.pending_decisions);
        }
      } catch {
        // ignore
      }
    }, 60_000);

    return () => { cancelled = true; clearInterval(interval); clearInterval(trackerInterval); };
  }, [authState?.authenticated]);

  const addActions = useCallback((items: ActionItem[]) => {
    setActions((prev) => {
      const ids = new Set(prev.map((a) => a.id));
      const merged = [...prev];
      for (const it of items) {
        if (ids.has(it.id)) continue;
        const overdueAtCreate = isDuePastOrToday(it.due);
        const status: Status = overdueAtCreate ? "Overdue" : it.status;
        const seeded: ActionItem = {
          ...it,
          status,
          lastUpdated: it.lastUpdated ?? "today · just now",
          history: it.history && it.history.length
            ? it.history
            : [{ at: it.lastUpdated ?? "today · just now", type: "created", by: it.sourceMeeting }],
        };
        merged.push(seeded);
      }
      return merged;
    });
  }, []);

  // Converts a completed PendingDecision into an ActionItem and injects it into
  // the actions array so it shows up in the Ledger's Completed tab.
  const completePendingDecision = useCallback((p: PendingDecision) => {
    // pending id is "tracker-pd-{backendId}", actions id is "tracker-{backendId}"
    const backendId = p.id.replace("tracker-pd-", "");
    const actionId = `tracker-${backendId}`;
    const bu: BU = (p.business_unit ? TRACKER_BU_MAP[p.business_unit] : undefined) ?? "BBRL";
    const now = new Date().toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" });

    setActions((prev) => {
      const existing = prev.find((a) => a.id === actionId);
      if (existing) {
        // Already in Ledger — just flip completed flag
        return prev.map((a) =>
          a.id === actionId ? { ...a, completed: true, status: "Completed" as Status } : a
        );
      }
      // Not yet in Ledger — inject a fresh completed entry
      const newItem: ActionItem = {
        id: actionId,
        title: p.text,
        owner: p.text.split(":")[0]?.trim() || "Unassigned",
        bu,
        sourceMeeting: p.business_unit || p.team,
        due: p.dueDate || now,
        dueRaw: p.dueRaw,
        status: "Completed" as Status,
        origin: "seed" as const,
        completed: true,
        lastUpdated: now,
        history: [{ at: now, type: "status_change", from: "open", to: "Completed", by: "CEO" }],
      };
      return [...prev, newItem];
    });
  }, []);

  // Re-evaluate overdue status on every mount/page load
  useEffect(() => {
    setActions((prev) => {
      let changed = false;
      const next = prev.map((a) => {
        if (a.status !== "Overdue" && isDuePastOrToday(a.due)) {
          changed = true;
          return { ...a, status: "Overdue" as Status };
        }
        return a;
      });
      return changed ? next : prev;
    });
    setCrossTags((prev) => {
      let changed = false;
      const next = prev.map((c) => {
        if (c.status !== "Overdue" && isDuePastOrToday(c.dueDate)) {
          changed = true;
          return { ...c, status: "Overdue" as Status };
        }
        return c;
      });
      return changed ? next : prev;
    });
  }, []);

  const updateAction = useCallback((id: string, patch: Partial<ActionItem>) => {
    setActions((prev) =>
      prev.map((a) => {
        if (a.id !== id) return a;
        let type: ChangeEvent["type"] = "note";
        let from: string | undefined;
        let to: string | undefined;
        if (patch.status && patch.status !== a.status) {
          type = "status_change"; from = a.status; to = patch.status;
        } else if (patch.due && patch.due !== a.due) {
          type = "due_changed"; from = a.due; to = patch.due;
        } else if (patch.owner && patch.owner !== a.owner) {
          type = "reassigned"; from = a.owner; to = patch.owner;
        }
        const ev: ChangeEvent = { at: "today · just now", type, from, to, by: "CEO 1:1 / today" };
        return { ...a, ...patch, lastUpdated: "today · just now", history: [...(a.history ?? []), ev] };
      })
    );
  }, []);

  // Removes an action item from the Ledger entirely. Used when undoing a
  // completion — rather than guessing a re-derived status client-side, the
  // synthesized "Completed" entry is dropped and the item is left to be
  // rebuilt fresh from the next backend poll (with the correct due-date-
  // derived status).
  const removeAction = useCallback((id: string) => {
    setActions((prev) => prev.filter((a) => a.id !== id));
  }, []);

  const addNotification = useCallback((n: AppNotification) => {
    setNotifications((prev) => [n, ...prev]);
  }, []);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 4000);
  }, []);

  const markNotifsRead = useCallback(() => {
    setReadNotifIds(new Set(notifications.map((n) => n.id)));
  }, [notifications]);

  const addPending = useCallback((p: PendingDecision) => setPending((prev) => [...prev, p]), []);
  const updatePending = useCallback((id: string, patch: Partial<PendingDecision>) =>
    setPending((prev) => prev.map((p) => (p.id === id ? { ...p, ...patch } : p))), []);

  const addMotion = useCallback((m: DecisionInMotion) => setMotion((prev) => [...prev, m]), []);
  const updateMotion = useCallback((id: string, patch: Partial<DecisionInMotion>) =>
    setMotion((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m))), []);

  const addCrossTag = useCallback((c: CrossTag) => setCrossTags((prev) => [...prev, isDuePastOrToday(c.dueDate) ? { ...c, status: "Overdue" as Status } : c]), []);
  const updateCrossTag = useCallback((id: string, patch: Partial<CrossTag>) =>
    setCrossTags((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch } : c))), []);

  const setQuestions = useCallback((dr: DrKey, items: string[]) => {
    setQuestionsState((prev) => ({ ...prev, [dr]: items }));
  }, []);

  const counts = useMemo(
    () => ({
      open: actions.length,
      overdue: actions.filter((a) => a.status === "Overdue").length,
      inProgress: actions.filter((a) => a.status === "In Progress").length,
      onTrack: actions.filter((a) => a.status === "On Track").length,
    }),
    [actions]
  );

  const value: AppStateValue = {
    actions, addActions, updateAction, removeAction,
    croGenerated, setCroGenerated,
    retailGenerated, setRetailGenerated,
    wholesaleGenerated, setWholesaleGenerated,
    retailAssetsGenerated, setRetailAssetsGenerated,
    notifications, addNotification,
    counts, toast, showToast,
    readNotifIds, markNotifsRead,
    pending, addPending, updatePending, completePendingDecision, refreshOpenTracker,
    motion, addMotion, updateMotion,
    crossTags, addCrossTag, updateCrossTag,
    questions, setQuestions,
    focusActionId, setFocusActionId,
    ledgerTeamFilter, setLedgerTeamFilter,
    meetingNotes, addMeetingNote,
    momEntries, addMomEntry,
    currentMeeting, backendConnected, todayMeetings,
    selectedMeetingBrief, loadingMeetingBrief, meetingBriefError, loadMeetingBrief, clearMeetingBrief,
    selectedMeeting, premeetingBrief, loadingPremeetingBrief, premeetingBriefError, loadPremeetingBrief, clearPremeetingBrief,
    emailSummary, loadingEmailSummary, emailSummaryError, refreshEmailSummary,
    authState, authChecked, authError, retryAuthCheck, loginRedirectExhausted, signIn,
  };

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const v = useContext(AppStateContext);
  if (!v) throw new Error("useAppState must be used inside AppStateProvider");
  return v;
}
