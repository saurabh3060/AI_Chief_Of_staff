/**
 * api.ts
 * ======
 * Typed fetch wrapper for all backend endpoints.
 * Base URL defaults to the Vite proxy prefix `/api` so
 * requests go through the dev server proxy in development.
 */

// Base URL for API + auth calls.
//   - Dev: leave VITE_API_BASE unset -> "/api", which the Vite dev server
//     proxies to the backend on :8516 (see vite.config.ts).
//   - Prod (cross-origin, Option B): set VITE_API_BASE to the backend's
//     absolute origin, e.g. "https://digitaoffice-dev.rbl.bank.in", so
//     both XHR and the full-page /auth/login navigation hit the backend
//     directly.
const BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") || "/api";

/**
 * Thrown when the backend reports the caller is not authenticated (HTTP
 * 401). Callers/UI can catch this specifically to trigger a redirect to
 * the Microsoft sign-in flow (`redirectToLogin()`).
 */
export class UnauthorizedError extends Error {
  constructor(path: string) {
    super(`Unauthorized: ${path}`);
    this.name = "UnauthorizedError";
  }
}

// Every request includes credentials so the HTTP-only session cookie set
// by /auth/callback rides along (the SPA is on a different origin, so
// the cookie only flows when credentials are explicitly included).
const CREDENTIALS: RequestCredentials = "include";

// ── Shared helpers ───────────────────────────────────────────────

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { credentials: CREDENTIALS });
  if (res.status === 401) throw new UnauthorizedError(path);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    credentials: CREDENTIALS,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (res.status === 401) throw new UnauthorizedError(path);
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
  return res.json();
}

async function patch<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "PATCH",
    credentials: CREDENTIALS,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (res.status === 401) throw new UnauthorizedError(path);
  if (!res.ok) throw new Error(`PATCH ${path} failed: ${res.status}`);
  return res.json();
}

async function del<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "DELETE",
    credentials: CREDENTIALS,
  });
  if (res.status === 401) throw new UnauthorizedError(path);
  if (!res.ok) throw new Error(`DELETE ${path} failed: ${res.status}`);
  return res.json();
}

// ── Backend response types ───────────────────────────────────────

export interface BackendCalendarEvent {
  id: string;
  subject: string;
  start: string;
  end: string;
  organizer_name: string | null;
  organizer_email: string | null;
  attendee_emails: string[];
  matched_view: string | null;
}

export interface BackendMeetingBrief {
  meeting_id: string;
  meeting_subject: string;
  meeting_start: string;
  generated_at: string;
  brief_text: string;
  questions: string[];
  source_thread_ids?: string[];
}

export interface BackendEmailItemSummary {
  message_id: string;
  subject: string;
  from_name: string | null;
  from_address: string | null;
  received_at: string;
  summary_text: string;
  action_items: string[];
  top_questions: string[];
}

export interface BackendEmailSummary {
  thread_id: string;
  subject: string;
  generated_at: string;
  summary_text: string;
  action_items: string[];
  top_questions: string[];
  /** Per-individual-email breakdown (one entry per message in the thread). */
  email_summaries: BackendEmailItemSummary[];
  action_candidates: { text: string; source_message_id: string | null }[];
}

/**
 * One row of `email_clusters.csv` — a synthesized topic cluster with
 * precomputed summary, questions, action items, and embedding.
 */
export interface BackendEmailClusterRecord {
  cluster_id: string;
  department: string;
  topic: string;
  /** "|"-joined summary bullets — see `splitBuField`. */
  summary: string;
  /** "|"-joined top questions — see `splitBuField`. */
  top_questions: string;
  /** "|"-joined action items — see `splitBuField`. */
  action_items: string;
  /** JSON list of BackendClusterEmailRef objects. */
  email_refs: string;
  email_count: number;
  updated_at: string;
}

/** Split a "|"-joined CSV field into a clean list, dropping empty entries. */
export function splitBuField(value: string): string[] {
  return value.split("|").map((s) => s.trim()).filter(Boolean);
}

/** Parse the JSON `email_refs` column from a cluster record. */
export function parseClusterEmailRefs(raw: string): BackendClusterEmailRef[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is BackendClusterEmailRef =>
      typeof item === "object" && item !== null && "subject" in item,
    );
  } catch {
    return [];
  }
}

export interface BackendActionItem {
  id: string;
  text: string;
  source_thread_id: string;
  source_subject: string;
  owner: string | null;
  due_date: string | null;
  status: "Overdue" | "In Progress" | "On Track" | "Completed";
  classified_at: string;
}

export interface BackendChatResponse {
  answer: string;
  tool_used: string | null;
  sources?: ChatSource[];
  is_greeting?: boolean;
  follow_up_questions?: string[];
  timings?: Record<string, number>;
  latency_ms?: number;
}

// ── Health ────────────────────────────────────────────────────────

export function fetchHealth() {
  return get<{ status: string; app: string; env: string }>("/health");
}

// ── Calendar ─────────────────────────────────────────────────────

export function fetchCurrentMeeting() {
  return get<{ current_meeting: BackendCalendarEvent | null; as_of: string }>("/agents/calendar/current");
}

export function fetchNextMeeting() {
  return get<{ next_meeting: BackendCalendarEvent | null; as_of: string }>("/agents/calendar/next");
}

export function fetchTodayEvents() {
  return get<{ events: BackendCalendarEvent[]; as_of: string }>("/agents/calendar/today");
}

export function fetchMeetingBrief() {
  return get<{ brief: BackendMeetingBrief | null; message?: string }>("/agents/calendar/brief");
}

/** Fetch the pre-meet brief for a specific meeting on today's calendar. */
export function fetchMeetingBriefById(meetingId: string) {
  return get<{ brief: BackendMeetingBrief }>(`/agents/calendar/brief/${encodeURIComponent(meetingId)}`);
}

/** Source email reference inside a precomputed cluster. */
export interface BackendClusterEmailRef {
  subject: string;
  from_address: string;
  date: string;
}

/**
 * Slim cluster-match result for one meeting (mirrors backend
 * `agent_email.premeeting.PreMeetingBriefResult`).
 */
export interface BackendPreMeetingClusterCard {
  cluster_id: string;
  department: string;
  topic: string;
  meeting_start: string;
  summary: string[];
  questions: string[];
  action_items: string[];
  email_refs: BackendClusterEmailRef[];
}

export interface BackendPreMeetingBrief {
  meeting_title: string;
  meeting_time: string;
  matched_department_view: string | null;
  matched: boolean;
  /**
   * One card per matched cluster. A recurring meeting series surfaces
   * every occurrence (April..August) as its own card; a one-off meeting
   * surfaces a single card; a meeting matching nothing falls back to the
   * whole department. The flat fields below mirror `clusters[0]` for
   * backward compatibility.
   */
  clusters: BackendPreMeetingClusterCard[];
  department: string;
  topic: string;
  summary: string[];
  questions: string[];
  action_items: string[];
  email_refs: BackendClusterEmailRef[];
}

/** Fetch the embedding-based cluster match for a specific calendar event. */
export function fetchPremeetingBriefForMeeting(meetingId: string) {
  return get<{ brief: BackendPreMeetingBrief }>(
    `/agents/premeeting-brief/${encodeURIComponent(meetingId)}`,
  );
}

// ── Meeting notes (CEO MoM) ──────────────────────────────────────

/** A persisted meeting note record (mirrors backend `MeetingNote`). */
export interface BackendMeetingNote {
  id: number;
  meeting_id: string;
  note_text: string;
  action_items: Record<string, unknown>[];
  /** Team/department tags (@mentions) the note was directed at, e.g. ["Risk", "BB&RL"]. */
  tags: string[];
  source: string;
  created_at: string;
  updated_at: string | null;
}

/** Request body for creating a meeting note (mirrors `MeetingNoteCreate`). */
export interface CreateMeetingNoteBody {
  meeting_id: string;
  note_text: string;
  /** Optional — backend LLM-extracts action items when omitted/empty. */
  action_items?: Record<string, unknown>[];
  /** Optional structured team/department tags parsed from @mentions in the note. */
  tags?: string[];
  /** Source identifier; defaults server-side to "ceo_note". */
  source?: string;
}

/**
 * Persist a CEO meeting note to Postgres. The `meeting_id` MUST be the
 * Graph event id so cluster synthesis can read the note back for the
 * right meeting.
 */
export function createMeetingNote(body: CreateMeetingNoteBody) {
  return post<BackendMeetingNote>("/agents/meeting-notes", body);
}

/** Return all persisted notes for a specific meeting (by Graph event id). */
export function fetchMeetingNotes(meetingId: string) {
  return get<{ notes: BackendMeetingNote[]; count: number }>(
    `/agents/meeting-notes/${encodeURIComponent(meetingId)}`,
  );
}

// ── Email ────────────────────────────────────────────────────────

export function fetchEmailSummary() {
  return get<{ summary: BackendEmailSummary | null; message?: string }>("/agents/email/summary");
}

export function fetchEmailSummaries() {
  return get<{ summaries: BackendEmailSummary[] }>("/agents/email/summaries");
}

/**
 * Return precomputed email cluster records, optionally filtered to one
 * `department`. Pure CSV read — never triggers an LLM call.
 */
export interface EmailClustersResponse {
  count: number;
  records: BackendEmailClusterRecord[];
  /**
   * False until the cluster pipeline has completed at least once in the
   * current backend process's lifetime. When false, `records` is always
   * empty — the backend is deliberately withholding potentially-stale
   * data left over from before a restart rather than serving it as current.
   */
  pipeline_fresh: boolean;
  pipeline_ready_at: string | null;
}

export function fetchEmailClusters(department?: string) {
  const qs = department ? `?department=${encodeURIComponent(department)}` : "";
  return get<EmailClustersResponse>(`/agents/email/clusters${qs}`);
}

/** Manually trigger one run of the email cluster synthesis pipeline. */
export function refreshEmailClusters() {
  return post<{ clusters_written: number }>("/agents/email/clusters/refresh", {});
}

// ── Action Tracker ───────────────────────────────────────────────

export function fetchActionItems(status?: string) {
  const qs = status ? `?status=${encodeURIComponent(status)}` : "";
  return get<{ action_items: BackendActionItem[] }>(`/agents/actions${qs}`);
}

export function patchActionItem(id: string, body: { status?: string; owner?: string }) {
  return patch<{ action_item: BackendActionItem }>(`/agents/actions/${id}`, body);
}

// ── Open Tracker ─────────────────────────────────────────────────

export interface OpenTrackerItem {
  id: string;
  text: string;
  owner: string;
  due_date: string | null;
  due_label: string;
  business_unit: string;
  status: "Overdue" | "Due Today" | "Upcoming";
  overdue_by_days: number | null;
  source_topic: string;
}

export interface PendingDecisionItem {
  id: string;
  text: string;
  owner: string;
  due_date: string | null;
  due_label: string;
  business_unit: string;
  source_topic: string;
  status: "Overdue" | "Due Today" | "Upcoming";
}

export interface OpenTrackerResponse {
  items: OpenTrackerItem[];
  pending_decisions: PendingDecisionItem[];
  overdue_count: number;
  due_today_count: number;
  upcoming_count: number;
  total_count: number;
  pending_overdue_count: number;
  pending_due_today_count: number;
  pending_upcoming_count: number;
  /**
   * False until the cluster pipeline has completed at least once in the
   * current backend process's lifetime. When false, all items/counts are
   * empty/zero — the backend is deliberately withholding potentially-stale
   * data left over from before a restart rather than serving it as current.
   */
  pipeline_fresh: boolean;
  pipeline_ready_at: string | null;
}

export function fetchOpenTracker() {
  return get<OpenTrackerResponse>("/agents/open-tracker");
}

/** Mark a single tracker item as complete. Returns the saved completed item. */
export function completeOpenTrackerItem(itemId: string) {
  return post<{ completed: boolean; item: {
    id: string;
    text: string;
    owner: string;
    business_unit: string;
    due_date: string | null;
    source_topic: string;
    completed_at: string;
  } }>(`/agents/open-tracker/${itemId}/complete`, {});
}

/**
 * Undo a previous completion of a tracker item. Removes it from the
 * completed store so it reappears in the next GET /agents/open-tracker
 * response (back in the open items list or the "Awaiting Your Decision"
 * queue, depending on its owner).
 */
export function uncompleteOpenTrackerItem(itemId: string) {
  return del<{ uncompleted: boolean; item: {
    id: string;
    text?: string;
    owner?: string;
    business_unit?: string;
    due_date?: string | null;
    due_label?: string;
    status?: string;
    overdue_by_days?: number | null;
    source_topic?: string;
    is_pending_decision?: boolean;
  } }>(`/agents/open-tracker/${itemId}/complete`);
}

/** Update the due date for a tracker item. Returns the updated item with re-classified status. */
export function updateOpenTrackerDueDate(itemId: string, dueDate: string) {
  return patch<{ updated: boolean; item: {
    id: string;
    text: string;
    owner: string;
    business_unit: string;
    due_date: string;
    due_label: string;
    status: string;
    overdue_by_days: number | null;
    source_topic: string;
    updated_at: string;
  } }>(`/agents/open-tracker/${itemId}/due-date`, { due_date: dueDate });
}

/** Edit action item fields (description, owner, team, due date). Returns updated item. */
export interface EditTrackerItemBody {
  text?: string;
  owner?: string;
  business_unit?: string;
  due_date?: string;
}

export function editOpenTrackerItem(itemId: string, body: EditTrackerItemBody) {
  return patch<{ updated: boolean; item: {
    id: string;
    text: string;
    owner: string;
    business_unit: string;
    due_date: string | null;
    due_label: string;
    status: string;
    overdue_by_days: number | null;
    source_topic: string;
    updated_at: string;
  } }>(`/agents/open-tracker/${itemId}/edit`, body);
}

// ── Chatbot ──────────────────────────────────────────────────────

/** Permanently delete a tracker action item. */
export function deleteOpenTrackerItem(itemId: string) {
  return del<{ deleted: boolean; item: { id: string; text: string; owner: string; business_unit: string } }>(
    `/agents/open-tracker/${itemId}`,
  );
}

export function postChatQuery(message: string, mode: "email" | "document" = "email") {
  return post<BackendChatResponse>("/chatbot/query", { message, mode });
}

/** A single persisted Q&A pair, as returned by GET /chatbot/history. */
export interface ChatHistoryEntry {
  question: string;
  answer: string;
  timestamp: number;
}

/**
 * Fetch persisted chat history (durable `chatbot_history` Postgres table,
 * fronted by GET /chatbot/history), oldest first. The backend wraps the
 * list in `{ history: [...] }`; this unwraps it so callers get the array
 * directly. Returns an empty array if history is unavailable - the
 * endpoint is designed never to error on a missing/unreachable store.
 */
export async function fetchChatHistory(limit = 50): Promise<ChatHistoryEntry[]> {
  const res = await get<{ history: ChatHistoryEntry[] }>(
    `/chatbot/history?limit=${encodeURIComponent(limit)}`,
  );
  return res.history ?? [];
}

/** Delete the persisted chat history (Redis) - used by the chat refresh button. */
export function clearChatHistory() {
  return del<{ cleared: boolean }>("/chatbot/history");
}

/** Source citation returned by the RAG chatbot. */
export interface ChatSource {
  subject: string;
  filename: string;
  score: number;
  source_type?: string;
  conversation_id?: string;
  message_count?: number;
  first_received_at?: string;
  last_received_at?: string;
  participants?: string[];
  matched_message_subject?: string;
  matched_message_received_at?: string;
}

/** SSE event types emitted by POST /chatbot/query/stream. */
export type ChatStreamEvent =
  | { type: "step"; step: string; message: string }
  | { type: "sources"; sources: ChatSource[] }
  | { type: "token"; content: string }
  | { type: "greeting"; content: string }
  | { type: "done"; timings: Record<string, number> }
  | { type: "error"; content: string }
  | { type: "heartbeat"; elapsed_s: number }
  | { type: "replace"; content: string };

/**
 * Stream a chatbot query via SSE. Calls the provided callback for each
 * parsed event, letting the UI update incrementally.
 *
 * Resolves when the stream is complete. Rejects on network/auth errors.
 */
export async function streamChatQuery(
  message: string,
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal,
  mode: "email" | "document" = "email",
): Promise<void> {
  const res = await fetch(`${BASE}/chatbot/query/stream`, {
    method: "POST",
    credentials: CREDENTIALS,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, mode }),
    signal,
  });

  if (res.status === 401) throw new UnauthorizedError("/chatbot/query/stream");
  if (!res.ok) throw new Error(`POST /chatbot/query/stream failed: ${res.status}`);

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No readable stream in response");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Parse SSE lines: each event is "data: {...}\n\n"
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data: ")) continue;
      try {
        const event = JSON.parse(line.slice(6)) as ChatStreamEvent;
        onEvent(event);
      } catch {
        // skip malformed events
      }
    }
  }
}

// ── Auth (delegated SSO, server-side session) ────────────────────

export interface AuthState {
  authenticated: boolean;
  mode: "delegated" | "app_only" | "local";
  user: { name?: string | null; email?: string | null } | null;
}

/**
 * Report the current sign-in state. Returns `{ authenticated: false }`
 * (rather than throwing) when there is no valid session, so callers can
 * branch on it to gate the app.
 */
/**
 * Thrown when the auth check could not be *resolved* - i.e. we could not
 * get a clean 200/401 answer from the backend within the deadline (a
 * timeout, a network failure, or a 5xx). This is deliberately distinct
 * from an authenticated/unauthenticated answer: the caller must treat it
 * as an *error* state (show a retry screen), NOT silently assume the user
 * is signed out and bounce them into another slow sign-in round trip.
 */
export class AuthCheckError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AuthCheckError";
  }
}

/**
 * Maximum time we'll wait for `/auth/me` before giving up. The pre-login
 * gate blocks the whole app on this call, so it must be bounded: if the
 * backend is slow or unreachable we surface an error state within this
 * window instead of spinning "Checking sign-in status…" indefinitely.
 */
export const AUTH_CHECK_TIMEOUT_MS = 5000;

/**
 * Report the current sign-in state, bounded by {@link AUTH_CHECK_TIMEOUT_MS}.
 *
 * Three outcomes, kept strictly distinct so the UI can render three
 * different states:
 *   - 200  -> resolves with `{ authenticated: true, ... }`
 *   - 401  -> resolves with `{ authenticated: false, ... }` (known signed out)
 *   - timeout / network error / any other non-OK status
 *          -> throws {@link AuthCheckError} (unresolved / error)
 */
export async function fetchAuthState(): Promise<AuthState> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), AUTH_CHECK_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(`${BASE}/auth/me`, {
      credentials: CREDENTIALS,
      signal: controller.signal,
      // Never let a stale/proxy-cached answer mask the real auth state.
      cache: "no-store",
    });
  } catch (err) {
    // AbortController fires an "AbortError" DOMException on timeout.
    // Guard by name rather than `instanceof DOMException` so this holds
    // in any environment where DOMException isn't a global.
    if (err instanceof Error && err.name === "AbortError") {
      throw new AuthCheckError(
        `GET /auth/me timed out after ${AUTH_CHECK_TIMEOUT_MS}ms`,
      );
    }
    throw new AuthCheckError(
      `GET /auth/me failed: ${err instanceof Error ? err.message : String(err)}`,
    );
  } finally {
    clearTimeout(timer);
  }

  if (res.status === 401) {
    return { authenticated: false, mode: "delegated", user: null };
  }
  if (!res.ok) {
    throw new AuthCheckError(`GET /auth/me failed: ${res.status}`);
  }
  return res.json();
}

/**
 * Begin the Microsoft sign-in flow. This is a full-page navigation (not
 * an XHR): the backend redirects to Microsoft, then back to
 * /auth/callback, which sets the session cookie and redirects to this
 * SPA (FRONTEND_URL).
 */
export function redirectToLogin(): void {
  window.location.href = `${BASE}/auth/login`;
}

/** Invalidate the server-side session and clear the cookie. */
export async function logout(): Promise<void> {
  await fetch(`${BASE}/auth/logout`, { method: "POST", credentials: CREDENTIALS });
}

export interface AppConfig {
  calendar_polling_interval_seconds: number;
}

export function fetchAppConfig(): Promise<AppConfig> {
  return get<AppConfig>("/auth/config");
}
