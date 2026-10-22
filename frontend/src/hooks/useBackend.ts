/**
 * useBackend.ts
 * =============
 * React Query hooks wrapping every backend endpoint.
 * Components call these hooks; the hooks call src/lib/api.ts.
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchHealth,
  fetchCurrentMeeting,
  fetchNextMeeting,
  fetchMeetingBrief,
  fetchEmailSummary,
  fetchEmailSummaries,
  fetchEmailClusters,
  refreshEmailClusters,
  fetchActionItems,
  patchActionItem,
  postChatQuery,
  fetchOpenTracker,
  completeOpenTrackerItem,
  uncompleteOpenTrackerItem,
  updateOpenTrackerDueDate,
  editOpenTrackerItem,
  deleteOpenTrackerItem,
  type EditTrackerItemBody,
} from "@/lib/api";

// ── Health ────────────────────────────────────────────────────────

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    retry: 1,
    staleTime: 30_000,
  });
}

// ── Calendar ─────────────────────────────────────────────────────

export function useCurrentMeeting() {
  return useQuery({
    queryKey: ["calendar", "current"],
    queryFn: fetchCurrentMeeting,
    refetchInterval: 60_000,
    retry: 1,
  });
}

export function useNextMeeting() {
  return useQuery({
    queryKey: ["calendar", "next"],
    queryFn: fetchNextMeeting,
    refetchInterval: 60_000,
    retry: 1,
  });
}

export function useMeetingBrief() {
  return useQuery({
    queryKey: ["calendar", "brief"],
    queryFn: fetchMeetingBrief,
    retry: 1,
    staleTime: 5 * 60_000,
  });
}

// ── Email ────────────────────────────────────────────────────────

export function useEmailSummary() {
  return useQuery({
    queryKey: ["email", "summary"],
    queryFn: fetchEmailSummary,
    retry: 1,
    staleTime: 5 * 60_000,
  });
}

export function useEmailSummaries() {
  return useQuery({
    queryKey: ["email", "summaries"],
    queryFn: fetchEmailSummaries,
    retry: 1,
    staleTime: 5 * 60_000,
  });
}

/**
 * Email cluster records for one `department`. Backs department tabs with
 * one Past meeting summaries row per cluster. Pure CSV read.
 */
export function useEmailClusters(department: string | undefined) {
  return useQuery({
    queryKey: ["email", "clusters", department ?? "all"],
    queryFn: () => fetchEmailClusters(department),
    enabled: !!department,
    retry: 1,
    staleTime: 60_000,
  });
}

/** Trigger one cluster pipeline run and invalidate cached cluster queries. */
export function useRefreshEmailClusters() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: refreshEmailClusters,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["email", "clusters"] });
    },
  });
}

// ── Action Tracker ───────────────────────────────────────────────

export function useActionItems(status?: string) {
  return useQuery({
    queryKey: ["actions", status ?? "all"],
    queryFn: () => fetchActionItems(status),
    retry: 1,
    staleTime: 30_000,
  });
}

export function usePatchActionItem() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: { status?: string; owner?: string } }) =>
      patchActionItem(id, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["actions"] });
    },
  });
}

// ── Chatbot ──────────────────────────────────────────────────────

export function useChatQuery() {
  return useMutation({
    mutationFn: (message: string) => postChatQuery(message),
  });
}

// ── Open Tracker ─────────────────────────────────────────────────

/**
 * Fetches the Open Actions Status Tracker data from
 * GET /agents/open-tracker (reads email_clusters.csv — no LLM call).
 * Auto-refreshes every 60 seconds so the Launchpad card stays current.
 */
export function useOpenTracker() {
  return useQuery({
    queryKey: ["open-tracker"],
    queryFn: fetchOpenTracker,
    refetchInterval: 60_000,
    retry: 1,
    staleTime: 30_000,
  });
}

/**
 * Mutation to mark a single tracker item complete.
 * On success, invalidates the open-tracker query so the card
 * refreshes and the completed item disappears from the open list.
 */
export function useCompleteTrackerItem() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (itemId: string) => completeOpenTrackerItem(itemId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["open-tracker"] });
    },
  });
}

/**
 * Mutation to undo ("Undo" toast) a previously completed tracker item.
 * On success, invalidates the open-tracker query so the item reappears
 * in the Open Actions Status Tracker or the "Awaiting Your Decision"
 * queue, whichever bucket its owner routes it to.
 */
export function useUncompleteTrackerItem() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (itemId: string) => uncompleteOpenTrackerItem(itemId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["open-tracker"] });
    },
  });
}

/**
 * Mutation to update the due date of a tracker item.
 * On success, invalidates the open-tracker query so both the Launchpad
 * card and the Ledger reflect the new date and re-classified status.
 */
export function useUpdateTrackerDueDate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ itemId, dueDate }: { itemId: string; dueDate: string }) =>
      updateOpenTrackerDueDate(itemId, dueDate),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["open-tracker"] });
    },
  });
}

/**
 * Mutation to edit a tracker item's fields (text, owner, team, due date).
 * On success, invalidates the open-tracker query so both Launchpad and
 * Ledger reflect the changes.
 */
export function useEditTrackerItem() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ itemId, body }: { itemId: string; body: EditTrackerItemBody }) =>
      editOpenTrackerItem(itemId, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["open-tracker"] });
    },
  });
}

/**
 * Permanently delete a tracker action item. Invalidates the open-tracker
 * cache on success so the item disappears from the Launchpad card.
 */
export function useDeleteTrackerItem() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (itemId: string) => deleteOpenTrackerItem(itemId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["open-tracker"] });
    },
  });
}
