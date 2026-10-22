import type { ActionItem } from "@/context/AppState";

/** Explicit mapping from seeded tracker/pending ids → seeded ledger action ids. */
const SEED_MAP: Record<string, string> = {
  DM1: "S1", // RBI unsecured-lending response
  DM2: "S2", // EWS go-live
  DM3: "S4", // Dormant CASA branch P&L
  DM4: "S7", // DigiSave rollout
  DM5: "S3", // NCLT recovery
};

/**
 * Resolve the ledger ActionItem id that corresponds to a tracker/pending item.
 * Tries: direct id match → explicit seed map → fuzzy title-token overlap.
 * Returns null when no confident match exists.
 */
export function resolveLedgerActionId(
  id: string,
  text: string,
  actions: ActionItem[],
): string | null {
  if (actions.some((a) => a.id === id)) return id;
  const mapped = SEED_MAP[id];
  if (mapped && actions.some((a) => a.id === mapped)) return mapped;

  const tokens = text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((w) => w.length > 3);
  if (tokens.length === 0) return null;

  let best: { id: string; score: number } | null = null;
  for (const a of actions) {
    const title = a.title.toLowerCase();
    const score = tokens.filter((t) => title.includes(t)).length;
    if (score >= 2 && (!best || score > best.score)) best = { id: a.id, score };
  }
  return best?.id ?? null;
}
