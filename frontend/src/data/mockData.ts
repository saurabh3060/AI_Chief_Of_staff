import type { ActionItem, AppNotification, BU, ChangeEvent } from "@/context/AppState";

const c = (at: string, type: ChangeEvent["type"], extra: Partial<ChangeEvent> = {}): ChangeEvent => ({ at, type, ...extra });

export const SEED_ACTIONS: ActionItem[] = [];

export const SEED_NOTIFICATIONS: AppNotification[] = [];

export const BU_COLORS: Record<BU, { bg: string; text: string }> = {
  "BBRL":                                    { bg: "#DBEAFE", text: "#1E40AF" },
  "Credit Cards":                            { bg: "#FEE2E2", text: "#991B1B" },
  "Retail Advances - Excluding Credit Cards":{ bg: "#FFE4D6", text: "#9A3412" },
  "Wholesale Banking":                       { bg: "#DCFCE7", text: "#166534" },
};

export const BU_ORDER: BU[] = ["BBRL", "Credit Cards", "Retail Advances - Excluding Credit Cards", "Wholesale Banking"];

export const TODAY_CALENDAR: { time: string; title: string; tone: "info" | "crimson"; view: string | null }[] = [];

// CRO chat
export const CRO_PRELOADED: { role: "user" | "ai"; text: string }[] = [];
export const CRO_CHIPS: string[] = [];
export const RETAIL_CHIPS: string[] = [];
export const CRO_LIVE_CHIPS: string[] = [];

export interface ChatResponse {
  text?: string;
  table?: { title: string; headers: string[]; rows: (string | number)[][] };
  footer?: string;
}

export function croResponse(_q: string): ChatResponse {
  return { text: "" };
}

export function retailResponse(_q: string): ChatResponse {
  return { text: "" };
}

export const QUARTER_LABELS = ["Q4FY25", "Q1FY26", "Q2FY26", "Q3FY26", "Q4FY26"];

// ===================== Wholesale =====================
export const WHOLESALE_PRELOADED: { role: "user" | "ai"; text: string }[] = [];
export const WHOLESALE_CHIPS: string[] = [];
export const WHOLESALE_LIVE_CHIPS: string[] = [];
export function wholesaleResponse(_q: string): ChatResponse {
  return { text: "" };
}

// ===================== Retail Assets =====================
export const RA_PRELOADED: { role: "user" | "ai"; text: string }[] = [];
export const RA_CHIPS: string[] = [];
export const RA_LIVE_CHIPS: string[] = [];
export function retailAssetsResponse(_q: string): ChatResponse {
  return { text: "" };
}