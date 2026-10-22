/**
 * sanitize.ts
 * =============
 * Defense-in-depth sanitization for text values that ultimately come
 * from an LLM response (email summary / action items / top questions).
 * The backend (`agent_email/prompts.py::_sanitize_text_value`) already
 * strips stray control characters before this data ever reaches the
 * frontend, but this mirrors that same narrow cleanup client-side in
 * case a value bypasses the backend path (e.g. a cached sessionStorage
 * value written by an older backend build).
 *
 * Deliberately narrow: only removes actual control/non-printable
 * characters (and BOM/replacement-character artifacts). Never touches
 * letters, digits, punctuation, currency symbols, or any other visible
 * Unicode text, so real content is never dropped.
 */

// Control characters illegal to display, plus BOM (\uFEFF) and the
// Unicode replacement character (\uFFFD) that can appear when a byte
// stream was decoded incorrectly upstream.
const STRAY_CONTROL_RE = /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufeff\ufffd]/g;

/**
 * Clean a single-line text value (a question, action item, or bullet)
 * of stray control characters. Embedded newlines/tabs are collapsed to
 * a single space (not deleted) so wrapped text never gets two words
 * glued together; runs of whitespace are then normalized to one space.
 */
export function sanitizeLine(value: string | null | undefined): string {
  if (!value) return "";
  const collapsed = value.replace(/[\r\n\t]+/g, " ");
  const cleaned = collapsed.replace(STRAY_CONTROL_RE, "");
  return cleaned.replace(/[ \t]{2,}/g, " ").trim();
}

/**
 * Clean a multi-line block (e.g. a full summary) of stray control
 * characters while preserving intentional newlines between bullets.
 */
export function sanitizeBlock(value: string | null | undefined): string {
  if (!value) return "";
  return value
    .split("\n")
    .map((line) => sanitizeLine(line))
    .join("\n")
    .trim();
}

/** Sanitize every string in an array via `sanitizeLine`, dropping any that end up empty. */
export function sanitizeLines(values: string[] | null | undefined): string[] {
  if (!values) return [];
  return values.map(sanitizeLine).filter((v) => v.length > 0);
}
