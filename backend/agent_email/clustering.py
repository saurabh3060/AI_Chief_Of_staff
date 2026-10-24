"""
agent_email/clustering.py
==========================
Embedding, clustering (components or knn), attachment extraction,
and batch splitting for the cluster synthesis pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from core.config import Settings, get_settings
from core.graph_client.client import GraphClient
from core.logging_config import get_agent_logger
from core.text_match import _ABBREVIATION_EXPANSIONS, meaningful_tokens, strip_email_noise, strip_synthesis_noise

from agent_email.attachment_filter import is_mom_attachment
from agent_email.extractors import extract_docx_text, extract_excel_text, extract_pdf_text, extract_pptx_text
from agent_email.prompts import topic_key_from_semantics
from agent_email.schema import EmailCandidate, EmailMessage
from agent_email.semantic_match import cosine_similarity, embed_text

logger = get_agent_logger("agent_email")


@dataclass
class MeetingOccurrence:
    """One calendar meeting instance used to pin emails to a past occurrence."""

    id: str
    subject: str
    start: datetime
    end: datetime
    embedding: np.ndarray | None = None


# Calendar / document-role tokens stripped when comparing subjects for a
# recurring-series merge. Generic: months, fiscal markers, MoM labels —
# never a product or meeting-series name.
_SERIES_CALENDAR_TOKENS = frozenset({
    "jan", "january", "feb", "february", "mar", "march", "apr", "april",
    "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept",
    "september", "oct", "october", "nov", "november", "dec", "december",
    "q1", "q2", "q3", "q4", "h1", "h2", "fy",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
})
_SERIES_DOC_ROLE_TOKENS = frozenset({
    "minutes", "mom", "notes", "note", "action", "actions", "key",
    "update", "updates", "discussion", "monthly", "weekly", "quarterly",
    "annual", "fyi",
})
_SERIES_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with",
    "at", "by", "from", "re", "fw", "fwd", "please", "where", "do", "we",
    "actually", "stand", "required", "view",
})
_CASA_SLASH_RE = re.compile(r"\bca\s*/\s*sa\b", re.IGNORECASE)
_NESTED_SUBJECT_RE = re.compile(r"(?im)^[ \t>]*subject:\s*(.+)$")


def series_subject_tokens(subject: str) -> set[str]:
    """
    Subject stem used to detect a recurring mail series.

    Strips Re/Fw, calendar tokens, and document-role words (minutes, MoM,
    actions) so "Wholesale Banking - July Review | Minutes" and
    "Actions from June's WB Review" share the same stem. Abbreviations
    such as WB expand to their spelled-out tokens. Does not hardcode any
    meeting or product name.
    """
    cleaned = normalize_subject(subject).lower()
    if _CASA_SLASH_RE.search(cleaned):
        cleaned = _CASA_SLASH_RE.sub("casa", cleaned)
    raw = re.findall(r"[a-z0-9]+", cleaned)
    kept: list[str] = []
    for token in raw:
        if token.isdigit() and len(token) <= 4:
            continue
        if token in _SERIES_CALENDAR_TOKENS or token in _SERIES_DOC_ROLE_TOKENS or token in _SERIES_STOPWORDS:
            continue
        if len(token) < 2:
            continue
        kept.append(token)
    out: set[str] = {t for t in kept if len(t) >= 3}
    for token in kept:
        out |= set(_ABBREVIATION_EXPANSIONS.get(token, ()))
    return out


def department_stem_tokens(settings: Settings | None = None) -> frozenset[str]:
    """
    Tokens from configured department names and summaries that must not
    alone evidence a recurring-series match (wholesale, banking, retail, …).
    """
    active = settings or get_settings()
    departments = getattr(active, "bu_departments", None)
    if not departments:
        return frozenset({"wholesale", "banking", "retail", "cards", "corporate", "excluding"})
    tokens: set[str] = set()
    for dept in departments:
        tokens |= meaningful_tokens(dept.name)
        tokens |= meaningful_tokens(dept.summary)
    return frozenset(tokens)


def stems_share_series(
    stem_a: set[str],
    stem_b: set[str],
    jaccard_min: float,
    dept_tokens: frozenset[str],
) -> bool:
    """True when two subject stems look like the same recurring series."""
    if not stem_a or not stem_b:
        return False
    shared = stem_a & stem_b
    if not shared:
        return False
    if not (shared - dept_tokens):
        return False
    union = stem_a | stem_b
    if not union:
        return False
    return len(shared) / len(union) >= jaccard_min


def _same_conversation(i: int, j: int, conversation_ids: list[str]) -> bool:
    cid_i = (conversation_ids[i] or "").strip()
    cid_j = (conversation_ids[j] or "").strip()
    return bool(cid_i and cid_j and cid_i == cid_j)


# --- Recurring-series occurrence period parsing (generic, name-free) ---
#
# A recurring series (monthly review, quarterly forum, ...) names its
# occurrence by a CALENDAR PERIOD in the subject: "... April Review",
# "MoM ... - May", "Actions from June's ...", "Q3 ... pack". The period
# token is what distinguishes one occurrence of a series from the next,
# and it is generic calendar vocabulary - never a product/meeting name.
# We normalize any month spelling to its 2-digit month, and any quarter/
# half token to a canonical form, so two emails that name the same period
# collapse to the same occurrence key regardless of wording.
_MONTH_TO_NUM: dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
# Quarter / half-year period tokens (q1-q4, h1-h2), canonicalized to themselves.
# A 2- or 4-digit year appearing next to a period token ("May 2026",
# "May-26", "FY26"). Bare years are only trusted when adjacent to a
# period token, so an arbitrary number in a subject is never a year.
_QUARTER_RE = re.compile(r"(?<![a-z0-9])(q[1-4]|h[12])(?![a-z0-9])")
# Separators allowed between a period token and its year: space ' ’ . _ / - ,
# (deliberately NOT "|" or ":", which separate unrelated parts of a subject).
_YEAR_SEP = r"[\s'’._/\-,]*"
_YEAR_AFTER_RE = re.compile(_YEAR_SEP + r"(20\d{2}|\d{2})(?!\d)")
_YEAR_AFTER_FY_RE = re.compile(_YEAR_SEP + r"(?:fy" + _YEAR_SEP + r")?(20\d{2}|\d{2})(?!\d)")
_YEAR_BEFORE_RE = re.compile(r"(?:(?:fy)?" + _YEAR_SEP + r"(20\d{2})|fy" + _YEAR_SEP + r"(\d{2}))" + _YEAR_SEP + r"$")


def _adjacent_year(before: str, after: str, allow_fy: bool) -> str | None:
    """Year text written directly after (or, if 4-digit / FY-prefixed, before) a period token.

    A 2-digit number directly AFTER the token is a year ("Sept 25" = Sept 2025).
    A bare 2-digit number BEFORE it is never a year ("25 Sept" is a day,
    "Branch 25 Sept" is not 2025), so only a 4-digit or FYnn year is accepted there.
    """
    m = (_YEAR_AFTER_FY_RE if allow_fy else _YEAR_AFTER_RE).match(after)
    if m:
        return m.group(1)
    m = _YEAR_BEFORE_RE.search(before)
    if m:
        return m.group(1) or (m.group(2) if allow_fy else None)
    return None


def _normalize_year(raw: str, fallback_year: int | None) -> int | None:
    """Map a matched year token ('2026', '26, "'26") to a full year."""
    digits = raw.strip().lstrip("'")
    if not digits.isdigit():
        return fallback_year
    val = int(digits)
    if val >= 2000:
        return val
    if val < 100:
        return 2000 + val
    return fallback_year


def series_period_token(
    subject: str,
    received_at: "datetime | None" = None,
    require_explicit: bool = False,
) -> str:
    """
    Canonical calendar period an email's subject refers to, e.g.
    "2026-04" (April), "2026-q3", or "" when no period can be inferred.

    Generic and name-free: it recognizes month names/abbreviations and
    quarter/half tokens only (the same calendar vocabulary already used
    to STRIP these words in `series_subject_tokens`), plus an optional
    adjacent year. When the subject names a month/quarter but no year,
    the year is taken from `received_at` (rolling back a year when the
    named month is far AHEAD of the received month, e.g. a December MoM
    that arrives the following January).

    When the subject names no period token:
      - `require_explicit=False` (default): fall back to the received
        month so every email lands in some period bucket (used for
        chronological ordering).
      - `require_explicit=True`: return "" - the email does NOT name a
        period, so it must NOT be treated as a recurring-series
        occurrence (used when SPLITTING a merged cluster, so two working
        papers sent the same month are not torn apart just because they
        share a received month).
    """
    cleaned = normalize_subject(subject).lower()

    recv_year = received_at.year if received_at is not None else None
    recv_month = received_at.month if received_at is not None else None

    # 1) Explicit month name anywhere in the subject. Matching on letter runs
    # (not whitespace tokens) makes "Sep26", "Sept'26", "Sep-26" and "Sept 26"
    # all resolve the same way; the first month named wins.
    month_num: int | None = None
    year_text: str | None = None
    for m in re.finditer(r"[a-z]+", cleaned):
        if m.group(0) in _MONTH_TO_NUM:
            month_num = _MONTH_TO_NUM[m.group(0)]
            year_text = _adjacent_year(cleaned[:m.start()], cleaned[m.end():], allow_fy=False)
            break

    # 2) Explicit quarter/half token (only used if no month is named).
    quarter_tok: str = ""
    if month_num is None:
        qm = _QUARTER_RE.search(cleaned)
        if qm:
            quarter_tok = qm.group(1)
            year_text = _adjacent_year(cleaned[:qm.start()], cleaned[qm.end():], allow_fy=True)

    # Year: only one written right next to the month/quarter, else the received
    # year. Any other number in the subject ("Top 15 reports") is not a year.
    year: int | None = _normalize_year(year_text, recv_year) if year_text else None
    explicit_year = year is not None
    if year is None:
        year = recv_year

    if month_num is not None:
        # Roll the year back when the subject names a month notably ahead
        # of the received month (a period referenced in arrears that
        # arrives early the next year). Only applies when we fell back to
        # the received year (no explicit year in the subject).
        if year is not None and not explicit_year and recv_month is not None:
            if month_num - recv_month >= 6:
                year -= 1
        return f"{year:04d}-{month_num:02d}" if year is not None else f"m{month_num:02d}"
    if quarter_tok:
        return f"{year:04d}-{quarter_tok}" if year is not None else quarter_tok

    # 3) No period token in the subject.
    if require_explicit:
        return ""
    if recv_year is not None and recv_month is not None:
        return f"{recv_year:04d}-{recv_month:02d}"
    return ""


def series_occurrence_key(
    subject: str,
    received_at: "datetime | None" = None,
    settings: Settings | None = None,
    require_explicit_period: bool = True,
) -> str:
    """
    Occurrence identity for a recurring-series email, derived purely from
    the email itself: the series stem + the calendar period it names.

    Two emails collapse to the same occurrence only when they share both
    the recurring-series stem (e.g. wholesale/banking/review, WB expanded)
    AND the same period (2026-04). This lets a monthly series split into
    one bucket per month straight from the emails, with no dependency on
    calendar history and no hardcoded meeting or product name. Returns ""
    when the subject has no recurring-series stem OR (by default) names no
    EXPLICIT period.

    `require_explicit_period=True` (default) is what makes this safe for
    splitting a merged cluster: an email is treated as a series occurrence
    only when its subject literally names a month/quarter, so two working
    papers sent in the same month (no period in their subjects) are not
    split apart just because they share a received month.
    """
    stem = series_subject_tokens(subject)
    if not stem:
        return ""
    period = series_period_token(subject, received_at, require_explicit=require_explicit_period)
    if not period:
        return ""
    stem_key = "+".join(sorted(stem))
    return f"{stem_key}|{period}"


def _union_find_groups(n: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[rj] = ri

    for i, j in pairs:
        _union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(_find(i), []).append(i)
    return list(groups.values())


def cluster_by_similarity_graph(vectors: list[np.ndarray], threshold: float) -> list[list[int]]:
    """Connected-components clustering: merge all pairs with cosine >= threshold."""
    n = len(vectors)
    if n <= 1:
        return [list(range(n))] if n else []

    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if cosine_similarity(vectors[i], vectors[j]) >= threshold:
                pairs.append((i, j))
    return _union_find_groups(n, pairs)


def cluster_by_nearest_neighbor(vectors: list[np.ndarray], threshold: float) -> list[list[int]]:
    """1-NN union-find: each email joins its nearest neighbor if cosine >= threshold."""
    n = len(vectors)
    if n <= 1:
        return [list(range(n))] if n else []

    pairs: list[tuple[int, int]] = []
    for i in range(n):
        best_j = -1
        best_score = -1.0
        for j in range(n):
            if i == j:
                continue
            score = cosine_similarity(vectors[i], vectors[j])
            if score > best_score:
                best_score = score
                best_j = j
        if best_j >= 0 and best_score >= threshold:
            pairs.append((i, best_j))
    return _union_find_groups(n, pairs)


def _pairwise_cosine_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """Full symmetric cosine similarity matrix for small candidate pools."""
    n = len(vectors)
    sim = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            score = cosine_similarity(vectors[i], vectors[j])
            sim[i, j] = score
            sim[j, i] = score
    return sim


def _complete_link_score(sim: np.ndarray, left: set[int], right: set[int]) -> float:
    """Minimum pairwise cosine between two index sets (complete-linkage distance)."""
    return min(float(sim[i, j]) for i in left for j in right)


def cluster_by_complete_linkage(vectors: list[np.ndarray], threshold: float) -> list[list[int]]:
    """
    Complete-linkage agglomerative clustering: repeatedly merge the two
    clusters whose worst-case (minimum) pairwise cosine is highest,
    stopping once the best merge would fall below `threshold`.

    Every pair within a final cluster is >= threshold similar — no
    single-linkage chaining through one bridging email.
    """
    n = len(vectors)
    if n <= 1:
        return [list(range(n))] if n else []

    sim = _pairwise_cosine_matrix(vectors)
    clusters: list[set[int]] = [{i} for i in range(n)]

    while len(clusters) > 1:
        best_score = -1.0
        best_pair = (-1, -1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                score = _complete_link_score(sim, clusters[i], clusters[j])
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
        if best_score < threshold:
            break
        merged = clusters[best_pair[0]] | clusters[best_pair[1]]
        remaining = [
            cluster
            for idx, cluster in enumerate(clusters)
            if idx not in best_pair
        ]
        remaining.append(merged)
        clusters = remaining

    return [sorted(cluster) for cluster in clusters]


def merge_by_conversation_id(
    n: int,
    conversation_ids: list[str] | None,
    groups: list[list[int]],
    run_id: str = "",
) -> list[list[int]]:
    """Join every pair that shares a Graph ``conversationId`` (reply thread)."""
    if n < 2 or not conversation_ids or len(conversation_ids) != n:
        return groups

    extra: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if _same_conversation(i, j, conversation_ids):
                extra.append((i, j))
                logger.info(
                    "[CLUSTER-THREAD run=%s] pair=%d-%d conversation_id=%r",
                    run_id, i, j, (conversation_ids[i] or "").strip(),
                )

    if not extra:
        return groups

    pairs: list[tuple[int, int]] = list(extra)
    for group in groups:
        for left, right in zip(group, group[1:]):
            pairs.append((left, right))
    return _union_find_groups(n, pairs)


def merge_recurring_series(
    vectors: list[np.ndarray],
    groups: list[list[int]],
    subjects: list[str],
    settings: Settings | None = None,
    run_id: str = "",
) -> list[list[int]]:
    """
    Join cosine near-misses whose subject stems match a recurring series.

    A pair is merged when stem Jaccard and shared-token floors clear,
    AND either cosine >= series_near_miss OR stem Jaccard is very high
    (>= 0.8). The Jaccard override is what joins a short "Actions from
    June's WB Review" into the monthly-review series when body/topic
    embeddings drifted. Adjacent-but-different threads (liabilities mix
    vs monthly review) share too little stem overlap to join.
    """
    n = len(vectors)
    if n < 2 or len(subjects) != n:
        return groups

    active = settings or get_settings()
    floor = float(getattr(active, "email_cluster_series_near_miss", 0.68) or 0.68)
    jaccard_min = float(getattr(active, "email_cluster_series_jaccard", 0.5) or 0.5)
    dept_tokens = department_stem_tokens(active)
    stems = [series_subject_tokens(subject) for subject in subjects]

    extra: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(vectors[i], vectors[j])
            a, b = stems[i], stems[j]
            if not stems_share_series(a, b, jaccard_min, dept_tokens):
                continue
            shared = a & b
            jaccard = len(shared) / len(a | b)
            if sim < floor and jaccard < 0.8:
                continue
            extra.append((i, j))
            logger.info(
                "[CLUSTER-SERIES run=%s] pair=%d-%d cosine=%.3f jaccard=%.2f shared=%s",
                run_id, i, j, sim, jaccard, sorted(shared),
            )

    if not extra:
        return groups

    pairs: list[tuple[int, int]] = list(extra)
    for group in groups:
        for left, right in zip(group, group[1:]):
            pairs.append((left, right))
    return _union_find_groups(n, pairs)


def _is_recurring_series_group(
    indices: list[int],
    stems: list[set[str]],
    jaccard_min: float,
    dept_tokens: frozenset[str],
) -> bool:
    """True when every pair in the group shares a recurring-series stem."""
    if len(indices) < 2:
        return False
    for pos, i in enumerate(indices):
        for j in indices[pos + 1:]:
            if not stems_share_series(stems[i], stems[j], jaccard_min, dept_tokens):
                return False
    return True


def merge_residual_near_matches(
    vectors: list[np.ndarray],
    groups: list[list[int]],
    subjects: list[str],
    settings: Settings | None = None,
    run_id: str = "",
) -> list[list[int]]:
    """
    Join leftover non-series emails whose cosine clears a slightly lower
    floor. Recurring-series clusters stay locked so a working-paper
    thread (NIM / CASA / balance-sheet) cannot be absorbed into MBR.
    """
    n = len(vectors)
    if n < 2 or len(subjects) != n:
        return groups

    active = settings or get_settings()
    residual = float(getattr(active, "email_cluster_residual_threshold", 0.72) or 0.0)
    if residual <= 0:
        return groups
    jaccard_min = float(getattr(active, "email_cluster_series_jaccard", 0.5) or 0.5)
    dept_tokens = department_stem_tokens(active)
    stems = [series_subject_tokens(subject) for subject in subjects]

    locked: set[int] = set()
    for group in groups:
        if _is_recurring_series_group(group, stems, jaccard_min, dept_tokens):
            locked.update(group)

    extra: list[tuple[int, int]] = []
    unlocked = [i for i in range(n) if i not in locked]
    for pos, i in enumerate(unlocked):
        for j in unlocked[pos + 1:]:
            sim = cosine_similarity(vectors[i], vectors[j])
            if sim < residual:
                continue
            extra.append((i, j))
            logger.info(
                "[CLUSTER-RESIDUAL run=%s] pair=%d-%d cosine=%.3f",
                run_id, i, j, sim,
            )

    if not extra:
        return groups

    pairs: list[tuple[int, int]] = list(extra)
    for group in groups:
        for left, right in zip(group, group[1:]):
            pairs.append((left, right))
    return _union_find_groups(n, pairs)


def _occurrence_split_key_for_index(
    idx: int,
    group: list[int],
    subjects: list[str],
    received_ats: list["datetime | None"] | None,
    conversation_ids: list[str] | None,
    settings: Settings | None,
) -> str:
    """
    Occurrence bucket for one member of a cluster group.

    A member is split out only when another member of the *same* group
    shares its recurring-series stem and names a *different* explicit
    calendar period. Compiled working-paper packs (different stems, or
    a discussion mail with no period in its subject) stay in one bucket.
    Members of the same ``conversationId`` are never split apart.
    """
    subject = subjects[idx]
    received_at = received_ats[idx] if received_ats and idx < len(received_ats) else None
    period_i = series_period_token(subject, received_at, require_explicit=True)
    if not period_i:
        return ""

    active = settings or get_settings()
    jaccard_min = float(getattr(active, "email_cluster_series_jaccard", 0.5) or 0.5)
    dept_tokens = department_stem_tokens(active)
    stem_i = series_subject_tokens(subject)

    for j in group:
        if j == idx:
            continue
        if conversation_ids and _same_conversation(idx, j, conversation_ids):
            continue
        period_j = series_period_token(
            subjects[j],
            received_ats[j] if received_ats and j < len(received_ats) else None,
            require_explicit=True,
        )
        if not period_j or period_j == period_i:
            continue
        stem_j = series_subject_tokens(subjects[j])
        if not stems_share_series(stem_i, stem_j, jaccard_min, dept_tokens):
            continue
        series_key = series_occurrence_key(subject, received_at, settings=settings)
        if series_key:
            return f"series:{series_key}"
    return ""


def _split_groups_by_meeting(
    groups: list[list[int]],
    meeting_ids: list[str] | None,
    subjects: list[str] | None = None,
    received_ats: list["datetime | None"] | None = None,
    conversation_ids: list[str] | None = None,
    settings: Settings | None = None,
    run_id: str = "",
) -> list[list[int]]:
    """
    Split a cluster only where it spans two DIFFERENT periods of the same
    recurring series (April vs May of a monthly review). Non-series
    members share one bucket and are never split, so clusters merged by
    semantics or subject-citation (a discussion-starter + the working
    papers it forwards, or a topic family like several notes about one
    product) stay intact. See `_occurrence_split_key_for_index`.
    """
    if not subjects:
        return groups

    out: list[list[int]] = []
    for group in groups:
        buckets: dict[str, list[int]] = {}
        for idx in group:
            key = _occurrence_split_key_for_index(
                idx, group, subjects, received_ats, conversation_ids, settings,
            )
            buckets.setdefault(key, []).append(idx)
        if len(buckets) > 1:
            logger.info(
                "[CLUSTER-OCCURRENCE run=%s] split group of %d into %d occurrence bucket(s): %s",
                run_id, len(group), len(buckets),
                {key or "<unanchored>": len(idxs) for key, idxs in buckets.items()},
            )
        for _key, idxs in buckets.items():
            out.append(idxs)
    return out


def cluster_candidates(
    vectors: list[np.ndarray],
    settings: Settings | None = None,
    run_id: str = "",
    subjects: list[str] | None = None,
    bodies: list[str] | None = None,
    meeting_ids: list[str] | None = None,
    received_ats: list["datetime | None"] | None = None,
    conversation_ids: list[str] | None = None,
) -> list[list[int]]:
    """Dispatch to components, knn, or agglomerative clustering per `email_cluster_algo`."""
    active = settings or get_settings()
    algo = (active.email_cluster_algo or "components").lower()
    threshold = float(active.email_cluster_similarity_threshold)

    n = len(vectors)
    if n >= 2:
        for i in range(n):
            best_j = -1
            best_score = -1.0
            for j in range(n):
                if i == j:
                    continue
                score = cosine_similarity(vectors[i], vectors[j])
                if score > best_score:
                    best_score = score
                    best_j = j
                if j > i and score >= threshold:
                    logger.info(
                        "[CLUSTER-EDGE run=%s] pair=%d-%d cosine=%.3f",
                        run_id, i, j, score,
                    )
            logger.info(
                "[CLUSTER-NN run=%s] i=%d nn=%d cosine=%.3f join=%s",
                run_id, i, best_j, best_score, best_score >= threshold,
            )

    if algo == "knn":
        groups = cluster_by_nearest_neighbor(vectors, threshold)
    elif algo == "agglomerative":
        groups = cluster_by_complete_linkage(vectors, threshold)
    else:
        groups = cluster_by_similarity_graph(vectors, threshold)

    if conversation_ids:
        groups = merge_by_conversation_id(n, conversation_ids, groups, run_id=run_id)

    if subjects:
        groups = merge_recurring_series(vectors, groups, subjects, settings=active, run_id=run_id)
        groups = merge_quoted_subject_links(
            groups, subjects, bodies, vectors=vectors, settings=active, run_id=run_id,
        )
        groups = merge_residual_near_matches(vectors, groups, subjects, settings=active, run_id=run_id)

    # Occurrence split: after the semantic/series merges have joined
    # everything they can, break any cluster that spans two distinct
    # occurrences so each occurrence surfaces as its own line item. The
    # split key prefers the email-derived recurring-series key (series
    # stem + calendar period) and falls back to the anchored meeting_id,
    # so a monthly series splits into one row per month FROM THE EMAILS
    # even when the calendar has only one (or zero) matching occurrence.
    # This runs last so the recurring-series merge can pull scattered
    # follow-ups together first, and the occurrence key then re-separates
    # them by which period/occurrence they belong to.
    anchored_enabled = bool(getattr(active, "meeting_anchored_clustering_enabled", False))
    has_meetings = bool(meeting_ids and any(m for m in meeting_ids))
    if subjects and (anchored_enabled or has_meetings):
        groups = _split_groups_by_meeting(
            groups,
            meeting_ids,
            subjects=subjects,
            received_ats=received_ats,
            conversation_ids=conversation_ids,
            settings=active,
            run_id=run_id,
        )

    sizes = sorted((len(g) for g in groups), reverse=True)
    logger.info(
        "[CLUSTER-ALGO run=%s] algo=%s threshold=%.3f n=%d clusters=%d sizes=%s anchored=%s",
        run_id, algo, threshold, len(vectors), len(groups), sizes, anchored_enabled,
    )
    return groups


def split_for_synthesis(indices: list[int], max_size: int) -> list[list[int]]:
    """Split a cluster into batches of at most `max_size` emails."""
    if max_size <= 0:
        return [indices]
    return [indices[i:i + max_size] for i in range(0, len(indices), max_size)]


def quoted_nested_subjects(body_text: str) -> list[str]:
    """Subject lines quoted in a forwarded/reply chain (before noise stripping)."""
    found: list[str] = []
    seen: set[str] = set()
    for match in _NESTED_SUBJECT_RE.finditer(body_text or ""):
        subject = normalize_subject(match.group(1) or "")
        key = subject.lower()
        if not subject or key in seen:
            continue
        seen.add(key)
        found.append(subject)
    return found


def _subject_is_cited(member_subject: str, quoted_subject: str) -> bool:
    """True when a nested Subject: line refers to this email's subject."""
    member = normalize_subject(member_subject).lower()
    quoted = normalize_subject(quoted_subject).lower()
    if not member or not quoted:
        return False
    if member == quoted:
        return True
    if len(member) >= 12 and member in quoted:
        return True
    if len(quoted) >= 12 and quoted in member:
        return True
    return False


def merge_quoted_subject_links(
    groups: list[list[int]],
    subjects: list[str],
    bodies: list[str] | None,
    vectors: list[np.ndarray] | None = None,
    settings: Settings | None = None,
    run_id: str = "",
) -> list[list[int]]:
    """
    Join emails that compile each other: a nested 'Subject:' line in one
    body matching another candidate's subject.

    This is how a discussion-starter that forwards several working-paper
    threads (balance sheet, targets, funding pack) becomes one cluster
    without naming those products. Recurring-series clusters stay locked
    so a quote inside minutes cannot swallow a different workstream.

    A cosine floor (`email_cluster_cite_min_cosine`, default 0.50) is
    applied on top of the subject-citation match so that a forwarded
    compilation email cannot act as a hub-and-spoke connector between
    completely unrelated topics purely through quoted Subject: headers.
    Set to 0.0 to restore the old string-match-only behaviour.
    """
    n = len(subjects)
    if n < 2 or not bodies or len(bodies) != n:
        return groups

    active = settings or get_settings()
    jaccard_min = float(getattr(active, "email_cluster_series_jaccard", 0.5) or 0.5)
    dept_tokens = department_stem_tokens(active)
    stems = [series_subject_tokens(subject) for subject in subjects]
    locked: set[int] = set()
    for group in groups:
        if _is_recurring_series_group(group, stems, jaccard_min, dept_tokens):
            locked.update(group)

    cite_floor = float(getattr(active, "email_cluster_cite_min_cosine", 0.50) or 0.0)
    has_vectors = vectors is not None and len(vectors) == n

    extra: list[tuple[int, int]] = []
    nested = [quoted_nested_subjects(body) for body in bodies]
    for i, quotes in enumerate(nested):
        if not quotes:
            continue
        for j, subject in enumerate(subjects):
            if i == j:
                continue
            if i in locked or j in locked:
                continue
            if not any(_subject_is_cited(subject, quoted) for quoted in quotes):
                continue
            # Cosine gate: subject-citation match alone is not enough when
            # a forwarded email compiles many unrelated threads. Require a
            # minimum semantic similarity so only genuinely related emails
            # are joined. Skip the gate when vectors are unavailable.
            if has_vectors and cite_floor > 0.0:
                sim = cosine_similarity(vectors[i], vectors[j])
                if sim < cite_floor:
                    logger.info(
                        "[CLUSTER-CITE-SKIP run=%s] from=%d to=%d quoted=%r cosine=%.3f < floor=%.2f",
                        run_id, i, j, normalize_subject(subject), sim, cite_floor,
                    )
                    continue
            extra.append((i, j))
            logger.info(
                "[CLUSTER-CITE run=%s] from=%d to=%d quoted=%r",
                run_id, i, j, normalize_subject(subject),
            )

    if not extra:
        return groups

    pairs: list[tuple[int, int]] = list(extra)
    for group in groups:
        for left, right in zip(group, group[1:]):
            pairs.append((left, right))
    return _union_find_groups(n, pairs)


def candidate_embed_text(candidate: EmailCandidate, settings: Settings | None = None) -> str:
    """
    Text fed into the clustering embedding model.

    Two strategies, selected by EMAIL_EMBED_SEMANTIC_ENABLED:

    1. Semantic (default): build the embedding from the LLM-extracted
       business semantics — business objective + topic + subtopics +
       entities + actions/decisions — so the vector captures the complete
       business CONTEXT of the email rather than its surface wording. This
       is what keeps two emails that share only a broad topic but drive
       DIFFERENT objectives from clustering together: the objective (the
       discriminator) is repeated `EMAIL_EMBED_OBJECTIVE_WEIGHT` times so it
       dominates the centroid, while topic/subtopic/entity/action lines add
       the surrounding context. A short slice of raw body text
       (EMAIL_EMBED_SEMANTIC_BODY_CONTEXT_CHARS) plus attachment text is
       appended as grounding so concrete figures/names still anchor the
       vector and cross-topic mentions (e.g. ORION inside WBR minutes) are
       preserved.
       See `build_semantic_embed_text`.

    2. Raw fallback: subject (once) + nested quoted subjects + stripped body
       + attachment text. Used when semantic embedding is disabled, or per
       email when its semantics are empty (topic tagging disabled or the
       extractor returned nothing) so the pipeline never emits an empty
       embedding. See `build_raw_embed_text`.

    Attachment text is capped to EMAIL_EMBED_ATTACHMENT_MAX_CHARS in both
    modes so a large MoM DOCX (e.g. 24 KB Digital Steerco minutes) does not
    drown out the primary signal.
    """
    active = settings or get_settings()

    if bool(getattr(active, "email_embed_semantic_enabled", True)):
        semantic = build_semantic_embed_text(candidate, active)
        if semantic:
            return semantic
        # No semantics for this email (tagging disabled/failed) — fall back
        # to raw text so this candidate still gets a meaningful embedding.
        logger.debug(
            "[CLUSTER-EMBED] no semantics for subject=%r; using raw embed text",
            candidate.message.subject,
        )

    return build_raw_embed_text(candidate, active)


def _dedupe_phrases(phrases: list[str]) -> list[str]:
    """Trim/drop-empty and remove case-insensitive duplicate phrases, order-preserving."""
    seen: set[str] = set()
    out: list[str] = []
    for phrase in phrases:
        cleaned = re.sub(r"\s+", " ", str(phrase)).strip(" \t-•·,;")
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


def build_semantic_embed_text(candidate: EmailCandidate, settings: Settings | None = None) -> str:
    """
    Compose the structured business-context embedding text for one email.

    Layout (labels are kept in the text so the embedding model sees each
    element's role, and so two emails are compared objective-to-objective,
    topic-to-topic, etc. rather than as one undifferentiated blob):

        Objective: <business objective>          (repeated N times)
        Topic: <topic 1>; <topic 2>; ...         (primary + subtopics)
        Entities: <entity 1>; <entity 2>; ...
        Actions: <action 1>; <action 2>; ...
        Context: <short raw-body slice>           (grounding, optional)
        <attachment text, capped>                 (grounding, optional)

    Returns "" when the email carries NO structured semantics at all
    (objective, topics, entities and actions are all empty) so the caller
    can fall back to raw-text embedding.
    """
    active = settings or get_settings()
    msg = candidate.message

    objective = re.sub(r"\s+", " ", (candidate.business_objective or "")).strip()
    topics = _dedupe_phrases(candidate.topics or [])
    entities = _dedupe_phrases(candidate.entities or [])
    actions = _dedupe_phrases(candidate.actions or [])

    if not objective and not topics and not entities and not actions:
        return ""

    weight = max(1, int(getattr(active, "email_embed_objective_weight", 2) or 1))
    body_ctx_cap = max(0, int(getattr(active, "email_embed_semantic_body_context_chars", 600) or 0))
    attach_cap = max(0, int(getattr(active, "email_embed_attachment_max_chars", 8000) or 0))

    parts: list[str] = []

    # Business objective — the discriminator. Repeated so it dominates the
    # centroid: this is what keeps same-topic / different-objective emails
    # in separate clusters. Fall back to the derived topic_key or the
    # subject only when the extractor produced no objective, so this line
    # is never empty.
    objective_line = objective or (candidate.topic_key or "").strip() or normalize_subject(msg.subject or "")
    if objective_line:
        for _ in range(weight):
            parts.append(f"Objective: {objective_line}")

    # Topic + subtopics. Lead with the subject as the primary surface topic
    # (once) then the extracted topics/subtopics.
    topic_terms: list[str] = []
    subject_norm = normalize_subject(msg.subject or "")
    if subject_norm:
        topic_terms.append(subject_norm)
    topic_terms.extend(topics)
    topic_terms = _dedupe_phrases(topic_terms)
    if topic_terms:
        parts.append("Topic: " + "; ".join(topic_terms))

    if entities:
        parts.append("Entities: " + "; ".join(entities))

    # Actions / decisions requested or taken.
    if actions:
        parts.append("Actions: " + "; ".join(actions))

    # Grounding: a short raw-body slice so concrete figures/names anchor the
    # vector without letting a long body overwhelm the structured signal.
    if body_ctx_cap:
        body = strip_email_noise(msg.body_text or "")
        if body:
            parts.append("Context: " + body[:body_ctx_cap])

    # Grounding: attachment (MoM) text, capped. Preserves cross-topic
    # mentions (e.g. ORION discussed inside July WBR minutes) so the WBR
    # card still surfaces in the ORION pre-meet brief.
    if candidate.attachment_text:
        text = candidate.attachment_text
        if attach_cap:
            text = text[:attach_cap]
        if text.strip():
            parts.append(text)

    return "\n".join(p for p in parts if p).strip()


def build_raw_embed_text(candidate: EmailCandidate, settings: Settings | None = None) -> str:
    """
    Raw-text embedding (previous default / fallback).

    Subject (once) + nested quoted subjects + stripped body + attachment
    text. Subject appears exactly once — repeating it distorts the centroid
    by over-weighting the subject relative to the actual content.

    Attachment text is capped to EMAIL_EMBED_ATTACHMENT_MAX_CHARS so a large
    MoM DOCX does not drown out the subject/body signal.
    """
    active = settings or get_settings()
    msg = candidate.message
    subject = (msg.subject or "").strip()
    body_cap = max(0, int(active.email_embed_body_max_chars))
    attach_cap = max(0, int(active.email_embed_attachment_max_chars))

    parts: list[str] = []
    # Subject exactly once — no repetition.
    if subject:
        parts.append(subject)
    for nested in quoted_nested_subjects(msg.body_text or ""):
        if nested.lower() != subject.lower():
            # Quoted subjects from forwarded chains: single copy so they
            # don't dilute the primary topic. A deeply-forwarded compilation
            # email otherwise becomes a blended centroid of many unrelated
            # topics, causing it to cosine-match with everything.
            parts.append(nested)
    body = strip_email_noise(msg.body_text or "")
    if body and body_cap:
        parts.append(body[:body_cap])
    elif body:
        parts.append(body)
    # Attachment text (MoM DOCX / PDF) capped so large files don't drown
    # out subject+body. This is what surfaces cross-topic mentions (e.g.
    # ORION discussed inside July WBR minutes → July WBR card appears in
    # the Project ORION pre-meet brief).
    if candidate.attachment_text:
        text = candidate.attachment_text
        if attach_cap:
            text = text[:attach_cap]
        parts.append(text)
    return "\n".join(p for p in parts if p).strip()


def _candidate_has_semantics(candidate: EmailCandidate) -> bool:
    """True when the candidate already carries any structured semantics.

    The relevance stage is the single extractor and normally fills these; a
    candidate with none (e.g. its extraction returned empty, or the candidate
    arrived from a path that did not extract) needs the DB fallback below.
    """
    return bool(
        candidate.business_objective
        or candidate.topics
        or candidate.entities
        or candidate.actions
    )


async def tag_candidates_with_topics(
    candidates: list[EmailCandidate],
    settings: Settings | None = None,
    run_id: str = "",
) -> None:
    """Attach the one-line topic key + structured semantics to each candidate.

    This stage makes NO LLM call. The structured semantics were already
    extracted ONCE at the relevance stage and ride on the candidate; this
    stage only derives the display ``topic_key`` from them and persists the
    per-mail row. A single bulk DB read (no N+1) backfills any candidate that
    reached here without in-memory semantics — e.g. after a crash/resume, or
    if an individual extraction came back empty — reusing the persisted
    extraction rather than re-running the model.
    """
    active = settings or get_settings()
    if not candidates:
        return

    # Bulk-backfill ONLY the candidates missing in-memory semantics, in a
    # single query over all their message_ids (never one query per email).
    missing = [c for c in candidates if not _candidate_has_semantics(c)]
    if missing:
        from agent_email import store

        missing_ids = [c.message.id for c in missing if c.message.id]
        try:
            by_id = await store.read_email_semantics(missing_ids, run_id=run_id, settings=active)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[TOPIC-TAG run=%s] semantics backfill read failed (%s); "
                "candidates without semantics fall back to raw-text embedding",
                run_id, exc,
            )
            by_id = {}
        filled = 0
        for candidate in missing:
            sem = by_id.get(candidate.message.id or "")
            if not sem:
                continue
            candidate.business_objective = sem.get("business_objective", "") or ""
            candidate.topics = list(sem.get("topics", []) or [])
            candidate.entities = list(sem.get("entities", []) or [])
            candidate.actions = list(sem.get("actions", []) or [])
            candidate.subtopics = list(sem.get("subtopics", []) or [])
            candidate.decisions = list(sem.get("decisions", []) or [])
            candidate.context = sem.get("context", "") or ""
            filled += 1
        logger.info(
            "[TOPIC-TAG run=%s] semantics backfill: %d missing, %d filled from DB",
            run_id, len(missing), filled,
        )

    # Derive the display topic_key from the (reused) semantics for every
    # candidate. No model call here.
    for candidate in candidates:
        semantics = {
            "business_objective": candidate.business_objective,
            "topics": candidate.topics,
        }
        candidate.topic_key = topic_key_from_semantics(semantics)
        logger.info(
            "[TOPIC-TAG run=%s] subject=%r topic_key=%r objective=%r topics=%s",
            run_id, candidate.message.subject, candidate.topic_key,
            candidate.business_objective, candidate.topics,
        )

    from agent_email import store

    await store.write_topic_records(candidates, run_id=run_id, settings=active)


async def embed_candidates(
    candidates: list[EmailCandidate],
    settings: Settings | None = None,
) -> list[np.ndarray]:
    """Embed the clustering text, never the relevance-gate vector.

    The gate vector decides which mail stays in the pool. Clustering and
    anchoring use `candidate_embed_text` (objective repeated, subject plus
    topics, entities, actions, a short body slice, and the minutes).
    """
    active = settings or get_settings()
    return list(await asyncio.gather(*(
        embed_text(candidate_embed_text(c, active), active) for c in candidates
    )))


def mean_embedding(vectors: list[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(vectors)
    centroid = stacked.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


async def extract_mom_attachment_text(
    raw_attachment: dict,
    settings: Settings | None = None,
) -> tuple[str, str]:
    """Direct text extraction for MoM attachments. Returns (type, text)."""
    name = raw_attachment.get("name", "unknown")
    content_bytes_b64 = raw_attachment.get("contentBytes")
    if not content_bytes_b64:
        return "unsupported", ""

    active = settings or get_settings()
    if not is_mom_attachment(name):
        return "unsupported", ""

    raw_bytes = base64.b64decode(content_bytes_b64)
    lower_name = name.lower()

    if lower_name.endswith(".pdf"):
        try:
            return "pdf", extract_pdf_text(raw_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed PDF extract '%s': %s", name, exc)
            return "pdf", ""

    if lower_name.endswith((".xlsx", ".xlsm")):
        try:
            return "excel", extract_excel_text(raw_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed Excel extract '%s': %s", name, exc)
            return "excel", ""

    if lower_name.endswith((".pptx", ".pptm")):
        try:
            return "pptx", extract_pptx_text(raw_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed PPTX extract '%s': %s", name, exc)
            return "pptx", ""

    if lower_name.endswith((".docx", ".docm", ".doc")):
        try:
            return "docx", extract_docx_text(raw_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed DOCX extract '%s': %s", name, exc)
            return "docx", ""

    return "unsupported", ""


async def enrich_candidates_with_attachments(
    candidates: list[EmailCandidate],
    client: GraphClient,
    bearer_token: str | None,
    settings: Settings | None = None,
) -> None:
    """Fetch and extract MoM attachment text for each candidate in-place."""
    active = settings or get_settings()

    async def _one(candidate: EmailCandidate) -> None:
        msg = candidate.message
        if not msg.has_attachments:
            return
        try:
            raw_attachments = await client.get_attachments(msg.id, bearer_token=bearer_token)
            texts: list[str] = []
            for raw_att in raw_attachments.get("value", []):
                raw_detail = await client.get_attachment_content(
                    msg.id, raw_att["id"], bearer_token=bearer_token
                )
                att_type, att_text = await extract_mom_attachment_text(raw_detail, settings=active)
                if att_text:
                    texts.append(att_text)
            if texts:
                candidate.attachment_text = "\n\n".join(texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Attachment fetch failed message_id=%s: %s", msg.id, exc)

    await asyncio.gather(*(_one(c) for c in candidates))


def _cap_text(text: str, limit: int) -> str:
    """Slice to `limit` chars. limit <= 0 means unlimited (full text)."""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit]


def format_emails_for_synthesis(
    candidates: list[EmailCandidate],
    indices: list[int],
    settings: Settings | None = None,
    meeting_id: str = "",
    notes_data: list[dict] | None = None,
) -> str:
    """
    Append full email text (subject / stripped body / MoM) for LLM synthesis.

    Caps come from `email_synth_*_max_chars` (0 = no truncation). Clustering
    embed caps are a separate setting and must not apply here.

    Args:
        candidates: List of email candidates in the cluster
        indices: Indices of candidates to include
        settings: Configuration for truncation limits
        meeting_id: Optional meeting_id to fetch associated notes
        notes_data: Pre-fetched notes data (to avoid N+1 queries)
                   Should be list of dicts with keys: note_text, created_at

    Returns:
        Formatted string for LLM prompt
    """
    active = settings or get_settings()
    body_cap = int(getattr(active, "email_synth_body_max_chars", 0) or 0)
    attach_cap = int(getattr(active, "email_synth_attachment_max_chars", 0) or 0)
    total = len(indices)
    blocks: list[str] = []
    
    # Append emails first
    for ordinal, idx in enumerate(indices, start=1):
        c = candidates[idx]
        msg = c.message
        body = _cap_text(strip_synthesis_noise(msg.body_text or ""), body_cap)
        attach = _cap_text(c.attachment_text or "", attach_cap)
        block = (
            f"--- Email {ordinal} of {total} ---\n"
            f"Subject: {msg.subject}\n"
            f"Date: {msg.received_at.isoformat() if msg.received_at else ''}\n"
            f"Body:\n{body}\n"
        )
        if attach:
            block += f"MoM Attachment:\n{attach}\n"
        blocks.append(block)
    
    # Append CEO notes if available
    if notes_data:
        for i, note in enumerate(notes_data, start=total + 1):
            note_text = note.get("note_text", "")
            if not note_text or not note_text.strip():
                continue
            created_at = note.get("created_at", "")
            # Structured team/department tags the CEO attached to the note
            # (e.g. ["Risk", "BB&RL"]). Surface them explicitly so synthesis
            # knows which teams the note was directed at instead of relying
            # on the "@Team" text buried in the note body.
            raw_tags = note.get("tags") or []
            tags = [str(t).strip() for t in raw_tags if str(t).strip()] if isinstance(raw_tags, list) else []
            tags_line = f"Tagged teams: {', '.join(tags)}\n" if tags else ""
            block = (
                f"--- CEO Note {i} ---\n"
                f"Date: {created_at}\n"
                f"{tags_line}"
                f"Note:\n{note_text}\n"
            )
            blocks.append(block)
    
    return "\n\n".join(blocks)


def normalize_subject(subject: str) -> str:
    normalized = (subject or "").strip()
    prefix = re.compile(r"^(re|fw|fwd)\s*:\s*", re.IGNORECASE)
    while True:
        new = prefix.sub("", normalized).strip()
        if new == normalized:
            break
        normalized = new
    return normalized
