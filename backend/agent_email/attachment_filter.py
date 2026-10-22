"""
agent_email/attachment_filter.py
====================================
Fuzzy-matching gate that decides whether an attachment's FILENAME looks
like a Minutes of Meeting (MoM) document, before any text extraction
(PyMuPDF/openpyxl/python-pptx/vision OCR) is ever attempted on it.

Why this exists: previously EVERY attachment on a matched BU email (any
PDF/Excel/PPTX/image, regardless of what it actually was - decks, stray
forwards, logos, unrelated reports) was extracted and its full text
dumped into the single LLM prompt for that email. That bloats the input
prompt with irrelevant content, which is exactly what produces vague or
incorrect action items/summaries - the model has to wade through
unrelated attachment text to find the one thing (the actual MoM) that
matters. This module lets `agent_email.ocr.parse_attachment` and
`agent_email.service._extract_bu_attachment_text` skip anything that
doesn't look like a MoM by filename, before spending any extraction
work on it.

Toggle via `Settings.mom_attachment_filter_enabled` (MOM_ATTACHMENT_FILTER_ENABLED
in `.env`) and tune matching via `Settings.mom_attachment_keywords_raw`
(MOM_ATTACHMENT_KEYWORDS) / `Settings.mom_attachment_fuzzy_threshold`
(MOM_ATTACHMENT_FUZZY_THRESHOLD) - no code change needed to adjust
either.

The fuzzy scoring below is implemented from scratch (plain Levenshtein
edit-distance) rather than pulling in a third-party fuzzy-matching
library, since this one small comparison doesn't warrant an extra
dependency.
"""

from core.config import Settings, get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("agent_email")


def _levenshtein_distance(a: str, b: str) -> int:
    """
    Classic dynamic-programming edit distance between two strings: the
    minimum number of single-character insertions, deletions, or
    substitutions needed to turn `a` into `b`. O(len(a) * len(b)) time
    and O(min(len(a), len(b))) space (single rolling row).
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Keep `b` as the shorter string so the rolling row is as small as
    # possible.
    if len(a) < len(b):
        a, b = b, a

    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            substitute_cost = previous_row[j - 1] + (0 if ca == cb else 1)
            current_row[j] = min(insert_cost, delete_cost, substitute_cost)
        previous_row = current_row

    return previous_row[-1]


def _similarity_ratio(a: str, b: str) -> float:
    """
    Convert edit distance between `a` and `b` into a 0-100 similarity
    score: 100 means identical, 0 means completely dissimilar. Same
    normalization convention as common fuzzy-matching libraries:
    `100 * (1 - distance / max(len(a), len(b)))`.
    """
    if not a and not b:
        return 100.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 100.0
    distance = _levenshtein_distance(a, b)
    return 100.0 * (1 - distance / max_len)


def _partial_ratio(keyword: str, text: str) -> float:
    """
    Fuzzy "does `keyword` appear inside `text`" score (0-100), so a short
    keyword like "MoM" can still score highly against a long filename
    like "mom_wholesalebanking_08dec2025.pdf" that contains far more
    characters than the keyword itself - a plain whole-string similarity
    ratio would be dragged down by all the extra characters around the
    match.

    Slides a window the length of `keyword` across `text` and returns
    the best similarity ratio found at any window position. `keyword` is
    assumed to be the shorter of the two strings; if `text` is shorter
    than `keyword`, falls back to a plain whole-string ratio.
    """
    if len(text) <= len(keyword):
        return _similarity_ratio(keyword, text)

    best = 0.0
    window_len = len(keyword)
    # Slide across every possible window position; filenames are short
    # (a handful of words) so this stays cheap even done exhaustively.
    for start in range(len(text) - window_len + 1):
        window = text[start:start + window_len]
        score = _similarity_ratio(keyword, window)
        if score > best:
            best = score
        if best == 100.0:
            break
    return best


def is_mom_attachment(filename: str, settings: Settings | None = None) -> bool:
    """
    Return True if `filename` fuzzy-matches at least one configured MoM
    keyword (e.g. "Minutes of Meeting", "MoM") at or above the configured
    threshold, OR if the filter is disabled entirely (in which case every
    attachment is treated as eligible, preserving the pre-filter
    behaviour).

    Matching is case-insensitive and tolerant of minor filename noise
    (extra words, underscores/dates around the keyword, small typos)
    via `_partial_ratio`.
    """
    active_settings = settings or get_settings()
    if not active_settings.mom_attachment_filter_enabled:
        return True

    keywords = active_settings.mom_attachment_keywords
    if not keywords:
        # No keywords configured - fail open rather than silently
        # skipping every attachment because of a misconfiguration.
        logger.warning("MoM attachment filter is enabled but no keywords are configured; allowing all attachments")
        return True

    name_lower = filename.lower()
    threshold = active_settings.mom_attachment_fuzzy_threshold

    best_score = 0.0
    best_keyword = ""
    for keyword in keywords:
        score = _partial_ratio(keyword.lower(), name_lower)
        if score > best_score:
            best_score = score
            best_keyword = keyword

    matched = best_score >= threshold
    logger.info(
        "MoM attachment filter: filename='%s' best_keyword='%s' score=%.1f threshold=%d -> %s",
        filename, best_keyword, best_score, threshold, "PARSE" if matched else "SKIP",
    )
    return matched
