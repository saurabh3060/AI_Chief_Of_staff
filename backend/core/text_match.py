"""
core/text_match.py
======================
Shared lightweight text-matching helpers used by BOTH the calendar
agent (meeting -> department, for view routing) and the email agent
(meeting -> related emails, for meeting-specific retrieval).

Kept dependency-free (no embeddings/ML here - see
`agent_email.semantic_match` for the embedding-based layer) so this
module can be imported from either agent without pulling in Ollama or
any I/O. Extracted out of `agent_calendar.service` so the same
stopword list / tokenizer / overlap scoring isn't duplicated (and
drift) between the two agents.
"""

import math
import re

# Common words that carry no real topic/department signal - excluded
# from token matching so a meeting/email's filler words ("meeting",
# "regarding", "update", "review", "business") can't create spurious
# overlap between genuinely unrelated items. This list is deliberately
# shared: it is the same set of generic words the requirements call out
# ("review", "update", "meeting", "business") that must NOT by
# themselves cause a match.
MATCH_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "at",
    "by", "from", "re", "fw", "fwd", "meeting", "meetings", "call", "sync",
    "review", "update", "updates", "follow", "up", "followup", "discussion",
    "regarding", "re:", "about", "session", "catchup", "catch", "invite",
    "agenda", "notes", "mom", "minutes", "data", "business", "tomorrow",
    "today", "yesterday", "questions", "question", "please", "thanks",
    "regards", "team", "dear", "hi", "hello",
})


def normalize_match_text(text: str) -> str:
    """Lowercase and collapse whitespace so fuzzy comparisons aren't thrown off by casing/spacing noise."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


# --- Noise stripping (quoted reply chains / signatures / disclaimers) ---
#
# Real-world mail bodies frequently carry along the ENTIRE prior reply
# chain (quoted "> " lines, "-----Original Message-----" blocks, "On
# ... wrote:" headers, or a raw "From:/Sent:/To:/Subject:" quoted
# header block) plus a fixed corporate disclaimer/signature at the very
# end. None of that is the sender's actual new content, and left in it
# dilutes whatever embedding is computed from the text - two unrelated
# emails that both quote the same earlier thread, or both carry the
# same bank-wide disclaimer, end up looking more similar than they
# really are purely because of shared boilerplate, not shared topic.
# This is intentionally regex-based (no HTML/MIME parsing) and only
# ever TRUNCATES or removes lines - it never rewrites remaining content
# - so a body with no such markers passes through unchanged.
_QUOTE_MARKERS = [
    re.compile(r"(?im)^[ \t'\"`]*-{2,}\s*original message\s*-{2,}.*", re.DOTALL),
    re.compile(r"(?im)^on .{0,120}wrote:\s*$", re.DOTALL),
    re.compile(r"(?im)^from:\s.*", re.DOTALL),
]

_DISCLAIMER_MARKERS = [
    re.compile(r"(?is)this e-?mail( message)?( and any attachments)?.*"),
    re.compile(r"(?is)confidentiality notice.*"),
    re.compile(r"(?is)disclaimer\s*:.*"),
]

_QUOTE_LINE_RE = re.compile(r"(?m)^[ \t]*>.*$")
# Outlook / Exchange prepends this banner to every inbound external mail.
# Left in, it is identical across unrelated messages and inflates cosine
# similarity of short-body emails above the clustering threshold.
_BANNER_LINE_RE = re.compile(r"(?im)^[ \t]*\[?external e-?mail\]?:?[ \t].*$")


def strip_email_noise(text: str) -> str:
    """
    Best-effort removal of quoted reply-chain history and common
    disclaimer/signature boilerplate from an email body BEFORE it is
    embedded (semantic matching, department tagging, cluster-merge
    comparisons). Never raises - a body with none of the known markers
    is returned unchanged (aside from whitespace collapsing), so this
    only ever removes noise, it never risks stripping genuine content
    from a body that doesn't match any pattern.

    This is deliberately lightweight (no HTML/MIME quote-depth
    parsing) - it targets the handful of boilerplate patterns that are
    common in Outlook-originated mail (this backend's actual mailbox
    source) and is meant to improve embedding signal quality, not to
    be a complete quote-stripping implementation.
    """
    if not text:
        return ""
    result = _BANNER_LINE_RE.sub("", text)
    for pattern in _QUOTE_MARKERS:
        match = pattern.search(result)
        if match:
            result = result[: match.start()]
    for pattern in _DISCLAIMER_MARKERS:
        match = pattern.search(result)
        if match:
            result = result[: match.start()]
    result = _QUOTE_LINE_RE.sub("", result)
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# Routing header lines in a forwarded/replied thread. These name the
# mailbox From/To/Cc, which must not be treated as meeting owners.
_ROUTING_HEADER_LINE_RE = re.compile(
    r"^[ \t]*(from|to|cc|bcc|sent|date|subject|thread-topic|thread-index)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)
_ORIGINAL_MESSAGE_LINE_RE = re.compile(
    r"^[ \t]*-{2,}\s*original message\s*-{2,}[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_synthesis_noise(text: str) -> str:
    """
    Light clean for cluster synthesis: drop the external-mail banner,
    trailing disclaimer, and From/To/Cc/Sent routing lines. Keep the
    full reply thread so figures and disagreements in earlier messages
    still reach the brief. Do not use strip_email_noise here - that
    truncates at the first From: and would hide the thread.
    """
    if not text:
        return ""
    result = _BANNER_LINE_RE.sub("", text)
    for pattern in _DISCLAIMER_MARKERS:
        match = pattern.search(result)
        if match:
            result = result[: match.start()]
    result = _ORIGINAL_MESSAGE_LINE_RE.sub("", result)
    result = _ROUTING_HEADER_LINE_RE.sub("", result)
    result = _QUOTE_LINE_RE.sub("", result)
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# Known department-name abbreviations that must be treated as equivalent
# to their spelled-out form for token-overlap purposes. Without this, a
# meeting titled "Wholesale Banking Monthly Business Review" (spelled
# out) shares ZERO tokens with an email subject like "WB Monthly Review
# - August | Key Actions" (abbreviated) once "wb" is dropped by the
# length filter below - even though both are unambiguously about the
# same business unit. This was flagged as a known gap in
# `weighted_token_overlap_score`'s docstring ("doesn't fix the
# abbreviation case by itself - that still needs a synonym/alias step")
# and is what caused genuinely relevant MBR emails using the "WB"
# shorthand to fall below `meeting_email_min_shared_tokens` and get
# dropped despite a strong semantic score. Expansion is additive (the
# raw tokens are still tokenized/filtered as before); it only ever adds
# extra signal, never removes any.
_ABBREVIATION_EXPANSIONS: dict[str, frozenset[str]] = {
    "wb": frozenset({"wholesale", "banking"}),
    # BB&RL (Branch Banking & Retail Liabilities) is written many ways in
    # real subjects/bodies: "BB&RL", "BBRL", "BB RL", "BB/RL". The "&" and
    # "/" separators, plus the fact that "bb"/"rl" are each < 3 chars, mean
    # every one of these tokenizes to NOTHING under the length filter, so a
    # subject that literally names the department shares zero tokens with
    # the "BB&RL" dictionary name. Collapse them all to one canonical token
    # so the token-overlap tie-break can route these to BB&RL. Note the "&"
    # form arrives here as the single token "bbrl" (findall on [a-z0-9]+
    # already drops the "&"), and the spaced/slashed forms arrive as "bb"
    # then "rl"; mapping all three keys covers every permutation.
    "bbrl": frozenset({"bbrl"}),
    "bb": frozenset({"bbrl"}),
    "rl": frozenset({"bbrl"}),
}

# "CA/SA" is the same product as "CASA", but the slash splits into two
# 2-char tokens that the length filter drops. Dictionary phrases use the
# glued form, so the slash form would otherwise contribute no overlap.
_CASA_SLASH_RE = re.compile(r"\bca\s*/\s*sa\b")


def meaningful_tokens(text: str) -> set[str]:
    """
    Split text into lowercased alphanumeric word tokens, dropping
    stopwords and tokens shorter than 3 chars. These are the words that
    actually carry topic/department signal (e.g. 'wholesale',
    'corporate', 'branch', 'collections', 'cards', 'acquisition').

    Known short abbreviations (see `_ABBREVIATION_EXPANSIONS`, e.g. "wb"
    for "Wholesale Banking") are additionally expanded into their
    spelled-out equivalent tokens, so an abbreviated subject/body still
    overlaps with query text that uses the full department name.
    The slash form "CA/SA" is treated as the same product token as "CASA".
    """
    normalized = normalize_match_text(text)
    raw_tokens = re.findall(r"[a-z0-9]+", normalized)
    tokens = {t for t in raw_tokens if len(t) >= 3 and t not in MATCH_STOPWORDS}
    for t in raw_tokens:
        tokens |= _ABBREVIATION_EXPANSIONS.get(t, frozenset())
    if _CASA_SLASH_RE.search(normalized):
        tokens.add("casa")
    return tokens


def token_overlap_score(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    """
    0-100 score = share of the QUERY's meaningful tokens that also
    appear in `candidate_tokens`. Anchored on the query's own words, so
    a candidate only scores if it actually mentions words the query
    uses - generic overlap on stopwords/short tokens can't inflate it
    since those are already excluded from both sets.

    Flat/unweighted: every shared token counts equally, regardless of
    how common that word is across the corpus being matched against.
    See `weighted_token_overlap_score` for a rarity-aware alternative -
    kept as a separate function (not a breaking change to this one)
    since some callers (e.g. `agent_calendar.service.match_meeting_to_
    department`, department-level matching) still want the simple,
    unweighted version.
    """
    if not query_tokens or not candidate_tokens:
        return 0.0
    matched = len(query_tokens & candidate_tokens)
    return 100.0 * matched / len(query_tokens)


def build_document_frequencies(token_sets: list[set[str]]) -> dict[str, int]:
    """
    Count, across a corpus of token sets (e.g. one set per candidate
    email), how many documents each token appears in at least once.
    This is the raw input to `weighted_token_overlap_score`'s
    inverse-document-frequency (IDF) weighting.

    A word that shows up in almost every document in the corpus (e.g.
    a BU/department name like "wholesale" or "banking", which appears
    on nearly every email from that BU) carries little discriminating
    power for telling two DIFFERENT topics within that BU apart. A word
    that shows up in only one or two documents (e.g. "deposit", "tat",
    "collections") is much stronger evidence that two texts are
    genuinely about the same specific topic, not just the same BU.
    """
    doc_freq: dict[str, int] = {}
    for tokens in token_sets:
        for token in tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    return doc_freq


def token_idf_weight(token: str, doc_freq: dict[str, int], corpus_size: int) -> float:
    """
    Inverse-document-frequency weight for one token: high for rare
    tokens (appear in few documents), low for common tokens (appear in
    most/all documents). Uses the standard smoothed-IDF formula
    `ln((N + 1) / (df + 1)) + 1` so every token gets a weight > 0 (a
    token that appears in EVERY document still counts a little, just
    far less than a token that appears in only one) and a token never
    seen in the corpus (df=0, e.g. a word only in the query) gets the
    maximum weight rather than dividing by zero.
    """
    if corpus_size <= 0:
        return 1.0
    df = doc_freq.get(token, 0)
    return math.log((corpus_size + 1) / (df + 1)) + 1.0


def weighted_token_overlap_score(
    query_tokens: set[str],
    candidate_tokens: set[str],
    doc_freq: dict[str, int],
    corpus_size: int,
) -> float:
    """
    Rarity-weighted alternative to `token_overlap_score`: instead of
    counting every shared token equally, each shared token contributes
    proportional to its IDF weight (see `token_idf_weight`) - so one
    shared RARE, distinctive word (e.g. "deposit", "tat") can outweigh
    two shared COMMON words that merely reflect the two texts belong to
    the same business unit (e.g. "wholesale", "banking").

    This directly targets the failure mode a flat token-count score
    (or a hard "must share >= N tokens" gate on top of it) is prone to:
    two texts about DIFFERENT topics within the same BU sharing just
    enough generic BU-name words to look like a real match, while two
    texts about the SAME topic that happen to use different generic
    wording (e.g. "WB Monthly Review" vs "Wholesale Banking Monthly
    Business Review" - "WB" vs "Wholesale"/"Banking" spelled out) score
    lower than they should on raw token count alone. Rarity weighting
    doesn't fix the abbreviation case by itself (that still needs a
    synonym/alias step - see `meaningful_tokens` callers), but it does
    make the score reflect topic-specificity rather than word-count,
    which is the more robust foundation for that abbreviation handling
    (or any future ensemble scoring with other signals) to sit on top of.

    Result is normalized to 0-100 by dividing by the query's own total
    IDF weight (analogous to `token_overlap_score` dividing by
    `len(query_tokens)`), so it stays comparable in scale.
    """
    if not query_tokens or not candidate_tokens:
        return 0.0
    matched = query_tokens & candidate_tokens
    if not matched:
        return 0.0
    matched_weight = sum(token_idf_weight(t, doc_freq, corpus_size) for t in matched)
    query_weight = sum(token_idf_weight(t, doc_freq, corpus_size) for t in query_tokens)
    if query_weight <= 0:
        return 0.0
    return 100.0 * matched_weight / query_weight


# HTML stripping (shared by email + calendar agents).
_HTML_DROP_BLOCK_RE = re.compile(
    r"<style\b[^>]*>.*?</\s*style\b[^>]*>"
    r"|<script\b[^>]*>.*?</\s*script\b[^>]*>"
    r"|<head\b[^>]*>.*?</\s*head\b[^>]*>"
    r"|<!--.*?-->",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITIES = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&apos;": "'", "&rsquo;": "'",
    "&lsquo;": "'", "&ldquo;": '"', "&rdquo;": '"', "&mdash;": "-",
    "&ndash;": "-", "&hellip;": "...",
}


def html_to_text(body_text: str, content_type: str = "") -> str:
    """
    Convert HTML email/calendar body to plain text. Removes style/script/
    blocks wholesale, strips tags, unescapes entities, collapses whitespace.
    """
    if not body_text:
        return ""

    looks_html = content_type.lower() == "html" or ("<" in body_text and ">" in body_text)
    if not looks_html:
        return body_text.strip()

    text = _HTML_DROP_BLOCK_RE.sub(" ", body_text)
    text = re.sub(r"(?i)<\s*br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</\s*(p|div|tr|li|h[1-6])\s*>", "\n", text)
    text = _HTML_TAG_RE.sub("", text)

    for entity, replacement in _HTML_ENTITIES.items():
        text = text.replace(entity, replacement)
    text = re.sub(r"&#x?[0-9a-fA-F]+;", " ", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
