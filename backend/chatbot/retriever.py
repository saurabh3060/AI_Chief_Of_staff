"""
chatbot/retriever.py
=======================
Hybrid retrieval for the chatbot's RAG pipeline: Weaviate semantic search
+ BM25 keyword search, merged and reranked, then grouped from message-level
matches into full conversation threads.

Ported from the standalone chatbot_final project's retriever.py,
adapted to:
  - read tuning knobs (TOP_K, RETRIEVAL_CANDIDATES, EMBEDDING_MODEL)
    from core.config.get_settings() instead of a module-local config.py
  - use this project's centralised logger
  - fetch Weaviate objects via chatbot.weaviate_store (the already-ported
    collection management module) instead of a module-local
    weaviate_client.py
"""

import functools
import re
import time
from dataclasses import dataclass

import numpy as np
import ollama

from chatbot.reranker import rerank
from core.config import get_settings
from core.logging_config import get_agent_logger

logger = get_agent_logger("chatbot")

# nomic-embed-text's model maximum is 2048 tokens. Passing a larger num_ctx
# makes Ollama log "requested context size too large" and clamp it anyway,
# so cap the embedding context here rather than inheriting a large chat
# default. Chat generation uses its own (bigger) window — this is embed only.
_EMBED_NUM_CTX = 2048


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string using Ollama and return a 2-D numpy array.

    Returns shape ``(1, dim)`` with L2-normalised vectors so cosine
    similarity reduces to a dot product.
    """
    settings = get_settings()
    client = ollama.Client(host=settings.ollama_host)
    response = client.embed(
        model=settings.embedding_model,
        input=text,
        options={"num_ctx": _EMBED_NUM_CTX},
    )
    vec = np.array(response["embeddings"], dtype=np.float32)   # shape (1, dim)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vec / norms


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed several strings in ONE Ollama call; returns shape ``(n, dim)`` with
    L2-normalised rows (cosine similarity is a dot product). Same client, model
    and context cap as embed_query; used by build_context to score blocks.
    """
    settings = get_settings()
    client = ollama.Client(host=settings.ollama_host)
    response = client.embed(
        model=settings.embedding_model,
        input=texts,
        options={"num_ctx": _EMBED_NUM_CTX},
    )
    vec = np.array(response["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vec / norms


def _chunk_key(chunk: dict) -> str:
    """Return the durable message identity used across retrieval."""
    message_id = str(chunk.get("message_id") or "").strip()
    if message_id:
        return f"message:{message_id}"

    # Compatibility for meeting records outside email messages.
    meeting_id = str(chunk.get("meeting_id") or "").strip()
    if meeting_id:
        return f"meeting:{meeting_id}"
    return f"legacy:{chunk.get('email_id', '')}:{chunk.get('subject', '')}"


def _thread_key(chunk: dict) -> str:
    """Return the durable conversation (thread) identity for grouping."""
    conversation_id = str(chunk.get("conversation_id") or "").strip()
    if conversation_id:
        return f"conversation:{conversation_id}"
    meeting_id = str(chunk.get("meeting_id") or "").strip()
    if meeting_id:
        return f"meeting:{meeting_id}"
    return f"legacy:{chunk.get('email_id', '')}:{chunk.get('subject', '')}"


def _has_answer_content(chunk: dict) -> bool:
    """Only messages with real body/attachment/thread text are useful as context."""
    return bool(
        str(chunk.get("thread_raw_content") or chunk.get("raw_content") or "").strip()
    )


def _thread_chunk_from_message(message_chunk: dict) -> dict:
    """Expand one matched message chunk into a full-thread chunk for the LLM.

    The matched message already carries the full rendered thread
    (thread_raw_content / messages_json) denormalised onto it, so no
    second lookup is needed - this just reshapes it into the
    conversation-level dict shape that format_context()/service._get_sources()
    expect, while preserving the message-level scores for ranking/UI.
    """
    thread_chunk = dict(message_chunk)
    thread_chunk["subject"] = message_chunk.get("thread_subject") or message_chunk.get("subject", "")
    thread_chunk["raw_content"] = message_chunk.get("thread_raw_content") or message_chunk.get("raw_content", "")
    thread_chunk["source_type"] = "email_conversation"
    thread_chunk["matched_message_id"] = message_chunk.get("message_id", "")
    thread_chunk["matched_message_subject"] = message_chunk.get("subject", "")
    thread_chunk["matched_message_received_at"] = message_chunk.get("received_at", "")
    return thread_chunk


def _group_messages_into_threads(candidates: list[dict], top_k: int) -> list[dict]:
    """Group message-level chunks by conversation thread and return top_k threads.

    Each thread is represented by its single best-scoring message, expanded
    into a full-thread chunk via ``_thread_chunk_from_message``.  The thread's
    rank is determined by the highest ``combined_score`` among its messages.
    """
    # Best message per thread (keyed by _thread_key).
    best_by_thread: dict[str, dict] = {}
    for chunk in candidates:
        key = _thread_key(chunk)
        existing = best_by_thread.get(key)
        if existing is None or chunk.get("combined_score", 0) > existing.get("combined_score", 0):
            best_by_thread[key] = chunk

    # Sort threads by their best message score, descending.
    ranked = sorted(
        best_by_thread.values(),
        key=lambda c: c.get("combined_score", 0),
        reverse=True,
    )[:top_k]

    # Expand each winning message into a full-thread chunk.
    return [_thread_chunk_from_message(msg) for msg in ranked]



# ── Weaviate hybrid retrieval ─────────────────────────────────

def _message_chunk_from_properties(props: dict, score: float = 0.0) -> dict:
    """Convert Weaviate message properties into the common message chunk shape."""
    return {
        "message_id":        props.get("message_id", ""),
        "conversation_id":   props.get("conversation_id", ""),
        "message_index":     props.get("message_index", 0) or 0,

        "subject":           props.get("subject", ""),
        "body":              props.get("body", ""),
        "attachment_data":   props.get("attachment_data", ""),
        "metadata_summary":  props.get("metadata_summary", ""),
        "received_at":       props.get("received_at", ""),
        "from_address":      props.get("from_address", ""),
        "has_attachment":    props.get("has_attachment", False) or False,

        "thread_subject":      props.get("thread_subject", ""),
        "thread_raw_content":  props.get("thread_raw_content", ""),
        "raw_content":         props.get("thread_raw_content", ""),  # downstream compatibility
        "first_received_at":   props.get("first_received_at", ""),
        "last_received_at":    props.get("last_received_at", ""),
        "participants":        props.get("participants", []) or [],
        "message_ids":         props.get("message_ids", []) or [],
        "message_count":       props.get("message_count", 0) or 0,
        "messages_json":       props.get("messages_json", "[]"),

        "source_type":      props.get("source_type", "email_message"),
        "email_id":         props.get("email_id", 0),
        "score":            float(score),
        "bm25_score":       0.0,
        "combined_score":   float(score),
    }


def fetch_all_chunks_weaviate(client) -> list[dict]:
    """Fetch all message chunks from Weaviate to build a local BM25 index.

    Called once at startup so BM25 re-ranking can run locally after
    Weaviate returns its top-k results. Thin wrapper around
    chatbot.weaviate_store.fetch_all_chunks (already filters for usable
    content), kept under this name for parity with the standalone
    project and main.py-style startup code.
    """
    from chatbot._vec_backend import vec_store
    return vec_store().fetch_all_chunks(client)


def retrieve_weaviate(
    query: str,
    client,
    top_k: int | None = None,
    bm25_index=None,
) -> list[dict]:
    """Retrieve top-k threads from Weaviate using hybrid search at message
    granularity, BM25 re-rank, then expand matched messages into full threads.

    Pipeline:
      1. Embed query
      2. Weaviate hybrid search over messages (75% semantic + 25% BM25) -> candidates
      3. BM25 re-rank the candidates locally (60% semantic + 40% BM25)
      4. Group matched messages by thread (conversation_id); each thread's
         rank is its single best-scoring message. Expand the top_k threads
         into full-conversation chunks using their denormalised thread content.

    Returns chunks in the same dict format consumed by chatbot.service and
    format_context() (source_type == "email_conversation").
    """
    from chatbot._vec_backend import vec_store
    vs = vec_store()

    from core.config import get_settings
    settings = get_settings()
    top_k = top_k if top_k is not None else settings.chatbot_top_k
    retrieval_candidates = settings.chatbot_retrieval_candidates

    t_total = time.perf_counter()

    # ── Step 1: Embed query ───────────────────────────────────
    query_embedding = embed_query(query)          # shape (1, dim), L2-normalised
    query_vector = query_embedding[0].tolist()     # plain list

    # ── Step 2: Search — Weaviate hybrid or FAISS vector-only ──
    if settings.use_faiss:
        # FAISS: vector-only search; BM25 re-rank below handles the text part.
        t_search = time.perf_counter()
        hits = vs.search_vectors(query_vector, limit=retrieval_candidates)
        search_ms = (time.perf_counter() - t_search) * 1000
        logger.info("FAISS vector search: %.1fms (%d results)", search_ms, len(hits))
        candidates: list[dict] = []
        for chunk, score in hits:
            c = dict(chunk)
            c["score"] = float(score)
            c["combined_score"] = float(score)
            if _has_answer_content(c) and c.get("message_id"):
                candidates.append(c)
    else:
        from weaviate.classes.query import MetadataQuery
        if vs.ensure_collection_exists(client, settings):
            logger.warning(
                "Weaviate collection '%s' was missing and has been recreated + "
                "refilled from PostgreSQL for this query.",
                settings.chatbot_weaviate_collection,
            )
        collection = vs.get_collection(client, settings)

        t_search = time.perf_counter()
        try:
            response = collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=0.75,
                limit=retrieval_candidates,
                query_properties=["subject", "body", "metadata_summary"],
                return_metadata=MetadataQuery(score=True),
            )
        except Exception as e:
            logger.error("Weaviate hybrid search failed (%s) - re-checking collection and retrying once", e)
            vs.ensure_collection_exists(client, settings)
            collection = vs.get_collection(client, settings)
            response = collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=0.75,
                limit=retrieval_candidates,
                query_properties=["subject", "body", "metadata_summary"],
                return_metadata=MetadataQuery(score=True),
            )
        search_ms = (time.perf_counter() - t_search) * 1000
        logger.info("Weaviate hybrid search: %.1fms (%d results)", search_ms, len(response.objects))

        candidates = []
        for obj in response.objects:
            score = obj.metadata.score if obj.metadata else 0.0
            chunk = _message_chunk_from_properties(obj.properties, score)
            if _has_answer_content(chunk) and chunk.get("message_id"):
                candidates.append(chunk)

    # ── Step 4: BM25 re-rank (message level) ──────────────────
    if bm25_index is not None:
        # Score each candidate against the query using the local BM25 index.
        # Transfer scores by stable message identity, never by subject.
        t_rerank = time.perf_counter()
        bm25_results = bm25_index.query(query, top_n=retrieval_candidates)
        bm25_score_map = {
            _chunk_key(r): r["bm25_score"]
            for r in bm25_results
            if _has_answer_content(r)
        }

        for chunk in candidates:
            chunk["bm25_score"] = bm25_score_map.get(_chunk_key(chunk), 0.0)

        reranked = rerank(query, candidates, bm25_index=bm25_index)
        rerank_ms = (time.perf_counter() - t_rerank) * 1000
        logger.info("BM25 re-rank: %.1fms", rerank_ms)
    else:
        # No BM25 index - fall back to legacy keyword re-ranker
        logger.warning("No BM25 index for Weaviate re-rank - using legacy keyword scorer.")
        reranked = rerank(query, candidates)

    # Defensive message-level deduplication. The collection should already
    # contain one object per message_id, but this protects the UI and LLM
    # context during migration or partial legacy states.
    deduped_by_id: dict[str, dict] = {}
    for chunk in reranked:
        if not _has_answer_content(chunk):
            continue
        key = _chunk_key(chunk)
        existing = deduped_by_id.get(key)
        if existing is None or chunk.get("combined_score", 0) > existing.get("combined_score", 0):
            deduped_by_id[key] = chunk

    deduped = sorted(
        deduped_by_id.values(),
        key=lambda chunk: chunk.get("combined_score", 0),
        reverse=True,
    )
    candidate_pool = deduped[:retrieval_candidates]

    # ── Step 5: Group matched messages into threads, take top_k ──
    results = _group_messages_into_threads(candidate_pool, top_k)

    total_ms = (time.perf_counter() - t_total) * 1000
    if results:
        logger.info(
            "Weaviate retrieval complete: %.1fms | %d threads | top combined_score=%.3f",
            total_ms, len(results), results[0]["combined_score"],
        )
    else:
        logger.warning("Weaviate retrieval returned no results.")

    # Attach the same bounded message pool shown in the UI.
    for chunk in results:
        chunk["_all_candidates"] = candidate_pool

    return results


# ── Prompt-size reduction: block-based context builder ────────
#
# The raw thread text (thread_raw_content / messages_json) repeats quoted
# reply history, headers, signatures and address blocks, so a 3-thread prompt
# ran 4,300-8,400 tokens and Ollama truncated it. build_context() works in
# four stages:
#   1. clean   - drop only what carries no information (headers, signatures,
#                address/url lines, separators, disclaimers, encoding
#                artifacts) and quoted repeats of earlier blocks. Nothing else
#                is rewritten, shortened or reordered.
#   2. blocks  - group the cleaned lines into atomic blocks: paragraphs,
#                bullet/numbered groups and tables. A label always shares a
#                block with the value lines under it.
#   3. select  - if every block of the chosen threads fits the budget, send it
#                all unchanged; otherwise rank blocks (BM25 + embedding
#                similarity against the question and each sub-question) and
#                fill the budget best-first, adding neighbours. Blocks are kept
#                or dropped whole; only a block larger than the room left is cut
#                (first part + figure lines), with a WARNING.
#   4. render  - original order, grouped under "[Email N: subject]".
# Every threshold lives in Settings (CHATBOT_CONTEXT_*) and reaches this code
# through ContextConfig; the constants below are email-boilerplate patterns.

# Generic words ignored when matching question words: they appear in almost
# every business email and give no discriminating signal.
_GENERIC_WORDS = {
    "q1", "q2", "q3", "q4", "fy", "fy25", "fy26",
    "actuals", "data", "please", "find", "below",
}

# Header/label lines that carry no answer content.
_HEADER_LINE_RE = re.compile(
    r"^\s*(from|sent|to|cc|bcc|subject|date|reply-to|importance)\s*:",
    re.IGNORECASE,
)
# Rendered-thread markers ("=== Message 1 of 2 ===", "Message-ID: ...") carry no answer content.
_MESSAGE_MARKER_RE = re.compile(r"^\s*(?:={3,}\s*Message\s+\d+\s+of\s+\d+\s*={3,}|Message-ID\s*:.*)$", re.IGNORECASE)
# Separator rules: ----, ====, ____, **** (3+ of the same char).
_SEPARATOR_RE = re.compile(r"^\s*[-=_*]{3,}\s*$")
# "-----Original Message-----" and "On ... wrote:" reply boundaries.
_REPLY_BOUNDARY_RE = re.compile(
    r"^\s*(-{2,}\s*original message\s*-{2,}|on .{3,250}\bwrote:\s*)$",
    re.IGNORECASE,
)
# A line that is only a sign-off ("Regards,", "Thanks and regards", "Sent from
# my ..."). The rest of that paragraph (name, title, phone) is the signature.
_SIGNOFF_RE = re.compile(
    r"^\s*(?:(?:(?:best|warm|kind|with)\s{1,3})?(?:regards|thanks|thank you|sincerely|cheers|br"
    r"|yours (?:faithfully|sincerely))(?:\s{0,3}(?:and|&)\s{0,3}regards)?\s{0,3}[,.!]?"
    r"|sent from my\b[^\n]{0,60})\s*$",
    re.IGNORECASE,
)
# Paragraphs that are legal/confidentiality boilerplate.
_DISCLAIMER_RE = re.compile(
    r"\b(disclaimer|confidentiality notice|intended recipient|privileged and confidential)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+\-]{1,64}@[\w\-]{1,255}\.[\w.\-]{1,255}")
# A short line that is almost entirely an address / phone block.
_ADDRESS_BLOCK_RE = re.compile(
    r"^\s*(?:tel|phone|mob|mobile|fax|ext|ph|m|t)\s{0,3}[:.]?\s{0,3}[\d +()\-]{6,40}$",
    re.IGNORECASE,
)
# Invisible characters and replacement marks that only come from bad decoding.
_ARTIFACT_RE = re.compile("[﻿​‌‍⁠�]")
# UTF-8 text that was decoded as cp1252/latin-1: a lead character followed by
# continuation-looking characters, e.g. an apostrophe turned into three symbols.
_MOJIBAKE_RE = re.compile(
    "(?:[Â-ô][\u0080-¿ŒœŠšŸŽž"
    "ƒˆ˜–—‘-„†-•…‰‹"
    "›€™]{1,3})+"
)
# Bullet / numbered-list markers: "- x", "* x", "• x", "1. x", "2) x", "a) x".
_BULLET_RE = re.compile(r"^\s*(?:[-*•·–—▪●○◦]|\(?\d{1,3}[.)]|\(?[A-Za-z][.)])\s+")
# A figure: a digit, a percent sign or a currency mark.
_FIGURE_RE = re.compile(r"\d|%|[₹$€£¥]")
_DIGIT_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*")
# Sub-question separators: commas (not inside numbers), ";", "&", "and" and
# list numbering such as "1)" / "2.".
_SUBQ_SPLIT_RE = re.compile(r"(?:,(?=\s|$)|;|&|\band\b|\b\d{1,2}[.)](?=\s))", re.IGNORECASE)
# Question text is capped at this many characters before the sub-question split
# (the pieces are stripped afterwards, so no surrounding \s* is needed in the pattern).
MAX_REGEX_INPUT_LEN = 10000

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can",
    "what", "when", "where", "who", "which", "how", "why",
    "in", "on", "at", "to", "for", "of", "with", "and", "or", "but",
    "from", "this", "that", "these", "those", "it", "its", "we", "our",
    "i", "you", "they", "them", "their", "not", "as", "if", "than", "then",
    "about", "into", "over", "per", "by", "any", "all", "some",
    # question phrasing that says what is asked, not what it is about
    "much", "many", "long", "percentage", "percent", "number", "tell",
    "show", "give",
}

# Gap between blocks when rendered (a blank line); counted against the budget.
_BLOCK_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class ContextConfig:
    """Every build_context threshold, resolved from Settings (CHATBOT_CONTEXT_*).

    Built once per request by ``from_settings``; tests construct it directly,
    so the builder itself never reads the environment.
    """

    max_chars: int              # CHATBOT_CONTEXT_MAX_CHARS
    top_threads: int            # CHATBOT_CONTEXT_TOP_THREADS
    neighbor_blocks: int        # CHATBOT_CONTEXT_NEIGHBOR_BLOCKS
    min_thread_share: float     # CHATBOT_CONTEXT_MIN_THREAD_SHARE
    figure_bonus: float         # CHATBOT_CONTEXT_FIGURE_BONUS
    use_embeddings: bool        # CHATBOT_CONTEXT_USE_EMBEDDINGS
    embedding_weight: float     # CHATBOT_CONTEXT_EMBEDDING_WEIGHT
    dedup_similarity: float     # CHATBOT_CONTEXT_DEDUP_SIMILARITY
    dedup_min_chars: int        # CHATBOT_CONTEXT_DEDUP_MIN_CHARS
    short_line_chars: int       # CHATBOT_CONTEXT_SHORT_LINE_CHARS
    table_min_lines: int        # CHATBOT_CONTEXT_TABLE_MIN_LINES
    table_min_numbers: int      # CHATBOT_CONTEXT_TABLE_MIN_NUMBERS
    block_max_chars: int        # CHATBOT_CONTEXT_BLOCK_MAX_CHARS
    partial_head_share: float   # CHATBOT_CONTEXT_PARTIAL_HEAD_SHARE
    thin_body_chars: int        # CHATBOT_CONTEXT_THIN_BODY_CHARS
    attachment_max_chars: int   # CHATBOT_CONTEXT_ATTACHMENT_MAX_CHARS
    signature_max_lines: int    # CHATBOT_CONTEXT_SIGNATURE_MAX_LINES
    signature_max_words: int    # CHATBOT_CONTEXT_SIGNATURE_MAX_WORDS
    signature_caps_share: float  # CHATBOT_CONTEXT_SIGNATURE_CAPS_SHARE
    signature_address_pattern: str  # CHATBOT_CONTEXT_SIGNATURE_ADDRESS_PATTERN
    ignore_words: frozenset = frozenset()   # CHATBOT_CONTEXT_IGNORE_WORDS

    @classmethod
    def from_settings(cls, settings=None) -> "ContextConfig":
        s = settings or get_settings()
        return cls(
            max_chars=s.chatbot_context_max_chars,
            top_threads=s.chatbot_context_top_threads,
            neighbor_blocks=s.chatbot_context_neighbor_blocks,
            min_thread_share=s.chatbot_context_min_thread_share,
            figure_bonus=s.chatbot_context_figure_bonus,
            use_embeddings=s.chatbot_context_use_embeddings,
            embedding_weight=s.chatbot_context_embedding_weight,
            dedup_similarity=s.chatbot_context_dedup_similarity,
            dedup_min_chars=s.chatbot_context_dedup_min_chars,
            short_line_chars=s.chatbot_context_short_line_chars,
            table_min_lines=s.chatbot_context_table_min_lines,
            table_min_numbers=s.chatbot_context_table_min_numbers,
            block_max_chars=s.chatbot_context_block_max_chars,
            partial_head_share=s.chatbot_context_partial_head_share,
            thin_body_chars=s.chatbot_context_thin_body_chars,
            attachment_max_chars=s.chatbot_context_attachment_max_chars,
            signature_max_lines=s.chatbot_context_signature_max_lines,
            signature_max_words=s.chatbot_context_signature_max_words,
            signature_caps_share=s.chatbot_context_signature_caps_share,
            signature_address_pattern=s.chatbot_context_signature_address_pattern,
            ignore_words=frozenset(
                w.strip().lower()
                for w in s.chatbot_context_ignore_words.split(",")
                if w.strip()
            ),
        )


@dataclass
class _Block:
    """One atomic unit of a cleaned thread: kept whole or dropped whole."""

    kind: str                   # "prose" | "bullets" | "table"
    lines: list
    pos: int = 0                # position within its thread, after dedup
    score: float = 0.0
    must: bool = False          # best match for the question or a sub-question
    tier: int = 3               # fill order: 1 best match/table, 2 neighbour, 3 rest
    labels: str = ""            # table sections: heading + row labels, scored against the question

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def has_figures(self) -> bool:
        return _FIGURE_RE.search(self.text) is not None


# ── Step 1: lossless cleaning ─────────────────────────────────

def _thread_body(chunk: dict) -> str:
    """Return ONLY the thread's own subject-free body text.

    build_context uses each thread's ``subject`` and the longer of ``body`` and
    ``raw_content`` (the full rendered email) - never messages_json,
    metadata_summary or any id field. A matched message's ``body`` can hold only
    the opening of an email whose table sits further down, so the fuller text
    wins whenever it is longer. attachment_data is consulted separately and only
    for a thin body.
    """
    body = str(chunk.get("body") or "").strip()
    raw = str(chunk.get("raw_content") or "").strip()
    return raw if len(raw) > len(body) else body


def _fix_encoding(text: str) -> str:
    """Remove decoding artifacts and repair UTF-8 that was read as cp1252/latin-1.

    Only runs that decode cleanly are rewritten; anything else is untouched.
    """
    text = _ARTIFACT_RE.sub("", text)

    def _repair(match: re.Match) -> str:
        raw = match.group(0)
        for enc in ("cp1252", "latin-1"):
            try:
                return raw.encode(enc).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        return raw

    return _MOJIBAKE_RE.sub(_repair, text)


def _is_boilerplate_line(line: str) -> bool:
    """Address/phone block, url, or a line that is mostly an email address."""
    if _ADDRESS_BLOCK_RE.match(line) or _URL_RE.search(line):
        return True
    return bool(_EMAIL_RE.search(line)) and len(_EMAIL_RE.sub("", line).strip()) < 15


@functools.lru_cache(maxsize=8)
def _address_regex(pattern: str):
    """Compiled CHATBOT_CONTEXT_SIGNATURE_ADDRESS_PATTERN, or None if invalid/empty."""
    if not pattern:
        return None
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        logger.warning("Invalid CHATBOT_CONTEXT_SIGNATURE_ADDRESS_PATTERN %r: %s", pattern, e)
        return None


def _is_signature_line(line: str, cfg: ContextConfig) -> bool:
    """True only for a line that is clearly part of a signature block: a short
    address line, or a short person-name / job-title / company line. A line that
    carries a figure (digit, %, currency mark) outside an address, ends like a
    sentence, or reads like ordinary words is NOT a signature line - when unsure,
    the line is kept."""
    if len(line) > cfg.short_line_chars or re.search(r"[%₹$€£¥]", line):
        return False
    address = _address_regex(cfg.signature_address_pattern)
    if address and address.search(line) and (_FIGURE_RE.search(line) or "," in line):
        return True
    if _FIGURE_RE.search(line) or line.rstrip().endswith((".", "!", "?", ":", ";")):
        return False
    words = line.split()
    letters = [w for w in words if w[:1].isalpha()]
    if not letters or len(words) > cfg.signature_max_words:
        return False
    return sum(1 for w in letters if w[:1].isupper()) / len(letters) >= cfg.signature_caps_share


def _is_signature_block(lines: list[str], cfg: ContextConfig) -> bool:
    """A signature block is 1..signature_max_lines lines that ALL look like
    signature lines; one doubtful line makes the whole block ordinary text."""
    return 1 <= len(lines) <= cfg.signature_max_lines and all(_is_signature_line(line, cfg) for line in lines)


def _clean_paragraphs(text: str, cfg: ContextConfig) -> list[list[str]]:
    """Return the paragraphs (lists of lines) left after removing headers,
    reply boundaries, separators, signature blocks, address/url lines and
    disclaimer paragraphs. Remaining lines are kept as written (only trimmed at
    the ends).

    A signature block is removed only when it directly follows a sign-off line,
    or is the last paragraph of a message (and not the message's only
    paragraph), and every one of its lines passes ``_is_signature_line``.
    """
    paragraphs: list[list[str]] = []
    current: list[str] = []
    signature: list[str] | None = None     # lines collected after a sign-off line
    message_start = 0                      # index in ``paragraphs`` where this message began

    def flush() -> None:
        nonlocal current, signature
        if signature is not None:
            if signature and not _is_signature_block(signature, cfg):
                current.extend(signature)  # not clearly a signature: keep the lines
            signature = None
        if current and not _DISCLAIMER_RE.search(" ".join(current)):
            paragraphs.append(current)
        current = []

    def end_message() -> None:
        nonlocal message_start
        flush()
        message = paragraphs[message_start:]
        if len(message) >= 2 and _is_signature_block(message[-1], cfg):
            paragraphs.pop()
        message_start = len(paragraphs)

    for raw in _fix_encoding(text).splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if (
            _REPLY_BOUNDARY_RE.match(line) or _HEADER_LINE_RE.match(line)
            or _SEPARATOR_RE.match(line) or _MESSAGE_MARKER_RE.match(line)
        ):
            end_message()
            continue
        if _is_boilerplate_line(line):
            continue
        if signature is not None:
            signature.append(line)
        elif len(line) <= cfg.short_line_chars and _SIGNOFF_RE.match(line):
            signature = []
        else:
            current.append(line)
    end_message()
    return paragraphs


# ── Step 2: block formation ───────────────────────────────────

def _ends_sentence(line: str) -> bool:
    """True for a line that closes a sentence (., ! or ?) - prose, not a table cell."""
    return line.rstrip().endswith((".", "!", "?"))


def _line_kind(line: str, cfg: ContextConfig) -> str:
    """B: bullet/numbered line. T: table-like (short, delimited, or number-heavy).
    P: ordinary prose."""
    if _BULLET_RE.match(line):
        return "B"
    if (
        len(line) <= cfg.short_line_chars
        or "|" in line
        or "\t" in line
        or len(_DIGIT_TOKEN_RE.findall(line)) >= cfg.table_min_numbers
    ):
        return "T"
    return "P"


def _split_oversize(lines: list[str], cfg: ContextConfig) -> list[list[str]]:
    """Split a long prose block at line boundaries into chunks of at most
    block_max_chars. A split is only allowed before an ordinary prose line and
    never after a line ending in ":", so short value lines stay with the line
    above them."""
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for line in lines:
        splittable = current and _line_kind(line, cfg) == "P" and not current[-1].endswith(":")
        if splittable and size + len(line) + 1 > cfg.block_max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append(current)
    return chunks


# A table cell that is a figure: 1,16,094 / -3,385 / (3,385) / 12.5% / a dash / NA.
_TABLE_VALUE_RE = re.compile(
    r"^\(?[-+–−]?\s?[₹$€£]?\s?\d[\d,]*(?:\.\d+)?\s?%?\)?$"
    r"|^[-–—]+$|^(?:n/?a|nil)$",
    re.IGNORECASE,
)
_TABLE_UNIT_PREFIXES = ("values in", "(values in", "figures in", "amounts in", "all figures", "all values")


def _is_table_value(line: str) -> bool:
    return _TABLE_VALUE_RE.match(line.strip()) is not None


# A period header cell: Mar'25, Jun'26, Sep-26, Q1, FY26, Q1 FY'27, Qtr_2_FY'27, H1_FY'27.
_PERIOD_CELL_RE = re.compile(
    r"^(?:[A-Za-z]{3,9}[\s'’\-_]*\d{2,4}|(?:Qtr|Q|H)[\s_]*[1-4](?:[\s_]*FY[\s_'’]*\d{2,4})?|FY[\s_'’]*\d{2,4})$",
    re.IGNORECASE,
)
# First words that begin a new value-column name when they open a header cell.
_COLUMN_START_WORDS = {"gap", "yoy", "y0y", "qoq", "mom", "ytd", "mtd", "achievement"}
# A header cell that can end a column name (a qualifier such as Actuals, Plan, Growth %).
_COLUMN_END_RE = re.compile(r"actual|plan|projection|achieve|growth|leverag|target|budget|%", re.IGNORECASE)
# What a column (or a question) is about, besides its period.
_COLUMN_KIND_PATTERNS = {
    "actual": re.compile(r"actual", re.IGNORECASE),
    "plan": re.compile(r"\bplan\b|\btarget\b", re.IGNORECASE),
    "budget": re.compile(r"\bbudget\b", re.IGNORECASE),
    "ytd": re.compile(r"\bytd\b", re.IGNORECASE),
    "mtd": re.compile(r"\bmtd\b", re.IGNORECASE),
    "h1": re.compile(r"\bh1\b", re.IGNORECASE),
    "h2": re.compile(r"\bh2\b", re.IGNORECASE),
    "q1": re.compile(r"\bq(?:tr)?\s*1\b", re.IGNORECASE),
    "q2": re.compile(r"\bq(?:tr)?\s*2\b", re.IGNORECASE),
    "q3": re.compile(r"\bq(?:tr)?\s*3\b", re.IGNORECASE),
    "q4": re.compile(r"\bq(?:tr)?\s*4\b", re.IGNORECASE),
    "fy": re.compile(r"\bfy\b|\bfy\s*['\u2019]?\d", re.IGNORECASE),
    "projection": re.compile(r"project", re.IGNORECASE),
    "gap": re.compile(r"\bgap\b", re.IGNORECASE),
    "growth": re.compile(r"\b(?:yoy|y0y)\b|growth|year[\s-]*on[\s-]*year", re.IGNORECASE),
    "achievement": re.compile(r"achievement", re.IGNORECASE),
    "leveraged": re.compile(r"leverag", re.IGNORECASE),
}
_HEADER_GROUP_VIOLATION = 10   # cost of a group not starting at, or holding a second, start cell


_BASE_THEN_ACHIEVEMENT_RE = re.compile(r"^(?:budget|plan|target)\s+achievement", re.IGNORECASE)
_BARE_BASE_RE = re.compile(r"^(?:budget|plan|target)$", re.IGNORECASE)


def _is_column_start(cell: str) -> bool:
    words = cell.split()
    return bool(
        _PERIOD_CELL_RE.match(cell) or _BASE_THEN_ACHIEVEMENT_RE.match(cell)
        or (words and words[0].lower() in _COLUMN_START_WORDS)
    )


def _join_base_before_achievement(cells: list[str]) -> list[str]:
    """A bare Budget / Plan cell that follows an "actuals" cell and precedes an
    Achievement cell belongs to the Achievement column ("Budget" + "Achievement %"
    -> "Budget Achievement %"), not to the actuals column before it."""
    out: list[str] = []
    i = 0
    while i < len(cells):
        if (
            _BARE_BASE_RE.match(cells[i]) and out and re.search("actual", out[-1], re.IGNORECASE)
            and i + 1 < len(cells) and cells[i + 1].lower().startswith("achievement")
        ):
            out.append(cells[i] + " " + cells[i + 1])
            i += 2
        else:
            out.append(cells[i])
            i += 1
    return out


def _group_header_cells(cells: list[str], n: int) -> list[list[str]] | None:
    """Split header cells into exactly ``n`` contiguous groups (one per value
    column), by dynamic programming. A group should start at a period cell or
    at a start word (Gap, YOY, Achievement ...), hold no second start cell, and
    preferably end with a qualifier (Actuals, Plan, Growth % ...). Returns None
    only when there are fewer cells than columns."""
    m = len(cells)
    if n < 1 or m < n:
        return None
    if m > n:
        joined = _join_base_before_achievement(cells)
        if len(joined) >= n:
            cells, m = joined, len(joined)
    if m == n:
        return [[c] for c in cells]

    def cost(i: int, j: int) -> int:
        group = cells[i:j]
        total = 0 if _is_column_start(group[0]) else _HEADER_GROUP_VIOLATION
        total += _HEADER_GROUP_VIOLATION * sum(_is_column_start(c) for c in group[1:])
        return total + (0 if _COLUMN_END_RE.search(group[-1]) else 1)

    inf = float("inf")
    best = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0
    for k in range(1, n + 1):
        for j in range(k, m - (n - k) + 1):
            for i in range(k - 1, j):
                if best[k - 1][i] == inf:
                    continue
                c = best[k - 1][i] + cost(i, j)
                if c < best[k][j]:
                    best[k][j], back[k][j] = c, i
    if best[n][m] == inf:
        return None
    groups: list[list[str]] = []
    j = m
    for k in range(n, 0, -1):
        i = back[k][j]
        groups.append(cells[i:j])
        j = i
    return groups[::-1]


def _column_names(groups: list[list[str]]) -> list[str]:
    """Join each group into a column name. A name that repeats (two "YOY Growth %")
    is prefixed with the nearest period-led column name to its left, so each is
    unique ("Sep'26 Actuals YOY Growth %")."""
    own = [" ".join(g) for g in groups]
    names = list(own)
    for idx, group in enumerate(groups):
        if own.count(own[idx]) > 1 and not _PERIOD_CELL_RE.match(group[0]):
            for j in range(idx - 1, -1, -1):
                if _PERIOD_CELL_RE.match(groups[j][0]):
                    names[idx] = f"{own[j]} {own[idx]}"
                    break
    return names


_MONTH_PERIOD_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[\s'’\-_.]*((?:20)?\d{2})\b",
    re.IGNORECASE,
)
_QUESTION_PERIOD_RE = re.compile(
    _MONTH_PERIOD_RE.pattern
    + r"|(?:Qtr|Q|H)[\s_]*[1-4](?:[\s_]*FY[\s_'’]*\d{2,4})?|FY[\s_'’]*\d{2,4}",
    re.IGNORECASE,
)


def _norm_key(text: str) -> str:
    """Comparable form of a period. Any spelling of a month and year gives one key:
    Dec'25 / Dec-25 / Dec 25 / Dec 2025 / December 2025 / Dec 25 qtr -> dec25 (Sept = Sep).
    Qtr_2_FY'27 -> q2fy27."""
    month = _MONTH_PERIOD_RE.search(text)
    if month:
        return month.group(1)[:3].lower() + month.group(2)[-2:]
    key = re.sub(r"[^a-z0-9]", "", text.lower())
    key = re.sub(r"^qtr", "q", key)
    return re.sub(r"(?<=[a-z])20(\d{2})$", r"\g<1>", key)


_DERIVED_KINDS = frozenset({"gap", "growth", "achievement", "leveraged"})


def _period_keys(text: str) -> set[str]:
    """Normalised keys of every period named in ``text`` (see _norm_key)."""
    return {_norm_key(m.group(0)) for m in _QUESTION_PERIOD_RE.finditer(text or "")}


def _kinds(text: str) -> frozenset:
    """Keywords (YTD, H1, Q1-Q4, FY, actual, plan, budget, projection, gap, growth,
    achievement, leveraged) that a header or a question contains."""
    text = text.replace("_", " ")
    return frozenset(k for k, pat in _COLUMN_KIND_PATTERNS.items() if pat.search(text))


# Words that say what kind of figure is asked for, not which row: ignored when
# matching a question to a row label ("Insurance fees" / "Insurance income" -> "Insurance").
_ROW_FILLER = frozenset({
    "fees", "fee", "income", "revenue", "actual", "actuals", "what", "was", "the", "from", "of",
})


def _best_rows(rows: list[tuple[str, list[str]]], q_words: set[str]) -> list[tuple[str, list[str]]]:
    """The row(s) whose label words (filler words ignored) are all in the question;
    only the most specific (most label words) are returned."""
    scored = []
    q_words = q_words - _ROW_FILLER
    for label, values in rows:
        words = _question_words(label) - _ROW_FILLER
        if words and words <= q_words:
            scored.append((len(words), label, values))
    if not scored:
        return []
    top = max(w for w, _l, _v in scored)
    return [(label, values) for w, label, values in scored if w == top]


def _matched_rows(rows: list[tuple[str, list[str]]], q_words: set[str]) -> list[str]:
    """"Matched row: label | v1 | ..." lines for the best matching row(s), copied as is."""
    return ["Matched row: " + " | ".join([label] + values) for label, values in _best_rows(rows, q_words)[:3]]


_LOOKUP_PREFIXES = ("Lookup result:", "Highest:", "Lowest:")
# With no heading line, a table section is named from its label column header.
_LABEL_HEADER_SECTIONS = {"product": "Liabilities", "fee category": "Fee Income", "category": "Asset"}
_SUPERLATIVE_RE = re.compile(r"\b(highest|lowest|largest|smallest|biggest|maximum|minimum|max|min)\b", re.IGNORECASE)
_HIGH_WORDS = {"highest", "largest", "biggest", "maximum", "max"}


def _to_number(cell: str) -> float | None:
    """Numeric value of a table cell for comparison only (never shown): 1,16,094 /
    -3,385 / (3,385) / 12.5%. None for a dash or text."""
    text = cell.strip()
    digits = re.sub(r"[^\d.]", "", text)
    if not digits or digits == ".":
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    negative = text.startswith(("-", "−", "–")) or (text.startswith("(") and text.endswith(")"))
    return -value if negative else value


def _column_periods(groups: list[list[str]]) -> list[set[str]]:
    """Periods of each value column: those written in its own header cells (for
    example "Achievement on Dec'25 Plan %"), else the nearest period to its left."""
    out: list[set[str]] = []
    inherited: set[str] = set()
    for group in groups:
        if _PERIOD_CELL_RE.match(group[0]):
            inherited = {_norm_key(group[0])}
        out.append(_period_keys(" ".join(group)) or inherited)
    return out


def _lookup_cell(
    heading: str, groups: list[list[str]], names: list[str],
    rows: list[tuple[str, list[str]]], q_words: set[str], question: str,
) -> str | None:
    """Find the column the question names - a period plus its kind (actuals / plan /
    projection / gap / YOY growth / achievement / leveraged) - and return either
    "Lookup result: <heading>, row <label>, column <name> = <value>" (one named row)
    or, for a highest/lowest question, "Highest: <row> = <value> (column <name>)".
    None unless the column (and, for a lookup, the row) is unique. Cell text is
    copied exactly."""
    wanted = _kinds(question)
    if not wanted:
        return None
    # With a period in the question only the columns of that period are candidates;
    # without one every column is.
    qperiods = _period_keys(question)
    columns = (
        [i for i, ps in enumerate(_column_periods(groups)) if qperiods & ps]
        if qperiods else list(range(len(groups)))
    )
    own = [_kinds(" ".join(g)) for g in groups]
    present = set().union(*(own[i] for i in columns)) if columns else set()
    # Keyword match: a column qualifies when its header holds every question keyword
    # that occurs in these headers; the one with the fewest other keywords wins
    # ("YTD actual" -> "YTD Actuals", not "H1 actuals Budget"). Full set of keywords
    # first, then only the derived kind (a gap / growth / achievement column is named
    # by its own keyword; "gap to Plan" only says what it is relative to). A tie skips.
    derived = wanted & _DERIVED_KINDS
    found: list[int] = []
    for required in (wanted & present, derived & present if derived != wanted else set()):
        qualified = [i for i in columns if required and required <= own[i]]
        if not qualified:
            continue
        fewest = min(len(own[i] - wanted) for i in qualified)
        found = [i for i in qualified if len(own[i] - wanted) == fewest]
        break
    if len(found) != 1:
        return None
    idx, n = found[0], len(groups)

    superlative = _SUPERLATIVE_RE.search(question)
    if superlative:
        candidates = [(label, values[idx]) for label, values in rows if len(values) == n]
        figures = [(label, cell, _to_number(cell)) for label, cell in candidates]
        figures = [f for f in figures if f[2] is not None]
        without_totals = [f for f in figures if not re.search(r"\btotal\b", f[0], re.IGNORECASE)]
        figures = without_totals or figures
        if not figures:
            return None
        high = superlative.group(1).lower() in _HIGH_WORDS
        target = (max if high else min)(f[2] for f in figures)
        top = [f for f in figures if f[2] == target]
        return f"{'Highest' if high else 'Lowest'}: {' / '.join(f[0] for f in top)} = {top[0][1]} (column {names[idx]})"

    best = _best_rows(rows, q_words)
    if len(best) != 1 or len(best[0][1]) != n:
        return None
    label, values = best[0]
    return f"Lookup result: {heading}, row {label}, column {names[idx]} = {values[idx]}"


def _fix_yoy(cell: str) -> str:
    """Header text "Y0Y" (zero) is YOY."""
    return re.sub(r"\bY0Y\b", "YOY", cell, flags=re.IGNORECASE)


def _structure_segment(seg: list[str], q_words: set[str] = frozenset(), question: str = "") -> list[str] | None:
    """Rebuild one flattened table (one cell per line) as "Columns: ..." plus
    "label | v1 | v2" rows, or return None when no label-and-values rows are found.
    Cell text is copied exactly.

    Leading heading lines (ending ":") and "Values in ..." lines stay in front.
    A data row is a label line followed by figure lines; the text lines before
    the first row's label are header cells. Values per row = the most common
    count. The first header cell names the label column; the rest are grouped
    into that many column names (see _group_header_cells). When they cannot be
    (fewer header cells than columns) the header cells are listed on one
    "Header cells in order:" line; the rows are always "label | v1 | v2 ...".
    A section with no heading is named from its label column (_LABEL_HEADER_SECTIONS).
    """
    if any("|" in line for line in seg):
        return None
    i = 0
    preamble: list[str] = []
    while i < len(seg) and not _is_table_value(seg[i]) and (
        seg[i].endswith(":") or seg[i].lower().startswith(_TABLE_UNIT_PREFIXES)
    ):
        preamble.append(seg[i])
        i += 1

    rows: list[tuple[str, list[str]]] = []
    header: list[str] = []
    pending: list[str] = []
    cur: tuple[str, list[str]] | None = None
    for line in seg[i:]:
        if _is_table_value(line):
            if cur is None:
                if not pending:
                    return None
                if not rows:
                    header, label = pending[:-1], pending[-1]
                else:
                    label = " ".join(pending)
                cur = (label, [line])
                rows.append(cur)
                pending = []
            else:
                cur[1].append(line)
        else:
            cur = None
            pending.append(line)
    if not rows:
        return None

    row_lines = [" | ".join([label] + values) for label, values in rows]
    if not header:
        return preamble + row_lines + pending

    counts = [len(values) for _label, values in rows]
    n = max(set(counts), key=counts.count)
    label_head, cells = header[0], header[1:]

    heading = next((p.rstrip(":") for p in preamble if p.endswith(":")), None)
    title: list[str] = []
    if heading is None:
        heading = _LABEL_HEADER_SECTIONS.get(label_head.strip().lower())
        if heading:
            title = [heading + ":"]

    groups = _group_header_cells(cells, n)
    lookup: list[str] = []
    if groups is not None:
        groups = [[_fix_yoy(c) for c in g] for g in groups]
        names = _column_names(groups)
        columns_line = ["Columns: " + " | ".join([label_head] + names)]
        found = _lookup_cell(heading or "table", groups, names, rows, q_words, question)
        lookup = [found] if found else []
    else:
        columns_line = ["Header cells in order: " + " | ".join(header)]
    return (
        title + preamble + lookup + columns_line + ([] if lookup else _matched_rows(rows, q_words))
        + row_lines + pending
    )


def _structure_table(lines: list[str], q_words: set[str] = frozenset(), question: str = "") -> list[list[str]]:
    """Split a table block into its sections and rebuild each with
    _structure_segment. A new section starts, after figures, at a heading line
    ending ":", at a "Values in ..." line, or at a label-column header such as
    "Product" / "Fee Category" / "Category". A section that cannot be rebuilt is
    kept exactly as it was. Every section keeps its own heading, "Values in" line
    and Columns line."""
    segments: list[list[str]] = []
    for idx, line in enumerate(lines):
        starts = (
            idx > 0 and not _is_table_value(line) and _is_table_value(lines[idx - 1])
            and (
                line.endswith(":")
                or line.lower().startswith(_TABLE_UNIT_PREFIXES)
                or line.strip().lower() in _LABEL_HEADER_SECTIONS
            )
        )
        if not segments or starts:
            segments.append([])
        segments[-1].append(line)
    return [_structure_segment(seg, q_words, question) or seg for seg in segments]


_TABLE_NOTE_PREFIXES = ("Columns:", "Matched row:", "Header cells in order:") + _LOOKUP_PREFIXES


def _has_table_rows(lines: list[str]) -> bool:
    """True when ``lines`` hold at least one data row (a line with a figure that is
    not the Columns / Lookup / header-list note)."""
    return any(
        _FIGURE_RE.search(line) and not line.startswith(("Columns:", "Header cells in order:") + _LOOKUP_PREFIXES)
        for line in lines
    )


def _table_labels(lines: list[str]) -> str:
    """Heading and row-label text of a table section, used to score it against
    the question (figures and the note lines above the rows are skipped)."""
    labels = []
    for line in lines:
        if line.startswith(_TABLE_NOTE_PREFIXES) or _is_table_value(line):
            continue
        labels.append(line.split(" | ", 1)[0])
    return " ".join(labels)


def _form_blocks(
    paragraphs: list[list[str]], cfg: ContextConfig, q_words: set[str] = frozenset(), question: str = "",
) -> list[_Block]:
    """Group cleaned paragraphs into atomic blocks.

    * adjacent table-like lines form one table block; across a blank line they
      still join when neither line ends a sentence (cells are often separated
      by blank lines, sentences are not); consecutive bullet lines form one
      bullet block
    * a table-like run shorter than table_min_lines continues the block above it
      (same paragraph, or that block's last line does not end a sentence) or
      prefixes the block below it (same paragraph), so a lone value line never
      becomes a block of its own; otherwise it stands alone
    * prose lines of one paragraph form one block, split only when longer than
      block_max_chars (see _split_oversize)
    * a line ending in ":" introduces the block that follows it and moves into it
    """
    runs: list[dict] = []
    for p_idx, para in enumerate(paragraphs):
        for line in para:
            kind = _line_kind(line, cfg)
            last = runs[-1] if runs else None
            joins = bool(last) and last["kind"] == kind and (
                last["last_para"] == p_idx
                or kind == "B"
                or (kind == "T" and not _ends_sentence(last["lines"][-1]) and not _ends_sentence(line))
            )
            if joins:
                last["lines"].append(line)
                last["last_para"] = p_idx
            else:
                runs.append({"kind": kind, "lines": [line], "first_para": p_idx, "last_para": p_idx})

    blocks: list[_Block] = []
    block_last_para: list[int] = []
    prefix: list[str] = []
    for i, run in enumerate(runs):
        if run["kind"] == "T" and len(run["lines"]) < cfg.table_min_lines:
            if blocks and (
                block_last_para[-1] == run["first_para"] or not _ends_sentence(blocks[-1].lines[-1])
            ):
                blocks[-1].lines.extend(run["lines"])
                block_last_para[-1] = run["last_para"]
                continue
            following = runs[i + 1] if i + 1 < len(runs) else None
            if following is not None and following["first_para"] == run["last_para"]:
                prefix = run["lines"]
                continue
            blocks.append(_Block(kind="prose", lines=run["lines"]))
            block_last_para.append(run["last_para"])
            continue
        kind = {"T": "table", "B": "bullets", "P": "prose"}[run["kind"]]
        blocks.append(_Block(kind=kind, lines=prefix + run["lines"]))
        block_last_para.append(run["last_para"])
        prefix = []

    # An introducing line ("... :") moves down into the block it introduces.
    for i in range(len(blocks) - 1):
        prev, nxt = blocks[i], blocks[i + 1]
        if prev.lines and prev.lines[-1].endswith(":"):
            nxt.lines.insert(0, prev.lines.pop())
    blocks = [b for b in blocks if b.lines]

    # One block per table section, so the section a question is about is kept
    # or dropped whole, with its heading and Columns line.
    sectioned: list[_Block] = []
    for block in blocks:
        if block.kind == "table":
            for section in _structure_table(block.lines, q_words, question):
                sectioned.append(_Block(kind="table", lines=section, labels=_table_labels(section)))
        else:
            sectioned.append(block)
    blocks = sectioned

    out: list[_Block] = []
    for block in blocks:
        if block.kind == "prose" and block.chars > cfg.block_max_chars:
            out.extend(_Block(kind="prose", lines=chunk) for chunk in _split_oversize(block.lines, cfg))
        else:
            out.append(block)
    return out


def _question_words(question: str, ignore_words=frozenset()) -> set[str]:
    """Significant (non-stopword, non-generic) words from the question, lowercased.

    Generic business words, question phrasing (much, many, percentage ...) and
    any configured ``ignore_words`` (CHATBOT_CONTEXT_IGNORE_WORDS) are dropped:
    they give no discriminating signal for matching.
    """
    words = re.findall(r"\w+", (question or "").lower())
    return {
        w for w in words
        if w not in _STOPWORDS and w not in _GENERIC_WORDS and w not in ignore_words
        and len(w) > 1
    }


def _prepare_thread(chunk: dict, q_words: set[str], cfg: ContextConfig, question: str = "") -> dict:
    """Clean one thread and split it into blocks.

    A thin body (< thin_body_chars after cleaning) also draws question-matching
    blocks from attachment_data, up to attachment_max_chars in total.
    """
    raw_body = _thread_body(chunk)
    blocks = _form_blocks(_clean_paragraphs(raw_body, cfg), cfg, q_words, question)

    if sum(b.chars for b in blocks) < cfg.thin_body_chars and q_words:
        attachment = str(chunk.get("attachment_data") or "").strip()
        used = 0
        for block in _form_blocks(_clean_paragraphs(attachment, cfg), cfg, q_words, question):
            if not (q_words & set(re.findall(r"\w+", block.text.lower()))):
                continue
            if used + block.chars > cfg.attachment_max_chars:
                break
            blocks.append(block)
            used += block.chars

    return {
        "subject": chunk.get("subject") or "(no subject)",
        "blocks": blocks,
        "raw_chars": len(raw_body),
    }


def _has_keyword_match(prepared: dict, q_words: set[str]) -> bool:
    """A real keyword match: a question word occurs in the subject or a block."""
    if not q_words:
        return True
    words = set(re.findall(r"\w+", prepared["subject"].lower()))
    for block in prepared["blocks"]:
        words |= set(re.findall(r"\w+", block.text.lower()))
    return bool(q_words & words)


def _dedup_blocks(selected: list[dict], cfg: ContextConfig) -> int:
    """Drop blocks that repeat an earlier block (quoted history), within a
    thread and across the threads being sent. Returns how many were dropped.

    A block repeats an earlier one when its words match exactly, or match at
    dedup_similarity or better AND carry exactly the same figures - an updated
    figure is new information, never a repeat. Blocks shorter than
    dedup_min_chars are never treated as repeats.
    """
    seen: list[tuple[set, set]] = []
    removed = 0
    for prepared in selected:
        kept: list[_Block] = []
        for block in prepared["blocks"]:
            if block.chars < cfg.dedup_min_chars:
                kept.append(block)
                continue
            words = set(re.findall(r"\w+", block.text.lower()))
            if not words:
                kept.append(block)
                continue
            figures = {w for w in words if any(c.isdigit() for c in w)}
            duplicate = any(
                figures == old_figures
                and len(words & old_words) / len(words | old_words) >= cfg.dedup_similarity
                for old_words, old_figures in seen
            )
            if duplicate:
                removed += 1
                continue
            seen.append((words, figures))
            kept.append(block)
        prepared["blocks"] = kept
    return removed


# ── Step 3: selection ─────────────────────────────────────────

def _sub_questions(question: str, ignore_words) -> list[str]:
    """The whole question first, then each part of a multi-part question
    (split on commas, ';', '&', 'and' and numbering) that still has
    significant words of its own."""
    queries = [question]
    parts = [p.strip() for p in _SUBQ_SPLIT_RE.split((question or "")[:MAX_REGEX_INPUT_LEN]) if p and p.strip()]
    if len(parts) > 1:
        for part in parts:
            if _question_words(part, ignore_words) and part not in queries:
                queries.append(part)
    return queries


_TABLE_EMBED_MAX_CHARS = 3200   # a table section (about 3,000 chars) is embedded whole


def _block_embed_chars(block: _Block, cfg: ContextConfig) -> int:
    """Characters of a block passed to the embedding model. The prose cap
    (block_max_chars) does not apply to a table section."""
    return max(cfg.block_max_chars, _TABLE_EMBED_MAX_CHARS) if block.kind == "table" else cfg.block_max_chars


def _score_blocks(blocks: list[_Block], queries: list[str], cfg: ContextConfig, stats: dict) -> None:
    """Set each block's score and mark the best block for every query.

    Hybrid score per (query, block): BM25 (built over this request's blocks,
    normalised to 0-1 per query) blended with embedding similarity when
    CHATBOT_CONTEXT_USE_EMBEDDINGS is on. A block's score is its best hybrid
    score over the whole question and every sub-question, plus figure_bonus
    when it contains figures. An embedding failure falls back to BM25 only.
    """
    from rank_bm25 import BM25Okapi

    from chatbot.bm25_index import _tokenise

    docs = [_tokenise(b.text) for b in blocks]
    bm25 = BM25Okapi(docs) if any(docs) else None

    sims = None
    if cfg.use_embeddings:
        try:
            vectors = embed_texts(queries + [b.text[: _block_embed_chars(b, cfg)] for b in blocks])
            sims = np.clip(vectors[len(queries):] @ vectors[: len(queries)].T, 0.0, None)  # (blocks, queries)
            stats["embeddings"] = "on"
        except Exception as e:  # embedding is an enhancement, never a requirement
            logger.warning("build_context: embedding failed (%s) - ranking blocks with BM25 only", e)
            stats["embeddings"] = "failed"
    else:
        stats["embeddings"] = "off"

    # A table section also scores on its heading and row labels (e.g. "deposits",
    # "CASA" -> the Liabilities section), whatever its numbers say.
    label_words = [_question_words(b.labels) if b.kind == "table" else set() for b in blocks]

    best_per_query = [(-1.0, -1)] * len(queries)
    for q_idx, query in enumerate(queries):
        q_set = _question_words(query, cfg.ignore_words)
        if bm25 is not None and (tokens := _tokenise(query)):
            raw = np.clip(np.asarray(bm25.get_scores(tokens), dtype=float), 0.0, None)
            bm_norm = raw / raw.max() if raw.max() > 0 else raw
        else:
            bm_norm = np.zeros(len(blocks))
        for b_idx, block in enumerate(blocks):
            hybrid = float(bm_norm[b_idx])
            if sims is not None:
                hybrid = (1.0 - cfg.embedding_weight) * hybrid + cfg.embedding_weight * float(sims[b_idx, q_idx])
            if q_set and label_words[b_idx]:
                hybrid += 0.5 * len(q_set & label_words[b_idx]) / len(q_set)
            block.score = max(block.score, hybrid)
            if hybrid > best_per_query[q_idx][0]:
                best_per_query[q_idx] = (hybrid, b_idx)

    for block in blocks:
        if block.has_figures:
            block.score += cfg.figure_bonus
    for _score, b_idx in best_per_query:
        if b_idx >= 0:
            blocks[b_idx].must = True


def _line_units(lines: list[str]) -> list[list[int]]:
    """Group line indexes so a label line is never separated from the figure
    lines directly below it: [label, value, value...], or a lone line."""
    units: list[list[int]] = []
    i = 0
    while i < len(lines):
        unit = [i]
        i += 1
        if _FIGURE_RE.search(lines[unit[0]]) is None:
            # " | " lines are complete rows (see _structure_table): each is its own unit
            while i < len(lines) and _FIGURE_RE.search(lines[i]) and " | " not in lines[i]:
                unit.append(i)
                i += 1
        units.append(unit)
    return units


def _partial_lines(lines: list[str], space: int, cfg: ContextConfig) -> list[str]:
    """Last resort for a block larger than the room left: keep its first part
    (up to partial_head_share of the room), then add the units that contain
    figures while they fit. Units keep a label with its value lines; only a
    single first line longer than the room is cut mid-line."""
    units = _line_units(lines)
    size = lambda unit: sum(len(lines[i]) + 1 for i in unit)  # noqa: E731
    keep: set[int] = set()
    used = 0
    head_room = int(space * cfg.partial_head_share)
    for unit in units:
        if keep and used + size(unit) > head_room and not lines[unit[0]].startswith("Columns:"):
            break
        if not keep and size(unit) > space:
            return [lines[unit[0]][: max(1, space)]]
        keep.update(unit)
        used += size(unit)
    for unit in units:
        if keep.issuperset(unit) or not any(_FIGURE_RE.search(lines[i]) for i in unit):
            continue
        if used + size(unit) <= space:
            keep.update(unit)
            used += size(unit)
    return [lines[i] for i in sorted(keep)]


def _select_blocks(selected: list[dict], queries: list[str], cfg: ContextConfig, budget: int, stats: dict) -> list[dict]:
    """Choose blocks for every thread being sent. Returns, per thread, a dict
    {block_pos: None (whole block) | list of lines (partial)}."""
    n = len(selected)
    flat = [(t, b) for t, p in enumerate(selected) for b in p["blocks"]]
    _score_blocks([b for _t, b in flat], queries, cfg, stats)

    overhead = [len(f"[Email {t + 1}: {p['subject']}]") + 1 for t, p in enumerate(selected)]
    usable = max(1, budget - sum(overhead) - len(_BLOCK_SEPARATOR) * (n - 1))

    # Budget share per thread: proportional to its best block score, but never
    # below min_thread_share, so no thread is dropped completely.
    relevance = [max((b.score for b in p["blocks"]), default=0.0) for p in selected]
    total = sum(relevance)
    min_share = min(cfg.min_thread_share, 1.0 / n)
    shares = [max(r / total if total > 0 else 1.0 / n, min_share) for r in relevance]
    caps = [int(usable * s / sum(shares)) for s in shares]

    chosen: list[dict] = [{} for _ in range(n)]
    used_by = [0] * n
    used = 0
    gap = len(_BLOCK_SEPARATOR)

    def try_add(t: int, block: _Block, use_caps: bool) -> bool:
        nonlocal used
        cost = block.chars + gap
        if block.pos in chosen[t] or used + cost > usable:
            return False
        if use_caps and used_by[t] + cost > caps[t]:
            return False
        chosen[t][block.pos] = None
        used += cost
        used_by[t] += cost
        return True

    def by_score(items):
        return sorted(items, key=lambda tb: -tb[1].score)

    def keep_partial(t: int, block: _Block, space: int, why: str) -> None:
        nonlocal used
        if space <= 0:
            logger.warning(
                "build_context: no room for block %d (%s) of thread %r - dropped whole (%s)",
                block.pos + 1, block.kind, selected[t]["subject"], why,
            )
            return
        lines = _partial_lines(block.lines, space, cfg)
        if block.kind == "table" and (
            not _has_table_rows(lines) or (len(lines) == 1 and len(lines[0]) < len(block.lines[0]))
        ):
            # A table is shortened by whole rows only: never a heading without rows
            # and never a line cut in the middle.
            logger.warning(
                "build_context: table block %d of thread %r does not fit even with rows dropped - "
                "dropped whole (%s)", block.pos + 1, selected[t]["subject"], why,
            )
            return
        cost = sum(len(line) + 1 for line in lines) + gap
        chosen[t][block.pos] = lines
        used += cost
        used_by[t] += cost
        stats["cuts"].append({
            "subject": selected[t]["subject"], "block": block.pos + 1, "kind": block.kind,
            "lines_kept": len(lines), "lines_total": len(block.lines),
            "chars_cut": max(0, block.chars - sum(len(line) + 1 for line in lines)),
        })
        logger.warning(
            "build_context: block %d (%s) of thread %r is larger than the room left (%s) - "
            "kept %d of %d lines (first part + figure lines), cut %d chars",
            block.pos + 1, block.kind, selected[t]["subject"], why,
            len(lines), len(block.lines), stats["cuts"][-1]["chars_cut"],
        )

    # Tier 1: the best block for the question and each sub-question, then tables.
    for _t, b in flat:
        b.tier = 1 if (b.must or b.kind == "table") else 3
    tier1 = by_score([x for x in flat if x[1].must]) + by_score(
        [x for x in flat if x[1].kind == "table" and not x[1].must]
    )
    # Best-match blocks first (whole within the thread share, else whole within the
    # overall budget), then the other tables the same way. A table section is cut
    # only when it cannot fit the overall budget at all; then only whole rows go.
    musts = [x for x in tier1 if x[1].must]
    # Tables of the two best-ranked emails come first, so their sections stay whole.
    tables = sorted((x for x in tier1 if not x[1].must), key=lambda x: (x[0] >= 2, -x[1].score))
    too_big: list[tuple[int, _Block]] = []
    for group in (musts, tables):
        over_cap = [(t, b) for t, b in group if not try_add(t, b, True)]
        too_big += [(t, b) for t, b in over_cap if not try_add(t, b, False) and b.must]
    for t, b in too_big:  # Step 3f, right away so lower tiers cannot use up the room
        keep_partial(t, b, min(usable - used, caps[t] - used_by[t]), "best match for the question")

    # Tier 2: neighbours of the blocks kept so far (labels and context come along).
    neighbours: list[tuple[int, _Block]] = []
    for t, b in by_score([x for x in tier1 if x[1].pos in chosen[x[0]]]):
        blocks = selected[t]["blocks"]
        for step in range(1, cfg.neighbor_blocks + 1):
            for pos in (b.pos - step, b.pos + step):
                if 0 <= pos < len(blocks) and pos not in chosen[t] and (t, blocks[pos]) not in neighbours:
                    neighbours.append((t, blocks[pos]))
    for t, b in neighbours:
        if try_add(t, b, True):
            b.tier = 2

    # Tier 3: everything else by score; then use any room the shares left over.
    for use_caps in (True, False):
        for t, b in by_score([x for x in flat if x[1].pos not in chosen[x[0]]]):
            try_add(t, b, use_caps)

    # A thread being sent is never left with nothing.
    for t, p in enumerate(selected):
        if not chosen[t] and p["blocks"]:
            best = max(p["blocks"], key=lambda blk: blk.score)
            keep_partial(t, best, max(1, caps[t]), "thread would otherwise be empty")
    return chosen


# ── Step 4: render ────────────────────────────────────────────

def _lookup_focus(selected: list[dict], prefer: frozenset = frozenset()) -> tuple[int, _Block] | None:
    """(thread index, table section) of the best-ranked section that carries a
    "Lookup result:" line, or None when no section does. Threads in ``prefer``
    (subject names the question's period) are looked at first."""
    for t in sorted(range(len(selected)), key=lambda i: (i not in prefer, i)):
        for block in selected[t]["blocks"]:
            if block.kind == "table" and any(line.startswith(_LOOKUP_PREFIXES) for line in block.lines):
                return t, block
    return None


def _render_lookup_context(selected: list[dict], focus: tuple[int, _Block], cfg: ContextConfig, stats: dict) -> str:
    """Context when the cell the question asks for was found in code: the lookup
    line, the email's short intro and the one table section it came from (whole,
    whatever the budget). Every other email is reduced to its subject line and
    the email's other tables are left out."""
    t, table = focus
    lookup_line = "\n".join(line for line in table.lines if line.startswith(_LOOKUP_PREFIXES))
    section = [line for line in table.lines if not line.startswith(("Matched row:",) + _LOOKUP_PREFIXES)]

    intro_block = next((b for b in selected[t]["blocks"] if b.kind != "table" and b.pos < table.pos), None)
    intro: list[str] = []
    if intro_block is not None:
        size = 0
        for line in intro_block.lines:  # whole lines, up to one prose block's length
            if intro and size + len(line) + 1 > cfg.block_max_chars:
                break
            intro.append(line)
            size += len(line) + 1

    parts: list[str] = []
    for i, p in enumerate(selected):
        header = f"[Email {i + 1}: {p['subject']}]"
        if i != t:
            parts.append(header)
            continue
        texts = [lookup_line] + (["\n".join(intro)] if intro else []) + ["\n".join(section)]
        parts.append(header + "\n" + _BLOCK_SEPARATOR.join(texts))

    context = _BLOCK_SEPARATOR.join(parts)
    kept = [{"pos": b.pos + 1, "kind": b.kind, "chars": b.chars, "tier": 1, "score": round(b.score, 3), "partial": False}
            for b in ([intro_block] if intro_block is not None else []) + [table]]
    stats.update(
        mode="lookup", threads_used=len(selected), blocks_kept=len(kept), chars_sent=len(context),
        kept_per_thread=[(p["rank"], p["subject"], len(kept) if i == t else 0, len(p["blocks"])) for i, p in enumerate(selected)],
        chosen=[{"rank": p["rank"], "subject": p["subject"], "blocks": kept if i == t else []} for i, p in enumerate(selected)],
    )
    return context


def build_context(
    threads: list[dict],
    question: str,
    cfg: ContextConfig | None = None,
    stats: dict | None = None,
    budget_chars: int | None = None,
) -> str:
    """Build the context string sent to the LLM from the retrieved threads.

    Uses at most ``cfg.top_threads`` threads (best-ranked first) that share a
    real keyword with the question. Their cleaned blocks are sent unchanged
    when they all fit ``budget_chars`` (default ``cfg.max_chars``), otherwise
    the best blocks are selected (see the module notes above). Output is in
    original order under ``[Email N: subject]`` (N counts the threads sent).

    ``stats`` (a dict), when given, is filled for logging and the debug dump:
      mode ("everything_fit" | "selection"), threads_in, threads_used,
      blocks_total, blocks_kept, chars_raw / chars_cleaned / chars_sent,
      dedup_removed, embeddings ("off" | "on" | "failed" | "skipped"),
      cuts [{subject, block, kind, lines_kept, lines_total, chars_cut}],
      kept_per_thread [(rank, subject, blocks_kept, blocks_total)],
      chosen [{rank, subject, blocks: [{pos, kind, chars, tier, score, partial}]}]
    """
    cfg = cfg or ContextConfig.from_settings()
    budget = max(1, int(budget_chars if budget_chars is not None else cfg.max_chars))
    stats = stats if stats is not None else {}
    stats.update(
        mode="selection", threads_in=len(threads), threads_used=0, blocks_total=0,
        blocks_kept=0, chars_raw=0, chars_cleaned=0, chars_sent=0, dedup_removed=0,
        embeddings="skipped", cuts=[], kept_per_thread=[], chosen=[],
    )
    empty = "(no relevant emails found)"
    if not threads:
        return empty

    q_words = _question_words(question, cfg.ignore_words)
    prepared = [_prepare_thread(chunk, q_words, cfg, question) for chunk in threads]
    for rank, p in enumerate(prepared, 1):
        p["rank"] = rank

    selected = [p for p in prepared if _has_keyword_match(p, q_words)][: max(1, cfg.top_threads)]
    stats["chars_raw"] = sum(p["raw_chars"] for p in selected)
    stats["dedup_removed"] = _dedup_blocks(selected, cfg)
    selected = [p for p in selected if p["blocks"]]
    for p in selected:
        for pos, block in enumerate(p["blocks"]):
            block.pos = pos
    if not selected:
        return empty

    stats["blocks_total"] = sum(len(p["blocks"]) for p in selected)
    stats["chars_cleaned"] = sum(b.chars for p in selected for b in p["blocks"])

    # Emails whose subject names a period the question names.
    q_periods = _period_keys(question)
    period_hits = []
    if q_periods:
        for t, p in enumerate(selected):
            head = "\n".join(b.text for b in p["blocks"])[:600]   # first 600 characters of the cleaned body
            if q_periods & (_period_keys(p["subject"]) | _period_keys(head)):
                period_hits.append(t)

    focus = _lookup_focus(selected, frozenset(period_hits))
    if focus is not None:
        return _render_lookup_context(selected, focus, cfg, stats)

    # No lookup, but the question names a period and some email's subject carries
    # it: only those emails are sent in full, the others as their subject line.
    brief: list[dict] = []
    # Only when a matched email actually holds a table; a prose-only match, or no
    # match, leaves every email as it is.
    if period_hits and any(b.kind == "table" for t in period_hits for b in selected[t]["blocks"]):
        brief = [p for t, p in enumerate(selected) if t not in period_hits]
        selected = [selected[t] for t in period_hits]

    headers = [f"[Email {t + 1}: {p['subject']}]" for t, p in enumerate(selected)]
    everything = len(_BLOCK_SEPARATOR) * (len(selected) - 1) + sum(
        len(h) + 1 + sum(b.chars for b in p["blocks"]) + len(_BLOCK_SEPARATOR) * (len(p["blocks"]) - 1)
        for h, p in zip(headers, selected)
    )

    if everything <= budget:
        stats["mode"] = "everything_fit"
        chosen: list[dict] = [{b.pos: None for b in p["blocks"]} for p in selected]
    else:
        queries = _sub_questions(question, cfg.ignore_words)
        chosen = _select_blocks(selected, queries, cfg, budget, stats)

    parts: list[str] = []
    for t, (header, p) in enumerate(zip(headers, selected)):
        texts, detail = [], []
        for block in p["blocks"]:
            if block.pos not in chosen[t]:
                continue
            partial = chosen[t][block.pos]
            texts.append(block.text if partial is None else "\n".join(partial))
            detail.append({
                "pos": block.pos + 1, "kind": block.kind, "chars": block.chars,
                "tier": block.tier, "score": round(block.score, 3), "partial": partial is not None,
            })
        stats["kept_per_thread"].append((p["rank"], p["subject"], len(texts), len(p["blocks"])))
        stats["chosen"].append({"rank": p["rank"], "subject": p["subject"], "blocks": detail})
        stats["blocks_kept"] += len(texts)
        if texts:
            parts.append(header + "\n" + _BLOCK_SEPARATOR.join(texts))
    for k, p in enumerate(brief, start=len(selected) + 1):
        parts.append(f"[Email {k}: {p['subject']}]")

    stats["threads_used"] = len(parts)
    context = _BLOCK_SEPARATOR.join(parts) if parts else empty
    stats["chars_sent"] = len(context)
    return context


def format_context(chunks: list[dict], question: str = "") -> str:
    """Format retrieved thread chunks into a compact context string for the LLM.

    Thin wrapper over :func:`build_context` kept under the existing name.
    """
    return build_context(chunks, question)
