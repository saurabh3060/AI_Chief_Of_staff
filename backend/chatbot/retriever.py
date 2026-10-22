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

import re
import time

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


# ── Prompt-size reduction: sentence-level context builder ─────
#
# The raw thread text (thread_raw_content / messages_json) repeats quoted
# reply history, headers, signatures and address blocks, so a 3-thread prompt
# ran 4,300-8,400 tokens and Ollama truncated it. build_context() sends only
# the best-ranked threads that share a real keyword with the question, and only
# their deduplicated, question-relevant sentences (plus protected table rows)
# from subject + body, capped to a character budget. The budget, thread count
# and table share come from Settings (CHATBOT_CONTEXT_*); the constants below
# are fixed heuristics of the text cleaner.

_MIN_SENTENCE_CHARS = 25          # drop anything shorter than this
_MAX_SENTENCE_CHARS = 400         # cap each kept sentence
_MIN_SENTENCES_PER_THREAD = 2     # never drop a thread completely
# A thread with fewer useful body chars than this also draws on its
# attachment_data (question-matching lines only, capped below).
_THIN_BODY_CHARS = 300
# Max chars of attachment_data pulled in for a thin-body thread.
_ATTACHMENT_FALLBACK_MAX_CHARS = 1500
# No single thread may occupy more than this fraction of the context budget,
# so one big email cannot crowd out the others.
_PER_THREAD_BUDGET_FRACTION = 0.35
# Table detection: a run of at least _TABLE_MIN_LINES consecutive short lines
# (or number-heavy rows) of which at least _TABLE_MIN_DIGIT_SHARE carry a digit.
_TABLE_CELL_MAX_CHARS = 60        # a label/cell line is short
_TABLE_ROW_MAX_CHARS = 200        # cap on one rendered table row
_TABLE_MIN_LINES = 6
_TABLE_MIN_DIGIT_SHARE = 0.3
_TABLE_MIN_NUMBERS_IN_ROW = 3     # a longer line still counts as a row if it holds this many numbers
_TABLE_VALUE_MAX_LETTERS = 4      # "1.65%", "12 cr", "FY25" are values; sentences are not
# Generic words ignored when scoring question overlap: they appear in almost
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
# Separator rules: ----, ====, ____, **** (3+ of the same char).
_SEPARATOR_RE = re.compile(r"^\s*[-=_*]{3,}\s*$")
# "-----Original Message-----" and "On ... wrote:" reply boundaries.
_REPLY_BOUNDARY_RE = re.compile(
    r"^\s*(-{2,}\s*original message\s*-{2,}|on .{3,250}\bwrote:\s*)$",
    re.IGNORECASE,
)
# "Sample N" prefix the test corpus injects.
_SAMPLE_PREFIX_RE = re.compile(r"^\s*sample\s+\d+\s*[:\-.]?\s*", re.IGNORECASE)
# Sign-offs / signature openers.
_SIGNATURE_RE = re.compile(
    r"^\s*(regards|thanks|thank you|best regards|warm regards|kind regards|"
    r"sincerely|cheers|br|yours (?:faithfully|sincerely)|sent from my)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")
# A short line that is almost entirely an address / phone / disclaimer block.
_ADDRESS_BLOCK_RE = re.compile(
    r"^\s*(tel|phone|mob|mobile|fax|ext|ph|m|t)\s*[:.]?\s*[\d +()\-]{6,}\s*$",
    re.IGNORECASE,
)
# Signal tokens: numbers, percentages, amounts, dates.
_NUMERIC_RE = re.compile(
    r"\d|%|₹|rs\.?|inr|usd|\$|crore|cr\b|lakh|lac|bps|"
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|"
    r"november|december|q[1-4]|fy\d|h[12])\b",
    re.IGNORECASE,
)

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


def _resolve_context_params(
    budget_chars: int | None,
    max_threads: int | None,
    table_fraction: float | None,
    ignore_words: set[str] | None,
) -> tuple[int, int, float, set[str]]:
    """Fill any parameter left as None from Settings (CHATBOT_CONTEXT_*).

    Settings is only consulted when something is missing, so callers (and unit
    tests) that pass every value explicitly never touch the environment.
    """
    if None in (budget_chars, max_threads, table_fraction, ignore_words):
        settings = get_settings()
        if budget_chars is None:
            budget_chars = settings.chatbot_context_max_chars
        if max_threads is None:
            max_threads = settings.chatbot_context_top_threads
        if table_fraction is None:
            table_fraction = settings.chatbot_context_table_fraction
        if ignore_words is None:
            ignore_words = {
                w.strip().lower()
                for w in settings.chatbot_context_ignore_words.split(",")
                if w.strip()
            }
    return (
        max(1, int(budget_chars)),
        max(1, int(max_threads)),
        min(max(float(table_fraction), 0.0), 1.0),
        ignore_words,
    )


def _thread_body(chunk: dict) -> str:
    """Return ONLY the thread's own subject-free body text.

    Per spec build_context must use each thread's ``subject`` and ``body``
    only - never thread_raw_content, messages_json, metadata_summary,
    attachment_data, or any id field.
    """
    return str(chunk.get("body") or "").strip()


def _split_sentences(text: str) -> list[str]:
    """Split a body into candidate sentences/lines.

    Split on newlines first (emails are line-oriented), then on sentence
    terminators so long paragraphs become individual sentences.
    """
    pieces: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split a line further on sentence boundaries (. ! ?) followed by space.
        for part in re.split(r"(?<=[.!?])\s+", line):
            part = part.strip()
            if part:
                pieces.append(part)
    return pieces


def _is_noise(sentence: str) -> bool:
    """True if a sentence is a header/separator/signature/address/url/noise line."""
    s = sentence.strip()
    if not s:
        return True
    if _HEADER_LINE_RE.match(s):
        return True
    if _SEPARATOR_RE.match(s):
        return True
    if _REPLY_BOUNDARY_RE.match(s):
        return True
    if _SIGNATURE_RE.match(s):
        return True
    if _ADDRESS_BLOCK_RE.match(s):
        return True
    if _URL_RE.search(s):
        return True
    # A line that is mostly an email address with little else.
    if _EMAIL_RE.search(s) and len(_EMAIL_RE.sub("", s).strip()) < 15:
        return True
    return False


def _clean_sentence(sentence: str) -> str:
    """Strip the 'Sample N' prefix and cap the sentence length."""
    s = _SAMPLE_PREFIX_RE.sub("", sentence).strip()
    if len(s) > _MAX_SENTENCE_CHARS:
        s = s[:_MAX_SENTENCE_CHARS].rstrip()
    return s


def _normalise_key(sentence: str) -> str:
    """Normalise a sentence for cross-thread dedup: lowercase, alnum only."""
    return re.sub(r"[^a-z0-9]+", "", sentence.lower())


def _question_words(question: str, ignore_words: set[str] | None = None) -> set[str]:
    """Significant (non-stopword, non-generic) words from the question, lowercased.

    Generic business words (q1-q4, fy, fy25/fy26, actuals, data, please, find,
    below), question phrasing (much, many, percentage ...) and any configured
    ``ignore_words`` (CHATBOT_CONTEXT_IGNORE_WORDS) are dropped too: they give
    no discriminating signal for thread/sentence scoring.
    """
    ignored = ignore_words or set()
    words = re.findall(r"\w+", (question or "").lower())
    return {
        w for w in words
        if w not in _STOPWORDS and w not in _GENERIC_WORDS and w not in ignored
        and len(w) > 1
    }


def _score_sentence(sentence: str, q_words: set[str]) -> float:
    """Overlap score between a sentence and the question words.

    Each distinct question word present scores 1; a small bonus is added for
    sentences carrying numeric/date/amount signal so figures are preferred
    when overlap ties.
    """
    if not q_words:
        base = 0.0
    else:
        s_words = set(re.findall(r"\w+", sentence.lower()))
        base = float(len(q_words & s_words))
    if _NUMERIC_RE.search(sentence):
        base += 0.5
    return base


def _has_numeric(sentence: str) -> bool:
    """True if the sentence contains a number, date, percentage or amount."""
    return bool(_NUMERIC_RE.search(sentence))


def _attachment_fallback_sentences(chunk: dict, q_words: set[str]) -> list[str]:
    """Return cleaned attachment_data lines that match the question words.

    Used only when a thread's body is thin (< _THIN_BODY_CHARS useful chars).
    Keeps only lines overlapping the (generic-filtered) question words, applies
    the same noise/length/clean rules as the body, and caps the total at
    _ATTACHMENT_FALLBACK_MAX_CHARS per thread.
    """
    attachment = str(chunk.get("attachment_data") or "").strip()
    if not attachment or not q_words:
        return []

    picked: list[str] = []
    used = 0
    for sent in _split_sentences(attachment):
        if _is_noise(sent):
            continue
        cleaned = _clean_sentence(sent)
        if len(cleaned) < _MIN_SENTENCE_CHARS:
            continue
        s_words = set(re.findall(r"\w+", cleaned.lower()))
        if not (q_words & s_words):
            continue
        if used + len(cleaned) > _ATTACHMENT_FALLBACK_MAX_CHARS:
            break
        picked.append(cleaned)
        used += len(cleaned)
    return picked


_DIGIT_GROUP_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _is_table_line(line: str) -> bool:
    """True for a line that can belong to a table: a short label/value cell, or
    a longer row holding several numbers. Header/signature/url noise is not."""
    if _is_noise(line):
        return False
    if len(line) <= _TABLE_CELL_MAX_CHARS:
        return True
    return (
        len(line) <= _TABLE_ROW_MAX_CHARS
        and len(_DIGIT_GROUP_RE.findall(line)) >= _TABLE_MIN_NUMBERS_IN_ROW
    )


def _looks_like_table(run: list[str]) -> bool:
    """A table is a long enough run of table-like lines with enough numbers."""
    if len(run) < _TABLE_MIN_LINES:
        return False
    with_digit = sum(1 for line in run if re.search(r"\d", line))
    return with_digit / len(run) >= _TABLE_MIN_DIGIT_SHARE


def _is_value_line(line: str) -> bool:
    """A mostly-numeric cell such as '1.65%', '12 cr' or 'FY25'."""
    return bool(re.search(r"\d", line)) and len(re.findall(r"[A-Za-z]", line)) <= _TABLE_VALUE_MAX_LETTERS


def _table_rows(run: list[str]) -> list[str]:
    """Join a table's lines into rows: a label line followed by value line(s)
    becomes ``label: v1 | v2``; any other line is kept as its own row."""
    rows: list[str] = []
    label: str | None = None
    values: list[str] = []

    def flush() -> None:
        nonlocal label, values
        if label is not None:
            rows.append(f"{label}: {' | '.join(values)}" if values else label)
        label, values = None, []

    for line in run:
        if _is_value_line(line):
            if label is None:
                rows.append(line)
            else:
                values.append(line)
        else:
            flush()
            label = line
    flush()
    return [row[:_TABLE_ROW_MAX_CHARS] for row in rows]


def _extract_tables(body: str) -> tuple[str, list[str]]:
    """Split a body into (prose text, table rows).

    Table rows bypass the noise filter, the minimum-length rule, dedup and
    scoring (see build_context); everything else is returned as prose.
    """
    prose: list[str] = []
    rows: list[str] = []
    run: list[str] = []
    blanks = 0

    def close() -> None:
        nonlocal run
        if _looks_like_table(run):
            rows.extend(_table_rows(run))
        else:
            prose.extend(run)
        run = []

    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            blanks += 1
            if blanks >= 2:
                close()
            continue
        blanks = 0
        if _is_table_line(line):
            run.append(line)
        else:
            close()
            prose.append(line)
    close()
    return "\n".join(prose), rows


def build_context(
    threads: list[dict],
    question: str,
    budget_chars: int | None = None,
    stats: dict | None = None,
    max_threads: int | None = None,
    table_fraction: float | None = None,
    ignore_words: set[str] | None = None,
) -> str:
    """Build a compact, deduplicated, question-relevant context string.

    Sends at most ``max_threads`` threads (best-ranked first) and only those
    that share a real keyword with the question; their best-scoring,
    deduplicated sentences plus protected table rows are grouped under
    ``[Email N: subject]`` with N counting the threads actually sent.
    Parameters left as None come from Settings (CHATBOT_CONTEXT_MAX_CHARS,
    CHATBOT_CONTEXT_TOP_THREADS, CHATBOT_CONTEXT_TABLE_FRACTION,
    CHATBOT_CONTEXT_IGNORE_WORDS).

    Rules implemented:
      * subject + body; never raw/messages_json/metadata/ids. A thread with a
        thin body (< 300 useful chars) also draws question-matching lines from
        its attachment_data, capped at 1500 chars for that thread
      * table rows (runs of short label/value lines or number-heavy rows) are
        joined as "label: value" and are NEVER dropped by the noise filter, the
        25-char minimum, dedup or scoring. They have their own budget
        (table_fraction of budget_chars); rows beyond it are cut from the end of
        the table and a WARNING naming the thread subject is logged
      * drop lines < 25 chars and header/separator/signature/address/url noise
        from the prose
      * dedup prose sentences across ALL threads (normalise lower + alnum)
      * score by question-word overlap (+ numeric bonus), ignoring generic and
        question words and configured ignore words
      * a thread has a real keyword match when a question word occurs in one of
        its sentences, its subject or its table rows; threads without one are
        dropped (skipped when the question has no significant words)
      * keep best prose sentences until the remaining budget; cap each sentence
        at 400 chars and each thread's prose at 35% of the budget
      * re-order kept sentences into original thread order
      * every sent thread gets >= 1 line: the top sentences by score, or, if the
        budget is already spent, its single best sentence (a small overshoot)
      * keep a numeric/date sentence that sits next to a kept sentence

    If ``stats`` (a dict) is passed, it is populated with diagnostics:
      stats["kept_per_thread"] = [(rank, subject, kept_count), ...]  (one tuple
          per input thread, in retrieval order)
      stats["threads_in"], stats["threads_used"]
      stats["table_rows_kept"], stats["table_rows_cut"]
    The returned context string is unaffected.
    """
    n_in = len(threads)
    if not threads:
        if stats is not None:
            stats.update(
                kept_per_thread=[], threads_in=0, threads_used=0,
                table_rows_kept=0, table_rows_cut=0,
            )
        return "(no relevant emails found)"

    budget, max_threads, table_fraction, ignore_words = _resolve_context_params(
        budget_chars, max_threads, table_fraction, ignore_words,
    )
    q_words = _question_words(question, ignore_words)

    # ── Per-thread extraction (preserving original order) ──
    # thread_sentences[t] = list of (orig_index, cleaned_sentence, score, norm_key)
    thread_sentences: list[list[tuple]] = []
    thread_tables: list[list[str]] = []
    thread_fallback: list[list[str]] = []
    for chunk in threads:
        sentences: list[tuple] = []

        def _append(text: str) -> None:
            cleaned = _clean_sentence(text)
            if len(cleaned) < _MIN_SENTENCE_CHARS:
                return
            key = _normalise_key(cleaned)
            if not key:
                return
            sentences.append((len(sentences), cleaned, _score_sentence(cleaned, q_words), key))

        prose, table_rows = _extract_tables(_thread_body(chunk))

        body_useful_chars = 0
        for sent in _split_sentences(prose):
            if _is_noise(sent):
                continue
            before = len(sentences)
            _append(sent)
            if len(sentences) > before:
                body_useful_chars += len(sentences[-1][1])

        # Thin body: supplement with question-matching attachment_data lines.
        if body_useful_chars < _THIN_BODY_CHARS:
            for line in _attachment_fallback_sentences(chunk, q_words):
                _append(line)

        thread_sentences.append(sentences)
        thread_tables.append(table_rows)
        # Last-resort lines (ignoring the length rule) for a sent thread that
        # would otherwise end up with nothing.
        thread_fallback.append(
            [_clean_sentence(p) for p in _split_sentences(prose) if not _is_noise(p)][:_MIN_SENTENCES_PER_THREAD]
        )

    # ── Choose the threads to send: real keyword match, best-ranked first ──
    def _has_keyword_match(t: int) -> bool:
        if not q_words:
            return True
        if any(score >= 1.0 for (_i, _c, score, _k) in thread_sentences[t]):
            return True
        subject_words = set(re.findall(r"\w+", str(threads[t].get("subject") or "").lower()))
        if q_words & subject_words:
            return True
        row_words = set(re.findall(r"\w+", " ".join(thread_tables[t]).lower()))
        return bool(q_words & row_words)

    sent_threads = [t for t in range(n_in) if _has_keyword_match(t)][:max_threads]
    chosen = set(sent_threads)
    for t in range(n_in):
        if t not in chosen:
            thread_sentences[t].clear()
            thread_tables[t] = []
            thread_fallback[t] = []

    # ── Global dedup of prose: keep the first occurrence of each sentence ──
    seen_keys: set[str] = set()
    for sentences in thread_sentences:
        kept = []
        for orig_idx, cleaned, score, key in sentences:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            kept.append((orig_idx, cleaned, score, key))
        sentences[:] = kept

    # ── Protected table rows (own budget; never dedup'd or scored away) ──
    table_budget = int(budget * table_fraction)
    table_used = 0
    rows_kept = 0
    rows_cut = 0
    table_kept: list[list[str]] = [[] for _ in threads]
    for t in sent_threads:
        rows = thread_tables[t]
        for i, row in enumerate(rows):
            if table_used + len(row) + 1 > table_budget:
                cut = len(rows) - i
                rows_cut += cut
                logger.warning(
                    "build_context: cut %d of %d table rows for thread %r - "
                    "table budget of %d chars exhausted (raise CHATBOT_CONTEXT_MAX_CHARS "
                    "or CHATBOT_CONTEXT_TABLE_FRACTION)",
                    cut, len(rows), threads[t].get("subject") or "(no subject)", table_budget,
                )
                break
            table_kept[t].append(row)
            table_used += len(row) + 1
            rows_kept += 1

    # ── Prose selection within what the tables left over ──
    prose_budget = max(0, budget - table_used)
    selected: list[set] = [set() for _ in thread_sentences]
    used_chars = 0
    thread_used: list[int] = [0 for _ in thread_sentences]
    per_thread_cap = max(1, int(budget * _PER_THREAD_BUDGET_FRACTION))

    def _try_add(t: int, orig_idx: int, cleaned: str) -> bool:
        nonlocal used_chars
        if orig_idx in selected[t]:
            return False
        if used_chars + len(cleaned) > prose_budget:
            return False
        if thread_used[t] + len(cleaned) > per_thread_cap:
            return False
        selected[t].add(orig_idx)
        used_chars += len(cleaned)
        thread_used[t] += len(cleaned)
        return True

    # 1) Guarantee the top N sentences from every sent thread (by score, then order).
    for t in sent_threads:
        floor = sorted(thread_sentences[t], key=lambda s: (-s[2], s[0]))[:_MIN_SENTENCES_PER_THREAD]
        for orig_idx, cleaned, _score, _key in floor:
            _try_add(t, orig_idx, cleaned)

    # 1b) A sent thread must never end up with 0 lines: force its single best
    #     sentence in even if that overshoots the budget slightly, else fall
    #     back to its first plain lines.
    forced_lines: dict[int, list[str]] = {}
    for t in sent_threads:
        if selected[t] or table_kept[t]:
            continue
        if thread_sentences[t]:
            best = min(thread_sentences[t], key=lambda s: (-s[2], s[0]))
            selected[t].add(best[0])
            used_chars += len(best[1])
            thread_used[t] += len(best[1])
        elif thread_fallback[t]:
            forced_lines[t] = thread_fallback[t]

    # 2) Fill remaining budget with the globally best-scoring sentences.
    ranked = sorted(
        (
            (t, orig_idx, cleaned, score)
            for t, sentences in enumerate(thread_sentences)
            for (orig_idx, cleaned, score, _key) in sentences
        ),
        key=lambda s: (-s[3], s[0], s[1]),
    )
    for t, orig_idx, cleaned, _score in ranked:
        _try_add(t, orig_idx, cleaned)

    # 3) Keep a numeric/date/amount sentence that sits right next to a kept one.
    for t, sentences in enumerate(thread_sentences):
        for orig_idx, cleaned, _score, _key in sentences:
            if orig_idx in selected[t] or not _has_numeric(cleaned):
                continue
            if (orig_idx - 1) in selected[t] or (orig_idx + 1) in selected[t]:
                _try_add(t, orig_idx, cleaned)

    # ── Render in original thread + sentence order, tables after the prose ──
    blocks: list[str] = []
    kept_per_thread: list[tuple] = []
    for t, chunk in enumerate(threads):
        subject = chunk.get("subject") or "(no subject)"
        by_idx = {orig_idx: cleaned for (orig_idx, cleaned, _s, _k) in thread_sentences[t]}
        lines = [by_idx[i] for i in sorted(selected[t]) if i in by_idx]
        lines += forced_lines.get(t, [])
        lines += table_kept[t]
        kept_per_thread.append((t + 1, subject, len(lines)))
        if t not in chosen:
            continue
        if not lines:
            logger.warning("build_context: thread %r has no usable body lines to send", subject)
            continue
        blocks.append(f"[Email {len(blocks) + 1}: {subject}]\n" + "\n".join(lines))

    if stats is not None:
        stats["kept_per_thread"] = kept_per_thread
        stats["threads_in"] = n_in
        stats["threads_used"] = len(blocks)
        stats["table_rows_kept"] = rows_kept
        stats["table_rows_cut"] = rows_cut

    return "\n\n".join(blocks) if blocks else "(no relevant emails found)"


def format_context(chunks: list[dict], question: str = "") -> str:
    """Format retrieved thread chunks into a compact context string for the LLM.

    Thin wrapper over :func:`build_context` kept under the existing name so
    callers in ``chatbot.service`` need only pass the user question. Keeps ALL
    retrieved threads but sends only deduplicated, question-relevant sentences
    from each thread's subject + body (see build_context for the rules).
    """
    return build_context(chunks, question)
