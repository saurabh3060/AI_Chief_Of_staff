"""
chatbot/reranker.py
======================
Re-rank retrieved chunks using BM25 + semantic similarity hybrid scoring.

Ported from the standalone chatbot_final project, unchanged apart from
using this project's centralised logger.
"""

import re
from typing import Dict, List

from core.logging_config import get_agent_logger

logger = get_agent_logger("chatbot")


def _normalise(scores: List[float]) -> List[float]:
    """Min-max normalise a list of scores to [0, 1]."""
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def rerank(
    query: str,
    retrieved_chunks: List[Dict],
    bm25_index=None,          # BM25Index instance - optional for backward compat
    bm25_top_n: int = 8,
) -> List[Dict]:
    """Re-rank chunks using a hybrid of semantic score + BM25 keyword score.

    Hybrid formula (after normalising both scores to [0, 1]):
        combined = 0.6 * semantic + 0.4 * bm25

    If no BM25 index is provided (backward-compatible fallback) the old
    hand-crafted month+year keyword scoring is used instead.

    Args:
        query:            User's question.
        retrieved_chunks: Candidates with 'score' (cosine similarity).
        bm25_index:       BM25Index instance built over all chunks.
        bm25_top_n:       How many BM25 results to fetch for merging.

    Returns:
        Re-ranked list sorted descending by combined_score.
    """
    if not retrieved_chunks:
        return retrieved_chunks

    # ── Path A: BM25 hybrid ───────────────────────────────────
    if bm25_index is not None:
        # 1. Get BM25 scores for all candidates already in the pool.
        #    BM25 was queried separately in retrieve(); each chunk that came
        #    from BM25 already has 'bm25_score'. Chunks that came only from
        #    semantic search get bm25_score=0.
        semantic_scores = [c.get("score", 0.0) for c in retrieved_chunks]
        bm25_scores     = [c.get("bm25_score", 0.0) for c in retrieved_chunks]

        # 2. Normalise both score lists independently to [0, 1]
        norm_semantic = _normalise(semantic_scores)
        norm_bm25     = _normalise(bm25_scores)

        # 3. Combine: 60% semantic + 40% BM25
        for i, chunk in enumerate(retrieved_chunks):
            chunk["semantic_score_norm"] = norm_semantic[i]
            chunk["bm25_score_norm"]     = norm_bm25[i]
            chunk["combined_score"] = (
                0.6 * norm_semantic[i] + 0.4 * norm_bm25[i]
            )
            chunk["keyword_score"] = bm25_scores[i]   # kept for logging/UI

    # ── Path B: fallback (no BM25 - legacy keyword scoring) ──
    else:
        logger.warning("BM25 index not provided - falling back to keyword scoring.")
        for chunk in retrieved_chunks:
            kw = _legacy_keyword_score(query, chunk)
            chunk["combined_score"] = 0.7 * chunk.get("score", 0.0) + 0.3 * kw
            chunk["keyword_score"]  = kw

    reranked = sorted(
        retrieved_chunks,
        key=lambda x: x.get("combined_score", 0),
        reverse=True,
    )

    logger.info(
        "Re-ranked %d chunks | top combined scores: %s",
        len(reranked), [round(c.get("combined_score", 0), 3) for c in reranked[:3]],
    )
    return reranked


# ── Legacy fallback (kept for backward compatibility) ─────────
def _legacy_keyword_score(query: str, chunk: Dict) -> float:
    """Original hand-crafted keyword scorer - used only if BM25 not available."""
    query_lower = query.lower()
    score = 0.0
    source_type = chunk.get("source_type", "email")

    if source_type in ("email", "email_conversation", "email_message"):
        subject     = chunk.get("subject", "").lower()
        raw_content = chunk.get("raw_content", "").lower()
        received_at = chunk.get("received_at", "").lower()

        query_words = set(re.findall(r"\w+", query_lower))
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "what", "when",
            "where", "who", "how", "in", "on", "at", "to", "for", "of",
            "with", "and", "or", "from", "this", "that",
        }
        query_words -= stopwords

        subject_words   = set(re.findall(r"\w+", subject))
        subject_overlap = len(query_words & subject_words)
        score += min(subject_overlap * 1.5, 3.0)

        MONTH_MAP = {
            "january": "01", "jan": "01", "february": "02", "feb": "02",
            "march": "03", "mar": "03", "april": "04", "apr": "04",
            "may": "05", "june": "06", "jun": "06", "july": "07", "jul": "07",
            "august": "08", "aug": "08", "september": "09", "sep": "09",
            "october": "10", "oct": "10", "november": "11", "nov": "11",
            "december": "12", "dec": "12",
        }
        query_months = [(m, n) for m, n in MONTH_MAP.items() if m in query_lower]
        query_years  = re.findall(r"\b(20\d{2})\b", query_lower)

        if query_months:
            searchable = subject + " " + received_at + " " + raw_content[:3000]
            for month_name, month_num in query_months:
                month_found = (
                    month_name in searchable or f"-{month_num}-" in searchable
                )
                year_found = (
                    any(yr in searchable for yr in query_years)
                    if query_years else True
                )
                if month_found and year_found:
                    score += 2.5
                elif month_found:
                    score += 1.0

        if query_words:
            content_words = set(re.findall(r"\w+", raw_content))
            score += len(query_words & content_words) * 0.1

    elif source_type == "meeting":
        meeting_subject = chunk.get("subject", "").lower()
        query_words = set(re.findall(r"\w+", query_lower)) - {
            "the", "a", "an", "is", "are", "was", "were", "what", "when",
            "where", "who", "how", "in", "on", "at", "to", "for", "of", "with",
        }
        subject_words = set(re.findall(r"\w+", meeting_subject))
        score += len(query_words & subject_words) * 0.5

    return score
