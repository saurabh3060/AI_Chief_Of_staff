"""
chatbot/bm25_index.py
========================
BM25 index over loaded message chunks, used alongside Weaviate
semantic search for hybrid retrieval.

Ported from the standalone chatbot_final project, unchanged apart from
using this project's centralised logger.
"""

import re

from rank_bm25 import BM25Okapi

from core.logging_config import get_agent_logger

logger = get_agent_logger("chatbot")

# Stopwords to strip before tokenising - keeps BM25 focused on signal terms
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "what", "when", "where", "who", "which", "how", "why",
    "in", "on", "at", "to", "for", "of", "with", "and", "or", "but",
    "from", "this", "that", "these", "those", "it", "its", "we", "our",
    "i", "you", "he", "she", "they", "them", "their", "my", "your",
    "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "as", "if", "than", "then", "about", "above", "across", "after",
    "also", "any", "each", "few", "more", "most", "other", "some",
    "such", "into", "through", "during", "before", "between",
}


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _chunk_document(chunk: dict) -> str:
    """Build searchable text for one individual message."""
    subject = chunk.get("subject", "") or chunk.get("thread_subject", "")
    body    = chunk.get("body", "") or chunk.get("raw_content", "")
    # Subject included once - no artificial boost, so body content
    # (which often contains cross-month data) gets a fair chance.
    return f"{subject} {body}"


class BM25Index:
    """Thin wrapper around BM25Okapi that keeps chunk references aligned."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        message_ids = [
            chunk.get("message_id") for chunk in chunks
            if chunk.get("source_type") == "email_message"
        ]
        if len(message_ids) != len(set(message_ids)):
            raise ValueError("BM25 received duplicate message_id values")

        logger.info("Building BM25 index over %d message documents...", len(chunks))
        corpus = [_tokenise(_chunk_document(c)) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        logger.info("BM25 index ready.")

    def query(self, query_text: str, top_n: int = 8) -> list[dict]:
        """Return up to top_n chunks ranked by BM25 score.

        Each returned dict is a shallow copy of the original chunk with an
        extra 'bm25_score' key added.
        """
        tokens = _tokenise(query_text)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Pair each score with its chunk index, sort descending
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored[:top_n]:
            if score <= 0:
                break   # BM25 score of 0 means no term overlap - not useful
            chunk = self.chunks[idx].copy()
            chunk["bm25_score"] = float(score)
            results.append(chunk)

        if results:
            logger.info(
                "BM25 query '%s': %d results, top score=%.3f",
                query_text[:60], len(results), results[0]["bm25_score"],
            )
        else:
            logger.info("BM25 query '%s': no results", query_text[:60])
        return results


def build_bm25(chunks: list[dict]) -> BM25Index:
    """Build and return a BM25Index from a list of chunk dicts."""
    return BM25Index(chunks)
