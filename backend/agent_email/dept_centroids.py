
"""
agent_email/dept_centroids.py
================================
Cached department embeddings for relevance scoring and cluster
department assignment.

Each BU department is represented two ways from the same dictionary
(summary + example phrases, no per-subject rules):

- **centroid**: mean of all seed vectors. Used by the relevance filter
  as a single "is this mail in-BU?" probe.
- **prototypes**: the individual seed vectors. Cluster department is
  the nearest prototype (max cosine to any seed of that department).

A mean centroid is a poor tagger for a broad department: Wholesale
Banking covers CASA, NIM, TAT, pipeline, trade, etc., so its average
sits in the middle of that mix. A narrow department whose seeds are
almost all CASA then wins any CASA-like mail by geometry, even when
the mail is wholesale CA/SA. Scoring against each phrase separately
avoids that dilution.

When the top two prototype scores are within a configurable margin,
token overlap against each department's own dictionary text is the
tie-break — still no per-mail rules.
"""

from __future__ import annotations

import asyncio

import numpy as np

from core.config import Department, Settings, get_settings
from core.logging_config import get_agent_logger
from core.text_match import meaningful_tokens, token_overlap_score

from agent_email.semantic_match import cosine_similarity, embed_text

logger = get_agent_logger("agent_email")

_index_cache: dict[str, dict] = {}
_index_lock = asyncio.Lock()


def _centroid_signature(settings: Settings) -> str:
    dept_part = "|".join(f"{d.name}:{d.summary}" for d in settings.bu_departments)
    phrases = settings.department_tagging_example_phrases
    phrase_part = "|".join(f"{k}:{','.join(v)}" for k, v in sorted(phrases.items()))
    model = getattr(settings, "embedding_model", "")
    return f"{model}::{dept_part}::{phrase_part}"


def _department_seed_texts(dept: Department, settings: Settings) -> list[str]:
    texts: list[str] = []
    if dept.summary.strip():
        texts.append(dept.summary.strip())
    for phrase in settings.department_tagging_example_phrases.get(dept.name, []):
        if phrase.strip():
            texts.append(phrase.strip())
    if not texts:
        texts.append(dept.name)
    return texts


def _l2_normalize_rows(stacked: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return stacked / norms


async def build_department_index(settings: Settings | None = None) -> dict:
    """
    Build (and cache) centroids, per-seed prototypes, and dictionary
    token sets for every configured department.
    """
    active = settings or get_settings()
    signature = _centroid_signature(active)
    cached = _index_cache.get(signature)
    if cached is not None:
        return cached

    async with _index_lock:
        cached = _index_cache.get(signature)
        if cached is not None:
            return cached

        departments = active.bu_departments
        logger.info(
            "[DEPT-CENTROID] building centroids for %d department(s) model=%s",
            len(departments), getattr(active, "embedding_model", ""),
        )
        centroids: dict[str, np.ndarray] = {}
        prototypes: dict[str, np.ndarray] = {}
        dept_tokens: dict[str, set[str]] = {}
        for dept in departments:
            seed_texts = _department_seed_texts(dept, active)
            vecs = await asyncio.gather(*(embed_text(t, active) for t in seed_texts))
            stacked = np.vstack(vecs)
            unit_rows = _l2_normalize_rows(stacked)
            prototypes[dept.name] = unit_rows
            centroid = stacked.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[dept.name] = centroid
            dept_tokens[dept.name] = meaningful_tokens(" ".join([dept.name, *seed_texts]))
            logger.info(
                "[DEPT-CENTROID] department=%r seed_texts=%d dim=%d",
                dept.name, len(seed_texts), centroid.shape[0],
            )

        index = {
            "centroids": centroids,
            "prototypes": prototypes,
            "dept_tokens": dept_tokens,
        }
        _index_cache[signature] = index
        return index


async def build_centroids(settings: Settings | None = None) -> dict[str, np.ndarray]:
    """Build (and cache) one averaged, L2-normalized centroid per department."""
    index = await build_department_index(settings)
    return index["centroids"]


def raw_cosine_scores(vec: np.ndarray, centroids: dict[str, np.ndarray]) -> dict[str, float]:
    """Raw cosine similarity (0-100, clamped) against each department centroid."""
    scores: dict[str, float] = {}
    for name, centroid in centroids.items():
        raw = cosine_similarity(vec, centroid)
        scores[name] = max(0.0, min(1.0, raw)) * 100.0
    return scores


def prototype_scores(vec: np.ndarray, prototypes: dict[str, np.ndarray]) -> dict[str, float]:
    """Max cosine (0-100, clamped) of `vec` against any seed of each department."""
    scores: dict[str, float] = {}
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return {name: 0.0 for name in prototypes}
    unit = vec / norm
    for name, seeds in prototypes.items():
        if seeds is None or seeds.size == 0:
            scores[name] = 0.0
            continue
        sims = seeds @ unit
        scores[name] = max(0.0, min(1.0, float(np.max(sims)))) * 100.0
    return scores


def pick_department(
    vec: np.ndarray,
    centroids: dict[str, np.ndarray],
    prototypes: dict[str, np.ndarray] | None = None,
    cluster_tokens: set[str] | None = None,
    dept_tokens: dict[str, set[str]] | None = None,
    margin: float = 0.0,
    min_score: float = 0.0,
) -> tuple[str, dict[str, float]]:
    """
    Assign a cluster (or email) to a department from the dictionary.

    Uses nearest prototype when `prototypes` is provided, else nearest
    mean centroid. No per-subject rules. Returns ("", {}) when empty.

    If the top two scores differ by `margin` or less and token sets are
    supplied, the close contenders are broken by token overlap against
    each department's own summary + phrases.
    """
    scores = prototype_scores(vec, prototypes) if prototypes else raw_cosine_scores(vec, centroids)
    if not scores:
        return "", {}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_name, best_score = ranked[0]
    if min_score > 0 and best_score < min_score:
        logger.info(
            "[CLUSTER-DEPT] best=%r score=%.1f below min=%.1f -> unlabeled",
            best_name, best_score, min_score,
        )
        return "", scores
    if (
        margin > 0
        and cluster_tokens is not None
        and dept_tokens
        and len(ranked) >= 2
    ):
        contenders = [
            name for name, score in ranked
            if best_score - score <= margin and (min_score <= 0 or score >= min_score)
        ]
        if len(contenders) > 1:
            token_scores = {
                name: token_overlap_score(cluster_tokens, dept_tokens.get(name, set()))
                for name in contenders
            }
            picked = max(contenders, key=lambda name: (token_scores[name], scores[name]))
            logger.info(
                "[CLUSTER-DEPT-TIEBREAK] semantic_best=%r picked=%r margin=%.1f "
                "contenders=%s token_scores=%s",
                best_name, picked, margin, contenders,
                {name: round(token_scores[name], 1) for name in contenders},
            )
            return picked, scores
    return best_name, scores


def department_reference_tokens(settings: Settings) -> set[str]:
    """Union of meaningful tokens from all department summaries + phrases."""
    tokens: set[str] = set()
    for dept in settings.bu_departments:
        tokens |= meaningful_tokens(dept.summary)
        for phrase in settings.department_tagging_example_phrases.get(dept.name, []):
            tokens |= meaningful_tokens(phrase)
        tokens |= meaningful_tokens(dept.name)
    return tokens
