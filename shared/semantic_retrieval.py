"""Semantic-only retrieval built on :mod:`shared.embeddings`.

This is the middle layer between raw vectors and the hybrid ranker. It adds
three things the embedding module should not own:

* a small, immutable result type (:class:`ScoredArticle`);
* input validation (blank/invalid queries return ``[]`` instead of raising);
* degradation (a missing model returns ``[]`` so callers can fall back).

Calibration note: on a five-article corpus, MiniLM scores are **not**
discriminative in absolute terms. Measured examples:

    "My package arrived broken."        -> 0.355 (correct)
    "The product is defective."         -> 0.185 (wrong article)
    "What time does the store open?"    -> 0.332 (should be irrelevant)

So ``min_score`` defaults to 0.0 and is only a caller-side convenience. Do not
use a global threshold as a relevance gate; use the semantic ordering to
re-rank candidates that rules/keywords already admitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from shared import embeddings

logger = logging.getLogger(__name__)

_warned: set[str] = set()


@dataclass(frozen=True)
class ScoredArticle:
    article_id: str
    score: float


def _warn_once(key: str, message: str, exc: Exception | None = None) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("%s (%s)", message, exc if exc is not None else "unknown cause")


def rank(
    text: str,
    *,
    limit: int | None = None,
    min_score: float = 0.0,
) -> list[ScoredArticle]:
    """Return articles ordered by cosine similarity, best first.

    Returns ``[]`` (never raises) for blank input or an unavailable model.
    Ties are broken by article id inside :func:`embeddings.semantic_scores`.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        scored = embeddings.semantic_scores(text)
    except embeddings.EmbeddingUnavailable as exc:
        _warn_once("unavailable", "semantic retrieval unavailable", exc)
        return []
    except Exception as exc:  # unexpected model/numpy failure: degrade, don't crash
        _warn_once("error", "semantic retrieval failed", exc)
        return []

    results = [ScoredArticle(aid, score) for aid, score in scored if score >= min_score]
    return results[:limit] if limit is not None else results


def article_ids(text: str, *, limit: int | None = None, min_score: float = 0.0) -> list[str]:
    """Convenience wrapper returning ids only."""
    return [item.article_id for item in rank(text, limit=limit, min_score=min_score)]


def is_available() -> bool:
    return embeddings.is_available()
