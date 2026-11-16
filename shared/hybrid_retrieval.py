"""Hybrid policy retrieval for ShopAssist.

Three signals, in strict priority order:

1. **Rules** (order facts). Authoritative. If a rule fires, its order is the
   answer and no similarity score can change it.
2. **Keywords.** Word-boundary policy terms (``refund``, ``cracked``, …).
3. **Semantic.** MiniLM cosine similarity, used to re-rank the candidates that
   keyword matching admitted — or to pick a candidate when no keyword matches.

Scoring formula (only for the non-rule path)::

    keyword_norm  = hits / max_hits_in_this_query        # 0..1
    semantic_norm = max(0, cosine_similarity)            # 0..1, raw
    score         = 0.5 * keyword_norm + 0.5 * semantic_norm

Why these weights:

* Keyword hits are a sparse, high-precision signal on this corpus (the policy
  terms are literally the domain vocabulary), so half the weight is a strong
  prior that keeps exact matches on top.
* Semantic is dense but noisy and uncalibrated (an unrelated question can score
  0.33 while a real complaint scores 0.18), so the other half only re-orders.
* We deliberately use *raw* cosine rather than min-max rescaling: rescaling
  would turn the best of five articles into "1.0" even for an unrelated query,
  inflating noise into a full vote.

Ties are broken by ``article_id`` ascending. ``limit`` is the same as before
(default 2) so prompt size does not grow.

When semantic is unavailable (missing dependency, disabled, load failure),
:func:`rank` delegates to :func:`shared.kb.select_article_ids`, i.e. exactly the
pre-existing rule + keyword behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared import semantic_retrieval
from shared.kb import article_ids, keyword_ranking, rule_candidates, select_article_ids

KEYWORD_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.5

SOURCE_RULE = "rule"
SOURCE_HYBRID = "hybrid"
SOURCE_FALLBACK = "fallback"


@dataclass(frozen=True)
class RankedArticle:
    article_id: str
    score: float
    source: str


def _semantic_map(text: str) -> dict[str, float] | None:
    """Cosine scores by article id, or ``None`` if semantic is unavailable."""
    results = semantic_retrieval.rank(text)
    if not results:
        return None
    return {item.article_id: max(0.0, item.score) for item in results}


def _fallback(text: str, order: dict, limit: int) -> list[RankedArticle]:
    ids = select_article_ids(text, order, limit)
    return [RankedArticle(aid, 0.0, SOURCE_FALLBACK) for aid in ids]


def rank(
    text: str,
    order: dict,
    *,
    limit: int = 2,
    use_semantic: bool = True,
) -> list[RankedArticle]:
    """Rank articles for a customer message + order, best first."""
    order = order or {}

    rules = rule_candidates(order)
    if rules:
        known = set(article_ids())
        return [
            RankedArticle(aid, 1.0, SOURCE_RULE) for aid in rules if aid in known
        ][:limit]

    if not use_semantic:
        return _fallback(text, order, limit)

    semantic = _semantic_map(text)
    if semantic is None:
        return _fallback(text, order, limit)

    hits = [(aid, score) for aid, score in keyword_ranking(text) if score > 0]
    if hits:
        max_hits = max(score for _, score in hits)
        candidates = [aid for aid, _ in hits]
    else:
        # No keyword admitted anything: let meaning supply the candidates.
        max_hits = 1
        candidates = [
            aid for aid, _ in sorted(semantic.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    hit_by_id = dict(hits)
    ranked = [
        RankedArticle(
            aid,
            round(
                KEYWORD_WEIGHT * (hit_by_id.get(aid, 0) / max_hits)
                + SEMANTIC_WEIGHT * semantic.get(aid, 0.0),
                6,
            ),
            SOURCE_HYBRID,
        )
        for aid in candidates
    ]
    ranked.sort(key=lambda item: (-item.score, item.article_id))
    return ranked[:limit]


def retrieve(
    text: str,
    order: dict,
    *,
    limit: int = 2,
    use_semantic: bool = True,
) -> list[str]:
    """Ids only, ready to pass to :func:`shared.kb.articles_blob`."""
    return [item.article_id for item in rank(text, order, limit=limit, use_semantic=use_semantic)]


def is_semantic_available() -> bool:
    return semantic_retrieval.is_available()
