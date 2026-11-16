from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from shared.tokens import count_tokens

KB_DIR = Path(__file__).resolve().parents[1] / "data" / "kb"


@lru_cache
def article_ids() -> list[str]:
    return sorted(p.stem for p in KB_DIR.glob("*.md"))


@lru_cache
def load_article(article_id: str) -> str:
    return (KB_DIR / f"{article_id}.md").read_text()


@lru_cache
def all_articles_blob() -> str:
    """Concatenated help-center articles for Question 1."""
    parts = []
    for aid in article_ids():
        parts.append(f"--- {aid} ---\n{load_article(aid)}\n")
    return "\n".join(parts)


_ARTICLE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "custom-items-no-replace",
        ("custom", "customized", "personalized", "personalised", "engraved", "printed"),
    ),
    (
        "damage-replacement-7-days",
        (
            "damage",
            "damaged",
            "cracked",
            "broken",
            "defective",
            # Functional-defect synonyms. Deliberately specific: a generic
            # "not working" would also match "the download isn't working",
            # where digital-license-no-refund should win.
            "malfunction",
            "malfunctioning",
            "faulty",
            "stopped working",
            "stopped functioning",
        ),
    ),
    (
        "digital-license-no-refund",
        ("digital", "ebook", "e-book", "license", "licence", "download"),
    ),
    ("subscription-cooling-off", ("subscription", "subscriptions", "renewal", "renew")),
    ("returns-within-30-days", ("refund", "refunds", "return", "returns", "unopened")),
)


def rule_candidates(order: dict) -> list[str] | None:
    """Order-fact rules. Returns ``None`` when no explicit rule applies."""
    if order.get("product_type") == "subscription":
        return ["subscription-cooling-off"]
    if order.get("license_revealed"):
        return ["digital-license-no-refund"]
    if order.get("customized") and order.get("damage_reported"):
        return ["custom-items-no-replace", "damage-replacement-7-days"]
    if order.get("damage_reported"):
        return ["damage-replacement-7-days", "custom-items-no-replace"]
    return None


def keyword_ranking(text: str) -> list[tuple[str, int]]:
    """Articles with their word-boundary keyword hit count, best first."""
    low = (text if isinstance(text, str) else "").lower()
    scored = sorted(
        (
            (
                sum(
                    1
                    for word in words
                    if re.search(rf"\b{re.escape(word)}\b", low)
                ),
                aid,
            )
            for aid, words in _ARTICLE_RULES
        ),
        reverse=True,
    )
    return [(aid, score) for score, aid in scored]


def select_article_ids(text: str, order: dict, limit: int = 2) -> list[str]:
    """Cheap deterministic shortlist of articles worth sending to the model.

    Order facts decide first (they are authoritative), keyword overlap is the
    fallback. Only these articles enter the prompt, instead of the whole KB.
    """
    ranked = rule_candidates(order or {})
    if ranked is None:
        ranked = [aid for aid, score in keyword_ranking(text) if score > 0]
        ranked = ranked or ["returns-within-30-days"]
    known = set(article_ids())
    return [aid for aid in ranked if aid in known][:limit]


def articles_blob(ids: Iterable[str]) -> str:
    """Concatenate only the selected articles."""
    return "\n".join(f"--- {aid} ---\n{load_article(aid).strip()}" for aid in ids)


@lru_cache
def expanded_help_center_blob() -> str:
    """Expanded help center for Question 2."""
    base = all_articles_blob()
    filler = (
        "## Internal routing note\n"
        "Carrier pickup windows, warehouse bin labels, and CS macro catalog. "
        "Does not change customer policy unless an article id is cited.\n"
    )
    chunks = [base]
    i = 0
    while count_tokens("\n".join(chunks)) < 210_000:
        chunks.append(filler + f"Batch {i}\n" + ("-" * 40 + "\n") * 30)
        i += 1
    return "\n".join(chunks)


def title_index(limit: int | None = 400) -> str:
    ids = article_ids()
    if limit:
        ids = ids[:limit]
    return "\n".join(f"{aid}" for aid in ids)
