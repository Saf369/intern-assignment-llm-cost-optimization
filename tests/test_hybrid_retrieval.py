from __future__ import annotations

import pytest

from shared import embeddings, hybrid_retrieval as hr
from shared.kb import select_article_ids
from shared.tools import format_thread, get_order, get_ticket

TICKETS = ["T-1001", "T-1002", "T-1003", "T-1004", "T-1091"]


def ticket_text(tid: str) -> str:
    return format_thread(get_ticket(tid))


def order_of(tid: str) -> dict:
    return get_order(get_ticket(tid)["order_id"])


@pytest.fixture
def semantic_ready(monkeypatch):
    monkeypatch.delenv(embeddings.DISABLE_ENV, raising=False)
    embeddings.reset_caches()
    if not hr.is_semantic_available():
        pytest.skip("sentence-transformers model is not available")
    yield
    embeddings.reset_caches()


def test_subscription_rule_beats_refund_keyword(semantic_ready):
    ranked = hr.rank("I want a refund for this.", {"product_type": "subscription"})
    assert ranked[0].article_id == "subscription-cooling-off"
    assert ranked[0].source == hr.SOURCE_RULE


def test_license_rule_beats_damage_keyword(semantic_ready):
    ranked = hr.rank("The item arrived broken.", {"license_revealed": True})
    assert ranked[0].article_id == "digital-license-no-refund"
    assert ranked[0].source == hr.SOURCE_RULE


def test_custom_rule_keeps_its_priority_order(semantic_ready):
    ranked = hr.rank(
        "My engraved mug is cracked.",
        {"customized": True, "damage_reported": True},
    )
    assert [item.article_id for item in ranked] == [
        "custom-items-no-replace",
        "damage-replacement-7-days",
    ]


def test_damage_rule(semantic_ready):
    ranked = hr.rank("It broke.", {"damage_reported": True})
    assert ranked[0].article_id == "damage-replacement-7-days"


def test_semantic_breaks_keyword_tie(semantic_ready):
    text = "I want a refund for my digital purchase."
    ranked = hr.rank(text, {})
    assert ranked[0].article_id == "digital-license-no-refund"
    assert ranked[0].source == hr.SOURCE_HYBRID


def test_keyword_only_keeps_old_tie_break():
    text = "I want a refund for my digital purchase."
    assert select_article_ids(text, {})[0] == "returns-within-30-days"
    assert hr.retrieve(text, {}, use_semantic=False)[0] == "returns-within-30-days"


def test_paraphrase_without_keywords_uses_semantics(semantic_ready):
    text = "My device stopped functioning after delivery."
    ranked = hr.rank(text, {})
    assert ranked[0].article_id == "damage-replacement-7-days"
    assert ranked[0].source == hr.SOURCE_HYBRID


def test_malfunctioning_maps_to_damage_article(semantic_ready):
    ranked = hr.rank("The item is malfunctioning.", {})
    assert ranked[0].article_id == "damage-replacement-7-days"
    assert ranked[0].source == hr.SOURCE_HYBRID


def test_defect_synonyms_are_damage_keywords():
    from shared.kb import keyword_ranking

    for word in ("malfunction", "malfunctioning", "faulty", "stopped working", "stopped functioning"):
        hits = dict(keyword_ranking(f"The item is {word}."))
        assert hits["damage-replacement-7-days"] > 0, word


def test_generic_not_working_does_not_claim_damage():
    from shared.kb import keyword_ranking

    hits = dict(keyword_ranking("The download is not working."))
    assert hits["damage-replacement-7-days"] == 0


def test_limit_is_respected(semantic_ready):
    assert len(hr.rank("broken item", {}, limit=1)) == 1
    assert len(hr.rank("broken item", {}, limit=2)) <= 2
    assert len(hr.rank("broken item", {})) <= 2


def test_scores_are_sorted_descending(semantic_ready):
    ranked = hr.rank("I want a refund for my digital purchase.", {})
    scores = [item.score for item in ranked]
    assert scores == sorted(scores, reverse=True)


def test_deterministic(semantic_ready):
    assert hr.rank("The product is defective.", {}) == hr.rank(
        "The product is defective.", {}
    )


def test_no_duplicate_articles(semantic_ready):
    ids = hr.retrieve("cracked personalized mug refund", {})
    assert len(ids) == len(set(ids))


def test_unavailable_semantic_matches_legacy_selection(monkeypatch):
    monkeypatch.setenv(embeddings.DISABLE_ENV, "1")
    embeddings.reset_caches()
    for tid in TICKETS:
        text, order = ticket_text(tid), order_of(tid)
        assert hr.retrieve(text, order) == select_article_ids(text, order)
        assert all(
            item.source in {hr.SOURCE_RULE, hr.SOURCE_FALLBACK}
            for item in hr.rank(text, order)
        )


def test_rules_still_work_with_blank_text(monkeypatch):
    monkeypatch.setenv(embeddings.DISABLE_ENV, "1")
    embeddings.reset_caches()
    ranked = hr.rank("", {"product_type": "subscription"})
    assert [item.article_id for item in ranked] == ["subscription-cooling-off"]


def test_blank_text_without_rules_falls_back():
    assert hr.retrieve("", {}, use_semantic=False) == ["returns-within-30-days"]
    assert hr.retrieve(None, {}) == ["returns-within-30-days"]
