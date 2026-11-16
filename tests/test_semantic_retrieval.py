from __future__ import annotations

import pytest

from shared import embeddings, semantic_retrieval as sr


@pytest.fixture
def model_ready(monkeypatch):
    monkeypatch.delenv(embeddings.DISABLE_ENV, raising=False)
    embeddings.reset_caches()
    if not sr.is_available():
        pytest.skip("sentence-transformers model is not available")
    yield
    embeddings.reset_caches()


def test_clear_query_ranks_correct_article_first(model_ready):
    ranked = sr.rank("My package arrived broken.")
    assert ranked[0].article_id == "damage-replacement-7-days"
    assert 0.0 < ranked[0].score <= 1.0


def test_results_are_sorted_and_limited(model_ready):
    ranked = sr.rank("I want a refund for my digital purchase.", limit=3)
    assert len(ranked) == 3
    scores = [item.score for item in ranked]
    assert scores == sorted(scores, reverse=True)


def test_all_five_articles_are_scored(model_ready):
    ranked = sr.rank("My device stopped working.")
    assert len(ranked) == 5
    assert len({item.article_id for item in ranked}) == 5


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None, 42])
def test_invalid_queries_return_empty(bad):
    assert sr.rank(bad) == []
    assert sr.article_ids(bad) == []


def test_min_score_filters(model_ready):
    strong = sr.rank("I want a refund for my digital purchase.", min_score=0.6)
    assert [item.article_id for item in strong] == ["digital-license-no-refund"]


def test_deterministic(model_ready):
    assert sr.rank("The product is defective.") == sr.rank("The product is defective.")


def test_disabled_model_degrades_to_empty(monkeypatch):
    monkeypatch.setenv(embeddings.DISABLE_ENV, "1")
    embeddings.reset_caches()
    assert sr.is_available() is False
    assert sr.rank("My package arrived broken.") == []


def test_load_failure_degrades_to_empty(monkeypatch):
    def boom():
        raise OSError("no model files")

    monkeypatch.delenv(embeddings.DISABLE_ENV, raising=False)
    monkeypatch.setattr(embeddings, "_load_model", boom)
    embeddings.reset_caches()
    assert sr.rank("My package arrived broken.") == []


def test_article_ids_matches_rank(model_ready):
    assert sr.article_ids("Please cancel my yearly plan.", limit=2) == [
        item.article_id for item in sr.rank("Please cancel my yearly plan.", limit=2)
    ]
