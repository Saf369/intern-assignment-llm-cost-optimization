from __future__ import annotations

import pytest

from shared import embeddings


@pytest.fixture
def model_ready(monkeypatch):
    monkeypatch.delenv(embeddings.DISABLE_ENV, raising=False)
    embeddings.reset_caches()
    if not embeddings.is_available():
        pytest.skip("sentence-transformers model is not available")
    yield
    embeddings.reset_caches()


def test_article_text_contains_id_as_words():
    text = embeddings.article_text("damage-replacement-7-days")
    assert "damage replacement 7 days" in text
    assert "replacement shipment" in text


def test_empty_query_is_rejected():
    with pytest.raises(ValueError):
        embeddings.embed_query("")
    with pytest.raises(ValueError):
        embeddings.embed_query("   ")


def test_disabled_by_environment(monkeypatch):
    monkeypatch.setenv(embeddings.DISABLE_ENV, "1")
    embeddings.reset_caches()
    with pytest.raises(embeddings.EmbeddingUnavailable):
        embeddings.get_model()
    assert embeddings.is_available() is False


def test_model_loading_failure_is_wrapped(monkeypatch):
    def boom():
        raise OSError("model files missing")

    monkeypatch.delenv(embeddings.DISABLE_ENV, raising=False)
    monkeypatch.setattr(embeddings, "_load_model", boom)
    embeddings.reset_caches()
    with pytest.raises(embeddings.EmbeddingUnavailable):
        embeddings.get_model()
    assert embeddings.is_available() is False


def test_model_is_loaded_once(model_ready):
    assert embeddings.get_model() is embeddings.get_model()


def test_article_vectors_are_cached(monkeypatch):
    calls: list[int] = []

    class FakeModel:
        def encode(self, texts, **_):
            import numpy as np

            calls.append(len(texts))
            return np.ones((len(texts), 4), dtype="float32")

    monkeypatch.setattr(embeddings, "_model", FakeModel())
    embeddings.article_vectors.cache_clear()
    first = embeddings.article_vectors()
    second = embeddings.article_vectors()
    assert first is second
    assert calls == [5]
    assert len(first[0]) == len(first[1]) == 5


def test_semantic_scores_are_normalised_and_sorted(model_ready):
    ids, matrix = embeddings.article_vectors()
    for row in matrix:
        assert abs(float((row * row).sum()) - 1.0) < 1e-4
    scored = embeddings.semantic_scores("My device stopped working.")
    assert len(scored) == len(ids)
    assert {aid for aid, _ in scored} == set(ids)
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_semantic_scores_are_deterministic(model_ready):
    a = embeddings.semantic_scores("The product is defective.")
    b = embeddings.semantic_scores("The product is defective.")
    assert a == b


def test_related_wording_matches_damage_policy(model_ready):
    scored = embeddings.semantic_scores("My device stopped functioning after delivery.")
    assert scored[0][0] == "damage-replacement-7-days"
