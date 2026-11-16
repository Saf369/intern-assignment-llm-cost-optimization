"""Local sentence embeddings for policy retrieval.

What an embedding is, in one sentence: a model turns a piece of text into a
list of numbers (a vector) so that texts with similar *meaning* end up close
together, even when they share no words.

This module is deliberately small and defensive:

* the model is loaded **lazily** (first use, not at import time) and only once;
* article vectors are computed once and cached for the life of the process;
* every failure path raises :class:`EmbeddingUnavailable`, which callers treat
  as "use the keyword/rule fallback" rather than crashing a request.

Set ``SHOPASSIST_DISABLE_EMBEDDINGS=1`` to force the fallback path (used by
tests and useful in production when the model is not deployed).
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache

from shared.kb import article_ids, load_article

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DISABLE_ENV = "SHOPASSIST_DISABLE_EMBEDDINGS"

_TRUTHY = {"1", "true", "yes", "on"}


class EmbeddingUnavailable(RuntimeError):
    """The local embedding model is disabled, missing, or failed to load."""


_model = None
_model_lock = threading.Lock()


def _disabled() -> bool:
    return os.getenv(DISABLE_ENV, "").strip().lower() in _TRUTHY


def _load_model():
    """Import and construct the model. Kept separate so tests can stub it.

    We try the local cache first so normal runs never touch the network; only
    the very first run (empty cache) falls back to downloading.
    """
    from sentence_transformers import SentenceTransformer

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:  # cosmetic only; never block loading
        pass

    try:
        return SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
    except Exception:
        return SentenceTransformer(MODEL_NAME, device="cpu")


def get_model():
    """Return the shared model instance, loading it on first use."""
    global _model
    if _disabled():
        raise EmbeddingUnavailable(f"{DISABLE_ENV} is set")
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:  # another thread may have loaded it while we waited
            try:
                _model = _load_model()
                logger.info("loaded embedding model %s", MODEL_NAME)
            except Exception as exc:  # import error, download error, OOM, ...
                raise EmbeddingUnavailable(
                    f"could not load {MODEL_NAME}: {type(exc).__name__}: {exc}"
                ) from exc
    return _model


def is_available() -> bool:
    """True if the model can be used. Loads it once, then answers from memory."""
    try:
        get_model()
        return True
    except EmbeddingUnavailable:
        return False


def article_text(article_id: str) -> str:
    """Text we embed for an article: its id with dashes opened up, plus the body.

    The id is included because it carries policy meaning
    (``damage-replacement-7-days``) that helps short articles.
    """
    title = article_id.replace("-", " ")
    body = load_article(article_id).strip()
    return f"{title}. {body}"


@lru_cache(maxsize=1)
def article_vectors():
    """Return ``(ids, matrix)`` for every article; computed once per process.

    ``matrix`` has one row per article, already L2-normalised, so cosine
    similarity is just a dot product.
    """
    model = get_model()
    ids = tuple(article_ids())
    vectors = model.encode(
        [article_text(aid) for aid in ids],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return ids, vectors


def embed_query(text: str):
    """Embed one customer query. Empty/blank input is rejected explicitly."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("query text must be a non-empty string")
    model = get_model()
    vector = model.encode(
        [text.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    return vector


def semantic_scores(text: str) -> list[tuple[str, float]]:
    """Article ids with cosine similarity to ``text``, best first.

    Deterministic tie-breaking: equal scores fall back to article id order.
    """
    ids, matrix = article_vectors()
    query = embed_query(text)
    sims = matrix @ query
    pairs = [(aid, float(score)) for aid, score in zip(ids, sims)]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return pairs


def reset_caches() -> None:
    """Drop the model and cached vectors. Intended for tests."""
    global _model
    with _model_lock:
        _model = None
    article_vectors.cache_clear()
