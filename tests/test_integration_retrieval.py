from __future__ import annotations

import logging

import pytest

import question1.agent as q1_agent
import question2.agent as q2_agent
from eval.reference import TICKET_EXPECTATIONS
from question1.agent import SupportCopilot
from question2.agent import SupportPipeline
from shared import embeddings, semantic_retrieval

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]
ALL = PUBLIC + ["T-1091"]

# Locked-in post-optimization totals. These are the regression contract.
Q1_TOTAL_POINTS = 6_396
Q2_TOTAL_POINTS = 4_908


@pytest.fixture
def semantics_off(monkeypatch):
    monkeypatch.setenv(embeddings.DISABLE_ENV, "1")
    embeddings.reset_caches()
    yield
    embeddings.reset_caches()


def test_q1_calls_the_hybrid_retriever(monkeypatch):
    seen: list[str] = []

    def spy(text, order, **_):
        seen.append(text)
        return ["returns-within-30-days"]

    monkeypatch.setattr(q1_agent, "retrieve_articles", spy)
    result = SupportCopilot().handle_ticket("T-1001")
    assert seen and "wireless headphones" in seen[0].lower()
    assert result.kb_article == "returns-within-30-days"


def test_q2_triage_calls_the_hybrid_retriever(monkeypatch):
    seen: list[str] = []

    def spy(text, order, **_):
        seen.append(text)
        return ["returns-within-30-days"]

    monkeypatch.setattr(q2_agent, "retrieve_articles", spy)
    result = SupportPipeline().handle_ticket("T-1001")
    assert len(seen) == 1
    assert result.kb_article == "returns-within-30-days"


def test_q1_fallback_matches_expected_resolutions(semantics_off):
    copilot = SupportCopilot()
    for tid in ALL:
        got = copilot.handle_ticket(tid)
        expected = TICKET_EXPECTATIONS[tid]
        assert (got.action, got.kb_article) == (expected.action, expected.kb_article)


def test_q2_fallback_matches_expected_resolutions(semantics_off):
    pipe = SupportPipeline()
    for tid in ALL:
        got = pipe.handle_ticket(tid)
        expected = TICKET_EXPECTATIONS[tid]
        assert (got.action, got.kb_article) == (expected.action, expected.kb_article)


def test_q1_fallback_cost_is_unchanged(semantics_off):
    copilot = SupportCopilot()
    for tid in PUBLIC:
        copilot.handle_ticket(tid)
    assert copilot.session.total_points() == Q1_TOTAL_POINTS
    assert len(copilot.session.calls) == len(PUBLIC)


def test_q2_fallback_cost_is_unchanged(semantics_off):
    pipe = SupportPipeline()
    for tid in PUBLIC:
        pipe.handle_ticket(tid)
    assert pipe.session.total_points() == Q2_TOTAL_POINTS
    assert len(pipe.session.calls) == len(PUBLIC)  # still one call per ticket


def test_fallback_is_silent_about_missing_model(semantics_off, caplog, monkeypatch):
    monkeypatch.setattr(semantic_retrieval, "_warned", set())
    caplog.set_level(logging.WARNING, logger="shared.semantic_retrieval")
    SupportCopilot().handle_ticket("T-1001")
    SupportCopilot().handle_ticket("T-1001")
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1  # warned once, not once per request
    assert "unavailable" in warnings[0].getMessage().lower()
