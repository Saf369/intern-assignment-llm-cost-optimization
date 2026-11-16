from __future__ import annotations

import json
from dataclasses import dataclass

from shared.billing import CostSession
from shared.hybrid_retrieval import retrieve as retrieve_articles
from shared.kb import article_ids, articles_blob
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import parse_resolution

VALID_ACTIONS = frozenset({"REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2"})


@dataclass
class TicketResolution:
    action: str
    kb_article: str


class SupportPipeline:
    """Same product as Question 1 — triage, retrieve, compose, audit.

    The four stages are kept, but only *compose* is worth a model call:
    triage is a deterministic shortlist over order facts, keywords and local
    embeddings, retrieve is a file read, and audit validates the answer locally.
    Sending 120k characters of help center into every node cost 69% of the
    baseline bill and changed no decision.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self._known_articles = frozenset(article_ids())

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, indent=2)

        candidates = self._triage(thread, order)
        retrieved = self._retrieve(candidates)
        draft = self._compose(thread, snapshot, retrieved)
        parsed = parse_resolution(draft)
        self._audit(parsed)
        return TicketResolution(action=parsed.action, kb_article=parsed.kb_article)

    def _triage(self, thread: str, order: dict) -> list[str]:
        """Node 1 — deterministic shortlist, no tokens spent."""
        return retrieve_articles(thread, order)

    def _retrieve(self, candidates: list[str]) -> str:
        """Node 2 — load only the shortlisted policy text."""
        return articles_blob(candidates)

    def _compose(self, thread: str, snapshot: str, retrieved: str) -> str:
        """Node 3 — the single model call, with the policy already narrowed."""
        messages = [
            ChatMessage(
                "system",
                "You are ShopAssist Tier-1 support. Reply with JSON only: "
                '{"action","kb_article","customer_reply"}.\n'
                f"POLICY:\n{retrieved}",
            ),
            ChatMessage(
                "user",
                f"THREAD:\n{thread}\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ),
        ]
        return self.llm.chat("compose", messages)

    def _audit(self, parsed) -> None:
        """Node 4 — local guardrail; raises instead of shipping a bad citation."""
        if parsed.action not in VALID_ACTIONS:
            raise ValueError(f"Audit failed: unknown action {parsed.action!r}")
        if parsed.kb_article not in self._known_articles:
            raise ValueError(f"Audit failed: unknown article {parsed.kb_article!r}")
