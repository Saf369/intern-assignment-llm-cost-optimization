"""Retrieval benchmark: legacy keyword selection vs hybrid retrieval.

Run:
    python eval/retrieval_benchmark.py
    python eval/retrieval_benchmark.py --markdown RETRIEVAL_BENCHMARK.md

Labels were written before running the benchmark; the 0.5/0.5 weights were
fixed in Phase 4. Nothing here tunes on these cases.

The corpus has five articles, so these numbers are small-sample and optimistic.
That is stated in the report rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.reference import TICKET_EXPECTATIONS  # noqa: E402
from shared import hybrid_retrieval as hybrid  # noqa: E402
from shared import semantic_retrieval  # noqa: E402
from shared.kb import articles_blob, select_article_ids  # noqa: E402
from shared.tokens import count_tokens  # noqa: E402
from shared.tools import format_thread, get_order, get_ticket  # noqa: E402

LIMIT = 2


@dataclass(frozen=True)
class Case:
    text: str
    expected: str | None  # None = no policy should apply (negative)
    kind: str
    order: dict = field(default_factory=dict)


def _ticket_case(ticket_id: str) -> Case:
    ticket = get_ticket(ticket_id)
    order = get_order(ticket["order_id"])
    return Case(format_thread(ticket), TICKET_EXPECTATIONS[ticket_id].kb_article, "ticketed", order)


CASES: list[Case] = [
    # --- direct keyword matches -------------------------------------------
    Case("I want a refund for the wireless headphones I haven't opened.",
         "returns-within-30-days", "exact keyword"),
    Case("Can I return an item I bought last week?", "returns-within-30-days", "exact keyword"),
    Case("The mug I ordered was cracked on arrival.", "damage-replacement-7-days", "exact keyword"),
    Case("The screen is damaged.", "damage-replacement-7-days", "exact keyword"),
    Case("I already revealed the license key, can I still get my money back?",
         "digital-license-no-refund", "exact keyword"),
    Case("I had my initials engraved on the item, can I exchange it?",
         "custom-items-no-replace", "exact keyword"),
    Case("Please cancel my annual subscription.", "subscription-cooling-off", "exact keyword"),
    # --- paraphrases / synonyms -------------------------------------------
    Case("My package arrived broken.", "damage-replacement-7-days", "paraphrase"),
    Case("The product is defective.", "damage-replacement-7-days", "paraphrase"),
    Case("The item is malfunctioning.", "damage-replacement-7-days", "paraphrase"),
    Case("My device stopped functioning after delivery.",
         "damage-replacement-7-days", "paraphrase"),
    Case("I changed my mind and want my money back.",
         "returns-within-30-days", "paraphrase"),
    Case("I want a refund for my digital purchase.", "digital-license-no-refund", "paraphrase"),
    Case("The ebook download is useless to me, can I get my money back?",
         "digital-license-no-refund", "paraphrase"),
    Case("It has my monogram on it, can I swap it for another one?",
         "custom-items-no-replace", "paraphrase"),
    Case("I renewed my yearly plan and want my money back.",
         "subscription-cooling-off", "paraphrase"),
    Case("My plan renewed automatically and I want to cancel it.",
         "subscription-cooling-off", "paraphrase"),
    # --- ticketed (rules must win) ----------------------------------------
    _ticket_case("T-1001"),
    _ticket_case("T-1002"),
    _ticket_case("T-1003"),
    _ticket_case("T-1004"),
    _ticket_case("T-1091"),
    # --- negatives: no policy should apply --------------------------------
    Case("What time does the store open?", None, "negative"),
    Case("Do you ship to Canada?", None, "negative"),
    Case("I forgot my password.", None, "negative"),
    Case("Where is my invoice?", None, "negative"),
]


def _ids(method: str, case: Case) -> list[str]:
    if method == "legacy":
        return select_article_ids(case.text, case.order, LIMIT)
    if method == "hybrid":
        return hybrid.retrieve(case.text, case.order, limit=LIMIT)
    if method == "semantic":
        return semantic_retrieval.article_ids(case.text, limit=LIMIT)
    raise ValueError(method)


def _metrics(method: str) -> dict:
    positives = [c for c in CASES if c.expected is not None]
    hit1 = hit2 = 0
    reciprocal = 0.0
    misses: list[tuple[str, str, list[str]]] = []
    for case in positives:
        ids = _ids(method, case)
        rank = ids.index(case.expected) + 1 if case.expected in ids else None
        hit1 += rank == 1
        hit2 += rank is not None and rank <= 2
        reciprocal += 1.0 / rank if rank else 0.0
        if rank != 1:
            misses.append((case.text, case.expected, ids))
    n = len(positives)
    return {
        "n": n,
        "hit@1": hit1 / n,
        "p@1": hit1 / n,
        "hit@2": hit2 / n,
        "p@2": (hit2 / n) / 2,
        "r@2": hit2 / n,
        "mrr": reciprocal / n,
        "misses": misses,
    }


def _latency(method: str) -> dict:
    # Warm the model once; measured calls then exclude load time.
    for case in CASES:
        _ids(method, case)
    samples: list[float] = []
    for case in CASES:
        timings = []
        for _ in range(3):  # 3 passes of 5 reps smooths CPU scheduling noise
            for _ in range(5):
                start = time.perf_counter()
                _ids(method, case)
                timings.append((time.perf_counter() - start) * 1000)
        samples.append(statistics.median(timings))
    return {
        "mean_ms": statistics.fmean(samples),
        "p95_ms": sorted(samples)[max(0, int(len(samples) * 0.95) - 1)],
    }


def _tokens(method: str) -> dict:
    counts = [len(_ids(method, case)) for case in CASES]
    tokens = [count_tokens(articles_blob(_ids(method, case))) for case in CASES]
    return {
        "mean_articles": statistics.fmean(counts),
        "mean_tokens": statistics.fmean(tokens),
        "total_tokens": sum(tokens),
    }


def _negatives(method: str) -> dict:
    rows = [c for c in CASES if c.expected is None]
    top = [_ids(method, c)[0] for c in rows]
    return {"returned_top1": dict((aid, top.count(aid)) for aid in sorted(set(top)))}


def main() -> None:
    methods = ["legacy", "semantic", "hybrid"]
    report = {}
    for method in methods:
        report[method] = {
            "quality": _metrics(method),
            "latency": _latency(method),
            "cost": _tokens(method),
            "negatives": _negatives(method),
        }

    lines: list[str] = []
    lines.append("# Retrieval benchmark — legacy keyword vs hybrid\n")
    lines.append(f"Cases: {len(CASES)} ({len([c for c in CASES if c.expected])} positives, "
                 f"{len([c for c in CASES if not c.expected])} negatives), limit={LIMIT}.\n")
    lines.append("| Method | P@1 | P@2 | R@2 | MRR | Hit@1 | Hit@2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method in methods:
        q = report[method]["quality"]
        lines.append(
            f"| {method} | {q['p@1']:.3f} | {q['p@2']:.3f} | {q['r@2']:.3f} | "
            f"{q['mrr']:.3f} | {q['hit@1']:.3f} | {q['hit@2']:.3f} |"
        )
    lines.append("")
    lines.append("| Method | mean latency | p95 latency | mean articles | mean tokens | total tokens |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method in methods:
        lat, cost = report[method]["latency"], report[method]["cost"]
        lines.append(
            f"| {method} | {lat['mean_ms']:.2f} ms | {lat['p95_ms']:.2f} ms | "
            f"{cost['mean_articles']:.2f} | {cost['mean_tokens']:.1f} | {cost['total_tokens']} |"
        )
    lines.append("")
    lines.append("## Negatives (no policy applies) — top-1 returned\n")
    for method in methods:
        lines.append(f"- **{method}**: {report[method]['negatives']['returned_top1']}")
    lines.append("")
    lines.append("## Top-1 misses\n")
    for method in methods:
        misses = report[method]["quality"]["misses"]
        lines.append(f"### {method} ({len(misses)})\n")
        for text, expected, got in misses:
            lines.append(f"- `{text}` → expected `{expected}`, got `{got}`")
        lines.append("")

    markdown = "\n".join(lines)
    print(markdown)
    print("\n## JSON\n")
    print(json.dumps(report, indent=2, default=str))

    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.markdown:
        args.markdown.write_text(markdown + "\n")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
