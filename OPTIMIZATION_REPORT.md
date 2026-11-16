# ShopAssist — cost optimization report

Billing model (from `README.md`):

```
points = uncached_input × 3 + cached_input × 0.3 + output × 15
```

The simulator, grading rules and `data/` were not modified. Only what the agents
send to the model changed.

## Summary

| Track | Before (pts/ticket) | After (pts/ticket) | Change | Budget | Budget used |
|---|---:|---:|---:|---:|---:|
| Q1 — single call | 8,810 | **1,599** | **−81.8%** | 3,000 | 53% |
| Q2 — pipeline | 73,350 | **1,227** | **−98.3%** | 2,500 | 49% |
| **Total, 4 tickets + 4 tickets** | **328,638** | **11,304** | **−96.6%** | — | — |
| Wall clock, `question2/run.py` | 143 s | **0.15 s** | −99.9% | — | — |

Correctness: all four public tickets plus the holdout T-1091 resolve to the same
`action` / `kb_article` as the baseline. `pytest` → 6 passed.

> Before-numbers are the averages of a fresh run of the untouched baseline
> (`git archive HEAD`). They vary by ~±0.6% between runs because the baseline puts
> a random `uuid4()` in every prompt, which tokenizes differently each time.

## Question 1 — one call per ticket

### What changed

`question1/agent.py`

| Change | Before | After |
|---|---|---|
| `Run id {uuid4()}` at the top of the system prompt | present | removed |
| Instruction line | `"Think step by step. Write full reasoning, quote any article…"` | `"Reply with JSON only: action, kb_article, customer_reply."` |
| Help-center context | all 5 articles (195 tok) | `select_article_ids()` → only the articles that can apply |

`shared/kb.py` (new helpers, `all_articles_blob()` left intact)

| Helper | Purpose |
|---|---|
| `select_article_ids(text, order, limit=2)` | order facts first (subscription / license_revealed / customized+damage / damage), word-boundary keyword scoring as fallback |
| `articles_blob(ids)` | render only the selected articles |

### Per-ticket result

| Ticket | Before in/out | Before | After in/out | After | Saved |
|---|---|---:|---|---:|---:|
| T-1001 | 525 / 485 | 8,850 | 331 / 46 | **1,683** | −81.0% |
| T-1002 | 519 / 480 | 8,757 | 317 / 39 | **1,536** | −82.5% |
| T-1003 | 502 / 494 | 8,916 | 341 / 39 | **1,608** | −82.0% |
| T-1004 | 505 / 480 | 8,715 | 343 / 36 | **1,569** | −82.0% |
| **avg** | 513 / 485 | **8,810** | 333 / 40 | **1,599** | **−81.8%** |

### Per-call breakdown, T-1001 (submission requirement)

| | tokens | price | points |
|---|---:|---:|---:|
| Before — input | 525 | ×3 | 1,575 |
| Before — output | 485 | ×15 | 7,275 |
| **Before total** | | | **8,850** |
| After — input | 331 | ×3 | 993 |
| After — output | 46 | ×15 | 690 |
| **After total** | | | **1,683** |

Output was 82% of the bill. `shared/llm.py` scans the whole prompt for phrases
such as `"step by step"`, `"full reasoning"`, `"quote the"` and, on a hit, pads the
completion to ≥480 tokens. Dropping those phrases returns the completion to the
~40-token JSON. At 15 points/token that single line was worth ~6,600 points per
ticket.

## Question 2 — four-stage pipeline

### What changed

`question2/agent.py` keeps the four named stages but only *compose* calls the model:

| Stage | Before | After |
|---|---|---|
| triage | LLM call + `help_center[:120_000]` (8.4k tok in) | deterministic shortlist, 0 tok |
| retrieve | LLM call returning 8 articles | `articles_blob()` file read, 0 tok |
| compose | LLM call + thread + snapshot + triage + articles + triage again | one terse LLM call with the shortlisted article |
| audit | LLM call + `help_center[:80_000]` (6.7k tok in) | local validation of action + article id, 0 tok |

`SupportPipeline.__init__` no longer builds `expanded_help_center_blob()`, which
also removes the 143-second O(n²) token-counting loop from every run.

Why the model calls had to go: the shortest possible completion is the 46-token
JSON = 690 points, so four LLM calls cost ≥2,760 points before any input — over
the 2,500 budget on their own. Three calls cannot fit either; one call leaves
comfortable margin.

### Per-ticket result

| Ticket | Before | After | Saved |
|---|---:|---:|---:|
| T-1001 | 73,548 | **1,311** | −98.2% |
| T-1002 | 73,374 | **1,164** | −98.4% |
| T-1003 | 72,999 | **1,236** | −98.3% |
| T-1004 | 73,479 | **1,197** | −98.4% |
| **avg** | **73,350** | **1,227** | **−98.3%** |

### Node breakdown — T-1001 (physical fulfillment path)

| Stage | Before in/out | Before pts | After in/out | After pts |
|---|---|---:|---|---:|
| triage | 8,458 / 488 | 32,694 | 0 / 0 | 0 |
| retrieve | 596 / 46 | 2,478 | 0 / 0 | 0 |
| compose | 1,193 / 488 | 10,899 | 207 / 46 | 1,311 |
| audit | 6,719 / 488 | 27,477 | 0 / 0 | 0 |
| **total** | 16,966 / 1,510 | **73,548** | 207 / 46 | **1,311** |

### Node breakdown — T-1002 (denial path)

| Stage | Before in/out | Before pts | After in/out | After pts |
|---|---|---:|---|---:|
| triage | 8,448 / 495 | 32,769 | 0 / 0 | 0 |
| retrieve | 606 / 39 | 2,403 | 0 / 0 | 0 |
| compose | 1,194 / 480 | 10,782 | 193 / 39 | 1,164 |
| audit | 6,705 / 487 | 27,420 | 0 / 0 | 0 |
| **total** | 16,953 / 1,501 | **73,374** | 193 / 39 | **1,164** |

## Correctness evidence

```
$ pytest -q -k "not help_center"
5 passed, 1 deselected in 0.11s

$ pytest -q            # includes the 210k-token help-center assertion
6 passed in 155.43s

$ python question1/run.py    → 6,396 total_points
$ python question2/run.py    → 4,908 total_points  (0.15 s)
```

Resolutions are identical to the baseline for T-1001…T-1004, and the optimized
pipeline also resolves the holdout T-1091 correctly (`SEND_ARTICLE` /
`subscription-cooling-off`, 1,155 points).

## Files changed

| File | Change |
|---|---|
| `shared/kb.py` | + `select_article_ids()`, `articles_blob()`; existing helpers untouched |
| `question1/agent.py` | selector wired in, UUID and verbose trigger removed |
| `question2/agent.py` | four stages, single model call, 143 s startup removed |
| `tests/test_question1.py` | `test_baseline_cost_is_high` → `test_cost_within_budget` (≤3,000/ticket) |
| `tests/test_question2.py` | `test_baseline_cost_is_high` → `test_cost_within_budget` (≤2,500/ticket) |

The two cost tests asserted that the *baseline was expensive*, so they were
replaced with the assignment's actual targets. The correctness tests were not
weakened.

## Follow-ups (not done)

- `shared/kb.py::expanded_help_center_blob()` re-tokenizes the full blob each
  iteration (O(n²)). Counting only the appended chunk yields the identical
  210k-token string and would cut the one remaining slow test from ~143 s to ~1 s.
- `select_article_ids()` encodes policy routing in Python. That is the intended
  trade (deterministic routing instead of model calls), but it must be kept in
  sync when policy changes; the model still makes the final decision and still
  reads the full text of the cited article.
