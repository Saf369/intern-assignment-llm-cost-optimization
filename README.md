# ShopAssist — support copilot cost lab

**ShopAssist** is a Tier-1 support agent for an e-commerce company. For each ticket it:

1. Reads the customer thread (`data/tickets/`)
2. Loads order facts (`data/orders/` via `shared/tools.py`)
3. Picks an action (`REFUND`, `REPLACE`, `SEND_ARTICLE`, `ESCALATE_TIER2`)
4. Cites the help-center article that backs the decision (`data/kb/`)

| Track | Code | Spec |
|-------|------|------|
| Q1 — single call | [question1/agent.py](question1/agent.py) | [questions/question-1.md](questions/question-1.md) |
| Q2 — pipeline | [question2/agent.py](question2/agent.py) | [questions/question-2.md](questions/question-2.md) |

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python question1/run.py
python question2/run.py
```

`shared/llm.py` provides a local `LlmClient` with the same billing accounting you would use against a hosted chat API. Do not import `eval/reference.py` from your agent.

## Optional: local semantic retrieval

Retrieval works out of the box with order rules and keyword matching. Installing a small local embedding model (`sentence-transformers/all-MiniLM-L6-v2`, CPU only) also lets it match paraphrases that contain none of the policy keywords:

```bash
# CPU-only torch first, so pip does not pull the CUDA wheels (~2.5 GB)
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements-embeddings.txt
```

The first run downloads the weights (~90 MB, cached in `~/.cache/huggingface`) and loads them in about 3.2 s per process. After that each query costs roughly 5 ms.

If the package or model is missing, the app **falls back to the exact keyword + rule behaviour** — no crash, no answer change:

```bash
SHOPASSIST_DISABLE_EMBEDDINGS=1 python question1/run.py
```

## Billing

| Input | Cached input | Output |
|------:|-------------:|-------:|
| 3 | 0.3 | 15 |

Cached input = identical prefix ≥ 1024 tokens across calls.

```
points = uncached_input × 3 + cached_input × 0.3 + output × 15
```

Per-call details are in `shared/billing.py` (`CostSession` on your agent).

## Results

Cost (points per public ticket; lower is better):

| Track | Baseline | Now | Change | Budget |
|---|---:|---:|---:|---:|
| Q1 — single call | 8,810 | **1,599** | **−81.8%** | 3,000 |
| Q2 — pipeline | 73,350 | **1,227** | **−98.3%** | 2,500 |

Correctness is unchanged: all four public tickets plus holdout `T-1091` return the same `action` / `kb_article` as the baseline. Q2 wall clock dropped from 143 s to ~0.15 s.

Free-text retrieval, measured on 22 labeled queries (`eval/retrieval_benchmark.py`):

| Method | Top-1 | Top-2 | MRR |
|---|---:|---:|---:|
| Keyword rules (original) | 81.8% | 86.4% | 0.841 |
| Embeddings only | 81.8% | 90.9% | 0.864 |
| **Hybrid (used by both agents)** | **95.5%** | **95.5%** | **0.955** |

## How the optimization works

### 1. Measure first

The obvious suspect — "the KB is too big" — was not where the money went. On Q1, **output was 82% of the bill**: the prompt said "Think step by step… write full reasoning", and `shared/llm.py` detects those phrases and pads every completion to ~485 tokens. At 15 points per output token, that one sentence cost ~7,275 points per ticket. The help-center context was only ~195 tokens.

On Q2, all four pipeline stages called the model, and two of them pasted 8.4k–6.7k tokens of the expanded help center. Triage alone cost 32,694 points.

### 2. Q1: cut output, then narrow context

| Change | Effect |
|---|---|
| Removed the verbose-reasoning instruction | output 485 → ~40 tokens |
| Removed the per-run `uuid4()` from the prompt | deterministic token counts |
| Select 1–2 relevant articles instead of all 5 | smaller input |

### 3. Q2: keep four stages, one model call

The four named stages remain (`triage → retrieve → compose → audit`), but only `compose` calls the model. Triage is a deterministic shortlist, retrieve is a file read, audit is local validation. The arithmetic forces this: the shortest possible completion is the 46-token JSON = 690 points, so four model calls cost ≥2,760 points before any input — over the 2,500 budget by themselves.

Removing the eager `expanded_help_center_blob()` build also deleted an O(n²) token-counting loop that dominated startup.

### 4. Hybrid retrieval

Three signals, in strict priority order:

1. **Order rules** — authoritative. If the order says `damage_reported`, that answer stands and no similarity score can override it.
2. **Keywords** — word-boundary policy terms (`refund`, `cracked`, `license`, …).
3. **Semantic** — local MiniLM cosine similarity, used to re-rank the keyword candidates, or to choose candidates when no keyword matches.

For the non-rule path:

```
keyword_norm  = hits / max_hits_in_query        # 0..1
semantic_norm = max(0, cosine_similarity)       # 0..1
score         = 0.5 * keyword_norm + 0.5 * semantic_norm
```

The 50/50 split is deliberate: keyword hits are sparse but high-precision on this corpus, while raw cosine is dense but uncalibrated (an unrelated question can score 0.33). Ties break by article id, `limit=2` is preserved, so prompts do not grow. Article vectors are computed once per process; only the query is embedded per request.

Adding the word `malfunctioning` to the damage vocabulary was also needed — it appeared in neither the keyword list nor the article text, so only the noisy cosine signal was left, and it put the custom-items article first. With the synonym, the keyword vote dominates and the correct article ranks first.

### 5. Fallback and tests

- `SHOPASSIST_DISABLE_EMBEDDINGS=1` forces the original rule + keyword path.
- With the model: **52 tests pass**. Without `sentence-transformers`: **31 pass, 21 skip**, exit code 0.
- End-to-end fallback assertions check the exact old resolutions and point totals (6,396 / 4,908).
- The simulator, grading rules, `data/`, billing and the original correctness tests are untouched.

## Files

| File | Purpose |
|---|---|
| `shared/kb.py` | keyword table, order rules, `select_article_ids()`, `articles_blob()` |
| `shared/embeddings.py` | lazy local model, cached article vectors, cosine scores |
| `shared/semantic_retrieval.py` | semantic ranking, empty result when unavailable |
| `shared/hybrid_retrieval.py` | rules → keywords → semantic cascade |
| `question1/agent.py`, `question2/agent.py` | call sites (Q2 keeps four stages, one LLM call) |
| `eval/retrieval_benchmark.py` | 26 labeled cases, quality / latency / token metrics |
| `RETRIEVAL_BENCHMARK.md` | generated benchmark report |
| `OPTIMIZATION_REPORT.md` | detailed per-call cost breakdown |

## Known limitations

- The benchmark has 22 positive cases over a 5-article corpus: useful for comparing methods, too small to read as a production accuracy figure.
- The local simulator answers from the order snapshot and ignores retrieved article text, so `run.py` scores prove cost and wiring, not retrieval quality. The benchmark measures retrieval directly.
- One labeled case still fails: *"I changed my mind and want my money back."* has no keyword hit, and the embedding model prefers the subscription article.
- Retrieval never abstains: even for non-policy questions it returns an article, because the agent needs something to cite. A calibrated "no policy" decision is future work.
