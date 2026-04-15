# Eval Methodology

## Philosophy

The goal of FinRAG Eval is to measure whether a RAG pipeline is **useful and trustworthy**, not just whether it produces answers that look plausible.

Three failure modes matter most in financial QA:

1. **Hallucination** — the model asserts something not in the retrieved evidence
2. **Wrong citation** — the answer cites sources that do not support the claim
3. **Missing evidence** — the retriever fails to surface the relevant chunk

Each is measured separately so a system cannot mask one failure with strength in another.

## Metrics

### Citation Precision

**What it measures:** Of the chunks the pipeline cited, what fraction actually supported the gold answer?

**Formula:**

```
citation_precision = |predicted_chunks ∩ gold_chunks| / |predicted_chunks|
```

Returns `0.0` if the pipeline cited no chunks.

**Interpretation:** Low precision means the pipeline is citing irrelevant or wrong sources.

---

### Citation Recall

**What it measures:** Of the gold chunks needed to answer the question, what fraction did the pipeline cite?

**Formula:**

```
citation_recall = |predicted_chunks ∩ gold_chunks| / |gold_chunks|
```

Returns `1.0` if the example has no gold citations (question needs no citation).

**Interpretation:** Low recall means the pipeline is missing key evidence.

---

### Exact Match

**What it measures:** Whether the gold answer string appears in the generated answer.

**Method:** Case-insensitive substring check. If `acceptable_aliases` are defined on the example, any alias match also counts.

**Use case:** Primarily useful for numerical and factual questions where there is one correct answer (e.g. revenue figures, dates, percentages).

**Limitation:** Exact match is a weak signal for open-ended or comparative questions. Use faithfulness score for those.

---

### Faithfulness Score

**What it measures:** Whether every claim in the generated answer is supported by the retrieved context.

**Method:** LLM-as-judge. The judge model receives the retrieved chunks and the generated answer, then returns a score from 0.0 to 1.0.

**Prompt version:** `faithfulness-v1` (see `src/evaluators/faithfulness.py`)

**Threshold:** A score ≥ 0.5 is counted as `faithful = True`.

**Interpretation:**
- `1.0` — every claim is directly supported by the retrieved context
- `0.0` — the answer makes claims not found in the context at all
- `None` — judge call failed; treated as `faithful = False` in aggregates

**Important:** The judge model and prompt version are saved in every run artifact. Do not compare faithfulness scores across runs that used different judge models or prompt versions.

---

### Latency

**What it measures:** Wall-clock time from start of generation call to response received.

**Unit:** milliseconds

**Captured:** In `PipelineResult.latency_ms` by the generation adapter. Does not include retrieval time in the current implementation.

---

### Cost

**What it measures:** Estimated USD cost of the generation call based on token usage.

**Formula:**

```
cost_usd = (prompt_tokens / 1000 * 0.01) + (completion_tokens / 1000 * 0.03)
```

**Note:** These rates approximate GPT-4 pricing and are a reference point, not exact billing. Actual costs depend on your Azure deployment and pricing tier.

**Captured fields:** `prompt_tokens`, `completion_tokens`, `cost_usd` per question; `total_cost_usd` in run aggregate.

---

## Aggregate Metrics

Each run produces the following aggregate metrics over all scored examples:

| Field | Description |
|---|---|
| `n_examples` | Total number of examples evaluated |
| `exact_match_rate` | Fraction of examples with exact match |
| `mean_faithfulness_score` | Mean faithfulness score (excludes None) |
| `pct_faithful` | Fraction of examples where `faithful = True` |
| `mean_citation_precision` | Mean citation precision across all examples |
| `mean_citation_recall` | Mean citation recall across all examples |
| `median_latency_ms` | Median generation latency in milliseconds |
| `total_cost_usd` | Sum of estimated USD cost for the full run |

## Scoring Independence

Retrieval metrics (citation precision, citation recall) and answer metrics (faithfulness, exact match) are computed in separate modules:

- `src/evaluators/citation.py` — retrieval-side metrics
- `src/evaluators/faithfulness.py` — answer-side metrics

This separation is intentional. A system can have high faithfulness but low citation recall (it gave a correct answer but retrieved the wrong chunks), or high citation recall but low faithfulness (it retrieved the right chunks but made up an answer anyway). Both failures matter and should be visible independently.

## What Is Not Measured (Yet)

- **Retrieval ranking quality** (MRR, NDCG) — planned
- **Multi-hop reasoning correctness** — flagged via `requires_multi_hop` field but not separately scored
- **Numerical accuracy** — `gold_answer_normalized` and `normalization_unit` fields exist for future use
- **Human preference** — out of scope for the offline benchmark runner

## Prompt Versioning

All prompts used in scoring are versioned with a string constant (e.g. `FAITHFULNESS_PROMPT_VERSION = "faithfulness-v1"`).

If a prompt is changed, the version string must be incremented. Leaderboard entries from different prompt versions must not be directly compared.
