# Real Financial QA Dataset — Design Spec

**Date:** 2026-04-04
**Scope:** Dataset construction pipeline — fetch SEC filings, chunk, generate Q&A candidates via LLM, human review loop, write to `datasets/full/`

---

## Problem

The MVP ships with 5 hand-written sample questions. That is enough to verify the pipeline runs end-to-end but not enough to produce meaningful benchmark results. A credible eval harness needs a dataset of ~50–100 grounded QA pairs drawn from real public filings.

## Goal

After this work, running `python scripts/build_dataset.py generate` followed by `python scripts/build_dataset.py review` produces a reviewed `datasets/full/qa_examples.jsonl` with ≥50 `BenchmarkExample` records grounded in real SEC 10-K and 10-Q excerpts, with correct gold citations pointing to real chunk IDs.

---

## Module Structure

```
src/ingest/
├── text_extract.py     NEW — strip HTML from raw SEC filing responses
├── qa_generator.py     NEW — LLM Q&A candidate generation per chunk
└── dataset_builder.py  NEW — orchestrates fetch → chunk → generate → write

scripts/
└── build_dataset.py    NEW — CLI: `generate` and `review` subcommands

configs/
└── dataset_build.yaml  NEW — tickers, fiscal years, sections, settings

datasets/full/          output directory (gitignored)
├── documents.jsonl
├── chunks.jsonl
└── qa_examples.jsonl

tests/
├── test_text_extract.py
├── test_qa_generator.py
├── test_dataset_builder.py
└── fixtures/
    └── sample_candidates.jsonl
```

`__init__.py` already present in `src/ingest/`. No new packages required.

---

## Data Contracts

### New schema: `QACandidate`

Intermediate record used during construction. Not persisted to the final dataset.

```python
class QACandidate(BaseModel):
    candidate_id: str          # "{chunk_id}-q{n}"
    chunk_id: str              # source chunk
    document_id: str
    company: str
    question: str
    gold_answer: str           # LLM-generated, human-reviewed
    difficulty: Literal["easy", "medium", "hard"]
    question_type: Literal["factual", "numerical", "comparative", "multi_hop"]
    gold_citations: list[str]  # chunk_ids that support the answer
    review_status: Literal["pending", "approved", "rejected", "edited"]
    reviewer_note: str = ""
```

### `dataset_build.yaml` format

```yaml
tickers: ["AAPL", "MSFT", "JPM"]
fiscal_years: [2023, 2024]
form_types: ["10-K", "10-Q"]
sections: ["Risk Factors", "MD&A", "Financial Statements"]
top_k_chunks_per_section: 5     # how many interesting chunks to sample per section
questions_per_chunk: 2
difficulty_mix: {easy: 0.5, medium: 0.3, hard: 0.2}
output_dir: datasets/full/
generation_adapter: azure_openai
```

### Output file relationships

| File | Schema | Notes |
|---|---|---|
| `documents.jsonl` | `DocumentMetadata` | Existing schema, no changes |
| `chunks.jsonl` | `Chunk` | Existing schema, no changes |
| `qa_examples.jsonl` | `BenchmarkExample` | Approved `QACandidate` records map 1:1 |

A `candidates.jsonl` scratch file persists in `output_dir` during construction to support interrupted/resumed review sessions. It is not the final dataset.

---

## Q&A Generation Logic

### Interesting chunk filtering

Before sending chunks to the LLM, filter to those worth asking about:
- Token count ≥ 80 (enough substance for a real question)
- Text contains at least one number, date, or percentage (financial signal)
- `section_title` is in the configured `sections` list

This avoids wasting LLM calls on boilerplate chunks (table of contents, signature pages, cover sheets).

### Generation prompt

One LLM call per chunk. Returns structured JSON array.

```
Given this excerpt from {company}'s {form_type} filing ({period}):

<chunk>
{text}
</chunk>

Generate {n} question-answer pairs. For each pair:
- The question must be answerable solely from the excerpt
- The answer must be a direct quote or close paraphrase from the excerpt
- Assign difficulty: easy (single fact lookup), medium (requires inference or
  calculation), hard (requires comparing multiple facts or multi-step reasoning)
- Assign type: factual | numerical | comparative | multi_hop

Return JSON array: [{"question": ..., "answer": ..., "difficulty": ..., "type": ...}]
```

The prompt is a versioned constant `QA_GENERATION_PROMPT_V1` in `qa_generator.py`, following the same pattern as the faithfulness prompt.

### Multi-hop questions

The generator only emits single-chunk candidates. Multi-hop pairs (difficulty=hard, type=multi_hop) with `gold_citations` pointing to 2+ chunks are constructed manually during the human review step by the reviewer editing the candidate and adding additional chunk IDs.

### Failure handling

If the LLM returns invalid JSON or an empty array, the chunk is skipped and the error is logged to stderr. No retry. Coverage gaps will be visible in the review phase (fewer candidates than expected).

---

## Human Review Loop

`scripts/build_dataset.py` exposes two subcommands, each independently resumable.

### Phase 1: Generate

```bash
python scripts/build_dataset.py generate --config configs/dataset_build.yaml
```

- Fetches filings via existing `sec_fetch.py`
- Extracts text via new `text_extract.py`
- Chunks via existing `chunker.py`
- Filters interesting chunks
- Calls LLM via `qa_generator.py`
- Appends new candidates to `candidates.jsonl`

Idempotent: skips already-fetched documents (checks `documents.jsonl` by `document_id`) and already-generated candidates (checks `candidates.jsonl` by `chunk_id`).

Progress output:
```
AAPL 10-K 2024 → 47 chunks → 12 interesting → 24 candidates
MSFT 10-K 2024 → 51 chunks → 14 interesting → 28 candidates
```

### Phase 2: Review

```bash
python scripts/build_dataset.py review --config configs/dataset_build.yaml
```

Interactive loop over all `pending` candidates. Each candidate displays:

```
[14/89 pending] AAPL 10-K 2024 | MD&A | easy | factual
CHUNK: "Net sales increased 2% to $391.0 billion..."
Q: What was Apple's total net sales for fiscal 2024?
A: $391.0 billion

(a)pprove  (e)dit  (r)eject  (s)kip  (q)uit
```

Actions:
- **approve** — writes to `qa_examples.jsonl` as `BenchmarkExample`, marks candidate approved
- **edit** — opens question/answer in `$EDITOR`, saves edited version, then approve
- **reject** — marks rejected with optional note, skips to next
- **skip** — leaves as pending, resumes in next session
- **quit** — saves state, exits cleanly

### State persistence

`review_status` in `candidates.jsonl` is updated in place after each decision. Re-running `review` shows only `pending` candidates. The `candidates.jsonl` file is the audit trail; it is never deleted.

---

## Testing

No live network or LLM calls required for any test.

| File | What it tests |
|---|---|
| `tests/test_text_extract.py` | HTML stripping: removes tags, preserves whitespace, handles empty input, handles malformed HTML |
| `tests/test_qa_generator.py` | Interesting chunk filter edge cases (short chunk, no numbers, wrong section); prompt construction; JSON parse success and failure fallback using `MockGenerationAdapter` |
| `tests/test_dataset_builder.py` | Candidate ID format; deduplication (skip already-generated chunk); `QACandidate` → `BenchmarkExample` mapping; round-trip serialization of `QACandidate` |

`tests/fixtures/sample_candidates.jsonl` — 5 `QACandidate` records (mix of pending/approved/rejected statuses) used as test fixture.

No test covers `scripts/build_dataset.py` directly — the interactive review loop is verified by manual use.

---

## Out of Scope

- Automated quality scoring of LLM-generated questions
- Crowd-sourcing or external annotation tools
- Automatic multi-hop pair construction
- Dataset versioning or provenance tracking beyond `candidates.jsonl`
- Any changes to existing `BenchmarkExample`, `Chunk`, or `DocumentMetadata` schemas
