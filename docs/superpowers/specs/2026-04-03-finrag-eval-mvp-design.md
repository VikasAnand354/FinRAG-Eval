# FinRAG Eval — MVP Design Spec

**Date:** 2026-04-03
**Scope:** Full MVP — fix project structure and implement all 8 milestones from CLAUDE.md

---

## Problem

The repo has 5 files in the wrong directory layout. `pyproject.toml` expects `src.*` packages but files sit at root-level `cli/`, `common/`, `ingest/`. None of the core modules exist: no retrieval, no generation, no evaluators, no leaderboard, no tests, no fixtures, no configs. Nothing runs end-to-end.

## Goal

After this work, `python -m src.cli.run_eval --config configs/smoke.yaml` executes a full benchmark run on a small sample dataset and produces a versioned JSON artifact with per-question scores, aggregate metrics, and a summary table.

---

## Project Structure

```
src/
├── common/
│   ├── models.py          # existing — move from root common/
│   ├── registry.py        # existing — move from root common/
│   └── config.py          # NEW: load .env + parse YAML RunConfig
├── ingest/
│   ├── sec_fetch.py       # existing — move from root ingest/
│   └── chunker.py         # existing — move from root ingest/
├── retrieval/
│   └── bm25.py            # NEW: BM25 baseline retriever
├── generation/
│   ├── base.py            # NEW: abstract GenerationAdapter
│   └── azure_openai.py    # NEW: Azure OpenAI adapter + MockGenerationAdapter
├── evaluators/
│   ├── latency.py         # NEW: latency + cost scoring (pure functions)
│   ├── citation.py        # NEW: citation precision/recall + exact match (pure functions)
│   ├── faithfulness.py    # NEW: LLM-based faithfulness judge
│   └── runner.py          # NEW: orchestrates all three, returns ScoredRow
├── leaderboard/
│   ├── artifacts.py       # NEW: write + validate versioned RunArtifact JSON
│   └── summary.py         # NEW: aggregate Markdown table
└── cli/
    ├── run_eval.py         # REWRITE: Typer app, --config flag, sequential run loop
    └── summarize_run.py    # NEW: Typer app, --run-dir flag, reprints summary

configs/
├── smoke.yaml             # NEW: sample fixtures, top_k=3
└── full.yaml              # NEW: same shape, full dataset placeholder

datasets/
└── sample/
    ├── documents.jsonl    # NEW: 3-5 hand-written 10-K excerpts
    ├── chunks.jsonl       # NEW: pre-chunked from documents.jsonl
    └── qa_examples.jsonl  # NEW: 5-10 benchmark QA pairs

tests/
├── test_models.py
├── test_chunker.py
├── test_bm25.py
├── test_citation.py
├── test_faithfulness.py
└── fixtures/
    └── golden_run.json

.env.example               # NEW: documents Azure env vars
```

`__init__.py` added to every `src/` package. `pyproject.toml` unchanged.

---

## Data Contracts

### Existing schemas (keep as-is)
- `DocumentMetadata` — document-level metadata
- `Chunk` — chunked text unit with token count
- `BenchmarkExample` — QA pair with gold answer and gold citations
- `GoldCitation` — `document_id`, `chunk_id`, `support_type`

### New schemas (add to `common/models.py`)

**`RunConfig`** — parsed from YAML
```
run_id: str
dataset_path: str
chunks_path: str
retriever: str              # "bm25"
top_k: int
generation_adapter: str     # "azure_openai" | "mock"
output_dir: str
```

**`PipelineResult`** — what the RAG pipeline returns per question
```
example_id: str
answer: str
citations: list[str]        # chunk_ids returned by pipeline
latency_ms: float
prompt_tokens: int
completion_tokens: int
```

**`ScoredRow`** — one row per question after evaluation
```
example_id: str
answer: str
citations: list[str]
latency_ms: float
prompt_tokens: int
completion_tokens: int
cost_usd: float
exact_match: bool
citation_precision: float
citation_recall: float
faithful: bool
faithfulness_score: float | None
```

**`RunArtifact`** — top-level leaderboard JSON
```
run_id: str
timestamp: datetime
config_snapshot: dict       # full YAML config + prompt versions
scores: list[ScoredRow]
aggregate: dict             # means/medians of all numeric fields
```

---

## Pipeline Adapter Interface

`src/generation/base.py` defines one abstract class:

```python
class GenerationAdapter(ABC):
    def generate(self, question: str, chunks: list[Chunk]) -> PipelineResult:
        ...
```

External RAG systems implement this to plug into the harness without touching evaluator logic.

`MockGenerationAdapter` in `src/generation/azure_openai.py` returns canned responses — used in all tests, no network required.

---

## Retrieval

**BM25** (`src/retrieval/bm25.py`)
- Dependency: `rank_bm25`
- Loads pre-chunked JSONL at startup, builds in-memory index
- `retrieve(query: str, top_k: int) -> list[Chunk]`
- Index rebuilt per run — no persistence needed at benchmark scale

---

## Generation

**Azure OpenAI adapter** (`src/generation/azure_openai.py`)
- Reads `AZURE_OPENAI_*` env vars via `src/common/config.py` (loaded once)
- System prompt + question + retrieved chunks → answer text
- Returns `PipelineResult` with answer, latency_ms, token counts
- Prompt string is a versioned constant in the module, saved into `config_snapshot`
- Graceful failure: logs error, returns answer="" with zero tokens

---

## Evaluators

All evaluators are pure functions except faithfulness. None share state.

**`src/evaluators/latency.py`**
- `score_latency(result: PipelineResult, price_per_1k_tokens: float) -> dict`
- Returns `latency_ms`, `prompt_tokens`, `completion_tokens`, `cost_usd`
- Price constants defined at top of file

**`src/evaluators/citation.py`**
- `score_citations(result: PipelineResult, example: BenchmarkExample) -> dict`
- Returns `citation_precision`, `citation_recall`, `exact_match`
- Exact match: normalized `gold_answer` substring present in answer
- No model calls

**`src/evaluators/faithfulness.py`**
- `score_faithfulness(result: PipelineResult, chunks: list[Chunk], adapter: GenerationAdapter) -> dict`
- Prompt: given chunks + answer, are all claims grounded? Returns 0–1 score
- `faithful = score >= 0.5`
- Prompt versioned as constant, included in `config_snapshot`
- Returns `faithfulness_score=None` on model failure

**`src/evaluators/runner.py`**
- `evaluate(result: PipelineResult, example: BenchmarkExample, chunks: list[Chunk], judge: GenerationAdapter, config: RunConfig) -> ScoredRow`
- Calls all three scorers, merges into `ScoredRow`

---

## Run Loop

`src/cli/run_eval.py`:
1. Load `RunConfig` from YAML
2. Load `BenchmarkExample` list from `dataset_path`
3. Load `Chunk` list from `chunks_path`, build BM25 index
4. Init `GenerationAdapter` from config
5. For each example: `retrieve → generate → evaluate` → collect `ScoredRow`
6. Compute aggregates
7. Write `RunArtifact` via `artifacts.write_artifact()`
8. Print summary table via `summary.generate_summary()`

Sequential, no parallelism.

---

## Leaderboard & Reporting

**`src/leaderboard/artifacts.py`**
- `write_artifact(run: RunArtifact, output_dir: Path)` — writes:
  - `outputs/<run_id>/artifact.json` — full `RunArtifact`
  - `outputs/<run_id>/scores.jsonl` — one `ScoredRow` per line
  - `outputs/<run_id>/config.json` — config snapshot
- `validate_artifact(path: Path)` — re-validates against `RunArtifact` schema

**`src/leaderboard/summary.py`**
- `generate_summary(artifact: RunArtifact) -> str` — Markdown table with:
  - mean faithfulness score, % faithful
  - citation precision, citation recall
  - exact match rate
  - median latency_ms
  - total cost_usd

---

## CLI

```
finrag-run --config configs/smoke.yaml
finrag-summarize --run-dir outputs/smoke-001
```

Both use Typer. `run_eval` prints per-question progress with `rich`. Both registered in `pyproject.toml` `[project.scripts]` (already present).

---

## Configs

**`configs/smoke.yaml`**
```yaml
run_id: smoke-001
dataset_path: datasets/sample/qa_examples.jsonl
chunks_path: datasets/sample/chunks.jsonl
retriever: bm25
top_k: 3
generation_adapter: azure_openai
output_dir: outputs/
```

**`configs/full.yaml`** — same shape, `dataset_path` points to full dataset (placeholder).

---

## Sample Fixtures

`datasets/sample/documents.jsonl` — 3 short excerpts from public 10-K filings (Apple, Microsoft, one other). Each: `document_id`, `company`, `text`, `filing_date`.

`datasets/sample/chunks.jsonl` — pre-chunked output from `chunker.chunk_document()` over the sample docs. Matches `Chunk` schema exactly.

`datasets/sample/qa_examples.jsonl` — 5 questions over sample docs. `gold_citations` reference real `chunk_id` values from `chunks.jsonl`.

---

## Tests

| File | What it tests |
|---|---|
| `test_models.py` | Round-trip JSON serialization of all schemas |
| `test_chunker.py` | Paragraph split edge cases, token count, chunk_id format |
| `test_bm25.py` | Top-k retrieval, empty corpus, query with no match |
| `test_citation.py` | Precision/recall: full match, partial, zero, extra citations |
| `test_faithfulness.py` | Scoring logic with `MockGenerationAdapter`, failure fallback |
| `fixtures/golden_run.json` | Saved `RunArtifact` for regression testing artifact shape |

All tests use mocks for model calls. No live network required for `pytest -q`.

---

## Environment

`.env.example`:
```
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

---

## pyproject.toml Changes Required

Add to `dependencies`:
- `rank-bm25>=0.2` — BM25 retriever
- `tiktoken>=0.7` — token counting in chunker (already used, not listed)
- `requests>=2.31` — SEC EDGAR fetcher (already used, not listed)

---

## Out of Scope

- Multi-tenant web app
- Distributed workers
- Human annotation UI
- Vector database service
- LangChain / LlamaIndex as core dependencies
- Parallel eval workers
