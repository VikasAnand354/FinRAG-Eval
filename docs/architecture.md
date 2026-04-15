# Architecture

## Overview

FinRAG Eval is an offline benchmark runner. It takes a dataset of financial QA examples, runs them through a configurable RAG pipeline, scores the outputs, and writes versioned artifacts.

There is no server, no database, and no persistent service. Everything runs as a Python process driven by a config file.

## System Diagram

```
configs/smoke.yaml
       │
       ▼
  run_eval CLI
       │
       ├─── loads dataset (JSONL BenchmarkExamples)
       ├─── loads chunks   (JSONL Chunks)
       │
       ▼
  Retriever (BM25)
       │  top-k Chunks per question
       ▼
  GenerationAdapter (Mock | AzureOpenAI)
       │  answer + cited chunk IDs + latency + tokens
       ▼
  Evaluator Runner
       ├─── citation scorer   → precision, recall, exact match
       ├─── faithfulness judge → 0–1 score via LLM
       └─── latency scorer    → ms, tokens, USD cost
       │
       ▼
  Artifact Writer
       ├─── outputs/<run_id>/artifact.json
       ├─── outputs/<run_id>/scores.jsonl
       └─── outputs/<run_id>/config.json
       │
       ▼
  summarize_run CLI  →  Markdown summary table
```

## Module Map

```
src/
├── common/
│   ├── models.py       — all Pydantic schemas
│   ├── config.py       — YAML + env var loading
│   └── registry.py     — company metadata lookup
│
├── ingest/
│   ├── sec_fetch.py    — SEC EDGAR filing downloader
│   ├── text_extract.py — HTML → plaintext
│   ├── chunker.py      — paragraph splitting + token counting
│   ├── qa_generator.py — LLM-based QA candidate generation
│   └── dataset_builder.py — JSONL management + human review helpers
│
├── retrieval/
│   └── bm25.py         — BM25 keyword retriever
│
├── generation/
│   ├── base.py         — GenerationAdapter ABC
│   └── azure_openai.py — Azure OpenAI + Mock implementations
│
├── evaluators/
│   ├── runner.py       — orchestrates all scorers
│   ├── citation.py     — citation precision, recall, exact match
│   ├── faithfulness.py — LLM judge faithfulness scoring
│   └── latency.py      — latency and cost calculation
│
├── leaderboard/
│   ├── artifacts.py    — aggregate computation + file writing
│   └── summary.py      — Markdown summary table generation
│
└── cli/
    ├── run_eval.py     — main eval entry point
    └── summarize_run.py — artifact summarization
```

## Key Design Decisions

### Adapter pattern for retrieval and generation

Both the retriever and generation adapter are resolved by name from config strings at runtime. Adding a new retriever or LLM provider requires only a new class and a one-line registration — it does not touch evaluator logic or schemas.

### Separation of retrieval metrics and answer metrics

Citation and faithfulness metrics are in separate modules. Retrieval quality (did we retrieve the right chunks?) is scored independently of answer quality (did the model use them faithfully?). This separation is intentional and must be preserved.

### LLM-as-judge for faithfulness

Faithfulness is scored by calling a judge LLM with a versioned prompt (`FAITHFULNESS_PROMPT_V1`). The prompt version is saved in each artifact so results from different prompt versions are not mixed in the leaderboard.

### JSONL for all datasets and scores

JSONL (one JSON object per line) is used throughout. This keeps datasets streamable, appendable, and diffable in git. It avoids loading entire datasets into memory.

### Immutable run artifacts

Each run writes to `outputs/<run_id>/`. Run IDs are set in config and must be unique. Artifacts are never modified after writing. This is what makes leaderboard entries reproducible.

## Data Flow for Dataset Construction

Dataset construction is a separate pipeline driven by `scripts/build_dataset.py`:

```
Ticker list (config)
       │
       ▼
SEC EDGAR API  →  raw .htm filings
       │
       ▼
HTML extractor  →  plaintext
       │
       ▼
Paragraph chunker  →  chunks.jsonl
       │
       ▼
QA generator (LLM)  →  candidates.jsonl (pending review)
       │
       ▼
Human review CLI  →  qa_examples.jsonl (approved)
```

The approved `qa_examples.jsonl` and the `chunks.jsonl` are the inputs to the eval pipeline.

## Extension Points

| What to extend | Where to add code |
|---|---|
| New retriever | `src/retrieval/` — implement `retrieve(query, chunks, top_k)` |
| New LLM provider | `src/generation/` — subclass `GenerationAdapter` |
| New eval metric | `src/evaluators/` — add a new module, call it from `runner.py` |
| New dataset | `datasets/` — follow `BenchmarkExample` schema, no evaluator changes needed |
| New leaderboard format | `src/leaderboard/` — add a new writer alongside `artifacts.py` |
