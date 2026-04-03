# FinRAG Eval

Open source benchmark and evaluation harness for financial RAG systems.

This repository exists to measure whether a financial RAG pipeline is useful and trustworthy, not just whether it can answer questions that look plausible.

Primary goals:
- measure grounded answer quality and hallucination rate
- measure citation accuracy and evidence coverage
- measure retrieval quality separately from answer quality
- record latency and cost per run
- publish reproducible leaderboard artifacts

## Product boundary

This repo is **not** a production finance assistant.
It is an eval framework plus reference baselines.

Keep the scope small:
- offline benchmark runner
- configurable pipeline adapters
- deterministic artifact generation
- static leaderboard outputs

Do not add until the MVP is stable:
- multi tenant web app
- complex orchestration frameworks
- distributed workers
- human annotation UI
- bespoke vector database service

## Solo builder constraints

Optimize for a single engineer.
Prefer simple Python modules and clear data contracts over clever abstractions.
Choose components that are cheap to run locally and easy to swap later.

Use Azure OpenAI where model based judgment is needed, but keep all providers behind adapters so the framework stays open source and portable.

## What success looks like

A good run should produce:
- per question scoring rows
- aggregate metric summary
- error analysis slices
- saved prompts and model settings
- latency and token usage
- versioned JSON artifacts suitable for a public leaderboard

A good repo should let a contributor:
- add a new dataset without changing evaluator logic
- add a new RAG pipeline adapter without changing core schemas
- reproduce a leaderboard run from a config file

## Architecture map

Read these only when needed:
- high level system design: @docs/architecture.md
- metric definitions and scoring philosophy: @docs/eval-methodology.md
- dataset format and contracts: @docs/dataset-schema.md
- leaderboard rules and artifact schema: @docs/leaderboard.md
- project overview and setup steps: @README.md
- dependencies and scripts: @pyproject.toml

## Core modules

- `src/ingest/`
  loaders, document chunking helpers, fixture prep

- `src/retrieval/`
  retrieval adapters and reference baselines

- `src/generation/`
  LLM provider wrappers and answer generation adapters

- `src/evaluators/`
  retrieval metrics, faithfulness checks, citation checks, latency and cost scoring

- `src/leaderboard/`
  artifact validation, summary tables, static leaderboard generation

- `src/common/`
  schemas, config parsing, logging, tracing helpers

- `src/cli/`
  entrypoints for running evals and summarizing results

## Build order

Build in this order and stop after each milestone works:

1. dataset schemas and sample fixtures
2. pipeline adapter interface
3. baseline retrieval plus generation runner
4. latency and cost capture
5. citation span checking
6. faithfulness judge
7. aggregate report generation
8. leaderboard artifact validator

Do not start with UI.
Do not add more than one baseline pipeline before the first end to end run works.

## Technical choices

Default stack:
- Python 3.11+
- Pydantic for schemas
- Typer for CLI
- Pandas or Polars for reporting
- Pytest for tests
- Ruff for lint and format

Keep framework dependencies minimal.
Only add LangChain or LlamaIndex adapters if they help test external pipelines. The core framework should not depend on them.

## Azure OpenAI guidance

Use Azure OpenAI for:
- judge style faithfulness checks
- optional answer grading
- optional synthetic question generation for future dataset expansion

Do not hardcode Azure specifics into evaluator logic.
All model calls must go through provider interfaces.
Environment variables should be read from `.env` only in config loading code.

Expected env vars:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT`

## Coding rules

- use type hints everywhere
- prefer pure functions in metrics code
- avoid hidden global state
- keep prompts versioned and explicit
- save all model inputs that affect scoring
- do not mix retrieval metrics with answer metrics in the same module
- write small modules with direct names

When changing schemas or metrics:
- update docs
- update tests
- update sample artifacts

## Testing rules

Every new metric or schema change needs:
- unit tests for normal cases
- unit tests for edge cases
- at least one golden fixture test for serialized output

Do not rely only on live model calls in tests.
Mock provider responses when testing evaluator behavior.

## Run commands

Install:
- `uv sync`
- or `pip install -e .[dev]`

Quality:
- `ruff check .`
- `ruff format .`
- `pytest -q`

Run eval:
- `python -m src.cli.run_eval --config configs/smoke.yaml`
- `python -m src.cli.run_eval --config configs/full.yaml`

Summarize results:
- `python -m src.cli.summarize_run --run-dir outputs/<run_id>`

## Default expectations for Claude

When working in this repo:
- preserve scope discipline
- prefer finishing the current milestone over adding optional features
- avoid introducing infrastructure that requires a team to maintain
- write code that can be understood quickly by a new contributor
- document non obvious decisions near the code and in docs when needed

Before major edits:
- check the relevant doc imported above
- keep interfaces stable unless there is a strong reason to change them

If there is ambiguity, bias toward the simplest implementation that preserves reproducibility and clean evaluation logic.
