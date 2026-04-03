# FinRAG Eval

FinRAG Eval is an open source benchmark and evaluation harness for financial RAG systems.

It helps answer a simple question: when a retrieval augmented QA system answers a finance question, is the answer grounded, cited correctly, fast enough, and cheap enough to be useful?

## What it measures

- hallucination rate and answer faithfulness
- citation accuracy and evidence coverage
- retrieval quality separate from generation quality
- latency and token or cost metrics
- reproducible leaderboard artifacts

## What this repo is not

This is not a production finance chatbot.
It is not a heavy web platform.
It is not a large scale annotation system.

The initial goal is a clean offline benchmark runner with a few reference baselines and a static leaderboard format.

## Why finance

Financial QA is harder than generic RAG.
Questions are often:
- numerical
- time sensitive
- evidence sensitive
- multi document
- compliance and trust sensitive

That makes it a good setting for serious evaluation instead of demo driven benchmarking.

## MVP scope

The first version should support:
- a dataset of financial documents plus QA examples
- a baseline retrieval and answer generation pipeline
- pipeline adapters for external RAG systems
- metrics for faithfulness, citation accuracy, latency, and cost
- machine readable run artifacts
- leaderboard summary generation

## Suggested stack

- Python 3.11+
- Pydantic
- Typer
- Pandas or Polars
- Azure OpenAI for model based judges
- Pytest
- Ruff

## Project structure

```text
finrag-eval/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .env.example
├── docs/
├── .claude/
├── src/
├── configs/
├── datasets/
├── scripts/
└── tests/
```

## Getting started

### 1. Create a virtual environment

```bash
uv venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
uv sync
```

or

```bash
pip install -e .[dev]
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in the values you plan to use.

### 4. Run the test suite

```bash
pytest -q
```

### 5. Run the smoke benchmark

```bash
python -m src.cli.run_eval --config configs/smoke.yaml
```

## Azure OpenAI setup

This repo is designed so Azure OpenAI can be used for judge style evaluators without baking Azure specific logic into the framework.

Expected variables:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT`

Use Azure for:
- faithfulness judgment
- optional answer grading
- future dataset expansion workflows

## Output contract

Each benchmark run should write:
- per question scores
- aggregate metrics
- prompt and model config snapshot
- latency and cost summary
- leaderboard ready artifact

## Implementation plan

1. define dataset schemas and fixtures
2. implement a baseline pipeline adapter
3. capture latency and cost
4. implement citation checking
5. implement faithfulness scoring
6. generate summary reports and leaderboard artifacts

## Contributing

Keep changes focused.
If you change schemas, metrics, or artifact structure, update the matching docs and tests in the same pull request.
