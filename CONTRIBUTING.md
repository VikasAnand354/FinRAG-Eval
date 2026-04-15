# Contributing to FinRAG Eval

Thanks for your interest in contributing. This project values focused, clean contributions that keep the evaluation framework trustworthy and reproducible.

## What to work on

Good contributions include:

- New dataset splits following the existing schema
- New retriever adapters (dense, hybrid, re-ranking)
- New generation adapters (other LLM providers)
- Bug fixes in metric calculations
- Documentation improvements
- Additional test cases for edge cases

Please avoid:

- Adding web UI or server infrastructure
- Adding heavy orchestration frameworks as core dependencies
- Changing schemas or metric definitions without updating tests and docs

If you are unsure whether something fits the project scope, open an issue first.

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/VikasAnand354/FinRAG-Eval.git
cd finrag-eval
python -m venv .venv
source .venv/bin/activate
```

### 2. Install in dev mode

```bash
pip install -e .[dev]
```

### 3. Configure environment variables (optional)

Only needed if you are working with Azure OpenAI features:

```bash
cp .env.example .env
# fill in AZURE_OPENAI_* values
```

### 4. Verify your setup

```bash
pytest -q
python -m src.cli.run_eval --config configs/smoke.yaml
```

Both should complete without errors. The smoke eval uses a mock adapter and requires no credentials.

---

## Making changes

### Adding a new dataset

1. Produce `qa_examples.jsonl` and `chunks.jsonl` following the schemas in [docs/dataset-schema.md](docs/dataset-schema.md)
2. Place them in `datasets/<your_dataset_name>/`
3. Add a config file in `configs/` pointing to the new paths
4. Add at least one fixture example to `tests/fixtures/` and a test in `tests/test_golden.py`

No evaluator code changes are needed to add a dataset.

### Adding a new retriever

1. Create a new module in `src/retrieval/`
2. Implement a `retrieve(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]` function
3. Register the adapter name in `src/cli/run_eval.py`
4. Add unit tests in `tests/`

### Adding a new generation adapter

1. Create a new module in `src/generation/`
2. Subclass `GenerationAdapter` from `src/generation/base.py`
3. Implement both `generate()` and `judge()` methods
4. Register the adapter name in `src/cli/run_eval.py`
5. Add unit tests using mocked responses — do not rely on live API calls in tests

### Changing a metric

Metric changes affect leaderboard comparability. If you change a metric:

1. Update the implementation in `src/evaluators/`
2. Update the metric definition in [docs/eval-methodology.md](docs/eval-methodology.md)
3. Update or add unit tests in `tests/`
4. Update any affected golden fixtures in `tests/fixtures/`
5. Note the change in your PR description — existing leaderboard entries may no longer be directly comparable

---

## Code standards

- **Type hints everywhere** — all function signatures must be fully typed
- **Pure functions for metrics** — evaluator functions should have no side effects
- **No hidden global state** — pass dependencies explicitly
- **Small, direct modules** — one clear responsibility per file
- **Versioned prompts** — any prompt used in scoring must have a version constant

Run before submitting:

```bash
ruff format .
ruff check .
pytest -q
```

All three must pass cleanly.

---

## Pull request checklist

Before opening a PR:

- [ ] `ruff format .` and `ruff check .` pass
- [ ] `pytest -q` passes
- [ ] New behaviour has unit tests covering normal and edge cases
- [ ] Schema or metric changes have updated docs
- [ ] Golden fixtures are updated if serialized output changed
- [ ] PR description explains what changed and why

---

## Project structure quick reference

```
src/common/        — schemas and config loading
src/ingest/        — dataset construction pipeline
src/retrieval/     — retriever adapters
src/generation/    — LLM adapters
src/evaluators/    — metric calculations
src/leaderboard/   — artifact generation and summaries
src/cli/           — entry points
tests/             — unit tests and fixtures
configs/           — run configuration files
docs/              — architecture and methodology docs
```

See [docs/architecture.md](docs/architecture.md) for a full module map and design decisions.
