# FinRAG Eval MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full FinRAG Eval MVP so that `python -m src.cli.run_eval --config configs/smoke.yaml` executes a complete benchmark run and produces a versioned JSON artifact with per-question scores, aggregate metrics, and a summary table.

**Architecture:** Sequential offline eval harness — BM25 retrieval over pre-chunked JSONL, Azure OpenAI (or mock) generation, three pure-function evaluators (latency/cost, citation, faithfulness), and a leaderboard artifact writer. All modules are isolated by responsibility and connected only through Pydantic schemas.

**Tech Stack:** Python 3.11+, Pydantic v2, Typer, rank-bm25, tiktoken, openai (Azure), rich, pytest, ruff

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Move | `common/models.py` → `src/common/models.py` | All Pydantic schemas |
| Move | `common/registry.py` → `src/common/registry.py` | Company registry loader |
| Move | `ingest/chunker.py` → `src/ingest/chunker.py` | Text chunking (update signature) |
| Move | `ingest/sec_fetch.py` → `src/ingest/sec_fetch.py` | SEC EDGAR downloader |
| Rewrite | `cli/run_eval.py` → `src/cli/run_eval.py` | Typer eval runner |
| Create | `src/__init__.py` | Package marker |
| Create | `src/common/__init__.py` | Package marker |
| Create | `src/common/config.py` | Load .env + parse YAML RunConfig |
| Create | `src/ingest/__init__.py` | Package marker |
| Create | `src/retrieval/__init__.py` | Package marker |
| Create | `src/retrieval/bm25.py` | BM25Retriever class |
| Create | `src/generation/__init__.py` | Package marker |
| Create | `src/generation/base.py` | GenerationAdapter ABC |
| Create | `src/generation/azure_openai.py` | AzureOpenAIAdapter + MockGenerationAdapter |
| Create | `src/evaluators/__init__.py` | Package marker |
| Create | `src/evaluators/latency.py` | score_latency() pure function |
| Create | `src/evaluators/citation.py` | score_citations() pure function |
| Create | `src/evaluators/faithfulness.py` | score_faithfulness() LLM judge |
| Create | `src/evaluators/runner.py` | evaluate() orchestrator |
| Create | `src/leaderboard/__init__.py` | Package marker |
| Create | `src/leaderboard/artifacts.py` | write_artifact, validate_artifact, compute_aggregate |
| Create | `src/leaderboard/summary.py` | generate_summary() |
| Create | `src/cli/__init__.py` | Package marker |
| Create | `src/cli/summarize_run.py` | Typer summarize CLI |
| Create | `configs/smoke.yaml` | Sample dataset run config |
| Create | `configs/full.yaml` | Full dataset run config (placeholder) |
| Create | `datasets/sample/documents.jsonl` | 3 hand-written 10-K excerpts |
| Create | `datasets/sample/chunks.jsonl` | Pre-chunked from sample docs |
| Create | `datasets/sample/qa_examples.jsonl` | 5 benchmark QA pairs |
| Create | `tests/__init__.py` | Package marker |
| Create | `tests/test_models.py` | Schema round-trip tests |
| Create | `tests/test_chunker.py` | Chunker unit tests |
| Create | `tests/test_bm25.py` | BM25Retriever unit tests |
| Create | `tests/test_citation.py` | Citation evaluator tests |
| Create | `tests/test_faithfulness.py` | Faithfulness evaluator tests |
| Create | `tests/test_artifacts.py` | Artifact writer tests |
| Create | `tests/fixtures/golden_run.json` | Regression fixture |
| Create | `.env.example` | Azure env var documentation |
| Modify | `pyproject.toml` | Add rank-bm25, tiktoken, requests deps |

---

## Task 1: Fix project structure

**Files:**
- Create: `src/__init__.py`, `src/common/__init__.py`, `src/ingest/__init__.py`, `src/retrieval/__init__.py`, `src/generation/__init__.py`, `src/evaluators/__init__.py`, `src/leaderboard/__init__.py`, `src/cli/__init__.py`
- Create: `tests/__init__.py`, `tests/fixtures/` directory
- Move: all 5 existing files into `src/`

- [ ] **Step 1: Create src/ package tree and move files**

```bash
mkdir -p src/common src/ingest src/retrieval src/generation src/evaluators src/leaderboard src/cli
mkdir -p tests/fixtures
touch src/__init__.py src/common/__init__.py src/ingest/__init__.py
touch src/retrieval/__init__.py src/generation/__init__.py src/evaluators/__init__.py
touch src/leaderboard/__init__.py src/cli/__init__.py
touch tests/__init__.py

# Move existing files
mv common/models.py src/common/models.py
mv common/registry.py src/common/registry.py
mv ingest/chunker.py src/ingest/chunker.py
mv ingest/sec_fetch.py src/ingest/sec_fetch.py
mv cli/run_eval.py src/cli/run_eval.py

# Remove now-empty root dirs
rmdir common ingest cli
```

- [ ] **Step 2: Update registry.py import path**

In `src/common/registry.py` the file uses only stdlib — no import changes needed. Verify with:

```bash
python -c "from src.common.registry import CompanyRegistry; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Verify models import works**

```bash
python -c "from src.common.models import Chunk, BenchmarkExample; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/ tests/ && git commit -m "chore: move existing files into src/ layout, add package __init__.py files"
```

---

## Task 2: Update pyproject.toml dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add missing dependencies**

Edit the `dependencies` list in `pyproject.toml` to add three entries:

```toml
[project]
dependencies = [
  "pydantic>=2.7",
  "typer>=0.12",
  "pandas>=2.2",
  "pyyaml>=6.0",
  "python-dotenv>=1.0",
  "httpx>=0.27",
  "rich>=13.7",
  "rank-bm25>=0.2",
  "tiktoken>=0.7",
  "requests>=2.31"
]
```

- [ ] **Step 2: Install updated dependencies**

```bash
pip install -e .[dev,azure]
```

Expected: installs without errors, `rank_bm25` and `tiktoken` appear in output.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml && git commit -m "chore: add rank-bm25, tiktoken, requests to dependencies"
```

---

## Task 3: Add new schemas to models.py + test_models.py

**Files:**
- Modify: `src/common/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models.py`:

```python
from datetime import datetime, timezone

import pytest


def test_run_config_round_trip():
    from src.common.models import RunConfig

    cfg = RunConfig(
        run_id="smoke-001",
        dataset_path="datasets/sample/qa_examples.jsonl",
        chunks_path="datasets/sample/chunks.jsonl",
        retriever="bm25",
        top_k=3,
        generation_adapter="mock",
        output_dir="outputs/",
    )
    assert RunConfig.model_validate_json(cfg.model_dump_json()) == cfg


def test_pipeline_result_round_trip():
    from src.common.models import PipelineResult

    r = PipelineResult(
        example_id="q001",
        answer="Revenue was $383.3 billion.",
        citations=["aapl_10k_2023__c0001"],
        latency_ms=250.0,
        prompt_tokens=100,
        completion_tokens=30,
    )
    assert PipelineResult.model_validate_json(r.model_dump_json()) == r


def test_scored_row_round_trip():
    from src.common.models import ScoredRow

    row = ScoredRow(
        example_id="q001",
        answer="Revenue was $383.3 billion.",
        citations=["aapl_10k_2023__c0001"],
        latency_ms=250.0,
        prompt_tokens=100,
        completion_tokens=30,
        cost_usd=0.00394,
        exact_match=True,
        citation_precision=1.0,
        citation_recall=1.0,
        faithful=True,
        faithfulness_score=0.95,
    )
    assert ScoredRow.model_validate_json(row.model_dump_json()) == row


def test_scored_row_nullable_faithfulness():
    from src.common.models import ScoredRow

    row = ScoredRow(
        example_id="q001",
        answer="test",
        citations=[],
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=2,
        cost_usd=0.0,
        exact_match=False,
        citation_precision=0.0,
        citation_recall=0.0,
        faithful=False,
        faithfulness_score=None,
    )
    assert row.faithfulness_score is None
    assert ScoredRow.model_validate_json(row.model_dump_json()) == row


def test_run_artifact_round_trip():
    from src.common.models import RunArtifact, ScoredRow

    row = ScoredRow(
        example_id="q001",
        answer="test",
        citations=[],
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=2,
        cost_usd=0.0001,
        exact_match=False,
        citation_precision=0.0,
        citation_recall=0.0,
        faithful=False,
        faithfulness_score=None,
    )
    artifact = RunArtifact(
        run_id="test-001",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        config_snapshot={"run_id": "test-001"},
        scores=[row],
        aggregate={"n_examples": 1},
    )
    assert RunArtifact.model_validate_json(artifact.model_dump_json()) == artifact
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_models.py -v
```

Expected: `ImportError` or `AttributeError` for `RunConfig`, `PipelineResult`, etc.

- [ ] **Step 3: Add new schemas to models.py**

Append to the bottom of `src/common/models.py`:

```python


class RunConfig(BaseModel):
    run_id: str
    dataset_path: str
    chunks_path: str
    retriever: str
    top_k: int
    generation_adapter: str
    output_dir: str


class PipelineResult(BaseModel):
    example_id: str
    answer: str
    citations: List[str]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


class ScoredRow(BaseModel):
    example_id: str
    answer: str
    citations: List[str]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    exact_match: bool
    citation_precision: float
    citation_recall: float
    faithful: bool
    faithfulness_score: Optional[float]


class RunArtifact(BaseModel):
    run_id: str
    timestamp: datetime
    config_snapshot: dict
    scores: List[ScoredRow]
    aggregate: dict
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/common/models.py tests/test_models.py
git commit -m "feat: add RunConfig, PipelineResult, ScoredRow, RunArtifact schemas"
```

---

## Task 4: Add common/config.py and .env.example

**Files:**
- Create: `src/common/config.py`
- Create: `.env.example`

- [ ] **Step 1: Create config.py**

```python
# src/common/config.py
import os

import yaml
from dotenv import load_dotenv

from src.common.models import RunConfig


def load_run_config(path: str) -> RunConfig:
    load_dotenv()
    with open(path) as f:
        data = yaml.safe_load(f)
    return RunConfig(**data)


def get_azure_settings() -> dict[str, str]:
    required = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
    return {
        "api_key": os.environ["AZURE_OPENAI_API_KEY"],
        "endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
        "deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
    }
```

- [ ] **Step 2: Create .env.example**

```
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

- [ ] **Step 3: Verify config imports cleanly**

```bash
python -c "from src.common.config import load_run_config, get_azure_settings; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/common/config.py .env.example
git commit -m "feat: add config loader for YAML RunConfig and Azure env vars"
```

---

## Task 5: Create sample fixtures

**Files:**
- Create: `datasets/sample/documents.jsonl`
- Create: `datasets/sample/chunks.jsonl`
- Create: `datasets/sample/qa_examples.jsonl`

- [ ] **Step 1: Create datasets/sample/ directory**

```bash
mkdir -p datasets/sample
```

- [ ] **Step 2: Create documents.jsonl**

Each line is a valid `DocumentMetadata` JSON object. Create `datasets/sample/documents.jsonl`:

```jsonl
{"document_id": "aapl_10k_2023", "company": "AAPL", "company_name": "Apple Inc.", "source_type": "sec_edgar", "source_url": null, "doc_family": "10-K", "fiscal_period": "FY2023", "calendar_date": "2023-09-30", "filing_date": "2023-11-03", "report_period_end": "2023-09-30", "language": "en", "format": "txt", "local_path": "datasets/sample/raw/aapl_10k_2023.txt", "sha256": null, "collected_at": "2024-01-01T00:00:00Z"}
{"document_id": "msft_10k_2023", "company": "MSFT", "company_name": "Microsoft Corporation", "source_type": "sec_edgar", "source_url": null, "doc_family": "10-K", "fiscal_period": "FY2023", "calendar_date": "2023-06-30", "filing_date": "2023-07-27", "report_period_end": "2023-06-30", "language": "en", "format": "txt", "local_path": "datasets/sample/raw/msft_10k_2023.txt", "sha256": null, "collected_at": "2024-01-01T00:00:00Z"}
{"document_id": "googl_10k_2023", "company": "GOOGL", "company_name": "Alphabet Inc.", "source_type": "sec_edgar", "source_url": null, "doc_family": "10-K", "fiscal_period": "FY2023", "calendar_date": "2023-12-31", "filing_date": "2024-01-31", "report_period_end": "2023-12-31", "language": "en", "format": "txt", "local_path": "datasets/sample/raw/googl_10k_2023.txt", "sha256": null, "collected_at": "2024-01-01T00:00:00Z"}
```

- [ ] **Step 3: Create chunks.jsonl**

Each line is a valid `Chunk` JSON object. The `chunk_id` format is `{document_id}__c{index:04d}`. Paragraph boundaries determined by `\n\n`. Create `datasets/sample/chunks.jsonl`:

```jsonl
{"chunk_id": "aapl_10k_2023__c0000", "document_id": "aapl_10k_2023", "company": "AAPL", "section_title": null, "section_path": null, "page_number": null, "paragraph_number": 0, "text": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The Company also sells a variety of related services.", "token_count": 37, "report_period_end": "2023-09-30", "filing_date": "2023-11-03"}
{"chunk_id": "aapl_10k_2023__c0001", "document_id": "aapl_10k_2023", "company": "AAPL", "section_title": null, "section_path": null, "page_number": null, "paragraph_number": 1, "text": "For the fiscal year ended September 30, 2023, the Company reported total net sales of $383.3 billion, compared to $394.3 billion in fiscal year 2022. iPhone net sales were $200.6 billion, representing approximately 52 percent of total net sales.", "token_count": 52, "report_period_end": "2023-09-30", "filing_date": "2023-11-03"}
{"chunk_id": "aapl_10k_2023__c0002", "document_id": "aapl_10k_2023", "company": "AAPL", "section_title": null, "section_path": null, "page_number": null, "paragraph_number": 2, "text": "The Company's reportable segments consist of the Americas, Europe, Greater China, Japan, and Rest of Asia Pacific.", "token_count": 23, "report_period_end": "2023-09-30", "filing_date": "2023-11-03"}
{"chunk_id": "msft_10k_2023__c0000", "document_id": "msft_10k_2023", "company": "MSFT", "section_title": null, "section_path": null, "page_number": null, "paragraph_number": 0, "text": "Microsoft Corporation is a technology company. We develop and support software, services, devices, and solutions that deliver value to people and organizations.", "token_count": 30, "report_period_end": "2023-06-30", "filing_date": "2023-07-27"}
{"chunk_id": "msft_10k_2023__c0001", "document_id": "msft_10k_2023", "company": "MSFT", "section_title": null, "section_path": null, "page_number": null, "paragraph_number": 1, "text": "Revenue for fiscal year 2023 was $211.9 billion, an increase of 7 percent compared to fiscal year 2022. Our Intelligent Cloud segment revenue was $87.9 billion, representing 41 percent of total revenue.", "token_count": 40, "report_period_end": "2023-06-30", "filing_date": "2023-07-27"}
{"chunk_id": "googl_10k_2023__c0000", "document_id": "googl_10k_2023", "company": "GOOGL", "section_title": null, "section_path": null, "page_number": null, "paragraph_number": 0, "text": "Alphabet Inc. was created as the holding company for Google and several other businesses commenced in August 2015.", "token_count": 24, "report_period_end": "2023-12-31", "filing_date": "2024-01-31"}
{"chunk_id": "googl_10k_2023__c0001", "document_id": "googl_10k_2023", "company": "GOOGL", "section_title": null, "section_path": null, "page_number": null, "paragraph_number": 1, "text": "Our revenues for the fiscal year ended December 31, 2023 were $307.4 billion, an increase of 9 percent year over year. Google Services revenues were $272.5 billion, driven by growth in Google Search and YouTube.", "token_count": 42, "report_period_end": "2023-12-31", "filing_date": "2024-01-31"}
```

- [ ] **Step 4: Create qa_examples.jsonl**

Create `datasets/sample/qa_examples.jsonl`:

```jsonl
{"example_id": "q001", "company": "AAPL", "question": "What were Apple's total net sales for fiscal year 2023?", "question_type": "numerical", "difficulty": "easy", "answer_type": "numeric", "gold_answer": "$383.3 billion", "gold_answer_normalized": 383.3, "normalization_unit": "billion USD", "gold_citations": [{"document_id": "aapl_10k_2023", "chunk_id": "aapl_10k_2023__c0001", "support_type": "direct"}], "acceptable_aliases": ["383.3 billion", "383.3B"], "time_sensitive": false, "requires_multi_hop": false, "source_split": "test"}
{"example_id": "q002", "company": "AAPL", "question": "What were Apple's iPhone net sales for fiscal year 2023?", "question_type": "numerical", "difficulty": "easy", "answer_type": "numeric", "gold_answer": "$200.6 billion", "gold_answer_normalized": 200.6, "normalization_unit": "billion USD", "gold_citations": [{"document_id": "aapl_10k_2023", "chunk_id": "aapl_10k_2023__c0001", "support_type": "direct"}], "acceptable_aliases": ["200.6 billion", "200.6B"], "time_sensitive": false, "requires_multi_hop": false, "source_split": "test"}
{"example_id": "q003", "company": "MSFT", "question": "What was Microsoft's total revenue for fiscal year 2023?", "question_type": "numerical", "difficulty": "easy", "answer_type": "numeric", "gold_answer": "$211.9 billion", "gold_answer_normalized": 211.9, "normalization_unit": "billion USD", "gold_citations": [{"document_id": "msft_10k_2023", "chunk_id": "msft_10k_2023__c0001", "support_type": "direct"}], "acceptable_aliases": ["211.9 billion", "211.9B"], "time_sensitive": false, "requires_multi_hop": false, "source_split": "test"}
{"example_id": "q004", "company": "MSFT", "question": "What was Microsoft's Intelligent Cloud segment revenue for fiscal year 2023?", "question_type": "numerical", "difficulty": "easy", "answer_type": "numeric", "gold_answer": "$87.9 billion", "gold_answer_normalized": 87.9, "normalization_unit": "billion USD", "gold_citations": [{"document_id": "msft_10k_2023", "chunk_id": "msft_10k_2023__c0001", "support_type": "direct"}], "acceptable_aliases": ["87.9 billion", "87.9B"], "time_sensitive": false, "requires_multi_hop": false, "source_split": "test"}
{"example_id": "q005", "company": "GOOGL", "question": "What were Alphabet's total revenues for fiscal year 2023?", "question_type": "numerical", "difficulty": "easy", "answer_type": "numeric", "gold_answer": "$307.4 billion", "gold_answer_normalized": 307.4, "normalization_unit": "billion USD", "gold_citations": [{"document_id": "googl_10k_2023", "chunk_id": "googl_10k_2023__c0001", "support_type": "direct"}], "acceptable_aliases": ["307.4 billion", "307.4B"], "time_sensitive": false, "requires_multi_hop": false, "source_split": "test"}
```

- [ ] **Step 5: Verify fixtures parse against schemas**

```bash
python -c "
from src.common.models import Chunk, BenchmarkExample, DocumentMetadata
with open('datasets/sample/chunks.jsonl') as f:
    chunks = [Chunk.model_validate_json(line) for line in f]
with open('datasets/sample/qa_examples.jsonl') as f:
    examples = [BenchmarkExample.model_validate_json(line) for line in f]
print(f'Chunks: {len(chunks)}, Examples: {len(examples)}')
"
```

Expected: `Chunks: 7, Examples: 5`

- [ ] **Step 6: Commit**

```bash
git add datasets/ && git commit -m "feat: add sample fixture documents, chunks, and QA examples"
```

---

## Task 6: Update chunker + test_chunker.py

**Files:**
- Modify: `src/ingest/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_chunker.py`:

```python
from src.ingest.chunker import build_chunk_id, chunk_document, count_tokens, simple_paragraph_split


def test_simple_paragraph_split_basic():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird."
    parts = simple_paragraph_split(text)
    assert parts == ["First paragraph.", "Second paragraph.", "Third."]


def test_simple_paragraph_split_ignores_blank_lines():
    text = "\n\nFirst.\n\n\n\nSecond.\n\n"
    parts = simple_paragraph_split(text)
    assert parts == ["First.", "Second."]


def test_build_chunk_id_zero_padded():
    assert build_chunk_id("doc1", 0) == "doc1__c0000"
    assert build_chunk_id("doc1", 42) == "doc1__c0042"
    assert build_chunk_id("doc1", 9999) == "doc1__c9999"


def test_count_tokens_nonempty():
    n = count_tokens("hello world")
    assert n > 0


def test_chunk_document_produces_full_dicts():
    text = "First paragraph.\n\nSecond paragraph."
    chunks = chunk_document("doc1", "AAPL", text)
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "doc1__c0000"
    assert chunks[0]["document_id"] == "doc1"
    assert chunks[0]["company"] == "AAPL"
    assert chunks[0]["text"] == "First paragraph."
    assert chunks[0]["token_count"] > 0
    assert chunks[0]["paragraph_number"] == 0
    assert chunks[1]["chunk_id"] == "doc1__c0001"
    assert chunks[1]["paragraph_number"] == 1


def test_chunk_document_with_filing_date():
    text = "Revenue was $100 billion."
    chunks = chunk_document("doc1", "AAPL", text, filing_date="2023-11-03")
    assert chunks[0]["filing_date"] == "2023-11-03"


def test_chunk_document_optional_fields_are_none():
    text = "Some text."
    chunks = chunk_document("doc1", "AAPL", text)
    assert chunks[0]["section_title"] is None
    assert chunks[0]["section_path"] is None
    assert chunks[0]["page_number"] is None
    assert chunks[0]["report_period_end"] is None
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_chunker.py -v
```

Expected: failures on `test_chunk_document_produces_full_dicts` (missing `document_id`, `company` keys).

- [ ] **Step 3: Update chunk_document signature**

Replace the `chunk_document` function in `src/ingest/chunker.py`:

```python
from typing import List, Optional

from tiktoken import get_encoding

ENC = get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def simple_paragraph_split(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def build_chunk_id(document_id: str, idx: int) -> str:
    return f"{document_id}__c{idx:04d}"


def chunk_document(
    document_id: str,
    company: str,
    text: str,
    filing_date: Optional[str] = None,
    report_period_end: Optional[str] = None,
) -> List[dict]:
    paragraphs = simple_paragraph_split(text)
    chunks = []
    for i, p in enumerate(paragraphs):
        chunks.append({
            "chunk_id": build_chunk_id(document_id, i),
            "document_id": document_id,
            "company": company,
            "section_title": None,
            "section_path": None,
            "page_number": None,
            "paragraph_number": i,
            "text": p,
            "token_count": count_tokens(p),
            "report_period_end": report_period_end,
            "filing_date": filing_date,
        })
    return chunks
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_chunker.py -v
```

Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ingest/chunker.py tests/test_chunker.py
git commit -m "feat: update chunk_document to produce full Chunk-compatible dicts"
```

---

## Task 7: BM25 retriever + test_bm25.py

**Files:**
- Create: `src/retrieval/bm25.py`
- Create: `tests/test_bm25.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_bm25.py`:

```python
from src.common.models import Chunk


def _make_chunk(chunk_id: str, text: str, company: str = "TEST") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("__")[0],
        company=company,
        text=text,
        token_count=len(text.split()),
    )


def test_bm25_retrieves_most_relevant_chunk():
    from src.retrieval.bm25 import BM25Retriever

    chunks = [
        _make_chunk("doc1__c0000", "Apple iPhone revenue was 200 billion dollars"),
        _make_chunk("doc2__c0000", "Microsoft cloud services annual revenue"),
        _make_chunk("doc3__c0000", "Alphabet Google advertising total revenue"),
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("Apple iPhone", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "doc1__c0000"


def test_bm25_top_k_respects_limit():
    from src.retrieval.bm25 import BM25Retriever

    chunks = [_make_chunk(f"doc{i}__c0000", f"text about financial topic number {i}") for i in range(5)]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("financial topic", top_k=3)
    assert len(results) == 3


def test_bm25_top_k_larger_than_corpus():
    from src.retrieval.bm25 import BM25Retriever

    chunks = [_make_chunk("doc1__c0000", "only one document exists here")]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("document", top_k=5)
    assert len(results) == 1


def test_bm25_empty_corpus():
    from src.retrieval.bm25 import BM25Retriever

    retriever = BM25Retriever([])
    results = retriever.retrieve("any query", top_k=3)
    assert results == []
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_bm25.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.retrieval.bm25'`

- [ ] **Step 3: Implement BM25Retriever**

Create `src/retrieval/bm25.py`:

```python
from rank_bm25 import BM25Okapi

from src.common.models import Chunk


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        if chunks:
            tokenized = [c.text.lower().split() for c in chunks]
            self._bm25: BM25Okapi | None = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        if not self._chunks or self._bm25 is None:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._chunks[i] for i in ranked[:top_k]]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_bm25.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/bm25.py tests/test_bm25.py
git commit -m "feat: add BM25Retriever baseline retriever"
```

---

## Task 8: Generation adapter base + MockGenerationAdapter

**Files:**
- Create: `src/generation/base.py`
- Create: `src/generation/azure_openai.py` (MockGenerationAdapter only for now)

- [ ] **Step 1: Create base.py**

```python
# src/generation/base.py
from abc import ABC, abstractmethod

from src.common.models import Chunk, PipelineResult


class GenerationAdapter(ABC):
    @abstractmethod
    def generate(self, example_id: str, question: str, chunks: list[Chunk]) -> PipelineResult:
        """Generate an answer given a question and retrieved chunks."""
        ...

    @abstractmethod
    def judge(self, prompt: str) -> tuple[str, float, int, int]:
        """
        Send a raw prompt and return (response_text, latency_ms, prompt_tokens, completion_tokens).
        Used by the faithfulness evaluator.
        """
        ...
```

- [ ] **Step 2: Create azure_openai.py with MockGenerationAdapter**

```python
# src/generation/azure_openai.py
import json
import logging
import time

from src.common.models import Chunk, PipelineResult
from src.generation.base import GenerationAdapter

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT_V1 = (
    "You are a financial analyst assistant. Answer the question using only the provided "
    "context passages. Be precise with numbers and dates. If the context does not contain "
    "enough information to answer, say so."
)

PROMPT_VERSION = "qa-v1"


class MockGenerationAdapter(GenerationAdapter):
    """Deterministic adapter for tests — no network calls."""

    def __init__(self, canned_answer: str = "mock answer", canned_score: float = 0.9) -> None:
        self._answer = canned_answer
        self._canned_score = canned_score

    def generate(self, example_id: str, question: str, chunks: list[Chunk]) -> PipelineResult:
        return PipelineResult(
            example_id=example_id,
            answer=self._answer,
            citations=[c.chunk_id for c in chunks],
            latency_ms=1.0,
            prompt_tokens=10,
            completion_tokens=5,
        )

    def judge(self, prompt: str) -> tuple[str, float, int, int]:
        return (json.dumps({"score": self._canned_score}), 1.0, 10, 5)


class AzureOpenAIAdapter(GenerationAdapter):
    def __init__(self) -> None:
        from openai import AzureOpenAI

        from src.common.config import get_azure_settings

        settings = get_azure_settings()
        self._client = AzureOpenAI(
            api_key=settings["api_key"],
            api_version=settings["api_version"],
            azure_endpoint=settings["endpoint"],
        )
        self._deployment = settings["deployment"]

    def generate(self, example_id: str, question: str, chunks: list[Chunk]) -> PipelineResult:
        context = "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks)
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT_V1},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=messages,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            answer = response.choices[0].message.content or ""
            return PipelineResult(
                example_id=example_id,
                answer=answer,
                citations=[c.chunk_id for c in chunks],
                latency_ms=latency_ms,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
        except Exception:
            logger.exception("Generation failed for example_id=%s", example_id)
            return PipelineResult(
                example_id=example_id,
                answer="",
                citations=[],
                latency_ms=(time.perf_counter() - start) * 1000,
                prompt_tokens=0,
                completion_tokens=0,
            )

    def judge(self, prompt: str) -> tuple[str, float, int, int]:
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000
        text = response.choices[0].message.content or ""
        return (text, latency_ms, response.usage.prompt_tokens, response.usage.completion_tokens)
```

- [ ] **Step 3: Verify imports**

```bash
python -c "from src.generation.azure_openai import MockGenerationAdapter; m = MockGenerationAdapter(); print(m.judge('test'))"
```

Expected: `('{"score": 0.9}', 1.0, 10, 5)`

- [ ] **Step 4: Commit**

```bash
git add src/generation/base.py src/generation/azure_openai.py
git commit -m "feat: add GenerationAdapter ABC, MockGenerationAdapter, AzureOpenAIAdapter"
```

---

## Task 9: Latency evaluator + tests

**Files:**
- Create: `src/evaluators/latency.py`
- Create: `tests/test_latency.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_latency.py`:

```python
from src.common.models import PipelineResult


def _make_result(latency_ms: float = 200.0, prompt_tokens: int = 100, completion_tokens: int = 50) -> PipelineResult:
    return PipelineResult(
        example_id="q001",
        answer="test",
        citations=[],
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def test_score_latency_returns_all_fields():
    from src.evaluators.latency import score_latency

    scores = score_latency(_make_result())
    assert "latency_ms" in scores
    assert "prompt_tokens" in scores
    assert "completion_tokens" in scores
    assert "cost_usd" in scores


def test_score_latency_values_match_input():
    from src.evaluators.latency import score_latency

    scores = score_latency(_make_result(latency_ms=123.4, prompt_tokens=200, completion_tokens=80))
    assert scores["latency_ms"] == 123.4
    assert scores["prompt_tokens"] == 200
    assert scores["completion_tokens"] == 80


def test_score_latency_cost_is_positive():
    from src.evaluators.latency import score_latency

    scores = score_latency(_make_result(prompt_tokens=1000, completion_tokens=500))
    assert scores["cost_usd"] > 0


def test_score_latency_zero_tokens_zero_cost():
    from src.evaluators.latency import score_latency

    scores = score_latency(_make_result(prompt_tokens=0, completion_tokens=0))
    assert scores["cost_usd"] == 0.0
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_latency.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.evaluators.latency'`

- [ ] **Step 3: Implement latency evaluator**

Create `src/evaluators/latency.py`:

```python
from typing import Any

from src.common.models import PipelineResult

# Price per 1000 tokens in USD (GPT-4 approximate; update as needed)
PROMPT_TOKEN_PRICE_PER_1K: float = 0.01
COMPLETION_TOKEN_PRICE_PER_1K: float = 0.03


def score_latency(result: PipelineResult) -> dict[str, Any]:
    cost = (
        result.prompt_tokens / 1000 * PROMPT_TOKEN_PRICE_PER_1K
        + result.completion_tokens / 1000 * COMPLETION_TOKEN_PRICE_PER_1K
    )
    return {
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_usd": round(cost, 6),
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_latency.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/evaluators/latency.py tests/test_latency.py
git commit -m "feat: add latency and cost evaluator"
```

---

## Task 10: Citation evaluator + test_citation.py

**Files:**
- Create: `src/evaluators/citation.py`
- Create: `tests/test_citation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_citation.py`:

```python
from src.common.models import BenchmarkExample, GoldCitation, PipelineResult


def _make_example(gold_chunk_ids: list[str], gold_answer: str = "$100 billion") -> BenchmarkExample:
    return BenchmarkExample(
        example_id="q001",
        company="AAPL",
        question="What was revenue?",
        question_type="numerical",
        difficulty="easy",
        answer_type="numeric",
        gold_answer=gold_answer,
        gold_citations=[
            GoldCitation(document_id="doc1", chunk_id=cid, support_type="direct")
            for cid in gold_chunk_ids
        ],
        source_split="test",
    )


def _make_result(citations: list[str], answer: str = "Revenue was $100 billion") -> PipelineResult:
    return PipelineResult(
        example_id="q001",
        answer=answer,
        citations=citations,
        latency_ms=100.0,
        prompt_tokens=50,
        completion_tokens=10,
    )


def test_perfect_citation_match():
    from src.evaluators.citation import score_citations

    scores = score_citations(_make_result(["doc1__c0001"]), _make_example(["doc1__c0001"]))
    assert scores["citation_precision"] == 1.0
    assert scores["citation_recall"] == 1.0


def test_partial_recall():
    from src.evaluators.citation import score_citations

    scores = score_citations(
        _make_result(["doc1__c0001"]),
        _make_example(["doc1__c0001", "doc1__c0002"]),
    )
    assert scores["citation_precision"] == 1.0
    assert scores["citation_recall"] == 0.5


def test_zero_overlap():
    from src.evaluators.citation import score_citations

    scores = score_citations(_make_result(["doc1__c0099"]), _make_example(["doc1__c0001"]))
    assert scores["citation_precision"] == 0.0
    assert scores["citation_recall"] == 0.0


def test_empty_predicted_citations():
    from src.evaluators.citation import score_citations

    scores = score_citations(_make_result([]), _make_example(["doc1__c0001"]))
    assert scores["citation_precision"] == 0.0
    assert scores["citation_recall"] == 0.0


def test_extra_citations_lower_precision():
    from src.evaluators.citation import score_citations

    scores = score_citations(
        _make_result(["doc1__c0001", "doc1__c0099"]),
        _make_example(["doc1__c0001"]),
    )
    assert scores["citation_precision"] == 0.5
    assert scores["citation_recall"] == 1.0


def test_exact_match_true():
    from src.evaluators.citation import score_citations

    scores = score_citations(
        _make_result([], answer="The total revenue was $100 billion for the year"),
        _make_example([], gold_answer="$100 billion"),
    )
    assert scores["exact_match"] is True


def test_exact_match_false():
    from src.evaluators.citation import score_citations

    scores = score_citations(
        _make_result([], answer="Revenue was $200 billion"),
        _make_example([], gold_answer="$100 billion"),
    )
    assert scores["exact_match"] is False


def test_exact_match_via_alias():
    from src.common.models import BenchmarkExample, GoldCitation
    from src.evaluators.citation import score_citations

    example = BenchmarkExample(
        example_id="q001",
        company="AAPL",
        question="Revenue?",
        question_type="numerical",
        difficulty="easy",
        answer_type="numeric",
        gold_answer="$383.3 billion",
        gold_citations=[],
        acceptable_aliases=["383.3B", "383.3 billion"],
        source_split="test",
    )
    result = _make_result([], answer="Apple revenue was 383.3B in 2023")
    scores = score_citations(result, example)
    assert scores["exact_match"] is True
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_citation.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.evaluators.citation'`

- [ ] **Step 3: Implement citation evaluator**

Create `src/evaluators/citation.py`:

```python
from typing import Any

from src.common.models import BenchmarkExample, PipelineResult


def score_citations(result: PipelineResult, example: BenchmarkExample) -> dict[str, Any]:
    predicted = set(result.citations)
    gold = {gc.chunk_id for gc in example.gold_citations}

    if not predicted:
        precision = 0.0
    else:
        precision = len(predicted & gold) / len(predicted)

    if not gold:
        recall = 1.0
    else:
        recall = len(predicted & gold) / len(gold)

    answer_lower = result.answer.lower()
    gold_lower = example.gold_answer.lower().strip()
    exact_match = gold_lower in answer_lower

    if not exact_match and example.acceptable_aliases:
        exact_match = any(alias.lower() in answer_lower for alias in example.acceptable_aliases)

    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "exact_match": exact_match,
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_citation.py -v
```

Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/evaluators/citation.py tests/test_citation.py
git commit -m "feat: add citation precision/recall and exact match evaluator"
```

---

## Task 11: Faithfulness evaluator + test_faithfulness.py

**Files:**
- Create: `src/evaluators/faithfulness.py`
- Create: `tests/test_faithfulness.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_faithfulness.py`:

```python
from src.common.models import Chunk, PipelineResult
from src.generation.azure_openai import MockGenerationAdapter


def _make_chunk(text: str = "Apple revenue was $383.3 billion in 2023.") -> Chunk:
    return Chunk(
        chunk_id="aapl_10k_2023__c0001",
        document_id="aapl_10k_2023",
        company="AAPL",
        text=text,
        token_count=len(text.split()),
    )


def _make_result(answer: str = "Apple revenue was $383.3 billion.") -> PipelineResult:
    return PipelineResult(
        example_id="q001",
        answer=answer,
        citations=["aapl_10k_2023__c0001"],
        latency_ms=100.0,
        prompt_tokens=50,
        completion_tokens=20,
    )


def test_faithful_above_threshold():
    from src.evaluators.faithfulness import score_faithfulness

    mock = MockGenerationAdapter(canned_score=0.9)
    scores = score_faithfulness(_make_result(), [_make_chunk()], mock)
    assert scores["faithfulness_score"] == 0.9
    assert scores["faithful"] is True


def test_unfaithful_below_threshold():
    from src.evaluators.faithfulness import score_faithfulness

    mock = MockGenerationAdapter(canned_score=0.3)
    scores = score_faithfulness(_make_result(), [_make_chunk()], mock)
    assert scores["faithfulness_score"] == 0.3
    assert scores["faithful"] is False


def test_exactly_at_threshold_is_faithful():
    from src.evaluators.faithfulness import score_faithfulness

    mock = MockGenerationAdapter(canned_score=0.5)
    scores = score_faithfulness(_make_result(), [_make_chunk()], mock)
    assert scores["faithful"] is True


def test_bad_json_response_returns_none():
    from src.generation.base import GenerationAdapter

    from src.evaluators.faithfulness import score_faithfulness

    class BadJudge(GenerationAdapter):
        def generate(self, example_id: str, question: str, chunks: list[Chunk]) -> PipelineResult:
            return PipelineResult(example_id=example_id, answer="", citations=[], latency_ms=0, prompt_tokens=0, completion_tokens=0)

        def judge(self, prompt: str) -> tuple[str, float, int, int]:
            return ("not valid json at all", 1.0, 0, 0)

    scores = score_faithfulness(_make_result(), [_make_chunk()], BadJudge())
    assert scores["faithfulness_score"] is None
    assert scores["faithful"] is False
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_faithfulness.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.evaluators.faithfulness'`

- [ ] **Step 3: Implement faithfulness evaluator**

Create `src/evaluators/faithfulness.py`:

```python
import json
import logging
from typing import TYPE_CHECKING, Any

from src.common.models import Chunk, PipelineResult

if TYPE_CHECKING:
    from src.generation.base import GenerationAdapter

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT_V1 = (
    "You are a faithfulness evaluator for financial question answering.\n\n"
    "Given the retrieved context passages and a generated answer, assess whether "
    "every claim in the answer is supported by the context.\n\n"
    "Context:\n{context}\n\n"
    "Answer:\n{answer}\n\n"
    "Rate the faithfulness of the answer on a scale from 0.0 to 1.0, where:\n"
    "- 1.0 means every claim is directly supported by the context\n"
    "- 0.0 means the answer makes claims not found in the context\n\n"
    'Respond with only a JSON object: {{"score": <float between 0 and 1>}}'
)

FAITHFULNESS_PROMPT_VERSION = "faithfulness-v1"
FAITHFULNESS_THRESHOLD = 0.5


def score_faithfulness(
    result: PipelineResult,
    chunks: list[Chunk],
    adapter: "GenerationAdapter",
) -> dict[str, Any]:
    context = "\n\n".join(c.text for c in chunks)
    prompt = FAITHFULNESS_PROMPT_V1.format(context=context, answer=result.answer)
    try:
        response_text, _, _, _ = adapter.judge(prompt)
        parsed = json.loads(response_text)
        score = float(parsed["score"])
        return {"faithfulness_score": score, "faithful": score >= FAITHFULNESS_THRESHOLD}
    except Exception:
        logger.warning("Faithfulness scoring failed for example_id=%s", result.example_id)
        return {"faithfulness_score": None, "faithful": False}
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_faithfulness.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/evaluators/faithfulness.py tests/test_faithfulness.py
git commit -m "feat: add LLM-based faithfulness evaluator with versioned prompt"
```

---

## Task 12: Evaluator runner

**Files:**
- Create: `src/evaluators/runner.py`

- [ ] **Step 1: Create evaluator runner**

```python
# src/evaluators/runner.py
from src.common.models import BenchmarkExample, Chunk, PipelineResult, RunConfig, ScoredRow
from src.evaluators.citation import score_citations
from src.evaluators.faithfulness import score_faithfulness
from src.evaluators.latency import score_latency
from src.generation.base import GenerationAdapter


def evaluate(
    result: PipelineResult,
    example: BenchmarkExample,
    chunks: list[Chunk],
    judge: GenerationAdapter,
    config: RunConfig,  # noqa: ARG001 — reserved for future per-run scorer config
) -> ScoredRow:
    latency = score_latency(result)
    citations = score_citations(result, example)
    faithfulness = score_faithfulness(result, chunks, judge)

    return ScoredRow(
        example_id=result.example_id,
        answer=result.answer,
        citations=result.citations,
        **latency,
        **citations,
        **faithfulness,
    )
```

- [ ] **Step 2: Verify runner imports**

```bash
python -c "from src.evaluators.runner import evaluate; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/evaluators/runner.py && git commit -m "feat: add evaluator runner that orchestrates latency, citation, faithfulness"
```

---

## Task 13: Leaderboard artifacts + test_artifacts.py

**Files:**
- Create: `src/leaderboard/artifacts.py`
- Create: `tests/test_artifacts.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_artifacts.py`:

```python
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.common.models import RunArtifact, ScoredRow


def _make_row(example_id: str = "q001") -> ScoredRow:
    return ScoredRow(
        example_id=example_id,
        answer="Revenue was $383.3 billion.",
        citations=["aapl_10k_2023__c0001"],
        latency_ms=100.0,
        prompt_tokens=50,
        completion_tokens=10,
        cost_usd=0.0008,
        exact_match=True,
        citation_precision=1.0,
        citation_recall=1.0,
        faithful=True,
        faithfulness_score=0.9,
    )


def _make_artifact(run_id: str = "test-001", n_rows: int = 2) -> RunArtifact:
    from src.leaderboard.artifacts import compute_aggregate

    rows = [_make_row(f"q{i:03d}") for i in range(1, n_rows + 1)]
    return RunArtifact(
        run_id=run_id,
        timestamp=datetime(2026, 4, 3, tzinfo=timezone.utc),
        config_snapshot={"run_id": run_id, "retriever": "bm25"},
        scores=rows,
        aggregate=compute_aggregate(rows),
    )


def test_write_artifact_creates_expected_files():
    from src.leaderboard.artifacts import write_artifact

    artifact = _make_artifact()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = write_artifact(artifact, Path(tmpdir))
        assert (out_dir / "artifact.json").exists()
        assert (out_dir / "scores.jsonl").exists()
        assert (out_dir / "config.json").exists()


def test_write_artifact_scores_jsonl_line_count():
    from src.leaderboard.artifacts import write_artifact

    artifact = _make_artifact(n_rows=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = write_artifact(artifact, Path(tmpdir))
        lines = (out_dir / "scores.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3


def test_validate_artifact_roundtrip():
    from src.leaderboard.artifacts import validate_artifact, write_artifact

    artifact = _make_artifact(run_id="roundtrip-001")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = write_artifact(artifact, Path(tmpdir))
        reloaded = validate_artifact(out_dir / "artifact.json")
        assert reloaded.run_id == "roundtrip-001"
        assert len(reloaded.scores) == 2


def test_compute_aggregate_values():
    from src.leaderboard.artifacts import compute_aggregate

    rows = [_make_row("q001"), _make_row("q002")]
    agg = compute_aggregate(rows)
    assert agg["n_examples"] == 2
    assert agg["exact_match_rate"] == 1.0
    assert agg["mean_faithfulness_score"] == 0.9
    assert agg["pct_faithful"] == 1.0
    assert agg["mean_citation_precision"] == 1.0
    assert agg["mean_citation_recall"] == 1.0
    assert agg["total_cost_usd"] == pytest.approx(0.0016, abs=1e-6)


def test_compute_aggregate_empty():
    from src.leaderboard.artifacts import compute_aggregate

    agg = compute_aggregate([])
    assert agg["n_examples"] == 0
```

Add `import pytest` at the top of the test file.

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_artifacts.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.leaderboard.artifacts'`

- [ ] **Step 3: Implement artifacts.py**

Create `src/leaderboard/artifacts.py`:

```python
import json
import statistics
from pathlib import Path
from typing import Any

from src.common.models import RunArtifact, ScoredRow


def compute_aggregate(scores: list[ScoredRow]) -> dict[str, Any]:
    n = len(scores)
    if n == 0:
        return {"n_examples": 0}

    faithfulness_scores = [s.faithfulness_score for s in scores if s.faithfulness_score is not None]

    return {
        "n_examples": n,
        "exact_match_rate": sum(s.exact_match for s in scores) / n,
        "mean_faithfulness_score": statistics.mean(faithfulness_scores) if faithfulness_scores else 0.0,
        "pct_faithful": sum(s.faithful for s in scores) / n,
        "mean_citation_precision": statistics.mean(s.citation_precision for s in scores),
        "mean_citation_recall": statistics.mean(s.citation_recall for s in scores),
        "median_latency_ms": statistics.median(s.latency_ms for s in scores),
        "total_cost_usd": round(sum(s.cost_usd for s in scores), 6),
    }


def write_artifact(run: RunArtifact, output_dir: Path) -> Path:
    run_dir = output_dir / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "artifact.json").write_text(run.model_dump_json(indent=2))

    with (run_dir / "scores.jsonl").open("w") as f:
        for row in run.scores:
            f.write(row.model_dump_json() + "\n")

    (run_dir / "config.json").write_text(json.dumps(run.config_snapshot, indent=2))

    return run_dir


def validate_artifact(path: Path) -> RunArtifact:
    data = json.loads(path.read_text())
    return RunArtifact.model_validate(data)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_artifacts.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/leaderboard/artifacts.py tests/test_artifacts.py
git commit -m "feat: add artifact writer, validator, and aggregate computation"
```

---

## Task 14: Leaderboard summary

**Files:**
- Create: `src/leaderboard/summary.py`
- Create: `tests/test_summary.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_summary.py`:

```python
from datetime import datetime, timezone

from src.common.models import RunArtifact, ScoredRow
from src.leaderboard.artifacts import compute_aggregate


def _make_artifact() -> RunArtifact:
    row = ScoredRow(
        example_id="q001",
        answer="Revenue was $383.3 billion.",
        citations=["aapl_10k_2023__c0001"],
        latency_ms=250.0,
        prompt_tokens=100,
        completion_tokens=30,
        cost_usd=0.0019,
        exact_match=True,
        citation_precision=1.0,
        citation_recall=1.0,
        faithful=True,
        faithfulness_score=0.92,
    )
    return RunArtifact(
        run_id="smoke-001",
        timestamp=datetime(2026, 4, 3, tzinfo=timezone.utc),
        config_snapshot={},
        scores=[row],
        aggregate=compute_aggregate([row]),
    )


def test_summary_contains_run_id():
    from src.leaderboard.summary import generate_summary

    summary = generate_summary(_make_artifact())
    assert "smoke-001" in summary


def test_summary_contains_metric_labels():
    from src.leaderboard.summary import generate_summary

    summary = generate_summary(_make_artifact())
    assert "Exact Match" in summary
    assert "Faithfulness" in summary
    assert "Citation" in summary
    assert "Latency" in summary
    assert "Cost" in summary
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_summary.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.leaderboard.summary'`

- [ ] **Step 3: Implement summary.py**

Create `src/leaderboard/summary.py`:

```python
from src.common.models import RunArtifact


def generate_summary(artifact: RunArtifact) -> str:
    agg = artifact.aggregate
    n = agg.get("n_examples", 0)

    lines = [
        f"## Run: {artifact.run_id}",
        f"**Timestamp:** {artifact.timestamp.isoformat()}  |  **Examples:** {n}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Exact Match Rate | {agg.get('exact_match_rate', 0):.1%} |",
        f"| Mean Faithfulness Score | {agg.get('mean_faithfulness_score', 0):.3f} |",
        f"| % Faithful | {agg.get('pct_faithful', 0):.1%} |",
        f"| Mean Citation Precision | {agg.get('mean_citation_precision', 0):.3f} |",
        f"| Mean Citation Recall | {agg.get('mean_citation_recall', 0):.3f} |",
        f"| Median Latency (ms) | {agg.get('median_latency_ms', 0):.0f} |",
        f"| Total Cost (USD) | ${agg.get('total_cost_usd', 0):.4f} |",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_summary.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/leaderboard/summary.py tests/test_summary.py
git commit -m "feat: add leaderboard summary Markdown table generator"
```

---

## Task 15: Create configs

**Files:**
- Create: `configs/smoke.yaml`
- Create: `configs/full.yaml`

- [ ] **Step 1: Create configs/ directory and smoke.yaml**

```bash
mkdir -p configs
```

Create `configs/smoke.yaml`:

```yaml
run_id: smoke-001
dataset_path: datasets/sample/qa_examples.jsonl
chunks_path: datasets/sample/chunks.jsonl
retriever: bm25
top_k: 3
generation_adapter: mock
output_dir: outputs/
```

Note: `generation_adapter: mock` for local runs without Azure credentials. Change to `azure_openai` when real scoring is needed.

- [ ] **Step 2: Create full.yaml**

Create `configs/full.yaml`:

```yaml
run_id: full-001
dataset_path: datasets/full/qa_examples.jsonl
chunks_path: datasets/full/chunks.jsonl
retriever: bm25
top_k: 5
generation_adapter: azure_openai
output_dir: outputs/
```

- [ ] **Step 3: Verify smoke.yaml parses correctly**

```bash
python -c "
from src.common.config import load_run_config
cfg = load_run_config('configs/smoke.yaml')
print(cfg)
"
```

Expected: `RunConfig(run_id='smoke-001', ...)`

- [ ] **Step 4: Commit**

```bash
git add configs/ && git commit -m "feat: add smoke and full run configs"
```

---

## Task 16: Rewrite CLI run_eval.py

**Files:**
- Rewrite: `src/cli/run_eval.py`

- [ ] **Step 1: Rewrite run_eval.py as Typer app**

Replace the full contents of `src/cli/run_eval.py` with:

```python
import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track

from src.common.config import load_run_config
from src.common.models import BenchmarkExample, Chunk, RunArtifact
from src.evaluators.faithfulness import FAITHFULNESS_PROMPT_VERSION
from src.evaluators.runner import evaluate
from src.leaderboard.artifacts import compute_aggregate, write_artifact
from src.leaderboard.summary import generate_summary
from src.retrieval.bm25 import BM25Retriever

app = typer.Typer()
console = Console()


@app.command()
def main(config: str = typer.Option(..., "--config", help="Path to YAML run config")) -> None:
    run_config = load_run_config(config)

    examples: list[BenchmarkExample] = []
    with open(run_config.dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(BenchmarkExample.model_validate_json(line))

    chunks: list[Chunk] = []
    with open(run_config.chunks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))

    retriever = BM25Retriever(chunks)

    if run_config.generation_adapter == "mock":
        from src.generation.azure_openai import MockGenerationAdapter
        adapter = MockGenerationAdapter()
    elif run_config.generation_adapter == "azure_openai":
        from src.generation.azure_openai import AzureOpenAIAdapter
        adapter = AzureOpenAIAdapter()
    else:
        raise ValueError(f"Unknown generation_adapter: {run_config.generation_adapter}")

    config_snapshot = {
        **run_config.model_dump(),
        "faithfulness_prompt_version": FAITHFULNESS_PROMPT_VERSION,
    }

    scored_rows = []
    for example in track(examples, description="Evaluating..."):
        retrieved = retriever.retrieve(example.question, top_k=run_config.top_k)
        result = adapter.generate(
            example_id=example.example_id,
            question=example.question,
            chunks=retrieved,
        )
        row = evaluate(
            result=result,
            example=example,
            chunks=retrieved,
            judge=adapter,
            config=run_config,
        )
        scored_rows.append(row)

    aggregate = compute_aggregate(scored_rows)
    artifact = RunArtifact(
        run_id=run_config.run_id,
        timestamp=datetime.now(timezone.utc),
        config_snapshot=config_snapshot,
        scores=scored_rows,
        aggregate=aggregate,
    )

    out_dir = write_artifact(artifact, Path(run_config.output_dir))
    console.print(generate_summary(artifact))
    console.print(f"\nArtifact saved to: {out_dir}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Verify CLI help text works**

```bash
python -m src.cli.run_eval --help
```

Expected: shows `--config` option description.

- [ ] **Step 3: Commit**

```bash
git add src/cli/run_eval.py && git commit -m "feat: rewrite run_eval as Typer app with full eval pipeline"
```

---

## Task 17: Add summarize_run CLI

**Files:**
- Create: `src/cli/summarize_run.py`

- [ ] **Step 1: Create summarize_run.py**

```python
# src/cli/summarize_run.py
from pathlib import Path

import typer
from rich.console import Console

from src.leaderboard.artifacts import validate_artifact
from src.leaderboard.summary import generate_summary

app = typer.Typer()
console = Console()


@app.command()
def main(run_dir: str = typer.Option(..., "--run-dir", help="Path to run output directory")) -> None:
    artifact_path = Path(run_dir) / "artifact.json"
    if not artifact_path.exists():
        console.print(f"[red]artifact.json not found in {run_dir}[/red]")
        raise typer.Exit(code=1)
    artifact = validate_artifact(artifact_path)
    console.print(generate_summary(artifact))


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Verify CLI help**

```bash
python -m src.cli.summarize_run --help
```

Expected: shows `--run-dir` option.

- [ ] **Step 3: Commit**

```bash
git add src/cli/summarize_run.py && git commit -m "feat: add summarize_run CLI for reprinting run summaries"
```

---

## Task 18: Golden fixture + full test suite + smoke run

**Files:**
- Create: `tests/fixtures/golden_run.json`
- Create: `tests/test_golden.py`

- [ ] **Step 1: Create golden_run.json**

Create `tests/fixtures/golden_run.json`:

```json
{
  "run_id": "golden-001",
  "timestamp": "2026-04-03T00:00:00Z",
  "config_snapshot": {
    "run_id": "golden-001",
    "dataset_path": "datasets/sample/qa_examples.jsonl",
    "chunks_path": "datasets/sample/chunks.jsonl",
    "retriever": "bm25",
    "top_k": 3,
    "generation_adapter": "mock",
    "output_dir": "outputs/",
    "faithfulness_prompt_version": "faithfulness-v1"
  },
  "scores": [
    {
      "example_id": "q001",
      "answer": "$383.3 billion",
      "citations": ["aapl_10k_2023__c0001", "aapl_10k_2023__c0002", "aapl_10k_2023__c0000"],
      "latency_ms": 1.0,
      "prompt_tokens": 10,
      "completion_tokens": 5,
      "cost_usd": 0.00025,
      "exact_match": true,
      "citation_precision": 0.333,
      "citation_recall": 1.0,
      "faithful": true,
      "faithfulness_score": 0.9
    }
  ],
  "aggregate": {
    "n_examples": 1,
    "exact_match_rate": 1.0,
    "mean_faithfulness_score": 0.9,
    "pct_faithful": 1.0,
    "mean_citation_precision": 0.333,
    "mean_citation_recall": 1.0,
    "median_latency_ms": 1.0,
    "total_cost_usd": 0.00025
  }
}
```

- [ ] **Step 2: Create test_golden.py**

Create `tests/test_golden.py`:

```python
import json
from pathlib import Path

from src.common.models import RunArtifact


def test_golden_run_artifact_schema_valid():
    path = Path("tests/fixtures/golden_run.json")
    data = json.loads(path.read_text())
    artifact = RunArtifact.model_validate(data)
    assert artifact.run_id == "golden-001"
    assert len(artifact.scores) >= 1
    assert artifact.aggregate["n_examples"] >= 1
```

- [ ] **Step 3: Run full test suite**

```bash
pytest -q
```

Expected: all tests pass, no failures.

- [ ] **Step 4: Run smoke benchmark end-to-end**

```bash
python -m src.cli.run_eval --config configs/smoke.yaml
```

Expected:
- Progress bar shows 5 examples evaluated
- Summary Markdown table printed to terminal
- `outputs/smoke-001/artifact.json` created
- `outputs/smoke-001/scores.jsonl` has 5 lines
- `outputs/smoke-001/config.json` exists

- [ ] **Step 5: Verify summarize CLI works on the run output**

```bash
python -m src.cli.summarize_run --run-dir outputs/smoke-001
```

Expected: same summary table reprinted.

- [ ] **Step 6: Run ruff lint**

```bash
ruff check . && ruff format --check .
```

Fix any lint errors before committing.

- [ ] **Step 7: Final commit**

```bash
git add tests/fixtures/golden_run.json tests/test_golden.py outputs/.gitkeep
git commit -m "feat: add golden regression fixture, complete test suite, verify smoke run"
```

---

## Verification Checklist

After all tasks complete:

- [ ] `pytest -q` — all tests pass
- [ ] `python -m src.cli.run_eval --config configs/smoke.yaml` — exits 0, writes artifact
- [ ] `python -m src.cli.summarize_run --run-dir outputs/smoke-001` — prints summary table
- [ ] `python -c "from src.common.models import RunConfig, PipelineResult, ScoredRow, RunArtifact; print('ok')`
- [ ] `ruff check .` — no lint errors
- [ ] `outputs/smoke-001/artifact.json` validates against `RunArtifact` schema
