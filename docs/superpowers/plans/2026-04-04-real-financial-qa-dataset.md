# Real Financial QA Dataset — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that fetches real SEC 10-K/10-Q filings, chunks them, generates LLM Q&A candidates per chunk, and provides an interactive review CLI that writes approved candidates to `datasets/full/qa_examples.jsonl`.

**Architecture:** Three new `src/ingest/` modules (text_extract, qa_generator, dataset_builder) handle extraction, candidate generation, and file I/O. A `scripts/build_dataset.py` Typer CLI ties them together with `generate` and `review` subcommands. All schemas live in `src/common/models.py`; config loading in `src/common/config.py`.

**Tech Stack:** Python 3.11+, Pydantic v2, Typer, rich, tiktoken, rank-bm25, python-dotenv, requests, html.parser (stdlib)

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `src/common/models.py` | Modify | Add `QACandidate` and `DatasetBuildConfig` |
| `src/common/config.py` | Modify | Add `load_dataset_build_config()` |
| `src/ingest/text_extract.py` | Create | Strip HTML from raw SEC filings |
| `src/ingest/qa_generator.py` | Create | Chunk filtering + LLM Q&A generation |
| `src/ingest/dataset_builder.py` | Create | JSONL I/O, dedup, candidate→example mapping |
| `configs/dataset_build.yaml` | Create | Tickers, years, settings |
| `.gitignore` | Modify | Ignore `datasets/full/` |
| `scripts/build_dataset.py` | Create | `generate` and `review` Typer subcommands |
| `tests/test_models.py` | Modify | Add `QACandidate` and `DatasetBuildConfig` round-trip tests |
| `tests/test_text_extract.py` | Create | HTML stripping edge cases |
| `tests/test_qa_generator.py` | Create | Chunk filter + generation with mock adapter |
| `tests/test_dataset_builder.py` | Create | Dedup, mapping, JSONL helpers |
| `tests/fixtures/sample_candidates.jsonl` | Create | 5 candidates for test fixture |

---

## Task 1: QACandidate + DatasetBuildConfig schemas

**Files:**
- Modify: `src/common/models.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

Add to the bottom of `tests/test_models.py`:

```python
def test_qa_candidate_round_trip():
    from src.common.models import QACandidate

    c = QACandidate(
        candidate_id="aapl_10k_2023__c0001-q0",
        chunk_id="aapl_10k_2023__c0001",
        document_id="aapl_10k_2023",
        company="AAPL",
        question="What were Apple's total net sales for fiscal 2023?",
        gold_answer="$383.3 billion",
        difficulty="easy",
        question_type="numerical",
        gold_citations=["aapl_10k_2023__c0001"],
    )
    assert QACandidate.model_validate_json(c.model_dump_json()) == c
    assert c.review_status == "pending"
    assert c.reviewer_note == ""


def test_qa_candidate_difficulty_values():
    from src.common.models import QACandidate
    import pytest

    with pytest.raises(Exception):
        QACandidate(
            candidate_id="x-q0",
            chunk_id="x",
            document_id="x",
            company="X",
            question="q",
            gold_answer="a",
            difficulty="invalid",
            question_type="factual",
            gold_citations=["x"],
        )


def test_dataset_build_config_round_trip():
    from src.common.models import DatasetBuildConfig

    cfg = DatasetBuildConfig(
        tickers=["AAPL", "MSFT"],
        fiscal_years=[2023, 2024],
        output_dir="datasets/full/",
    )
    assert DatasetBuildConfig.model_validate_json(cfg.model_dump_json()) == cfg
    assert cfg.questions_per_chunk == 2
    assert cfg.generation_adapter == "azure_openai"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models.py::test_qa_candidate_round_trip tests/test_models.py::test_dataset_build_config_round_trip -v
```

Expected: FAIL with `ImportError` or `ValidationError` — `QACandidate` doesn't exist yet.

- [ ] **Step 3: Add schemas to models.py**

Add the following imports at the top of `src/common/models.py` (after existing imports):

```python
from typing import Literal
```

Then append these two classes at the bottom of `src/common/models.py`:

```python
class QACandidate(BaseModel):
    candidate_id: str
    chunk_id: str
    document_id: str
    company: str
    question: str
    gold_answer: str
    difficulty: Literal["easy", "medium", "hard"]
    question_type: Literal["factual", "numerical", "comparative", "multi_hop"]
    gold_citations: list[str]
    review_status: Literal["pending", "approved", "rejected", "edited"] = "pending"
    reviewer_note: str = ""


class DatasetBuildConfig(BaseModel):
    tickers: list[str]
    fiscal_years: list[int]
    form_types: list[str] = ["10-K", "10-Q"]
    sections: list[str] = ["Risk Factors", "MD&A", "Financial Statements"]
    top_k_chunks_per_section: int = 5
    questions_per_chunk: int = 2
    difficulty_mix: dict[str, float] = {"easy": 0.5, "medium": 0.3, "hard": 0.2}
    output_dir: str = "datasets/full/"
    generation_adapter: str = "azure_openai"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: all tests PASS (including 3 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/common/models.py tests/test_models.py
git commit -m "feat: add QACandidate and DatasetBuildConfig schemas"
```

---

## Task 2: HTML text extraction

**Files:**
- Create: `src/ingest/text_extract.py`
- Create: `tests/test_text_extract.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_text_extract.py`:

```python
def test_strip_html_removes_tags():
    from src.ingest.text_extract import strip_html

    result = strip_html("<p>Hello <b>world</b></p>")
    assert "Hello" in result
    assert "world" in result
    assert "<" not in result
    assert ">" not in result


def test_strip_html_empty_string():
    from src.ingest.text_extract import strip_html

    assert strip_html("") == ""


def test_strip_html_plain_text_unchanged():
    from src.ingest.text_extract import strip_html

    text = "Net sales increased 2% to $391.0 billion."
    assert strip_html(text) == text


def test_strip_html_collapses_extra_spaces():
    from src.ingest.text_extract import strip_html

    result = strip_html("<p>Net   sales</p>")
    assert "  " not in result


def test_strip_html_malformed():
    from src.ingest.text_extract import strip_html

    # Should not raise even with malformed HTML
    result = strip_html("<p>Unclosed tag <b>bold")
    assert "Unclosed tag" in result
    assert "bold" in result
    assert "<" not in result


def test_strip_html_preserves_financial_content():
    from src.ingest.text_extract import strip_html

    html = "<td>Revenue</td><td>$391.0 billion</td>"
    result = strip_html(html)
    assert "Revenue" in result
    assert "$391.0 billion" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_text_extract.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement text_extract.py**

Create `src/ingest/text_extract.py`:

```python
import re
from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def strip_html(html: str) -> str:
    """Strip HTML tags and return plain text.

    Collapses multiple spaces. Preserves all text content including
    numbers, percentages, and currency values.
    """
    if not html:
        return ""
    extractor = _TextExtractor()
    extractor.feed(html)
    text = extractor.get_text()
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_text_extract.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingest/text_extract.py tests/test_text_extract.py
git commit -m "feat: add HTML text extraction for SEC filings"
```

---

## Task 3: QA candidate generation

**Files:**
- Create: `src/ingest/qa_generator.py`
- Create: `tests/test_qa_generator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_qa_generator.py`:

```python
import json
import pytest
from src.common.models import Chunk, DatasetBuildConfig, QACandidate
from src.generation.azure_openai import MockGenerationAdapter


# A mock that returns controlled Q&A JSON from judge()
class _QAMockAdapter(MockGenerationAdapter):
    def __init__(self, response: str) -> None:
        super().__init__()
        self._response = response

    def judge(self, prompt: str) -> tuple[str, float, int, int]:
        return (self._response, 1.0, 10, 5)


def _make_chunk(token_count: int = 120, section_title: str | None = "MD&A",
                text: str = "Revenue was $391.0 billion, up 5%.") -> Chunk:
    return Chunk(
        chunk_id="aapl_10k_2023__c0001",
        document_id="aapl_10k_2023",
        company="AAPL",
        section_title=section_title,
        text=text,
        token_count=token_count,
        report_period_end="2023-09-30",
        filing_date="2023-11-03",
    )


def _default_config() -> DatasetBuildConfig:
    return DatasetBuildConfig(
        tickers=["AAPL"],
        fiscal_years=[2023],
        sections=["MD&A", "Risk Factors"],
    )


# --- is_interesting_chunk tests ---

def test_interesting_chunk_passes():
    from src.ingest.qa_generator import is_interesting_chunk

    chunk = _make_chunk(token_count=120)
    assert is_interesting_chunk(chunk, ["MD&A"]) is True


def test_interesting_chunk_too_short():
    from src.ingest.qa_generator import is_interesting_chunk

    chunk = _make_chunk(token_count=50)
    assert is_interesting_chunk(chunk, ["MD&A"]) is False


def test_interesting_chunk_no_financial_signal():
    from src.ingest.qa_generator import is_interesting_chunk

    chunk = _make_chunk(text="The company focuses on innovation and growth in markets.")
    assert is_interesting_chunk(chunk, ["MD&A"]) is False


def test_interesting_chunk_wrong_section():
    from src.ingest.qa_generator import is_interesting_chunk

    chunk = _make_chunk(section_title="Cover Page")
    assert is_interesting_chunk(chunk, ["MD&A", "Risk Factors"]) is False


def test_interesting_chunk_no_section_passes():
    """Chunk with section_title=None is not filtered by section."""
    from src.ingest.qa_generator import is_interesting_chunk

    chunk = _make_chunk(section_title=None, token_count=120)
    assert is_interesting_chunk(chunk, ["MD&A"]) is True


# --- generate_candidates tests ---

def test_generate_candidates_success():
    from src.ingest.qa_generator import generate_candidates

    response = json.dumps([
        {"question": "What was revenue?", "answer": "$391.0 billion",
         "difficulty": "easy", "type": "numerical"},
        {"question": "By how much did revenue grow?", "answer": "5%",
         "difficulty": "medium", "type": "numerical"},
    ])
    adapter = _QAMockAdapter(response)
    chunk = _make_chunk()
    cfg = _default_config()
    candidates = generate_candidates(chunk, adapter, cfg)

    assert len(candidates) == 2
    assert all(isinstance(c, QACandidate) for c in candidates)
    assert candidates[0].candidate_id == "aapl_10k_2023__c0001-q0"
    assert candidates[0].gold_citations == ["aapl_10k_2023__c0001"]
    assert candidates[0].review_status == "pending"
    assert candidates[1].candidate_id == "aapl_10k_2023__c0001-q1"


def test_generate_candidates_invalid_json_returns_empty():
    from src.ingest.qa_generator import generate_candidates

    adapter = _QAMockAdapter("not valid json")
    chunk = _make_chunk()
    cfg = _default_config()
    candidates = generate_candidates(chunk, adapter, cfg)
    assert candidates == []


def test_generate_candidates_empty_array_returns_empty():
    from src.ingest.qa_generator import generate_candidates

    adapter = _QAMockAdapter("[]")
    chunk = _make_chunk()
    cfg = _default_config()
    candidates = generate_candidates(chunk, adapter, cfg)
    assert candidates == []


def test_generate_candidates_prompt_contains_chunk_text():
    """Verify the prompt sent to the adapter contains chunk text."""
    from src.ingest.qa_generator import generate_candidates

    captured_prompts: list[str] = []

    class _CapturingAdapter(_QAMockAdapter):
        def judge(self, prompt: str) -> tuple[str, float, int, int]:
            captured_prompts.append(prompt)
            return ("[]", 1.0, 0, 0)

    chunk = _make_chunk(text="Revenue was $391.0 billion, up 5%.")
    generate_candidates(chunk, _CapturingAdapter("[]"), _default_config())
    assert len(captured_prompts) == 1
    assert "Revenue was $391.0 billion" in captured_prompts[0]
    assert "AAPL" in captured_prompts[0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_qa_generator.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement qa_generator.py**

Create `src/ingest/qa_generator.py`:

```python
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from src.common.models import Chunk, DatasetBuildConfig, QACandidate

if TYPE_CHECKING:
    from src.generation.base import GenerationAdapter

logger = logging.getLogger(__name__)

QA_GENERATION_PROMPT_V1 = """\
Given this excerpt from {company}'s {form_type} filing ({period}):

<chunk>
{text}
</chunk>

Generate {n} question-answer pairs. For each pair:
- The question must be answerable solely from the excerpt
- The answer must be a direct quote or close paraphrase from the excerpt
- Assign difficulty: easy (single fact lookup), medium (requires inference \
or calculation), hard (requires comparing multiple facts or multi-step reasoning)
- Assign type: factual | numerical | comparative | multi_hop

Return a JSON array only, no other text:
[{{"question": "...", "answer": "...", \
"difficulty": "easy|medium|hard", "type": "factual|numerical|comparative|multi_hop"}}]
"""

QA_GENERATION_PROMPT_VERSION = "qa-gen-v1"

_FINANCIAL_SIGNAL = re.compile(r"[\d%$£€]")


def is_interesting_chunk(chunk: Chunk, sections: list[str]) -> bool:
    """Return True if the chunk is worth generating Q&A from.

    Filters out:
    - Chunks with fewer than 80 tokens (too short for a real question)
    - Chunks with no financial signal (numbers, %, currency symbols)
    - Chunks whose section_title is set but not in the allowed sections list
    """
    if chunk.token_count < 80:
        return False
    if not _FINANCIAL_SIGNAL.search(chunk.text):
        return False
    if chunk.section_title is not None and chunk.section_title not in sections:
        return False
    return True


def generate_candidates(
    chunk: Chunk,
    adapter: GenerationAdapter,
    config: DatasetBuildConfig,
    form_type: str = "10-K",
    period: str = "",
) -> list[QACandidate]:
    """Generate QA candidates for a single chunk using the adapter.

    Returns an empty list if the LLM response cannot be parsed.
    """
    prompt = QA_GENERATION_PROMPT_V1.format(
        company=chunk.company,
        form_type=form_type,
        period=period or chunk.report_period_end or "unknown period",
        text=chunk.text,
        n=config.questions_per_chunk,
    )
    try:
        raw, _, _, _ = adapter.judge(prompt)
        pairs = json.loads(raw)
    except Exception:
        logger.warning("Failed to parse QA candidates for chunk %s", chunk.chunk_id)
        return []

    candidates: list[QACandidate] = []
    for i, pair in enumerate(pairs):
        try:
            candidates.append(
                QACandidate(
                    candidate_id=f"{chunk.chunk_id}-q{i}",
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    company=chunk.company,
                    question=pair["question"],
                    gold_answer=pair["answer"],
                    difficulty=pair["difficulty"],
                    question_type=pair["type"],
                    gold_citations=[chunk.chunk_id],
                    review_status="pending",
                )
            )
        except Exception:
            logger.warning(
                "Skipping malformed candidate %d for chunk %s", i, chunk.chunk_id
            )
    return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_qa_generator.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingest/qa_generator.py tests/test_qa_generator.py
git commit -m "feat: add QA candidate generation with chunk filtering"
```

---

## Task 4: Dataset builder helpers + fixtures

**Files:**
- Create: `src/ingest/dataset_builder.py`
- Create: `tests/fixtures/sample_candidates.jsonl`
- Create: `tests/test_dataset_builder.py`

- [ ] **Step 1: Create fixture file**

Create `tests/fixtures/sample_candidates.jsonl` with this exact content (one JSON object per line):

```
{"candidate_id": "aapl_10k_2023__c0001-q0", "chunk_id": "aapl_10k_2023__c0001", "document_id": "aapl_10k_2023", "company": "AAPL", "question": "What were Apple's total net sales for fiscal 2023?", "gold_answer": "$383.3 billion", "difficulty": "easy", "question_type": "numerical", "gold_citations": ["aapl_10k_2023__c0001"], "review_status": "pending", "reviewer_note": ""}
{"candidate_id": "aapl_10k_2023__c0001-q1", "chunk_id": "aapl_10k_2023__c0001", "document_id": "aapl_10k_2023", "company": "AAPL", "question": "What percentage of net sales did iPhone represent?", "gold_answer": "approximately 52 percent", "difficulty": "easy", "question_type": "numerical", "gold_citations": ["aapl_10k_2023__c0001"], "review_status": "approved", "reviewer_note": ""}
{"candidate_id": "msft_10k_2023__c0001-q0", "chunk_id": "msft_10k_2023__c0001", "document_id": "msft_10k_2023", "company": "MSFT", "question": "What was Microsoft's revenue for fiscal year 2023?", "gold_answer": "$211.9 billion", "difficulty": "easy", "question_type": "numerical", "gold_citations": ["msft_10k_2023__c0001"], "review_status": "pending", "reviewer_note": ""}
{"candidate_id": "msft_10k_2023__c0001-q1", "chunk_id": "msft_10k_2023__c0001", "document_id": "msft_10k_2023", "company": "MSFT", "question": "What share of total revenue did Intelligent Cloud represent?", "gold_answer": "41 percent", "difficulty": "medium", "question_type": "numerical", "gold_citations": ["msft_10k_2023__c0001"], "review_status": "rejected", "reviewer_note": "ambiguous phrasing"}
{"candidate_id": "googl_10k_2023__c0001-q0", "chunk_id": "googl_10k_2023__c0001", "document_id": "googl_10k_2023", "company": "GOOGL", "question": "By what percentage did Alphabet's revenues grow in 2023?", "gold_answer": "9 percent", "difficulty": "easy", "question_type": "numerical", "gold_citations": ["googl_10k_2023__c0001"], "review_status": "pending", "reviewer_note": ""}
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_dataset_builder.py`:

```python
import json
from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures"
CANDIDATES_FIXTURE = FIXTURES / "sample_candidates.jsonl"


def test_load_existing_chunk_ids_empty(tmp_path):
    from src.ingest.dataset_builder import load_existing_chunk_ids

    assert load_existing_chunk_ids(tmp_path / "missing.jsonl") == set()


def test_load_existing_chunk_ids_from_fixture():
    from src.ingest.dataset_builder import load_existing_chunk_ids

    ids = load_existing_chunk_ids(CANDIDATES_FIXTURE)
    assert "aapl_10k_2023__c0001" in ids
    assert "msft_10k_2023__c0001" in ids
    assert "googl_10k_2023__c0001" in ids
    assert len(ids) == 3  # 3 unique chunk_ids across 5 candidates


def test_load_existing_document_ids_empty(tmp_path):
    from src.ingest.dataset_builder import load_existing_document_ids

    assert load_existing_document_ids(tmp_path / "missing.jsonl") == set()


def test_load_pending_candidates():
    from src.ingest.dataset_builder import load_pending_candidates

    pending = load_pending_candidates(CANDIDATES_FIXTURE)
    assert len(pending) == 3
    assert all(c.review_status == "pending" for c in pending)
    ids = {c.candidate_id for c in pending}
    assert "aapl_10k_2023__c0001-q0" in ids
    assert "msft_10k_2023__c0001-q0" in ids
    assert "googl_10k_2023__c0001-q0" in ids


def test_candidate_to_example_basic():
    from src.common.models import QACandidate
    from src.ingest.dataset_builder import candidate_to_example

    candidate = QACandidate(
        candidate_id="aapl_10k_2023__c0001-q0",
        chunk_id="aapl_10k_2023__c0001",
        document_id="aapl_10k_2023",
        company="AAPL",
        question="What were net sales?",
        gold_answer="$383.3 billion",
        difficulty="easy",
        question_type="numerical",
        gold_citations=["aapl_10k_2023__c0001"],
    )
    example = candidate_to_example(candidate, "q0001")

    assert example.example_id == "q0001"
    assert example.company == "AAPL"
    assert example.question == "What were net sales?"
    assert example.gold_answer == "$383.3 billion"
    assert example.difficulty == "easy"
    assert example.question_type == "numerical"
    assert len(example.gold_citations) == 1
    assert example.gold_citations[0].chunk_id == "aapl_10k_2023__c0001"
    assert example.gold_citations[0].document_id == "aapl_10k_2023"
    assert example.gold_citations[0].support_type == "direct"
    assert example.requires_multi_hop is False


def test_candidate_to_example_multi_hop():
    from src.common.models import QACandidate
    from src.ingest.dataset_builder import candidate_to_example

    candidate = QACandidate(
        candidate_id="x-q0",
        chunk_id="x__c0001",
        document_id="x",
        company="X",
        question="Compare two periods",
        gold_answer="revenue grew",
        difficulty="hard",
        question_type="multi_hop",
        gold_citations=["x__c0001", "x__c0002"],
    )
    example = candidate_to_example(candidate, "q0002")
    assert example.requires_multi_hop is True
    assert len(example.gold_citations) == 2


def test_append_jsonl_creates_file(tmp_path):
    from src.common.models import QACandidate
    from src.ingest.dataset_builder import append_jsonl

    c = QACandidate(
        candidate_id="x-q0",
        chunk_id="x__c0001",
        document_id="x",
        company="X",
        question="q",
        gold_answer="a",
        difficulty="easy",
        question_type="factual",
        gold_citations=["x__c0001"],
    )
    out = tmp_path / "out.jsonl"
    append_jsonl([c], out)
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["candidate_id"] == "x-q0"


def test_append_jsonl_appends(tmp_path):
    from src.common.models import QACandidate
    from src.ingest.dataset_builder import append_jsonl

    def _make(cid: str) -> QACandidate:
        return QACandidate(
            candidate_id=cid,
            chunk_id="x",
            document_id="x",
            company="X",
            question="q",
            gold_answer="a",
            difficulty="easy",
            question_type="factual",
            gold_citations=["x"],
        )

    out = tmp_path / "out.jsonl"
    append_jsonl([_make("x-q0")], out)
    append_jsonl([_make("x-q1")], out)
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 2


def test_update_candidate_status(tmp_path):
    from src.ingest.dataset_builder import update_candidate_status

    import shutil
    path = tmp_path / "candidates.jsonl"
    shutil.copy(CANDIDATES_FIXTURE, path)

    update_candidate_status(path, "aapl_10k_2023__c0001-q0", "approved", "looks good")

    lines = [l for l in path.read_text().splitlines() if l.strip()]
    updated = {json.loads(l)["candidate_id"]: json.loads(l) for l in lines}
    assert updated["aapl_10k_2023__c0001-q0"]["review_status"] == "approved"
    assert updated["aapl_10k_2023__c0001-q0"]["reviewer_note"] == "looks good"
    # Other records untouched
    assert updated["msft_10k_2023__c0001-q0"]["review_status"] == "pending"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_dataset_builder.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement dataset_builder.py**

Create `src/ingest/dataset_builder.py`:

```python
import json
from pathlib import Path

from src.common.models import BenchmarkExample, GoldCitation, QACandidate


def load_existing_chunk_ids(candidates_path: Path) -> set[str]:
    """Return set of chunk_ids already present in candidates_path."""
    if not candidates_path.exists():
        return set()
    ids: set[str] = set()
    with open(candidates_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["chunk_id"])
    return ids


def load_existing_document_ids(documents_path: Path) -> set[str]:
    """Return set of document_ids already present in documents_path."""
    if not documents_path.exists():
        return set()
    ids: set[str] = set()
    with open(documents_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["document_id"])
    return ids


def load_pending_candidates(candidates_path: Path) -> list[QACandidate]:
    """Load all candidates with review_status == 'pending'."""
    if not candidates_path.exists():
        return []
    result: list[QACandidate] = []
    with open(candidates_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if data.get("review_status") == "pending":
                    result.append(QACandidate(**data))
    return result


def candidate_to_example(candidate: QACandidate, example_id: str) -> BenchmarkExample:
    """Convert an approved QACandidate to a BenchmarkExample."""
    return BenchmarkExample(
        example_id=example_id,
        company=candidate.company,
        question=candidate.question,
        question_type=candidate.question_type,
        difficulty=candidate.difficulty,
        answer_type="text",
        gold_answer=candidate.gold_answer,
        gold_answer_normalized=None,
        normalization_unit=None,
        gold_citations=[
            GoldCitation(
                document_id=candidate.document_id,
                chunk_id=cid,
                support_type="direct",
            )
            for cid in candidate.gold_citations
        ],
        acceptable_aliases=[],
        time_sensitive=False,
        requires_multi_hop=candidate.question_type == "multi_hop",
        source_split="test",
    )


def append_jsonl(records: list, path: Path) -> None:
    """Append Pydantic model records to a JSONL file, creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


def update_candidate_status(
    candidates_path: Path,
    candidate_id: str,
    status: str,
    reviewer_note: str = "",
) -> None:
    """Update review_status and reviewer_note for a single candidate in-place."""
    lines = candidates_path.read_text().splitlines()
    updated: list[str] = []
    for line in lines:
        if not line.strip():
            updated.append(line)
            continue
        data = json.loads(line)
        if data["candidate_id"] == candidate_id:
            data["review_status"] = status
            data["reviewer_note"] = reviewer_note
            updated.append(json.dumps(data))
        else:
            updated.append(line)
    candidates_path.write_text("\n".join(updated) + "\n")


def update_candidate_full(candidates_path: Path, candidate: QACandidate) -> None:
    """Rewrite a candidate record in-place with all updated fields."""
    lines = candidates_path.read_text().splitlines()
    updated: list[str] = []
    for line in lines:
        if not line.strip():
            updated.append(line)
            continue
        data = json.loads(line)
        if data["candidate_id"] == candidate.candidate_id:
            updated.append(candidate.model_dump_json())
        else:
            updated.append(line)
    candidates_path.write_text("\n".join(updated) + "\n")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_dataset_builder.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 6: Run full test suite to check nothing broken**

```bash
pytest -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ingest/dataset_builder.py tests/test_dataset_builder.py tests/fixtures/sample_candidates.jsonl
git commit -m "feat: add dataset builder helpers and test fixtures"
```

---

## Task 5: Config file, .gitignore, and config loader

**Files:**
- Create: `configs/dataset_build.yaml`
- Modify: `.gitignore`
- Modify: `src/common/config.py`

- [ ] **Step 1: Create configs/dataset_build.yaml**

Create `configs/dataset_build.yaml`:

```yaml
tickers:
  - AAPL
  - MSFT
  - JPM
fiscal_years:
  - 2023
  - 2024
form_types:
  - 10-K
  - 10-Q
sections:
  - Risk Factors
  - MD&A
  - Financial Statements
top_k_chunks_per_section: 5
questions_per_chunk: 2
difficulty_mix:
  easy: 0.5
  medium: 0.3
  hard: 0.2
output_dir: datasets/full/
generation_adapter: azure_openai
```

- [ ] **Step 2: Add datasets/full/ to .gitignore**

Add this block to `.gitignore` after the `# Outputs` section:

```
# Generated dataset (fetched filings + LLM candidates)
datasets/full/
```

- [ ] **Step 3: Add load_dataset_build_config() to config.py**

Add to `src/common/config.py` (after the existing imports and `load_run_config` function):

```python
from src.common.models import DatasetBuildConfig


def load_dataset_build_config(path: str) -> DatasetBuildConfig:
    load_dotenv()
    with open(path) as f:
        data = yaml.safe_load(f)
    return DatasetBuildConfig(**data)
```

Note: `RunConfig` is already imported at the top of config.py; change that line to import both:

```python
from src.common.models import DatasetBuildConfig, RunConfig
```

- [ ] **Step 4: Verify config loads correctly**

```bash
python -c "
from src.common.config import load_dataset_build_config
cfg = load_dataset_build_config('configs/dataset_build.yaml')
print(cfg.tickers, cfg.fiscal_years, cfg.questions_per_chunk)
"
```

Expected output: `['AAPL', 'MSFT', 'JPM'] [2023, 2024] 2`

- [ ] **Step 5: Run tests to verify nothing broken**

```bash
pytest -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/dataset_build.yaml .gitignore src/common/config.py
git commit -m "feat: add dataset build config file and loader"
```

---

## Task 6: build_dataset.py — generate and review subcommands

**Files:**
- Create: `scripts/build_dataset.py`

- [ ] **Step 1: Implement the full CLI script**

Create `scripts/build_dataset.py`:

```python
"""Dataset construction CLI for FinRAG Eval.

Commands:
  generate  Fetch SEC filings, chunk, and generate QA candidates.
  review    Interactively review candidates and write approved examples.
"""

import time
from datetime import UTC, datetime
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console

from src.common.config import load_dataset_build_config
from src.common.models import Chunk, DocumentMetadata, QACandidate
from src.ingest.chunker import chunk_document
from src.ingest.dataset_builder import (
    append_jsonl,
    candidate_to_example,
    load_existing_chunk_ids,
    load_existing_document_ids,
    load_pending_candidates,
    update_candidate_full,
    update_candidate_status,
)
from src.ingest.qa_generator import generate_candidates, is_interesting_chunk
from src.ingest.sec_fetch import (
    download_filing,
    fetch_company_submissions,
    filter_filings,
    get_cik_from_ticker,
)
from src.ingest.text_extract import strip_html

app = typer.Typer(help="Build the FinRAG Eval financial QA dataset.")
console = Console()


def _init_adapter(adapter_name: str):
    if adapter_name == "mock":
        from src.generation.azure_openai import MockGenerationAdapter

        return MockGenerationAdapter()
    if adapter_name == "azure_openai":
        from src.generation.azure_openai import AzureOpenAIAdapter

        return AzureOpenAIAdapter()
    raise ValueError(f"Unknown adapter: {adapter_name}")


@app.command()
def generate(
    config: str = typer.Option(..., "--config", help="Path to dataset_build.yaml"),
) -> None:
    """Fetch SEC filings, extract text, chunk, and generate QA candidates."""
    load_dotenv()
    cfg = load_dataset_build_config(config)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"

    docs_path = output_dir / "documents.jsonl"
    chunks_path = output_dir / "chunks.jsonl"
    candidates_path = output_dir / "candidates.jsonl"

    existing_doc_ids = load_existing_document_ids(docs_path)
    existing_chunk_ids = load_existing_chunk_ids(candidates_path)

    adapter = _init_adapter(cfg.generation_adapter)

    for ticker in cfg.tickers:
        console.print(f"\n[bold]Processing {ticker}...[/bold]")
        cik = get_cik_from_ticker(ticker)
        submissions = fetch_company_submissions(cik)
        company_name: str = submissions.get("name", ticker)

        filings = filter_filings(
            submissions,
            form_types=tuple(cfg.form_types),
            limit=len(cfg.fiscal_years) * len(cfg.form_types) * 3,
        )
        filings = [
            f for f in filings if int(f["filing_date"][:4]) in cfg.fiscal_years
        ]

        for filing in filings:
            year = filing["filing_date"][:4]
            form_slug = filing["form"].replace("-", "").lower()
            document_id = f"{ticker.lower()}_{form_slug}_{year}"

            if document_id in existing_doc_ids:
                console.print(f"  [dim]Skip {document_id} (already processed)[/dim]")
                continue

            console.print(f"  Fetching {ticker} {filing['form']} {year}...")
            filing_path = download_filing(cik, filing, raw_dir / ticker)

            html = filing_path.read_bytes().decode("utf-8", errors="replace")
            text = strip_html(html)

            doc_meta = DocumentMetadata(
                document_id=document_id,
                company=ticker,
                company_name=company_name,
                source_type=filing["form"],
                source_url=None,
                doc_family=filing["form"],
                fiscal_period=year,
                calendar_date=None,
                filing_date=filing["filing_date"],
                report_period_end=None,
                format="html",
                local_path=str(filing_path),
                sha256=None,
                collected_at=datetime.now(UTC),
            )
            append_jsonl([doc_meta], docs_path)

            chunk_dicts = chunk_document(
                document_id, ticker, text, filing_date=filing["filing_date"]
            )
            chunks = [Chunk(**d) for d in chunk_dicts]
            append_jsonl(chunks, chunks_path)

            interesting = [
                c
                for c in chunks
                if is_interesting_chunk(c, cfg.sections)
                and c.chunk_id not in existing_chunk_ids
            ][: cfg.top_k_chunks_per_section]

            new_candidates: list[QACandidate] = []
            for chunk in interesting:
                cands = generate_candidates(
                    chunk, adapter, cfg, form_type=filing["form"], period=year
                )
                new_candidates.extend(cands)
                existing_chunk_ids.add(chunk.chunk_id)

            if new_candidates:
                append_jsonl(new_candidates, candidates_path)

            existing_doc_ids.add(document_id)
            console.print(
                f"  {ticker} {filing['form']} {year} → "
                f"{len(chunks)} chunks → "
                f"{len(interesting)} interesting → "
                f"{len(new_candidates)} candidates"
            )
            time.sleep(0.5)

    total = sum(
        1
        for line in candidates_path.open()
        if line.strip()
    ) if candidates_path.exists() else 0
    console.print(f"\n[green]Done. {total} total candidates in {candidates_path}[/green]")


@app.command()
def review(
    config: str = typer.Option(..., "--config", help="Path to dataset_build.yaml"),
) -> None:
    """Interactively review QA candidates and write approved ones to qa_examples.jsonl."""
    load_dotenv()
    cfg = load_dataset_build_config(config)
    output_dir = Path(cfg.output_dir)
    candidates_path = output_dir / "candidates.jsonl"
    examples_path = output_dir / "qa_examples.jsonl"

    pending = load_pending_candidates(candidates_path)
    if not pending:
        console.print("[green]No pending candidates.[/green]")
        raise typer.Exit()

    existing_count = (
        sum(1 for line in examples_path.open() if line.strip())
        if examples_path.exists()
        else 0
    )

    console.print(f"[bold]{len(pending)} candidates pending review[/bold]\n")

    for i, candidate in enumerate(pending):
        console.rule(
            f"[bold cyan][{i + 1}/{len(pending)}][/bold cyan] "
            f"{candidate.company} | {candidate.difficulty} | {candidate.question_type}"
        )
        console.print(f"[dim]Chunk:[/dim] {candidate.chunk_id}")
        console.print(f"[yellow]Q:[/yellow] {candidate.question}")
        console.print(f"[green]A:[/green] {candidate.gold_answer}")
        console.print("\n[dim](a)pprove  (e)dit  (r)eject  (s)kip  (q)uit[/dim]")

        choice = ""
        while choice not in ("a", "e", "r", "s", "q"):
            choice = input("> ").strip().lower()

        if choice == "q":
            console.print("[yellow]Exiting. Progress saved.[/yellow]")
            break
        if choice == "s":
            continue
        if choice == "r":
            note = input("Rejection note (optional): ").strip()
            update_candidate_status(candidates_path, candidate.candidate_id, "rejected", note)
            console.print("[red]Rejected.[/red]")
            continue
        if choice == "e":
            new_q = input(f"Question [{candidate.question}]: ").strip()
            new_a = input(f"Answer [{candidate.gold_answer}]: ").strip()
            if new_q:
                candidate.question = new_q
            if new_a:
                candidate.gold_answer = new_a
            update_candidate_full(candidates_path, candidate)

        # approve (choice == "a" or after edit)
        existing_count += 1
        example_id = f"q{existing_count:04d}"
        example = candidate_to_example(candidate, example_id)
        append_jsonl([example], examples_path)
        update_candidate_status(candidates_path, candidate.candidate_id, "approved")
        console.print(f"[green]Approved as {example_id}[/green]")

    final_count = (
        sum(1 for line in examples_path.open() if line.strip())
        if examples_path.exists()
        else 0
    )
    console.print(
        f"\n[bold]Done. {final_count} total examples in {examples_path}[/bold]"
    )


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Verify the script loads without errors**

```bash
python scripts/build_dataset.py --help
```

Expected output shows two commands: `generate` and `review`.

```bash
python scripts/build_dataset.py generate --help
```

Expected: shows `--config` option.

- [ ] **Step 3: Run tests to verify nothing broken**

```bash
pytest -q
```

Expected: all tests PASS.

- [ ] **Step 4: Run ruff lint**

```bash
ruff check scripts/build_dataset.py
```

Expected: no errors. If any, fix them before committing.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_dataset.py
git commit -m "feat: add build_dataset CLI with generate and review subcommands"
```

---

## Task 7: Full pipeline smoke test with mock adapter

**Files:**
- No new files — verifies the complete pipeline end-to-end using mock adapter and sample fixtures.

- [ ] **Step 1: Run lint on all new files**

```bash
ruff check src/ingest/text_extract.py src/ingest/qa_generator.py src/ingest/dataset_builder.py src/common/models.py src/common/config.py scripts/build_dataset.py
```

Expected: no errors.

- [ ] **Step 2: Run the full test suite**

```bash
pytest -q
```

Expected: all tests PASS with 0 failures.

- [ ] **Step 3: Verify the generate command runs with mock adapter**

Create a temp config:

```bash
python -c "
import yaml, tempfile, os
cfg = {
    'tickers': ['AAPL'],
    'fiscal_years': [2023],
    'form_types': ['10-K'],
    'sections': ['Risk Factors', 'MD&A', 'Financial Statements'],
    'top_k_chunks_per_section': 2,
    'questions_per_chunk': 2,
    'output_dir': '/tmp/finrag_test/',
    'generation_adapter': 'mock',
}
with open('/tmp/test_dataset_build.yaml', 'w') as f:
    yaml.dump(cfg, f)
print('Config written.')
"
```

Run generate (this will make real SEC EDGAR requests — requires internet):

```bash
python scripts/build_dataset.py generate --config /tmp/test_dataset_build.yaml
```

Expected: output like:
```
Processing AAPL...
  Fetching AAPL 10-K 2023...
  AAPL 10-K 2023 → N chunks → M interesting → K candidates
Done. K total candidates in /tmp/finrag_test/candidates.jsonl
```

- [ ] **Step 4: Inspect generated output**

```bash
head -3 /tmp/finrag_test/candidates.jsonl | python -m json.tool
```

Expected: valid JSON with `candidate_id`, `question`, `gold_answer`, `review_status: "pending"`.

Note: With `mock` adapter, all answers will be `"mock answer"`. That's expected — the mock confirms the pipeline structure. Real answers require `generation_adapter: azure_openai` with valid env vars.

- [ ] **Step 5: Commit final**

```bash
git add -u
git commit -m "chore: dataset pipeline verified end-to-end with mock adapter"
```

If no changes (all already committed), skip this step.
