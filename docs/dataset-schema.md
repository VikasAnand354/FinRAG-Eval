# Dataset Schema

## Overview

The benchmark dataset consists of two JSONL files that are always used together:

- `qa_examples.jsonl` — one `BenchmarkExample` per line
- `chunks.jsonl` — one `Chunk` per line

Both files must be present for a run. The eval pipeline loads all chunks into memory and matches them to retrieved results by `chunk_id`.

---

## BenchmarkExample

Each line in `qa_examples.jsonl` is a JSON object matching this schema.

```python
class BenchmarkExample(BaseModel):
    example_id: str               # unique identifier, e.g. "aapl_2023_10k_001"
    company: str                  # ticker symbol, e.g. "AAPL"
    question: str                 # the question to answer
    question_type: str            # "factual" | "numerical" | "comparative" | "multi_hop"
    difficulty: str               # "easy" | "medium" | "hard"
    answer_type: str              # "short" | "long" | "numerical"
    gold_answer: str              # the correct answer string
    gold_answer_normalized: float | None   # numeric value if applicable
    normalization_unit: str | None         # unit for normalized answer, e.g. "USD_billions"
    gold_citations: list[GoldCitation]     # which chunks support the answer
    acceptable_aliases: list[str] | None   # alternative correct answer strings
    time_sensitive: bool          # answer may change across fiscal periods
    requires_multi_hop: bool      # answer requires combining multiple chunks
    source_split: str             # "train" | "dev" | "test"
```

### GoldCitation

Each `gold_citations` entry identifies a specific chunk that supports the answer.

```python
class GoldCitation(BaseModel):
    document_id: str    # matches Chunk.document_id
    chunk_id: str       # matches Chunk.chunk_id
    support_type: str   # "primary" | "supporting"
```

### Example

```json
{
  "example_id": "aapl_2023_10k_001",
  "company": "AAPL",
  "question": "What was Apple's total net sales for fiscal year 2023?",
  "question_type": "numerical",
  "difficulty": "easy",
  "answer_type": "numerical",
  "gold_answer": "$383.3 billion",
  "gold_answer_normalized": 383.3,
  "normalization_unit": "USD_billions",
  "gold_citations": [
    {
      "document_id": "aapl_10k_2023",
      "chunk_id": "aapl_10k_2023__c0042",
      "support_type": "primary"
    }
  ],
  "acceptable_aliases": ["383.3 billion", "383,285 million"],
  "time_sensitive": true,
  "requires_multi_hop": false,
  "source_split": "test"
}
```

---

## Chunk

Each line in `chunks.jsonl` is a JSON object matching this schema.

```python
class Chunk(BaseModel):
    chunk_id: str                  # unique identifier, e.g. "aapl_10k_2023__c0042"
    document_id: str               # parent document identifier
    company: str                   # ticker symbol
    section_title: str | None      # e.g. "Management Discussion and Analysis"
    section_path: list[str] | None # hierarchical section path
    page_number: int | None        # page in original filing
    paragraph_number: int | None   # paragraph index within document
    text: str                      # chunk text content
    token_count: int               # tiktoken cl100k_base token count
    report_period_end: str | None  # e.g. "2023-09-30"
    filing_date: str | None        # e.g. "2023-11-02"
```

### chunk_id Format

Chunk IDs use the format: `{document_id}__{cNNNN}` where NNNN is a zero-padded four-digit paragraph index.

Example: `aapl_10k_2023__c0042`

This format ensures chunk IDs are:
- unique within and across documents
- human-readable (company + filing + position)
- sortable in document order

### Example

```json
{
  "chunk_id": "aapl_10k_2023__c0042",
  "document_id": "aapl_10k_2023",
  "company": "AAPL",
  "section_title": "Financial Statements",
  "section_path": ["Item 8", "Financial Statements"],
  "page_number": 41,
  "paragraph_number": 42,
  "text": "Net sales for fiscal 2023 were $383.3 billion...",
  "token_count": 187,
  "report_period_end": "2023-09-30",
  "filing_date": "2023-11-02"
}
```

---

## Dataset Directory Layout

```
datasets/
├── sample/                    # small fixture dataset for smoke testing
│   ├── qa_examples.jsonl      # ~5 examples
│   └── chunks.jsonl           # ~20 chunks
└── full/                      # full benchmark dataset (git-ignored)
    ├── qa_examples.jsonl
    └── chunks.jsonl
```

The `full/` directory is excluded from git because it contains content derived from SEC filings. It must be generated locally using `scripts/build_dataset.py`.

---

## Adding a New Dataset

To add a new dataset without changing evaluator logic:

1. Produce a `qa_examples.jsonl` where each line is a valid `BenchmarkExample`
2. Produce a `chunks.jsonl` where each line is a valid `Chunk`
3. Ensure every `chunk_id` in `gold_citations` exists in `chunks.jsonl`
4. Place both files in `datasets/<your_dataset_name>/`
5. Create a new config file in `configs/` pointing to your dataset paths

No changes to evaluator code are needed.

---

## DocumentMetadata (optional reference)

During dataset construction, document-level metadata is tracked. This is not required for eval runs but is written during the build pipeline for provenance.

```python
class DocumentMetadata(BaseModel):
    document_id: str
    company: str
    company_name: str
    source_type: str          # e.g. "sec_filing"
    source_url: str | None
    doc_family: str           # e.g. "10-K"
    fiscal_period: str | None
    calendar_date: str | None
    filing_date: str | None
    report_period_end: str | None
    language: str             # default "en"
    format: str               # e.g. "html"
    local_path: str
    sha256: str | None
    collected_at: datetime
```
