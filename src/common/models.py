from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    document_id: str
    company: str
    company_name: str
    source_type: str
    source_url: str | None
    doc_family: str
    fiscal_period: str | None
    calendar_date: str | None
    filing_date: str | None
    report_period_end: str | None
    language: str = "en"
    format: str
    local_path: str
    sha256: str | None
    collected_at: datetime


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    company: str
    section_title: str | None = None
    section_path: list[str] | None = None
    page_number: int | None = None
    paragraph_number: int | None = None
    text: str
    token_count: int
    report_period_end: str | None = None
    filing_date: str | None = None


class GoldCitation(BaseModel):
    document_id: str
    chunk_id: str
    support_type: str


class BenchmarkExample(BaseModel):
    example_id: str
    company: str
    question: str
    question_type: str
    difficulty: str
    answer_type: str
    gold_answer: str
    gold_answer_normalized: float | None
    normalization_unit: str | None
    gold_citations: list[GoldCitation]
    acceptable_aliases: list[str] | None = []
    time_sensitive: bool = False
    requires_multi_hop: bool = False
    source_split: str


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
    citations: list[str]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


class ScoredRow(BaseModel):
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


class RunArtifact(BaseModel):
    run_id: str
    timestamp: datetime
    config_snapshot: dict
    scores: list[ScoredRow]
    aggregate: dict


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
