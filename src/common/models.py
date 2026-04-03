from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class DocumentMetadata(BaseModel):
    document_id: str
    company: str
    company_name: str
    source_type: str
    source_url: Optional[str]
    doc_family: str
    fiscal_period: Optional[str]
    calendar_date: Optional[str]
    filing_date: Optional[str]
    report_period_end: Optional[str]
    language: str = "en"
    format: str
    local_path: str
    sha256: Optional[str]
    collected_at: datetime

class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    company: str
    section_title: Optional[str] = None
    section_path: Optional[List[str]] = None
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    text: str
    token_count: int
    report_period_end: Optional[str] = None
    filing_date: Optional[str] = None

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
    gold_answer_normalized: Optional[float]
    normalization_unit: Optional[str]
    gold_citations: List[GoldCitation]
    acceptable_aliases: Optional[List[str]] = []
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
