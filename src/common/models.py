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
    section_title: Optional[str]
    section_path: Optional[List[str]]
    page_number: Optional[int]
    paragraph_number: Optional[int]
    text: str
    token_count: int
    report_period_end: Optional[str]
    filing_date: Optional[str]

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
