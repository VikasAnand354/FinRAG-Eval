from tiktoken import get_encoding

ENC = get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def simple_paragraph_split(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def build_chunk_id(document_id: str, idx: int) -> str:
    return f"{document_id}__c{idx:04d}"


def chunk_document(
    document_id: str,
    company: str,
    text: str,
    filing_date: str | None = None,
    report_period_end: str | None = None,
) -> list[dict]:
    paragraphs = simple_paragraph_split(text)
    chunks = []
    for i, p in enumerate(paragraphs):
        chunks.append(
            {
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
            }
        )
    return chunks
