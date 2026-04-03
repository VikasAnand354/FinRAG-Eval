from typing import List
from tiktoken import get_encoding

ENC = get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(ENC.encode(text))

def simple_paragraph_split(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def build_chunk_id(document_id: str, idx: int) -> str:
    return f"{document_id}__c{idx:04d}"

def chunk_document(document_id: str, text: str):
    paragraphs = simple_paragraph_split(text)
    chunks = []
    for i, p in enumerate(paragraphs):
        chunks.append({
            "chunk_id": build_chunk_id(document_id, i),
            "text": p,
            "token_count": count_tokens(p)
        })
    return chunks
