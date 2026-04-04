from src.common.models import Chunk


def _make_chunk(chunk_id: str, text: str, company: str = "TEST") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("__")[0],
        company=company,
        section_title="test_section",
        section_path=["section"],
        page_number=1,
        paragraph_number=1,
        text=text,
        token_count=len(text.split()),
        report_period_end="2024-01-01",
        filing_date="2024-01-01",
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

    chunks = [
        _make_chunk(f"doc{i}__c0000", f"text about financial topic number {i}") for i in range(5)
    ]
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
