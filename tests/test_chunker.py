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
