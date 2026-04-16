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
        gold_answer_normalized=None,
        normalization_unit=None,
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
