from src.common.models import BenchmarkExample, GoldCitation, PipelineResult


def _make_example(gold_answer: str = "$100 billion", aliases: list[str] | None = None):
    return BenchmarkExample(
        example_id="q001",
        company="AAPL",
        question="What was revenue?",
        question_type="numerical",
        difficulty="easy",
        answer_type="numerical",
        gold_answer=gold_answer,
        gold_answer_normalized=None,
        normalization_unit=None,
        gold_citations=[
            GoldCitation(document_id="doc1", chunk_id="doc1__c0001", support_type="direct")
        ],
        acceptable_aliases=aliases or [],
        source_split="test",
    )


def _make_result(answer: str) -> PipelineResult:
    return PipelineResult(
        example_id="q001",
        answer=answer,
        citations=[],
        latency_ms=100.0,
        prompt_tokens=50,
        completion_tokens=10,
    )


def test_exact_match_true():
    from src.evaluators.answer import score_exact_match

    scores = score_exact_match(
        _make_result("The total revenue was $100 billion for the year"),
        _make_example("$100 billion"),
    )
    assert scores["exact_match"] is True


def test_exact_match_false():
    from src.evaluators.answer import score_exact_match

    scores = score_exact_match(
        _make_result("Revenue was $200 billion"),
        _make_example("$100 billion"),
    )
    assert scores["exact_match"] is False


def test_exact_match_case_insensitive():
    from src.evaluators.answer import score_exact_match

    scores = score_exact_match(
        _make_result("Revenue was $100 BILLION"),
        _make_example("$100 billion"),
    )
    assert scores["exact_match"] is True


def test_exact_match_via_alias():
    from src.evaluators.answer import score_exact_match

    scores = score_exact_match(
        _make_result("Apple revenue was 383.3B in 2023"),
        _make_example("$383.3 billion", aliases=["383.3B", "383.3 billion"]),
    )
    assert scores["exact_match"] is True


def test_exact_match_empty_answer():
    from src.evaluators.answer import score_exact_match

    scores = score_exact_match(_make_result(""), _make_example("$100 billion"))
    assert scores["exact_match"] is False


def test_exact_match_empty_gold():
    from src.evaluators.answer import score_exact_match

    scores = score_exact_match(_make_result("some answer"), _make_example(""))
    assert scores["exact_match"] is False
