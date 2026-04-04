from src.common.models import Chunk, PipelineResult
from src.generation.azure_openai import MockGenerationAdapter


def _make_chunk(text: str = "Apple revenue was $383.3 billion in 2023.") -> Chunk:
    return Chunk(
        chunk_id="aapl_10k_2023__c0001",
        document_id="aapl_10k_2023",
        company="AAPL",
        text=text,
        token_count=len(text.split()),
    )


def _make_result(answer: str = "Apple revenue was $383.3 billion.") -> PipelineResult:
    return PipelineResult(
        example_id="q001",
        answer=answer,
        citations=["aapl_10k_2023__c0001"],
        latency_ms=100.0,
        prompt_tokens=50,
        completion_tokens=20,
    )


def test_faithful_above_threshold():
    from src.evaluators.faithfulness import score_faithfulness

    mock = MockGenerationAdapter(canned_score=0.9)
    scores = score_faithfulness(_make_result(), [_make_chunk()], mock)
    assert scores["faithfulness_score"] == 0.9
    assert scores["faithful"] is True


def test_unfaithful_below_threshold():
    from src.evaluators.faithfulness import score_faithfulness

    mock = MockGenerationAdapter(canned_score=0.3)
    scores = score_faithfulness(_make_result(), [_make_chunk()], mock)
    assert scores["faithfulness_score"] == 0.3
    assert scores["faithful"] is False


def test_exactly_at_threshold_is_faithful():
    from src.evaluators.faithfulness import score_faithfulness

    mock = MockGenerationAdapter(canned_score=0.5)
    scores = score_faithfulness(_make_result(), [_make_chunk()], mock)
    assert scores["faithful"] is True


def test_bad_json_response_returns_none():
    from src.common.models import Chunk, PipelineResult
    from src.evaluators.faithfulness import score_faithfulness
    from src.generation.base import GenerationAdapter

    class BadJudge(GenerationAdapter):
        def generate(self, example_id: str, question: str, chunks: list[Chunk]) -> PipelineResult:
            return PipelineResult(
                example_id=example_id,
                answer="",
                citations=[],
                latency_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
            )

        def judge(self, prompt: str) -> tuple[str, float, int, int]:
            return ("not valid json at all", 1.0, 0, 0)

    scores = score_faithfulness(_make_result(), [_make_chunk()], BadJudge())
    assert scores["faithfulness_score"] is None
    assert scores["faithful"] is False
