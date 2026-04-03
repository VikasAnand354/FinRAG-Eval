from src.common.models import PipelineResult


def _make_result(latency_ms: float = 200.0, prompt_tokens: int = 100, completion_tokens: int = 50) -> PipelineResult:
    return PipelineResult(
        example_id="q001",
        answer="test",
        citations=[],
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def test_score_latency_returns_all_fields():
    from src.evaluators.latency import score_latency

    scores = score_latency(_make_result())
    assert "latency_ms" in scores
    assert "prompt_tokens" in scores
    assert "completion_tokens" in scores
    assert "cost_usd" in scores


def test_score_latency_values_match_input():
    from src.evaluators.latency import score_latency

    scores = score_latency(_make_result(latency_ms=123.4, prompt_tokens=200, completion_tokens=80))
    assert scores["latency_ms"] == 123.4
    assert scores["prompt_tokens"] == 200
    assert scores["completion_tokens"] == 80


def test_score_latency_cost_is_positive():
    from src.evaluators.latency import score_latency

    scores = score_latency(_make_result(prompt_tokens=1000, completion_tokens=500))
    assert scores["cost_usd"] > 0


def test_score_latency_zero_tokens_zero_cost():
    from src.evaluators.latency import score_latency

    scores = score_latency(_make_result(prompt_tokens=0, completion_tokens=0))
    assert scores["cost_usd"] == 0.0
