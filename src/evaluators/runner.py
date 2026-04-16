from src.common.models import BenchmarkExample, Chunk, PipelineResult, RunConfig, ScoredRow
from src.evaluators.answer import score_exact_match
from src.evaluators.citation import score_citations
from src.evaluators.faithfulness import score_faithfulness
from src.evaluators.latency import score_latency
from src.generation.base import GenerationAdapter


def evaluate(
    result: PipelineResult,
    example: BenchmarkExample,
    chunks: list[Chunk],
    judge: GenerationAdapter,
    config: RunConfig,  # noqa: ARG001 — reserved for future per-run scorer config
) -> ScoredRow:
    latency = score_latency(result)
    citations = score_citations(result, example)
    exact = score_exact_match(result, example)
    faithfulness = score_faithfulness(result, chunks, judge)

    return ScoredRow(
        example_id=result.example_id,
        answer=result.answer,
        citations=result.citations,
        **latency,
        **citations,
        **exact,
        **faithfulness,
    )
