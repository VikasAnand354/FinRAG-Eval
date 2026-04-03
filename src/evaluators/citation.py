from typing import Any

from src.common.models import BenchmarkExample, PipelineResult


def score_citations(result: PipelineResult, example: BenchmarkExample) -> dict[str, Any]:
    predicted = set(result.citations)
    gold = {gc.chunk_id for gc in example.gold_citations}

    if not predicted:
        precision = 0.0
    else:
        precision = len(predicted & gold) / len(predicted)

    if not gold:
        recall = 1.0
    else:
        recall = len(predicted & gold) / len(gold)

    answer_lower = result.answer.lower()
    gold_lower = example.gold_answer.lower().strip()
    exact_match = gold_lower in answer_lower

    if not exact_match and example.acceptable_aliases:
        exact_match = any(alias.lower() in answer_lower for alias in example.acceptable_aliases)

    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "exact_match": exact_match,
    }
