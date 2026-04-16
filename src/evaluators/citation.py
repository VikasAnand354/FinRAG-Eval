"""Citation quality metrics: precision and recall.

Measures retrieval-side quality only — whether the pipeline cited the right
chunks. Answer quality (exact match, faithfulness) lives in separate modules.
"""

from typing import Any

from src.common.models import BenchmarkExample, PipelineResult


def score_citations(result: PipelineResult, example: BenchmarkExample) -> dict[str, Any]:
    """Compute citation precision and recall.

    Precision: of chunks the pipeline cited, what fraction are in the gold set.
    Recall:    of gold chunks needed, what fraction were cited.

    Returns:
        {"citation_precision": float, "citation_recall": float}
    """
    predicted = set(result.citations)
    gold = {gc.chunk_id for gc in example.gold_citations}

    precision = 0.0 if not predicted else len(predicted & gold) / len(predicted)
    recall = 1.0 if not gold else len(predicted & gold) / len(gold)

    return {
        "citation_precision": precision,
        "citation_recall": recall,
    }
