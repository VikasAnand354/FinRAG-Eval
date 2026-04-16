"""Answer quality metrics: exact match.

Kept separate from citation.py per the project's scoring-independence rule:
retrieval metrics (citation precision/recall) must not be mixed with answer
quality metrics (exact match, faithfulness).
"""

from typing import Any

from src.common.models import BenchmarkExample, PipelineResult


def score_exact_match(result: PipelineResult, example: BenchmarkExample) -> dict[str, Any]:
    """Check whether the gold answer appears in the generated answer.

    Uses case-insensitive substring matching. Also checks acceptable_aliases
    if defined on the example.

    Returns:
        {"exact_match": bool}
    """
    answer_lower = result.answer.lower()
    gold_lower = example.gold_answer.lower().strip()
    exact_match = bool(gold_lower) and (gold_lower in answer_lower)

    if not exact_match and example.acceptable_aliases:
        exact_match = any(alias.lower() in answer_lower for alias in example.acceptable_aliases)

    return {"exact_match": exact_match}
