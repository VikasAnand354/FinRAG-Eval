from typing import Any

from src.common.models import PipelineResult

# Price per 1000 tokens in USD (GPT-4 approximate; update as needed)
PROMPT_TOKEN_PRICE_PER_1K: float = 0.01
COMPLETION_TOKEN_PRICE_PER_1K: float = 0.03


def score_latency(result: PipelineResult) -> dict[str, Any]:
    cost = (
        result.prompt_tokens / 1000 * PROMPT_TOKEN_PRICE_PER_1K
        + result.completion_tokens / 1000 * COMPLETION_TOKEN_PRICE_PER_1K
    )
    return {
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_usd": round(cost, 6),
    }
