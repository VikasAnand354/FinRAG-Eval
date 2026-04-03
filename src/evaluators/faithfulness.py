import json
import logging
from typing import TYPE_CHECKING, Any

from src.common.models import Chunk, PipelineResult

if TYPE_CHECKING:
    from src.generation.base import GenerationAdapter

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT_V1 = (
    "You are a faithfulness evaluator for financial question answering.\n\n"
    "Given the retrieved context passages and a generated answer, assess whether "
    "every claim in the answer is supported by the context.\n\n"
    "Context:\n{context}\n\n"
    "Answer:\n{answer}\n\n"
    "Rate the faithfulness of the answer on a scale from 0.0 to 1.0, where:\n"
    "- 1.0 means every claim is directly supported by the context\n"
    "- 0.0 means the answer makes claims not found in the context\n\n"
    'Respond with only a JSON object: {{"score": <float between 0 and 1>}}'
)

FAITHFULNESS_PROMPT_VERSION = "faithfulness-v1"
FAITHFULNESS_THRESHOLD = 0.5


def score_faithfulness(
    result: PipelineResult,
    chunks: list[Chunk],
    adapter: "GenerationAdapter",
) -> dict[str, Any]:
    context = "\n\n".join(c.text for c in chunks)
    prompt = FAITHFULNESS_PROMPT_V1.format(context=context, answer=result.answer)
    try:
        response_text, _, _, _ = adapter.judge(prompt)
        parsed = json.loads(response_text)
        score = float(parsed["score"])
        return {"faithfulness_score": score, "faithful": score >= FAITHFULNESS_THRESHOLD}
    except Exception:
        logger.warning("Faithfulness scoring failed for example_id=%s", result.example_id)
        return {"faithfulness_score": None, "faithful": False}
