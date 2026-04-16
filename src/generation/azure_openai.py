import json
import logging
import time

from src.common.models import Chunk, PipelineResult
from src.generation.base import GenerationAdapter

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT_V1 = (
    "You are a financial analyst assistant. Answer the question using only the provided "
    "context passages. Be precise with numbers and dates. If the context does not contain "
    "enough information to answer, say so.\n\n"
    "Each passage is prefixed with its chunk ID in square brackets, e.g. [chunk_id].\n"
    "You MUST respond with a JSON object in this exact format:\n"
    '{"answer": "<your answer>", "citations": ["<chunk_id_1>", "<chunk_id_2>"]}\n'
    "Only include chunk IDs that directly support your answer."
)

PROMPT_VERSION = "qa-v2"

_RETRY_DELAYS = [5, 15, 30]  # seconds between retries on 429


def _call_with_retry(fn, *args, **kwargs):
    """Call fn with exponential backoff on RateLimitError (429).

    Retries up to len(_RETRY_DELAYS) times before re-raising.
    """
    from openai import RateLimitError

    last_exc: Exception | None = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            logger.warning("Rate limited — retrying in %ds (attempt %d)", delay, attempt)
            time.sleep(delay)
        try:
            return fn(*args, **kwargs)
        except RateLimitError as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


class MockGenerationAdapter(GenerationAdapter):
    """Deterministic adapter for tests — no network calls."""

    def __init__(self, canned_answer: str = "mock answer", canned_score: float = 0.9) -> None:
        self._answer = canned_answer
        self._canned_score = canned_score

    def generate(self, example_id: str, question: str, chunks: list[Chunk]) -> PipelineResult:
        return PipelineResult(
            example_id=example_id,
            answer=self._answer,
            citations=[chunks[0].chunk_id] if chunks else [],
            latency_ms=1.0,
            prompt_tokens=10,
            completion_tokens=5,
        )

    def judge(self, prompt: str) -> tuple[str, float, int, int]:
        return (json.dumps({"score": self._canned_score}), 1.0, 10, 5)


class AzureOpenAIAdapter(GenerationAdapter):
    def __init__(self) -> None:
        from openai import AzureOpenAI

        from src.common.config import get_azure_settings

        settings = get_azure_settings()
        self._client = AzureOpenAI(
            api_key=settings["api_key"],
            api_version=settings["api_version"],
            azure_endpoint=settings["endpoint"],
        )
        self._deployment = settings["deployment"]

    def generate(self, example_id: str, question: str, chunks: list[Chunk]) -> PipelineResult:
        context = "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks)
        valid_ids = {c.chunk_id for c in chunks}
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT_V1},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        start = time.perf_counter()
        try:
            response = _call_with_retry(
                self._client.chat.completions.create,
                model=self._deployment,
                messages=messages,
                response_format={"type": "json_object"},
            )
            latency_ms = (time.perf_counter() - start) * 1000
            raw = response.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            answer = parsed.get("answer", "")
            # Only keep cited IDs that were actually in the retrieved set
            cited = [c for c in parsed.get("citations", []) if c in valid_ids]
            return PipelineResult(
                example_id=example_id,
                answer=answer,
                citations=cited,
                latency_ms=latency_ms,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
        except Exception:
            logger.exception("Generation failed for example_id=%s", example_id)
            return PipelineResult(
                example_id=example_id,
                answer="",
                citations=[],
                latency_ms=(time.perf_counter() - start) * 1000,
                prompt_tokens=0,
                completion_tokens=0,
            )

    def judge(self, prompt: str) -> tuple[str, float, int, int]:
        start = time.perf_counter()
        try:
            response = _call_with_retry(
                self._client.chat.completions.create,
                model=self._deployment,
                messages=[{"role": "user", "content": prompt}],
            )
            latency_ms = (time.perf_counter() - start) * 1000
            text = response.choices[0].message.content or ""
            return (
                text,
                latency_ms,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
        except Exception:
            logger.exception("Judge call failed")
            return ("", (time.perf_counter() - start) * 1000, 0, 0)
