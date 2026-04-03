from abc import ABC, abstractmethod

from src.common.models import Chunk, PipelineResult


class GenerationAdapter(ABC):
    @abstractmethod
    def generate(self, example_id: str, question: str, chunks: list[Chunk]) -> PipelineResult:
        """Generate an answer given a question and retrieved chunks."""
        ...

    @abstractmethod
    def judge(self, prompt: str) -> tuple[str, float, int, int]:
        """
        Send a raw prompt and return (response_text, latency_ms, prompt_tokens, completion_tokens).
        Used by the faithfulness evaluator.
        """
        ...
