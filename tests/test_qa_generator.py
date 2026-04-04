import json

from src.common.models import Chunk, DatasetBuildConfig, QACandidate
from src.generation.azure_openai import MockGenerationAdapter


# A mock that returns controlled Q&A JSON from judge()
class _QAMockAdapter(MockGenerationAdapter):
    def __init__(self, response: str) -> None:
        super().__init__()
        self._response = response

    def judge(self, prompt: str) -> tuple[str, float, int, int]:
        return (self._response, 1.0, 10, 5)


def _make_chunk(
    token_count: int = 120,
    section_title: str | None = "MD&A",
    text: str = "Revenue was $391.0 billion, up 5%.",
) -> Chunk:
    return Chunk(
        chunk_id="aapl_10k_2023__c0001",
        document_id="aapl_10k_2023",
        company="AAPL",
        section_title=section_title,
        text=text,
        token_count=token_count,
        report_period_end="2023-09-30",
        filing_date="2023-11-03",
    )


def _default_config() -> DatasetBuildConfig:
    return DatasetBuildConfig(
        tickers=["AAPL"],
        fiscal_years=[2023],
        sections=["MD&A", "Risk Factors"],
    )


# --- is_interesting_chunk tests ---


def test_interesting_chunk_passes():
    from src.ingest.qa_generator import is_interesting_chunk

    chunk = _make_chunk(token_count=120)
    assert is_interesting_chunk(chunk, ["MD&A"]) is True


def test_interesting_chunk_too_short():
    from src.ingest.qa_generator import is_interesting_chunk

    chunk = _make_chunk(token_count=50)
    assert is_interesting_chunk(chunk, ["MD&A"]) is False


def test_interesting_chunk_no_financial_signal():
    from src.ingest.qa_generator import is_interesting_chunk

    chunk = _make_chunk(text="The company focuses on innovation and growth in markets.")
    assert is_interesting_chunk(chunk, ["MD&A"]) is False


def test_interesting_chunk_wrong_section():
    from src.ingest.qa_generator import is_interesting_chunk

    chunk = _make_chunk(section_title="Cover Page")
    assert is_interesting_chunk(chunk, ["MD&A", "Risk Factors"]) is False


def test_interesting_chunk_no_section_passes():
    """Chunk with section_title=None is not filtered by section."""
    from src.ingest.qa_generator import is_interesting_chunk

    chunk = _make_chunk(section_title=None, token_count=120)
    assert is_interesting_chunk(chunk, ["MD&A"]) is True


# --- generate_candidates tests ---


def test_generate_candidates_success():
    from src.ingest.qa_generator import generate_candidates

    response = json.dumps(
        [
            {
                "question": "What was revenue?",
                "answer": "$391.0 billion",
                "difficulty": "easy",
                "type": "numerical",
            },
            {
                "question": "By how much did revenue grow?",
                "answer": "5%",
                "difficulty": "medium",
                "type": "numerical",
            },
        ]
    )
    adapter = _QAMockAdapter(response)
    chunk = _make_chunk()
    cfg = _default_config()
    candidates = generate_candidates(chunk, adapter, cfg)

    assert len(candidates) == 2
    assert all(isinstance(c, QACandidate) for c in candidates)
    assert candidates[0].candidate_id == "aapl_10k_2023__c0001-q0"
    assert candidates[0].gold_citations == ["aapl_10k_2023__c0001"]
    assert candidates[0].review_status == "pending"
    assert candidates[1].candidate_id == "aapl_10k_2023__c0001-q1"


def test_generate_candidates_invalid_json_returns_empty():
    from src.ingest.qa_generator import generate_candidates

    adapter = _QAMockAdapter("not valid json")
    chunk = _make_chunk()
    cfg = _default_config()
    candidates = generate_candidates(chunk, adapter, cfg)
    assert candidates == []


def test_generate_candidates_empty_array_returns_empty():
    from src.ingest.qa_generator import generate_candidates

    adapter = _QAMockAdapter("[]")
    chunk = _make_chunk()
    cfg = _default_config()
    candidates = generate_candidates(chunk, adapter, cfg)
    assert candidates == []


def test_generate_candidates_prompt_contains_chunk_text():
    """Verify the prompt sent to the adapter contains chunk text."""
    from src.ingest.qa_generator import generate_candidates

    captured_prompts: list[str] = []

    class _CapturingAdapter(_QAMockAdapter):
        def judge(self, prompt: str) -> tuple[str, float, int, int]:
            captured_prompts.append(prompt)
            return ("[]", 1.0, 0, 0)

    chunk = _make_chunk(text="Revenue was $391.0 billion, up 5%.")
    generate_candidates(chunk, _CapturingAdapter("[]"), _default_config())
    assert len(captured_prompts) == 1
    assert "Revenue was $391.0 billion" in captured_prompts[0]
    assert "AAPL" in captured_prompts[0]
