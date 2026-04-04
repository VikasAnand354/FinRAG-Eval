from datetime import UTC, datetime


def test_run_config_round_trip():
    from src.common.models import RunConfig

    cfg = RunConfig(
        run_id="smoke-001",
        dataset_path="datasets/sample/qa_examples.jsonl",
        chunks_path="datasets/sample/chunks.jsonl",
        retriever="bm25",
        top_k=3,
        generation_adapter="mock",
        output_dir="outputs/",
    )
    assert RunConfig.model_validate_json(cfg.model_dump_json()) == cfg


def test_pipeline_result_round_trip():
    from src.common.models import PipelineResult

    r = PipelineResult(
        example_id="q001",
        answer="Revenue was $383.3 billion.",
        citations=["aapl_10k_2023__c0001"],
        latency_ms=250.0,
        prompt_tokens=100,
        completion_tokens=30,
    )
    assert PipelineResult.model_validate_json(r.model_dump_json()) == r


def test_scored_row_round_trip():
    from src.common.models import ScoredRow

    row = ScoredRow(
        example_id="q001",
        answer="Revenue was $383.3 billion.",
        citations=["aapl_10k_2023__c0001"],
        latency_ms=250.0,
        prompt_tokens=100,
        completion_tokens=30,
        cost_usd=0.00394,
        exact_match=True,
        citation_precision=1.0,
        citation_recall=1.0,
        faithful=True,
        faithfulness_score=0.95,
    )
    assert ScoredRow.model_validate_json(row.model_dump_json()) == row


def test_scored_row_nullable_faithfulness():
    from src.common.models import ScoredRow

    row = ScoredRow(
        example_id="q001",
        answer="test",
        citations=[],
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=2,
        cost_usd=0.0,
        exact_match=False,
        citation_precision=0.0,
        citation_recall=0.0,
        faithful=False,
        faithfulness_score=None,
    )
    assert row.faithfulness_score is None
    assert ScoredRow.model_validate_json(row.model_dump_json()) == row


def test_run_artifact_round_trip():
    from src.common.models import RunArtifact, ScoredRow

    row = ScoredRow(
        example_id="q001",
        answer="test",
        citations=[],
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=2,
        cost_usd=0.0001,
        exact_match=False,
        citation_precision=0.0,
        citation_recall=0.0,
        faithful=False,
        faithfulness_score=None,
    )
    artifact = RunArtifact(
        run_id="test-001",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        config_snapshot={"run_id": "test-001"},
        scores=[row],
        aggregate={"n_examples": 1},
    )
    assert RunArtifact.model_validate_json(artifact.model_dump_json()) == artifact


def test_qa_candidate_round_trip():
    from src.common.models import QACandidate

    c = QACandidate(
        candidate_id="aapl_10k_2023__c0001-q0",
        chunk_id="aapl_10k_2023__c0001",
        document_id="aapl_10k_2023",
        company="AAPL",
        question="What were Apple's total net sales for fiscal 2023?",
        gold_answer="$383.3 billion",
        difficulty="easy",
        question_type="numerical",
        gold_citations=["aapl_10k_2023__c0001"],
    )
    assert QACandidate.model_validate_json(c.model_dump_json()) == c
    assert c.review_status == "pending"
    assert c.reviewer_note == ""


def test_qa_candidate_difficulty_values():
    import pytest
    from pydantic import ValidationError

    from src.common.models import QACandidate

    with pytest.raises(ValidationError):
        QACandidate(
            candidate_id="x-q0",
            chunk_id="x",
            document_id="x",
            company="X",
            question="q",
            gold_answer="a",
            difficulty="invalid",
            question_type="factual",
            gold_citations=["x"],
        )


def test_dataset_build_config_round_trip():
    from src.common.models import DatasetBuildConfig

    cfg = DatasetBuildConfig(
        tickers=["AAPL", "MSFT"],
        fiscal_years=[2023, 2024],
        output_dir="datasets/full/",
    )
    assert DatasetBuildConfig.model_validate_json(cfg.model_dump_json()) == cfg
    assert cfg.questions_per_chunk == 2
    assert cfg.generation_adapter == "azure_openai"
