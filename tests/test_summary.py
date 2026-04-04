"""Tests for leaderboard summary generation."""

from datetime import UTC, datetime

from src.common.models import RunArtifact, ScoredRow
from src.leaderboard.artifacts import compute_aggregate


def _make_artifact() -> RunArtifact:
    """Create a test RunArtifact with a single scored example."""
    row = ScoredRow(
        example_id="q001",
        answer="Revenue was $383.3 billion.",
        citations=["aapl_10k_2023__c0001"],
        latency_ms=250.0,
        prompt_tokens=100,
        completion_tokens=30,
        cost_usd=0.0019,
        exact_match=True,
        citation_precision=1.0,
        citation_recall=1.0,
        faithful=True,
        faithfulness_score=0.92,
    )
    return RunArtifact(
        run_id="smoke-001",
        timestamp=datetime(2026, 4, 3, tzinfo=UTC),
        config_snapshot={},
        scores=[row],
        aggregate=compute_aggregate([row]),
    )


def test_summary_contains_run_id():
    """Test that the summary contains the run ID."""
    from src.leaderboard.summary import generate_summary

    summary = generate_summary(_make_artifact())
    assert "smoke-001" in summary


def test_summary_contains_metric_labels():
    """Test that the summary contains all metric labels."""
    from src.leaderboard.summary import generate_summary

    summary = generate_summary(_make_artifact())
    assert "Exact Match" in summary
    assert "Faithfulness" in summary
    assert "Citation" in summary
    assert "Latency" in summary
    assert "Cost" in summary
