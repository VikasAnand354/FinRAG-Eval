import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.common.models import RunArtifact, ScoredRow


def _make_row(example_id: str = "q001") -> ScoredRow:
    return ScoredRow(
        example_id=example_id,
        answer="Revenue was $383.3 billion.",
        citations=["aapl_10k_2023__c0001"],
        latency_ms=100.0,
        prompt_tokens=50,
        completion_tokens=10,
        cost_usd=0.0008,
        exact_match=True,
        citation_precision=1.0,
        citation_recall=1.0,
        faithful=True,
        faithfulness_score=0.9,
    )


def _make_artifact(run_id: str = "test-001", n_rows: int = 2) -> RunArtifact:
    from src.leaderboard.artifacts import compute_aggregate

    rows = [_make_row(f"q{i:03d}") for i in range(1, n_rows + 1)]
    return RunArtifact(
        run_id=run_id,
        timestamp=datetime(2026, 4, 3, tzinfo=UTC),
        config_snapshot={"run_id": run_id, "retriever": "bm25"},
        scores=rows,
        aggregate=compute_aggregate(rows),
    )


def test_write_artifact_creates_expected_files():
    from src.leaderboard.artifacts import write_artifact

    artifact = _make_artifact()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = write_artifact(artifact, Path(tmpdir))
        assert (out_dir / "artifact.json").exists()
        assert (out_dir / "scores.jsonl").exists()
        assert (out_dir / "config.json").exists()


def test_write_artifact_scores_jsonl_line_count():
    from src.leaderboard.artifacts import write_artifact

    artifact = _make_artifact(n_rows=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = write_artifact(artifact, Path(tmpdir))
        lines = (out_dir / "scores.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3


def test_validate_artifact_roundtrip():
    from src.leaderboard.artifacts import validate_artifact, write_artifact

    artifact = _make_artifact(run_id="roundtrip-001")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = write_artifact(artifact, Path(tmpdir))
        reloaded = validate_artifact(out_dir / "artifact.json")
        assert reloaded.run_id == "roundtrip-001"
        assert len(reloaded.scores) == 2


def test_compute_aggregate_values():
    from src.leaderboard.artifacts import compute_aggregate

    rows = [_make_row("q001"), _make_row("q002")]
    agg = compute_aggregate(rows)
    assert agg["n_examples"] == 2
    assert agg["exact_match_rate"] == 1.0
    assert agg["mean_faithfulness_score"] == pytest.approx(0.9, abs=1e-6)
    assert agg["pct_faithful"] == 1.0
    assert agg["mean_citation_precision"] == 1.0
    assert agg["mean_citation_recall"] == 1.0
    assert agg["total_cost_usd"] == pytest.approx(0.0016, abs=1e-6)


def test_compute_aggregate_empty():
    from src.leaderboard.artifacts import compute_aggregate

    agg = compute_aggregate([])
    assert agg["n_examples"] == 0
