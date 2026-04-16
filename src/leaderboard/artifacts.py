"""Artifact generation and aggregation for leaderboard runs."""

import json
import statistics
from pathlib import Path
from typing import Any

from src.common.models import RunArtifact, ScoredRow


def compute_aggregate(scores: list[ScoredRow]) -> dict[str, Any]:
    """Compute aggregate metrics from scored rows.

    Raises:
        ValueError: If scores is empty — a run with zero examples is invalid.
    """
    n = len(scores)
    if n == 0:
        raise ValueError(
            "Cannot compute aggregate: no scored examples. "
            "Check dataset_path and chunks_path in your config."
        )

    faithfulness_scores = [s.faithfulness_score for s in scores if s.faithfulness_score is not None]

    return {
        "n_examples": n,
        "exact_match_rate": sum(s.exact_match for s in scores) / n,
        "mean_faithfulness_score": (
            statistics.mean(faithfulness_scores) if faithfulness_scores else 0.0
        ),
        "pct_faithful": sum(s.faithful for s in scores) / n,
        "mean_citation_precision": statistics.mean(s.citation_precision for s in scores),
        "mean_citation_recall": statistics.mean(s.citation_recall for s in scores),
        "median_latency_ms": statistics.median(s.latency_ms for s in scores),
        "total_cost_usd": round(sum(s.cost_usd for s in scores), 6),
    }


def write_artifact(run: RunArtifact, output_dir: Path) -> Path:
    """Write artifact to disk with separate files for scores, config, and summary."""
    run_dir = output_dir / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "artifact.json").write_text(run.model_dump_json(indent=2))

    with (run_dir / "scores.jsonl").open("w") as f:
        for row in run.scores:
            f.write(row.model_dump_json() + "\n")

    (run_dir / "config.json").write_text(json.dumps(run.config_snapshot, indent=2))

    return run_dir


def validate_artifact(path: Path) -> RunArtifact:
    """Load and validate an artifact from disk via full Pydantic round-trip."""
    data = json.loads(path.read_text())
    return RunArtifact.model_validate(data)
