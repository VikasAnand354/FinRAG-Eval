"""Artifact generation and aggregation for leaderboard runs."""

import json
import statistics
from pathlib import Path
from typing import Any

from src.common.models import RunArtifact, ScoredRow


def compute_aggregate(scores: list[ScoredRow]) -> dict[str, Any]:
    """Compute aggregate metrics from scored rows.

    Args:
        scores: List of scored examples from a run.

    Returns:
        Dictionary of aggregate metrics.
    """
    n = len(scores)
    if n == 0:
        return {"n_examples": 0}

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
    """Write artifact to disk with separate files for scores, config, and summary.

    Args:
        run: The run artifact to write.
        output_dir: Parent directory where run directory will be created.

    Returns:
        Path to the created run directory.
    """
    run_dir = output_dir / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "artifact.json").write_text(run.model_dump_json(indent=2))

    with (run_dir / "scores.jsonl").open("w") as f:
        for row in run.scores:
            f.write(row.model_dump_json() + "\n")

    (run_dir / "config.json").write_text(json.dumps(run.config_snapshot, indent=2))

    return run_dir


def validate_artifact(path: Path) -> RunArtifact:
    """Load and validate an artifact from disk.

    Args:
        path: Path to artifact.json file.

    Returns:
        Validated RunArtifact instance.
    """
    data = json.loads(path.read_text())
    return RunArtifact.model_validate(data)
