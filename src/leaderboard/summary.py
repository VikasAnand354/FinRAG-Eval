"""Leaderboard summary generation."""

from src.common.models import RunArtifact


def generate_summary(artifact: RunArtifact) -> str:
    """Generate a Markdown summary table from a run artifact.

    Args:
        artifact: Completed run artifact with scores and aggregate metrics.

    Returns:
        Markdown-formatted summary table.
    """
    agg = artifact.aggregate
    n = agg.get("n_examples", 0)

    lines = [
        f"## Run: {artifact.run_id}",
        f"**Timestamp:** {artifact.timestamp.isoformat()}  |  **Examples:** {n}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Exact Match Rate | {agg.get('exact_match_rate', 0):.1%} |",
        f"| Mean Faithfulness Score | {agg.get('mean_faithfulness_score', 0):.3f} |",
        f"| % Faithful | {agg.get('pct_faithful', 0):.1%} |",
        f"| Mean Citation Precision | {agg.get('mean_citation_precision', 0):.3f} |",
        f"| Mean Citation Recall | {agg.get('mean_citation_recall', 0):.3f} |",
        f"| Median Latency (ms) | {agg.get('median_latency_ms', 0):.0f} |",
        f"| Total Cost (USD) | ${agg.get('total_cost_usd', 0):.4f} |",
    ]
    return "\n".join(lines)
