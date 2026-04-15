from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track

from src.common.config import load_run_config
from src.common.models import BenchmarkExample, Chunk, RunArtifact
from src.evaluators.faithfulness import FAITHFULNESS_PROMPT_VERSION
from src.evaluators.runner import evaluate
from src.leaderboard.artifacts import compute_aggregate, write_artifact
from src.leaderboard.summary import generate_summary
from src.retrieval.bm25 import BM25Retriever

app = typer.Typer()
console = Console()


@app.command()
def main(config: str = typer.Option(..., "--config", help="Path to YAML run config")) -> None:
    run_config = load_run_config(config)

    examples: list[BenchmarkExample] = []
    with open(run_config.dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(BenchmarkExample.model_validate_json(line))

    chunks: list[Chunk] = []
    with open(run_config.chunks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))

    retriever = BM25Retriever(chunks)

    if run_config.generation_adapter == "mock":
        from src.generation.azure_openai import MockGenerationAdapter

        adapter = MockGenerationAdapter()
    elif run_config.generation_adapter == "azure_openai":
        from src.generation.azure_openai import AzureOpenAIAdapter

        adapter = AzureOpenAIAdapter()
    else:
        raise ValueError(f"Unknown generation_adapter: {run_config.generation_adapter}")

    config_snapshot = {
        **run_config.model_dump(),
        "faithfulness_prompt_version": FAITHFULNESS_PROMPT_VERSION,
    }

    scored_rows = []
    for example in track(examples, description="Evaluating..."):
        retrieved = retriever.retrieve(example.question, top_k=run_config.top_k)
        result = adapter.generate(
            example_id=example.example_id,
            question=example.question,
            chunks=retrieved,
        )
        row = evaluate(
            result=result,
            example=example,
            chunks=retrieved,
            judge=adapter,
            config=run_config,
        )
        scored_rows.append(row)

    aggregate = compute_aggregate(scored_rows)
    artifact = RunArtifact(
        run_id=run_config.run_id,
        timestamp=datetime.now(UTC),
        config_snapshot=config_snapshot,
        scores=scored_rows,
        aggregate=aggregate,
    )

    out_dir = write_artifact(artifact, Path(run_config.output_dir))
    console.print(generate_summary(artifact))
    console.print(f"\nArtifact saved to: {out_dir}")


if __name__ == "__main__":
    app()
