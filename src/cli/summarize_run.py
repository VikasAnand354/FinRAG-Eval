from pathlib import Path

import typer
from rich.console import Console

from src.leaderboard.artifacts import validate_artifact
from src.leaderboard.summary import generate_summary

app = typer.Typer()
console = Console()


@app.command()
def main(run_dir: str = typer.Option(..., "--run-dir", help="Path to run output directory")) -> None:
    artifact_path = Path(run_dir) / "artifact.json"
    if not artifact_path.exists():
        console.print(f"[red]artifact.json not found in {run_dir}[/red]")
        raise typer.Exit(code=1)
    artifact = validate_artifact(artifact_path)
    console.print(generate_summary(artifact))


if __name__ == "__main__":
    app()
