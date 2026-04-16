"""Dataset construction CLI for FinRAG Eval.

Commands:
  generate  Fetch SEC filings, chunk, and generate QA candidates.
  review    Interactively review candidates and write approved examples.
"""

import time
from datetime import UTC, datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from src.common.config import load_dataset_build_config
from src.common.models import Chunk, DocumentMetadata, QACandidate
from src.ingest.chunker import chunk_document
from src.ingest.dataset_builder import (
    append_jsonl,
    candidate_to_example,
    load_existing_chunk_ids,
    load_existing_document_ids,
    load_pending_candidates,
    update_candidate_full,
    update_candidate_status,
)
from src.ingest.qa_generator import generate_candidates, is_interesting_chunk
from src.ingest.sec_fetch import (
    download_filing,
    fetch_company_submissions,
    filter_filings,
    get_cik_from_ticker,
)
from src.ingest.text_extract import strip_html

app = typer.Typer(help="Build the FinRAG Eval financial QA dataset.")
console = Console()


def _init_adapter(adapter_name: str):
    if adapter_name == "mock":
        from src.generation.azure_openai import MockGenerationAdapter

        return MockGenerationAdapter()
    if adapter_name == "azure_openai":
        from src.generation.azure_openai import AzureOpenAIAdapter

        return AzureOpenAIAdapter()
    raise ValueError(f"Unknown adapter: {adapter_name}")


@app.command()
def generate(
    config: str = typer.Option(..., "--config", help="Path to dataset_build.yaml"),
) -> None:
    """Fetch SEC filings, extract text, chunk, and generate QA candidates."""
    load_dotenv()
    cfg = load_dataset_build_config(config)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"

    docs_path = output_dir / "documents.jsonl"
    chunks_path = output_dir / "chunks.jsonl"
    candidates_path = output_dir / "candidates.jsonl"

    existing_doc_ids = load_existing_document_ids(docs_path)
    existing_chunk_ids = load_existing_chunk_ids(candidates_path)

    adapter = _init_adapter(cfg.generation_adapter)

    for ticker in cfg.tickers:
        console.print(f"\n[bold]Processing {ticker}...[/bold]")
        cik = get_cik_from_ticker(ticker)
        submissions = fetch_company_submissions(cik)
        company_name: str = submissions.get("name", ticker)

        filings = filter_filings(
            submissions,
            form_types=tuple(cfg.form_types),
            limit=len(cfg.fiscal_years) * len(cfg.form_types) * 3,
            scan_limit=30000,
        )
        filings = [f for f in filings if int(f["filing_date"][:4]) in cfg.fiscal_years]

        for filing in filings:
            # Use full filing date (YYYYMMDD) to avoid collisions between
            # multiple 10-Qs in the same year (e.g. aapl_10q_20240629).
            form_slug = filing["form"].replace("-", "").lower()
            date_slug = filing["filing_date"].replace("-", "")
            document_id = f"{ticker.lower()}_{form_slug}_{date_slug}"

            if document_id in existing_doc_ids:
                console.print(f"  [dim]Skip {document_id} (already processed)[/dim]")
                continue

            console.print(f"  Fetching {ticker} {filing['form']} {filing['filing_date']}...")
            filing_path = download_filing(cik, filing, raw_dir / ticker)

            html = filing_path.read_bytes().decode("utf-8", errors="replace")
            text = strip_html(html)

            doc_meta = DocumentMetadata(
                document_id=document_id,
                company=ticker,
                company_name=company_name,
                source_type=filing["form"],
                source_url=None,
                doc_family=filing["form"],
                fiscal_period=filing["filing_date"][:4],
                calendar_date=None,
                filing_date=filing["filing_date"],
                report_period_end=None,
                format="html",
                local_path=str(filing_path),
                sha256=None,
                collected_at=datetime.now(UTC),
            )
            append_jsonl([doc_meta], docs_path)

            chunk_dicts = chunk_document(
                document_id, ticker, text, filing_date=filing["filing_date"]
            )
            chunks = [Chunk(**d) for d in chunk_dicts]
            append_jsonl(chunks, chunks_path)

            interesting = [
                c
                for c in chunks
                if is_interesting_chunk(c, cfg.sections) and c.chunk_id not in existing_chunk_ids
            ][: cfg.top_k_chunks_per_section]

            new_candidates: list[QACandidate] = []
            for chunk in interesting:
                cands = generate_candidates(
                    chunk, adapter, cfg, form_type=filing["form"], period=filing["filing_date"][:4]
                )
                new_candidates.extend(cands)
                existing_chunk_ids.add(chunk.chunk_id)

            if new_candidates:
                append_jsonl(new_candidates, candidates_path)

            existing_doc_ids.add(document_id)
            console.print(
                f"  {ticker} {filing['form']} {filing['filing_date']} → "
                f"{len(chunks)} chunks → "
                f"{len(interesting)} interesting → "
                f"{len(new_candidates)} candidates"
            )
            time.sleep(0.5)

    total = (
        sum(1 for line in candidates_path.open() if line.strip()) if candidates_path.exists() else 0
    )
    console.print(f"\n[green]Done. {total} total candidates in {candidates_path}[/green]")


@app.command()
def review(
    config: str = typer.Option(..., "--config", help="Path to dataset_build.yaml"),
) -> None:
    """Interactively review QA candidates and write approved ones to qa_examples.jsonl."""
    load_dotenv()
    cfg = load_dataset_build_config(config)
    output_dir = Path(cfg.output_dir)
    candidates_path = output_dir / "candidates.jsonl"
    examples_path = output_dir / "qa_examples.jsonl"

    pending = load_pending_candidates(candidates_path)
    if not pending:
        console.print("[green]No pending candidates.[/green]")
        raise typer.Exit()

    existing_count = (
        sum(1 for line in examples_path.open() if line.strip()) if examples_path.exists() else 0
    )

    console.print(f"[bold]{len(pending)} candidates pending review[/bold]\n")

    for i, candidate in enumerate(pending):
        console.rule(
            f"[bold cyan][{i + 1}/{len(pending)}][/bold cyan] "
            f"{candidate.company} | {candidate.difficulty} | {candidate.question_type}"
        )
        console.print(f"[dim]Chunk:[/dim] {candidate.chunk_id}")
        console.print(f"[yellow]Q:[/yellow] {candidate.question}")
        console.print(f"[green]A:[/green] {candidate.gold_answer}")
        console.print("\n[dim](a)pprove  (e)dit  (r)eject  (s)kip  (q)uit[/dim]")

        choice = ""
        while choice not in ("a", "e", "r", "s", "q"):
            choice = input("> ").strip().lower()

        if choice == "q":
            console.print("[yellow]Exiting. Progress saved.[/yellow]")
            break
        if choice == "s":
            continue
        if choice == "r":
            note = input("Rejection note (optional): ").strip()
            update_candidate_status(candidates_path, candidate.candidate_id, "rejected", note)
            console.print("[red]Rejected.[/red]")
            continue
        if choice == "e":
            new_q = input(f"Question [{candidate.question}]: ").strip()
            new_a = input(f"Answer [{candidate.gold_answer}]: ").strip()
            if new_q:
                candidate.question = new_q
            if new_a:
                candidate.gold_answer = new_a
            update_candidate_full(candidates_path, candidate)

        # approve (choice == "a" or after edit)
        existing_count += 1
        example_id = f"q{existing_count:04d}"
        example = candidate_to_example(candidate, example_id)
        append_jsonl([example], examples_path)
        update_candidate_status(candidates_path, candidate.candidate_id, "approved")
        console.print(f"[green]Approved as {example_id}[/green]")

    final_count = (
        sum(1 for line in examples_path.open() if line.strip()) if examples_path.exists() else 0
    )
    console.print(f"\n[bold]Done. {final_count} total examples in {examples_path}[/bold]")


if __name__ == "__main__":
    app()
