# Leaderboard

## Purpose

The leaderboard provides a reproducible, versioned record of how different RAG pipelines perform on the FinRAG benchmark. It is built entirely from static JSON artifacts — there is no live service, no database, and no backend.

---

## Artifact Schema

Every benchmark run produces three files under `outputs/<run_id>/`:

### artifact.json

The complete run record. This is the canonical leaderboard entry.

```python
class RunArtifact(BaseModel):
    run_id: str              # matches the run directory name
    timestamp: datetime      # UTC time when the run completed
    config_snapshot: dict    # full copy of RunConfig at run time
    scores: list[ScoredRow]  # one row per example
    aggregate: dict          # summary metrics over all rows
```

### scores.jsonl

One `ScoredRow` JSON object per line. Used for per-question analysis and error slicing.

```python
class ScoredRow(BaseModel):
    example_id: str
    answer: str
    citations: list[str]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    exact_match: bool
    citation_precision: float
    citation_recall: float
    faithful: bool
    faithfulness_score: float | None
```

### config.json

A copy of the `RunConfig` used for the run, saved separately for quick reference without parsing the full artifact.

```python
class RunConfig(BaseModel):
    run_id: str
    dataset_path: str
    chunks_path: str
    retriever: str
    top_k: int
    generation_adapter: str
    output_dir: str
```

---

## Aggregate Metrics

The `aggregate` field in `artifact.json` contains:

| Field | Type | Description |
|---|---|---|
| `n_examples` | int | Number of examples evaluated |
| `exact_match_rate` | float | Fraction with exact match (0–1) |
| `mean_faithfulness_score` | float | Mean faithfulness score, excludes None |
| `pct_faithful` | float | Fraction where faithful = True (0–1) |
| `mean_citation_precision` | float | Mean citation precision (0–1) |
| `mean_citation_recall` | float | Mean citation recall (0–1) |
| `median_latency_ms` | float | Median latency in milliseconds |
| `total_cost_usd` | float | Total estimated USD cost for the run |

---

## Run ID Convention

Run IDs should be descriptive and unique. Suggested format:

```
{pipeline-name}-{dataset-name}-{YYYY-MM-DD}
```

Examples:
- `bm25-mock-sample-2026-04-07`
- `azure-bm25-full-2026-04-07`

Run IDs are used as directory names and leaderboard keys. Do not reuse a run ID — create a new one for each run.

---

## Leaderboard Rules

For a run to be eligible for the public leaderboard:

1. **Dataset must be the full benchmark** — smoke/sample runs are for development only
2. **Run artifact must validate** — `validate_artifact()` must pass without errors
3. **Config snapshot must be complete** — all fields in `RunConfig` must be present
4. **Prompt version must be recorded** — faithfulness results must reference a named prompt version
5. **No cherry-picking** — the run must cover all examples in the dataset, not a subset
6. **Generation adapter must be documented** — model name, provider, and API version must be identifiable from the config

---

## Submitting a Leaderboard Entry

1. Run the full benchmark:
   ```bash
   python -m src.cli.run_eval --config configs/full.yaml
   ```

2. Validate the artifact:
   ```python
   from src.leaderboard.artifacts import validate_artifact
   from pathlib import Path
   artifact = validate_artifact(Path("outputs/<run_id>/artifact.json"))
   ```

3. Summarize the run:
   ```bash
   python -m src.cli.summarize_run --run-dir outputs/<run_id>
   ```

4. Open a pull request adding your `outputs/<run_id>/artifact.json` to the `leaderboard/` directory

---

## Reproducibility Requirements

A leaderboard entry is only reproducible if a reader can reconstruct the exact run from the artifact. This requires:

- The `config_snapshot` fully describes what was run
- The dataset used is publicly accessible (or the sample fixtures suffice)
- The generation adapter and model version are identified
- The prompt version used for faithfulness scoring is named

If any of these are missing, the entry cannot be independently verified.

---

## Artifact Validation

Use `validate_artifact()` to check an artifact before submission:

```python
from pathlib import Path
from src.leaderboard.artifacts import validate_artifact

artifact = validate_artifact(Path("outputs/my-run/artifact.json"))
print(f"Validated: {artifact.run_id}, {artifact.aggregate['n_examples']} examples")
```

This performs a full Pydantic round-trip deserialization. If validation raises, the artifact is malformed and cannot be submitted.
