import json
from pathlib import Path

from src.common.models import RunArtifact


def test_golden_run_artifact_schema_valid():
    path = Path("tests/fixtures/golden_run.json")
    data = json.loads(path.read_text())
    artifact = RunArtifact.model_validate(data)
    assert artifact.run_id == "golden-001"
    assert len(artifact.scores) >= 1
    assert artifact.aggregate["n_examples"] >= 1
