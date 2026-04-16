import json
from pathlib import Path

from src.common.models import BenchmarkExample, GoldCitation, QACandidate

_ANSWER_TYPE_MAP = {
    "numerical": "numerical",
    "comparative": "long",
    "multi_hop": "long",
    "factual": "short",
}


def load_existing_chunk_ids(candidates_path: Path) -> set[str]:
    """Return set of chunk_ids already present in candidates_path."""
    if not candidates_path.exists():
        return set()
    ids: set[str] = set()
    with open(candidates_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["chunk_id"])
    return ids


def load_existing_document_ids(documents_path: Path) -> set[str]:
    """Return set of document_ids already present in documents_path."""
    if not documents_path.exists():
        return set()
    ids: set[str] = set()
    with open(documents_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["document_id"])
    return ids


def load_pending_candidates(candidates_path: Path) -> list[QACandidate]:
    """Load all candidates with review_status == 'pending'."""
    if not candidates_path.exists():
        return []
    result: list[QACandidate] = []
    with open(candidates_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if data.get("review_status") == "pending":
                    result.append(QACandidate(**data))
    return result


def candidate_to_example(
    candidate: QACandidate,
    example_id: str,
    source_split: str = "test",
) -> BenchmarkExample:
    """Convert an approved QACandidate to a BenchmarkExample.

    Args:
        candidate:    Approved QACandidate record.
        example_id:   Unique ID to assign (e.g. "q0001").
        source_split: Dataset split — "train", "dev", or "test".
    """
    answer_type = _ANSWER_TYPE_MAP.get(candidate.question_type, "short")
    return BenchmarkExample(
        example_id=example_id,
        company=candidate.company,
        question=candidate.question,
        question_type=candidate.question_type,
        difficulty=candidate.difficulty,
        answer_type=answer_type,
        gold_answer=candidate.gold_answer,
        gold_answer_normalized=None,
        normalization_unit=None,
        gold_citations=[
            GoldCitation(
                document_id=candidate.document_id,
                chunk_id=cid,
                support_type="direct",
            )
            for cid in candidate.gold_citations
        ],
        acceptable_aliases=[],
        time_sensitive=False,
        requires_multi_hop=candidate.question_type == "multi_hop",
        source_split=source_split,
    )


def append_jsonl(records: list, path: Path) -> None:
    """Append Pydantic model records to a JSONL file, creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


def update_candidate_status(
    candidates_path: Path,
    candidate_id: str,
    status: str,
    reviewer_note: str = "",
) -> None:
    """Update review_status and reviewer_note for a single candidate in-place."""
    lines = candidates_path.read_text().splitlines()
    updated: list[str] = []
    for line in lines:
        if not line.strip():
            updated.append(line)
            continue
        data = json.loads(line)
        if data["candidate_id"] == candidate_id:
            data["review_status"] = status
            data["reviewer_note"] = reviewer_note
            updated.append(json.dumps(data))
        else:
            updated.append(line)
    candidates_path.write_text("\n".join(updated) + "\n")


def update_candidate_full(candidates_path: Path, candidate: QACandidate) -> None:
    """Rewrite a candidate record in-place with all updated fields."""
    lines = candidates_path.read_text().splitlines()
    updated: list[str] = []
    for line in lines:
        if not line.strip():
            updated.append(line)
            continue
        data = json.loads(line)
        if data["candidate_id"] == candidate.candidate_id:
            updated.append(candidate.model_dump_json())
        else:
            updated.append(line)
    candidates_path.write_text("\n".join(updated) + "\n")
