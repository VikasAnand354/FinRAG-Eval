import json
import shutil
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
CANDIDATES_FIXTURE = FIXTURES / "sample_candidates.jsonl"


def test_load_existing_chunk_ids_empty(tmp_path):
    from src.ingest.dataset_builder import load_existing_chunk_ids

    assert load_existing_chunk_ids(tmp_path / "missing.jsonl") == set()


def test_load_existing_chunk_ids_from_fixture():
    from src.ingest.dataset_builder import load_existing_chunk_ids

    ids = load_existing_chunk_ids(CANDIDATES_FIXTURE)
    assert "aapl_10k_2023__c0001" in ids
    assert "msft_10k_2023__c0001" in ids
    assert "googl_10k_2023__c0001" in ids
    assert len(ids) == 3  # 3 unique chunk_ids across 5 candidates


def test_load_existing_document_ids_empty(tmp_path):
    from src.ingest.dataset_builder import load_existing_document_ids

    assert load_existing_document_ids(tmp_path / "missing.jsonl") == set()


def test_load_pending_candidates():
    from src.ingest.dataset_builder import load_pending_candidates

    pending = load_pending_candidates(CANDIDATES_FIXTURE)
    assert len(pending) == 3
    assert all(c.review_status == "pending" for c in pending)
    ids = {c.candidate_id for c in pending}
    assert "aapl_10k_2023__c0001-q0" in ids
    assert "msft_10k_2023__c0001-q0" in ids
    assert "googl_10k_2023__c0001-q0" in ids


def test_candidate_to_example_basic():
    from src.common.models import QACandidate
    from src.ingest.dataset_builder import candidate_to_example

    candidate = QACandidate(
        candidate_id="aapl_10k_2023__c0001-q0",
        chunk_id="aapl_10k_2023__c0001",
        document_id="aapl_10k_2023",
        company="AAPL",
        question="What were net sales?",
        gold_answer="$383.3 billion",
        difficulty="easy",
        question_type="numerical",
        gold_citations=["aapl_10k_2023__c0001"],
    )
    example = candidate_to_example(candidate, "q0001")

    assert example.example_id == "q0001"
    assert example.company == "AAPL"
    assert example.question == "What were net sales?"
    assert example.gold_answer == "$383.3 billion"
    assert example.difficulty == "easy"
    assert example.question_type == "numerical"
    assert len(example.gold_citations) == 1
    assert example.gold_citations[0].chunk_id == "aapl_10k_2023__c0001"
    assert example.gold_citations[0].document_id == "aapl_10k_2023"
    assert example.gold_citations[0].support_type == "direct"
    assert example.requires_multi_hop is False


def test_candidate_to_example_multi_hop():
    from src.common.models import QACandidate
    from src.ingest.dataset_builder import candidate_to_example

    candidate = QACandidate(
        candidate_id="x-q0",
        chunk_id="x__c0001",
        document_id="x",
        company="X",
        question="Compare two periods",
        gold_answer="revenue grew",
        difficulty="hard",
        question_type="multi_hop",
        gold_citations=["x__c0001", "x__c0002"],
    )
    example = candidate_to_example(candidate, "q0002")
    assert example.requires_multi_hop is True
    assert len(example.gold_citations) == 2


def test_append_jsonl_creates_file(tmp_path):
    from src.common.models import QACandidate
    from src.ingest.dataset_builder import append_jsonl

    c = QACandidate(
        candidate_id="x-q0",
        chunk_id="x__c0001",
        document_id="x",
        company="X",
        question="q",
        gold_answer="a",
        difficulty="easy",
        question_type="factual",
        gold_citations=["x__c0001"],
    )
    out = tmp_path / "out.jsonl"
    append_jsonl([c], out)
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["candidate_id"] == "x-q0"


def test_append_jsonl_appends(tmp_path):
    from src.common.models import QACandidate
    from src.ingest.dataset_builder import append_jsonl

    def _make(cid: str) -> QACandidate:
        return QACandidate(
            candidate_id=cid,
            chunk_id="x",
            document_id="x",
            company="X",
            question="q",
            gold_answer="a",
            difficulty="easy",
            question_type="factual",
            gold_citations=["x"],
        )

    out = tmp_path / "out.jsonl"
    append_jsonl([_make("x-q0")], out)
    append_jsonl([_make("x-q1")], out)
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 2


def test_update_candidate_status(tmp_path):
    from src.ingest.dataset_builder import update_candidate_status

    path = tmp_path / "candidates.jsonl"
    shutil.copy(CANDIDATES_FIXTURE, path)

    update_candidate_status(path, "aapl_10k_2023__c0001-q0", "approved", "looks good")

    lines = [line for line in path.read_text().splitlines() if line.strip()]
    updated = {json.loads(line)["candidate_id"]: json.loads(line) for line in lines}
    assert updated["aapl_10k_2023__c0001-q0"]["review_status"] == "approved"
    assert updated["aapl_10k_2023__c0001-q0"]["reviewer_note"] == "looks good"
    # Other records untouched
    assert updated["msft_10k_2023__c0001-q0"]["review_status"] == "pending"
