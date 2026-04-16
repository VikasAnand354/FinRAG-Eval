from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from src.common.models import Chunk, DatasetBuildConfig, QACandidate
from src.ingest.text_extract import is_prose_paragraph

if TYPE_CHECKING:
    from src.generation.base import GenerationAdapter

logger = logging.getLogger(__name__)

QA_GENERATION_PROMPT_V1 = """\
Given this excerpt from {company}'s {form_type} filing ({period}):

<chunk>
{text}
</chunk>

Generate {n} question-answer pairs. For each pair:
- The question must be answerable solely from the excerpt
- The answer must be a direct quote or close paraphrase from the excerpt
- Assign difficulty: easy (single fact lookup), medium (requires inference \
or calculation), hard (requires comparing multiple facts or multi-step reasoning)
- Assign type: factual | numerical | comparative | multi_hop

Return a JSON array only, no other text:
[{{"question": "...", "answer": "...", \
"difficulty": "easy|medium|hard", "type": "factual|numerical|comparative|multi_hop"}}]
"""

QA_GENERATION_PROMPT_VERSION = "qa-gen-v1"

_FINANCIAL_SIGNAL = re.compile(r"[\d%$£€]")


def is_interesting_chunk(chunk: Chunk, sections: list[str]) -> bool:
    """Return True if the chunk is worth generating Q&A from.

    Filters out:
    - Chunks with fewer than 80 tokens (too short for a real question)
    - Chunks with no financial signal (numbers, %, currency symbols)
    - Chunks dominated by XBRL namespace tokens (not readable prose)
    - Chunks whose section_title is set but not in the allowed sections list
    """
    if chunk.token_count < 80:
        return False
    if not _FINANCIAL_SIGNAL.search(chunk.text):
        return False
    if not is_prose_paragraph(chunk.text):
        return False
    return chunk.section_title is None or chunk.section_title in sections


def generate_candidates(
    chunk: Chunk,
    adapter: GenerationAdapter,
    config: DatasetBuildConfig,
    form_type: str = "10-K",
    period: str = "",
) -> list[QACandidate]:
    """Generate QA candidates for a single chunk using the adapter.

    Returns an empty list if the LLM response cannot be parsed.
    """
    prompt = QA_GENERATION_PROMPT_V1.format(
        company=chunk.company,
        form_type=form_type,
        period=period or chunk.report_period_end or "unknown period",
        text=chunk.text,
        n=config.questions_per_chunk,
    )
    try:
        raw, _, _, _ = adapter.judge(prompt)
        pairs = json.loads(raw)
    except Exception:
        logger.warning("Failed to parse QA candidates for chunk %s", chunk.chunk_id)
        return []

    candidates: list[QACandidate] = []
    for i, pair in enumerate(pairs):
        try:
            candidates.append(
                QACandidate(
                    candidate_id=f"{chunk.chunk_id}-q{i}",
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    company=chunk.company,
                    question=pair["question"],
                    gold_answer=pair["answer"],
                    difficulty=pair["difficulty"],
                    question_type=pair["type"],
                    gold_citations=[chunk.chunk_id],
                    review_status="pending",
                    qa_prompt_version=QA_GENERATION_PROMPT_VERSION,
                )
            )
        except Exception:
            logger.warning("Skipping malformed candidate %d for chunk %s", i, chunk.chunk_id)
    return candidates
