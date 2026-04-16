from rank_bm25 import BM25Okapi

from src.common.models import Chunk


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        if chunks:
            tokenized = [c.text.lower().split() for c in chunks]
            self._bm25: BM25Okapi | None = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def retrieve(self, query: str, top_k: int, company: str | None = None) -> list[Chunk]:
        """Retrieve top-k chunks by BM25 score.

        Args:
            query:   The question text to match against.
            top_k:   Maximum number of chunks to return.
            company: If provided, restrict retrieval to chunks from this company.
                     Prevents cross-company contamination in citation metrics.
        """
        if not self._chunks or self._bm25 is None:
            return []

        if company is not None:
            # Score all chunks but only return hits from the target company
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)
            ranked = sorted(
                (i for i, c in enumerate(self._chunks) if c.company == company),
                key=lambda i: scores[i],
                reverse=True,
            )
            return [self._chunks[i] for i in ranked[:top_k]]

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._chunks[i] for i in ranked[:top_k]]
