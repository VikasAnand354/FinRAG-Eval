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

    def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        if not self._chunks or self._bm25 is None:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._chunks[i] for i in ranked[:top_k]]
