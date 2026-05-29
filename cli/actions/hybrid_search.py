import os

from actions.indexer import InvertedIndex
from actions.chunked_semantic_search import ChunkedSemanticSearch
from actions.libs import normalize_scores
from actions.vars import INDEX_DATA


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(INDEX_DATA):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25 = self.idx.bm25_search(query=query, limit=limit * 500)
        semantics = self.semantic_search.search_chunks(query=query, limit=limit * 500)
        keyword_scores = normalize_scores(bm25.values())
        semantics_scores = normalize_scores([item["score"] for item in semantics])
        results = {}
        for doc in self.documents:
            docId = doc["id"]
            bm25_index = 0
            bm25_score = 0.0
            for id25 in bm25.keys():
                if id25 == docId:
                    bm25_score = keyword_scores[bm25_index]
                    break
                bm25_index += 1

            semantic_s = 0.0
            semantic_index = 0
            for res in semantics:
                if res["id"] == docId:
                    semantic_s = semantics_scores[semantic_index]
                    break
                semantic_index += 1

            results[docId] = {
                "document": doc,
                "bm25": bm25_score,
                "semantic": semantic_s,
                "hybrid": self.__hybrid_score(
                    bm25_score=bm25_score, semantic_score=semantic_s, alpha=alpha
                ),
            }
        return sorted(results.values(), key=lambda item: item["hybrid"], reverse=True)[
            :limit
        ]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

    def __hybrid_score(
        self, bm25_score: float, semantic_score: float, alpha: float = 0.5
    ) -> float:
        return alpha * bm25_score + (1 - alpha) * semantic_score
