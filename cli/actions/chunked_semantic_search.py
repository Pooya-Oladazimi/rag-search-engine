from collections import defaultdict
from actions.semantic_search import SemanticSearch
import re
from actions.vars import CHUNKS_EMBEDDINGS_CACHE, CHUNKS_DOCS_METADATA, SCORE_PRECISION
import numpy as np
import json
import os


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def search_chunks(self, query: str, limit: int = 10):
        query = query.strip()
        if not query:
            return []
        q_embedding = self.generate_embedding(text=query)
        chunks_scores = []
        idx = 0
        for ch_embed in self.chunk_embeddings:
            score = self.cosine_similarity(q_embedding, ch_embed)
            res = {}
            res["chunk_idx"] = idx
            res["movie_idx"] = self.chunk_metadata["chunks"][idx]["movie_idx"]
            res["score"] = score
            idx += 1
            chunks_scores.append(res)

        movie_scores = defaultdict(float)
        for score in chunks_scores:
            if score["score"] > movie_scores[score["movie_idx"]]:
                movie_scores[score["movie_idx"]] = score["score"]

        sorted_movie_scores = sorted(
            movie_scores.items(), key=lambda item: item[1], reverse=True
        )
        search_results = []
        for s in sorted_movie_scores[:limit]:
            idx = s[0]
            score = s[1]
            search_results.append(
                {
                    "id": self.documents[idx]["id"],
                    "title": self.documents[idx]["title"],
                    "description": self.documents[idx]["description"][:100],
                    "score": round(score, SCORE_PRECISION),
                    "metadata": self.documents[idx],
                }
            )
        return search_results

    def build_chunk_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        chunks = []
        self.chunk_metadata = []
        doc_idx = 0
        for doc in documents:
            if not doc["description"]:
                continue
            doc_chunks = self.semantic_chunk(
                text=doc["description"], max_chunk_size=4, overlap=1
            )
            chunk_idx = 0
            for chu in doc_chunks:
                chunks.append(chu)
                self.chunk_metadata.append(
                    {
                        "movie_idx": doc_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(doc_chunks),
                    }
                )
                chunk_idx += 1

            doc_idx += 1

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        with open(CHUNKS_EMBEDDINGS_CACHE, "wb") as f:
            np.save(f, self.chunk_embeddings)
        with open(CHUNKS_DOCS_METADATA, "w") as f:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(chunks)},
                f,
                indent=2,
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(CHUNKS_EMBEDDINGS_CACHE) and os.path.exists(
            CHUNKS_DOCS_METADATA
        ):
            with open(CHUNKS_EMBEDDINGS_CACHE, "rb") as f:
                self.chunk_embeddings = np.load(f)
            with open(CHUNKS_DOCS_METADATA, "r") as f:
                self.chunk_metadata = json.loads(f.read())
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def semantic_chunk(self, text: str, max_chunk_size: int, overlap: int):
        text = text.strip()
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        i = 0
        n_sentences = len(sentences)
        if n_sentences == 1 and not sentences[0].endswith((".", "!", "?")):
            return sentences[0]
        while i < n_sentences:
            chunk_sentences = sentences[i : i + max_chunk_size]
            if chunks and len(chunk_sentences) <= overlap:
                break
            cleaned_sentences = []
            for s in chunk_sentences:
                s = s.strip()
                cleaned_sentences.append(s)
            ch = " ".join(cleaned_sentences)
            if ch:
                chunks.append(ch)
            i += max_chunk_size - overlap
        return chunks
