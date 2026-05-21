from sentence_transformers.sentence_transformer.datasets import sentences
from actions.semantic_search import SemanticSearch
import re
from actions.vars import CHUNKS_EMBEDDINGS_CACHE, CHUNKS_DOCS_METADATA
import numpy as np
import json
import os


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

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
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        last_processed_index = 0
        for i in range(0, len(sentences), max_chunk_size - overlap):
            if i == 0:
                last_processed_index = max_chunk_size
                chunks.append(" ".join(sentences[i:last_processed_index]))
            else:
                last_processed_index = i + max_chunk_size
                chunks.append(" ".join(sentences[i:last_processed_index]))
        if last_processed_index < len(sentences):
            chunks.append(" ".join(sentences[last_processed_index - overlap :]))
        return chunks
