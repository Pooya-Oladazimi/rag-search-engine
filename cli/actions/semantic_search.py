import os
from sentence_transformers import SentenceTransformer
import numpy as np
from actions.vars import DOCS_EMBEDDINGS_CACHE
from pathlib import Path


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty for embedding.")
        embeded = self.model.encode([text])
        return embeded[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        docs_text_list = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            docs_text_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(docs_text_list, show_progress_bar=True)
        with open(DOCS_EMBEDDINGS_CACHE, "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        cache_full_path = os.path.abspath(DOCS_EMBEDDINGS_CACHE)
        p = Path(cache_full_path)
        if p.is_file():
            with open(DOCS_EMBEDDINGS_CACHE, "rb") as f:
                self.embeddings = np.load(f)
                if len(self.embeddings) == len(self.documents):
                    return self.embeddings
        else:
            return self.build_embeddings(documents=documents)

    def cosine_similarity(self, vec1, vec2) -> float:
        prod = np.dot(vec1, vec2)
        mag_vec1 = np.linalg.norm(vec1)
        mag_vec2 = np.linalg.norm(vec2)
        if not mag_vec1 or not mag_vec2:
            return 0.0
        return prod / (mag_vec1 * mag_vec2)

    def search(self, query: str, limit: int):
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_emedding = self.generate_embedding(text=query)
        scores = []
        for i in range(len(self.documents)):
            doc_vec = self.embeddings[i]
            doc = self.documents[i]
            sim = self.cosine_similarity(query_emedding, doc_vec)
            t = tuple([sim, doc])
            scores.append(t)
        tops = sorted(scores, key=lambda x: x[0], reverse=True)[:limit]
        res = []
        for s in tops:
            res.append(
                {
                    "score": s[0],
                    "title": s[1]["title"],
                    "description": s[1]["description"],
                }
            )
        return res
