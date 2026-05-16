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
