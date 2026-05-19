from actions.semantic_search import SemanticSearch
from actions.libs import load_dataset
from actions.chunked_semantic_search import ChunkedSemanticSearch


class SemanticCliFunctions:
    def __init__(self) -> None:
        self.search_engine = SemanticSearch()
        self.chunked_search_engine = ChunkedSemanticSearch()

    def verify_model(self):
        print(f"Model loaded: {self.search_engine.model}")
        print(f"Max sequence length: {self.search_engine.model.max_seq_length}")

    def embed_text(self, text: str):
        return self.search_engine.generate_embedding(text=text)

    def verify_embeddings(self):
        documents = load_dataset()
        return len(documents), self.search_engine.load_or_create_embeddings(
            documents=documents
        )

    def search(self, query: str, limit: int = 5):
        documents = load_dataset()
        self.search_engine.load_or_create_embeddings(documents)
        tops = self.search_engine.search(query=query, limit=limit)
        counter = 1
        for res in tops:
            print(
                f"{counter}. {res['title']} (score: {res['score']:.4f})\n{res['description'][:200]} ..\n"
            )
            counter += 1
        return True

    def chunk(self, text: str, chunk_size: int, overlap: int = 0) -> list[str]:
        res = []
        chars = text.split(" ")
        for i in range(0, len(chars), chunk_size):
            if i == 0:
                res.append(" ".join(chars[i : i + chunk_size]))
            else:
                res.append(" ".join(chars[i - overlap : i + chunk_size]))
        return res

    def semantic_chunk(self, text: str, max_chunk_size: int, overlap: int):
        return self.chunked_search_engine.semantic_chunk(text, max_chunk_size, overlap)

    def embed_chunks(self):
        documents = load_dataset()
        return self.chunked_search_engine.load_or_create_chunk_embeddings(documents)
