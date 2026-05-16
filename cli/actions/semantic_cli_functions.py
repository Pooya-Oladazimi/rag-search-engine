from actions.semantic_search import SemanticSearch
from actions.libs import load_dataset


class SemanticCliFunctions:
    def __init__(self) -> None:
        self.search_engine = SemanticSearch()

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
