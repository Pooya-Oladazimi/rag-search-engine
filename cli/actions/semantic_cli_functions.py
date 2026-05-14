from actions.semantic_search import SemanticSearch


class SemanticCliFunctions:
    def __init__(self) -> None:
        self.search_engine = SemanticSearch()

    def verify_model(self):
        print(f"Model loaded: {self.search_engine.model}")
        print(f"Max sequence length: {self.search_engine.model.max_seq_length}")
