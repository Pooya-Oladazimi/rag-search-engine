from actions.indexer import InvertedIndex
from actions.indexer import InvertedIndex
from actions.text_processor import TextProcessor


class CliFunctions:
    def __init__(self):
        self.indexer = InvertedIndex()
        self.index_is_loaded = False
        self.search_result_limit = 5

    def __load(self):
        if not self.index_is_loaded:
            self.indexer.load()
            self.index_is_loaded = True

    def __render_movies_titles(self, movies):
        for i in range(len(movies)):
            movie = movies[i]
            print(f"{i+1}. {movie['title']}")
        return True

    def __search(self, query):
        result = []
        text_proc = TextProcessor()
        text_proc.process(query)
        q_tokens = text_proc.tokens
        self.__load()
        for t in q_tokens:
            docs = self.indexer.get_documents(t)
            for dId in docs:
                result.append(self.indexer.docmap[dId])
                if len(result) == self.search_result_limit:
                    return result

        return result

    def __render_bm25_search_results(self, scores: dict[int, float]):
        counter = 1
        for docId, score in scores.items():
            print(
                f"{counter}. ({docId}) {self.indexer.docmap[docId]['title']} - Score: {score:.2f}"
            )
            counter += 1

    def search(self, query):
        print(f"Searching for: {query}")
        self.__render_movies_titles(self.__search(query))

    def build(self):
        self.indexer.build()
        self.indexer.save()
        self.index_is_loaded = False
        return True

    def tf(self, docId, term) -> int:
        self.__load()
        tf_res = self.indexer.get_tf(docId, term)
        return tf_res

    def idf(self, term) -> float:
        self.__load()
        idf = self.indexer.get_idf(term)
        return idf

    def tf_idf(self, docId, term) -> float:
        idf = self.idf(term)
        tf = self.tf(docId, term)
        return tf * idf

    def bm25idf(self, term: str) -> float:
        self.__load()
        return self.indexer.get_bm25_idf(term)

    def bm25tf(self, docId: int, term: str, k1: float, b: float) -> float:
        self.__load()
        self.indexer.k1 = k1
        self.indexer.b = b
        return self.indexer.get_bm25_tf(term=term, docId=docId)

    def bm25search(self, query: str, limit: int):
        self.__load()
        results = self.indexer.bm25_search(query=query, limit=limit)
        self.__render_bm25_search_results(results)
