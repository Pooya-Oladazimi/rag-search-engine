from actions.indexer import InvertedIndex
import math
from actions.indexer import InvertedIndex
from actions.text_processor import cleaned_tokenize


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
        q_tokens = cleaned_tokenize(query)
        self.__load()
        for t in q_tokens:
            docs = self.indexer.get_documents(t)
            for dId in docs:
                result.append(self.indexer.docmap[dId])
                if len(result) == self.search_result_limit:
                    return result

        return result

    def search(self, query):
        print(f"Searching for: {query}")
        self.__render_movies_titles(self.__search(query))

    def build(self):
        self.indexer.build()
        self.indexer.save()
        self.index_is_loaded = False
        return True

    def tf(self, docId, term):
        self.__load()
        print(self.indexer.get_tf(docId, term))
        return True

    def idf(self, term):
        self.__load()
        total_doc_count = len(self.indexer.docmap.keys()) + 1
        doc_frq = len(self.indexer.get_documents(term)) + 1
        idf = math.log(total_doc_count / doc_frq)
        print(f"Inverse document frequency of '{term}': {idf:.2f}")
        return True
