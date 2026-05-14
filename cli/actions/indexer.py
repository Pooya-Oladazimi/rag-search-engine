from collections import defaultdict, Counter

from actions.vars import (
    DATASET,
    INDEX_DATA,
    DOCMAP_DATA,
    CACHE_DIR,
    TERM_FEQUENCIES_DATA,
    DOCS_LENGTH_CACHE,
    BM25_B,
    BM25_K1,
)
import json
import pickle
import os
import pathlib
from actions.text_processor import TextProcessor
import math


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(list)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = defaultdict(int)
        self.text_proc = TextProcessor()
        self.k1 = BM25_K1
        self.b = BM25_B

    def __add_document(self, doc_id, text):
        self.text_proc.process(text)
        self.doc_lengths[doc_id] = len(self.text_proc.tokens)
        for t in self.text_proc.tokens:
            if doc_id not in self.index[t]:
                self.index[t].append(doc_id)
        self.text_proc.tokens = []

    def get_documents(self, term: str) -> list[int]:
        # we assume each term is one token
        term = term.lower()
        self.text_proc.process(term)
        tokens = self.text_proc.tokens
        if len(tokens) != 1:
            raise Exception("input has to be one token")
        res = self.index.get(tokens[0], [])
        res.sort()
        return res

    def get_tf(self, doc_id, term) -> int:
        # term has to be one token
        self.text_proc.process(term)
        tokens = self.text_proc.tokens
        if len(tokens) != 1:
            raise Exception("search term has to be one token.")
        tf = self.term_frequencies[doc_id].get(tokens[0], 0)
        return tf

    def get_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap.keys()) + 1
        doc_frq = len(self.get_documents(term)) + 1
        return math.log(total_doc_count / doc_frq)

    def get_bm25_idf(self, term: str) -> float:
        df = len(self.get_documents(term))
        total_docs = len(self.docmap.keys())
        return math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, term: str, docId: int) -> float:
        tf = self.get_tf(doc_id=docId, term=term)
        length_norm = (
            1
            - self.b
            + self.b * (self.doc_lengths.get(docId, 0) / self.__get_avg_doc_length())
        )
        return (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)

    def bm25_score(self, term: str, docId: int):
        return self.get_bm25_tf(term=term, docId=docId) * self.get_bm25_idf(term=term)

    def bm25_search(self, query: str, limit: int = 5) -> dict[int, float]:
        self.text_proc.process(query)
        query_tokens = self.text_proc.tokens
        scores = defaultdict(float)
        for t in query_tokens:
            for docId in self.index[t]:
                scores[docId] += self.bm25_score(term=t, docId=docId)
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit])

    def build(self):
        with open(DATASET, "r") as f:
            dataset = json.loads(f.read())
            movies = dataset["movies"]
            for m in movies:
                doc_text = f"{m['title']} {m['description']}"
                self.__add_document(m["id"], doc_text)
                self.text_proc.process(doc_text)
                self.docmap[m["id"]] = m
                self.term_frequencies[m["id"]].update(self.text_proc.tokens)
                self.text_proc.tokens = []

    def save(self):
        target_dir = os.path.join(os.getcwd(), CACHE_DIR)
        p = pathlib.Path(target_dir)
        if not p.is_dir():
            os.mkdir(target_dir)
        with open(INDEX_DATA, "wb") as f:
            pickle.dump(self.index, f)

        with open(DOCMAP_DATA, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(TERM_FEQUENCIES_DATA, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(DOCS_LENGTH_CACHE, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if not self.__cache_file_exists(INDEX_DATA):
            raise Exception("index is missing. Build it first")
        if not self.__cache_file_exists(DOCMAP_DATA):
            raise Exception("docmap is missing. Build it first")
        if not self.__cache_file_exists(TERM_FEQUENCIES_DATA):
            raise Exception("term freq is missing. Build the index first")
        if not self.__cache_file_exists(DOCS_LENGTH_CACHE):
            raise Exception("Docs Length cache is missing. Build it first.")

        with open(INDEX_DATA, "rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_DATA, "rb") as f:
            self.docmap = pickle.load(f)
        with open(TERM_FEQUENCIES_DATA, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(DOCS_LENGTH_CACHE, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def __cache_file_exists(self, file_path: str):
        abs_file_path = os.path.join(os.getcwd(), file_path)
        p_file_path = pathlib.Path(abs_file_path)
        return p_file_path.is_file()

    def __get_avg_doc_length(self) -> float:
        docs_count = len(self.doc_lengths.keys())
        if not docs_count:
            return 0.0
        return sum(self.doc_lengths.values()) / docs_count
