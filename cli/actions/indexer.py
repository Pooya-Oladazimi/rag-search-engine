from collections import defaultdict, Counter

from actions.vars import (
    DATASET,
    INDEX_DATA,
    DOCMAP_DATA,
    CACHE_DIR,
    TERM_FEQUENCIES_DATA,
)
import json
import pickle
import os
import pathlib
from actions.text_processor import cleaned_tokenize


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(list)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id, text):
        tokens = cleaned_tokenize(text)
        for t in tokens:
            if doc_id not in self.index[t]:
                self.index[t].append(doc_id)

    def get_documents(self, term: str) -> list[int]:
        # we assume each term is one token
        term = term.lower()
        tokens = cleaned_tokenize(term)
        if len(tokens) != 1:
            raise Exception("input has to be one token")
        res = self.index.get(tokens[0], [])
        res.sort()
        return res

    def get_tf(self, doc_id, term):
        # term has to be one token
        tokens = cleaned_tokenize(term)
        if len(tokens) != 1:
            raise Exception("search term has to be one token.")
        tf = self.term_frequencies[doc_id].get(tokens[0])
        if not tf:
            return "O"
        return tf

    def build(self):
        with open(DATASET, "r") as f:
            dataset = json.loads(f.read())
            movies = dataset["movies"]
            for m in movies:
                doc_text = f"{m['title']} {m['description']}"
                self.__add_document(m["id"], doc_text)
                self.docmap[m["id"]] = m
                self.term_frequencies[m["id"]].update(cleaned_tokenize(doc_text))

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

    def load(self):
        if not self.__cache_file_exists(INDEX_DATA):
            raise Exception("index is missing. Build it first")
        if not self.__cache_file_exists(DOCMAP_DATA):
            raise Exception("docmap is missing. Build it first")
        if not self.__cache_file_exists(TERM_FEQUENCIES_DATA):
            raise Exception("term freq is missing. Build the index first")

        with open(INDEX_DATA, "rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_DATA, "rb") as f:
            self.docmap = pickle.load(f)
        with open(TERM_FEQUENCIES_DATA, "rb") as f:
            self.term_frequencies = pickle.load(f)

    def __cache_file_exists(self, file_path: str):
        abs_file_path = os.path.join(os.getcwd(), file_path)
        p_file_path = pathlib.Path(abs_file_path)
        return p_file_path.is_file()
