import json
from actions.vars import DATASET
from actions.indexer import InvertedIndex
from actions.text_processor import cleaned_tokenize


def search(q: str, limit=5):
    with open(DATASET, "r") as dts:
        movies = json.loads(dts.read())
        movies = movies["movies"]
        result = []
        q_tokens = cleaned_tokenize(q)
        index = InvertedIndex()
        index.load()
        for t in q_tokens:
            docs = index.get_documents(t)
            for dId in docs:
                result.append(index.docmap[dId])
                if len(result) == limit:
                    return result

        return result
