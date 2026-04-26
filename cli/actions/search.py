import json
import string
from nltk.stem import PorterStemmer

DATASET = "data/movies.json"
STOPWORDS_DATASET = "data/stopwords.txt"


def search(q: str, limit=5):
    with open(DATASET, "r") as dts:
        movies = json.loads(dts.read())
        movies = movies["movies"]
        result = []
        q_tokens = stem_tokens(remove_stopwords(tokenize(clean_string(q))))
        for record in movies:
            title_tokens = stem_tokens(
                remove_stopwords(tokenize(clean_string(record.get("title", ""))))
            )
            for t in q_tokens:
                if any(t in title for title in title_tokens):
                    result.append(record)
            if len(result) == limit:
                break
        return result


def clean_string(s: str) -> str:
    sc = s.lower()
    sc_table = str.maketrans("", "", string.punctuation)
    sc = sc.translate(sc_table)
    return sc


def tokenize(s: str) -> list[str]:
    tokens = []
    for t in s.split(" "):
        if t:
            tokens.append(t)
    return tokens


def remove_stopwords(tokens: list[str]) -> list[str]:
    with open(STOPWORDS_DATASET, "r") as f:
        st_words = f.read().splitlines()
        clean_tokens = []
        for t in tokens:
            if t not in st_words:
                clean_tokens.append(t)
        return clean_tokens


def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    res = []
    for t in tokens:
        res.append(stemmer.stem(t))
    return list(set(res))
