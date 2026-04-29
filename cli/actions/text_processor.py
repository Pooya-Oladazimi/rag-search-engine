from nltk.stem import PorterStemmer
import string
from actions.vars import STOPWORDS_DATASET


def cleaned_tokenize(s: str) -> list[str]:
    return stem_tokens(remove_stopwords(tokenize(clean_string(s))))


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
    return res
