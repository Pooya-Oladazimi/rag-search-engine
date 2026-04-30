from nltk.stem import PorterStemmer
import string
from actions.vars import STOPWORDS_DATASET


class TextProcessor:
    def __init__(self) -> None:
        self.__cleaned_string = ""
        self.__raw_tokens = []
        self.__raw_tokens_without_stopwords = []
        self.tokens = []

    def process(self, s: str):
        self.__cleaned_string = self.__clean_string(s)
        self.__raw_tokens = self.__tokenize(self.__cleaned_string)
        self.__raw_tokens_without_stopwords = self.__remove_stopwords(self.__raw_tokens)
        self.tokens = self.__stem(self.__raw_tokens_without_stopwords)
        return True

    def __clean_string(self, s: str) -> str:
        sc = s.lower()
        sc_table = str.maketrans("", "", string.punctuation)
        sc = sc.translate(sc_table)
        return sc

    def __tokenize(self, s: str) -> list[str]:
        tokens = []
        for t in s.split(" "):
            if t:
                tokens.append(t)
        return tokens

    def __remove_stopwords(self, tokens: list[str]) -> list[str]:
        with open(STOPWORDS_DATASET, "r") as f:
            st_words = f.read().splitlines()
            clean_tokens = []
            for t in tokens:
                if t not in st_words:
                    clean_tokens.append(t)
            return clean_tokens

    def __stem(self, tokens: list[str]) -> list[str]:
        stemmer = PorterStemmer()
        res = []
        for t in tokens:
            res.append(stemmer.stem(t))
        return res
