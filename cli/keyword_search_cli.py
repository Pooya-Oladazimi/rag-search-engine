import argparse
from actions.cli_functions import CliFunctions
from actions.vars import BM25_K1


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    commands = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = commands.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    commands.add_parser("build", help="Build the search index")
    tf_parser = commands.add_parser(
        "tf", help="Get the term frequency of a term in a document."
    )

    term_args_keys = {"type": str, "help": "The target term"}
    docid_args_keys = {"type": int, "help": "The target document id"}

    tf_parser.add_argument("docId", **docid_args_keys)
    tf_parser.add_argument("term", **term_args_keys)
    idf_parser = commands.add_parser("idf", help="Get idf for a term")
    idf_parser.add_argument("term", **term_args_keys)
    tfidf_parser = commands.add_parser("tfidf", help="Get tf-idf for a term in a doc")
    tfidf_parser.add_argument("docId", **docid_args_keys)
    tfidf_parser.add_argument("term", **term_args_keys)
    bm25idf_parser = commands.add_parser("bm25idf", help="Get the Bm25idf for a term.")
    bm25idf_parser.add_argument("term", **term_args_keys)
    bm25tf_parser = commands.add_parser(
        "bm25tf", help="Get the Bm25tf for a term in a doc"
    )
    bm25tf_parser.add_argument("docId", **docid_args_keys)
    bm25tf_parser.add_argument("term", **term_args_keys)
    bm25tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )

    args = parser.parse_args()
    cli = CliFunctions()

    match args.command:
        case "search":
            cli.search(args.query)
        case "build":
            cli.build()
        case "tf":
            tf = cli.tf(args.docId, args.term)
            print(tf if tf else "O")
        case "idf":
            idf = cli.idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = cli.tf_idf(args.docId, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.docId}': {tfidf:.2f}"
            )
        case "bm25idf":
            print(f"BM25 IDF score of '{args.term}': {cli.bm25idf(args.term):.2f}")
        case "bm25tf":
            bm25tf = cli.bm25tf(term=args.term, docId=args.docId, k1=args.k1)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.docId}': {bm25tf:.2f}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
