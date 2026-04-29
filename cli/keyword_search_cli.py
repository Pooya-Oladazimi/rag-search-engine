import argparse
from actions.search import search
from actions.render import render_movies_titles
from actions.indexer import InvertedIndex
import math


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    commands = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = commands.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    commands.add_parser("build", help="Build the search index")
    tf_parser = commands.add_parser(
        "tf", help="Get the term frequency of a term in a document."
    )
    tf_parser.add_argument("docId", type=int, help="The document id")
    tf_parser.add_argument("term", type=str, help="Target term")
    idf_parser = commands.add_parser("idf", help="Get idf for a term")
    idf_parser.add_argument("term", type=str, help="The target term")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            render_movies_titles(search(args.query))
        case "build":
            invdex = InvertedIndex()
            invdex.build()
            invdex.save()
        case "tf":
            index = InvertedIndex()
            index.load()
            print(index.get_tf(args.docId, args.term))
        case "idf":
            index = InvertedIndex()
            index.load()
            total_doc_count = len(index.docmap.keys()) + 1
            doc_frq = len(index.get_documents(args.term)) + 1
            idf = math.log(total_doc_count / doc_frq)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
