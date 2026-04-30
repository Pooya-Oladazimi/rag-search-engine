import argparse
from actions.cli_functions import CliFunctions


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
    cli = CliFunctions()

    match args.command:
        case "search":
            cli.search(args.query)
        case "build":
            cli.build()
        case "tf":
            cli.tf(args.docId, args.term)
        case "idf":
            cli.idf(args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
