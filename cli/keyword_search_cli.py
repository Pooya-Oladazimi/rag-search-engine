import argparse
from actions.search import search
from actions.render import render_movies_titles


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            render_movies_titles(search(args.query, 50))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
