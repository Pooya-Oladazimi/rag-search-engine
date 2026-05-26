import argparse
import sys
from actions.libs import normalize_scores, load_dataset
from actions.hybrid_search import HybridSearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    commands = parser.add_subparsers(dest="command", help="Available commands")

    normalize_command = commands.add_parser(
        "normalize", help="Normalize a set of numbers based min-max approach."
    )
    normalize_command.add_argument(
        "numbers", nargs="*", type=float, help="Set of numbers."
    )

    weighted_search_command = commands.add_parser(
        "weighted-search", help="Hybrid search."
    )
    weighted_search_command.add_argument("query", type=str, help="search query.")
    weighted_search_command.add_argument(
        "--alpha", nargs="?", default=0.5, type=float, help="The alpha constant."
    )
    weighted_search_command.add_argument(
        "--limit", nargs="?", type=int, default=5, help="Number of results."
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_scores(args.numbers)
            if not scores:
                print()
                sys.exit(0)
            for score in scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            h = HybridSearch(load_dataset())
            results = h.weighted_search(
                query=args.query, alpha=args.alpha, limit=args.limit
            )
            i = 1
            for res in results:
                print(
                    f"{i}. {res['title']} \n Hybrid Score: {res['hybrid']} \n BM25: {res['bm25']}, Semantic: {res['semantic']}v\n{res['description'][:200]}"
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
