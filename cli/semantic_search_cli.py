import argparse
from actions.semantic_cli_functions import SemanticCliFunctions


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    commands = parser.add_subparsers(dest="command", description="CLI commands")

    commands.add_parser("verify", help="Verify the embedding model")
    embed_text_parser = commands.add_parser(
        "embed_text", help="Embed a text and get the vector."
    )
    embed_text_parser.add_argument(
        "text", type=str, help="The input text for embedding"
    )
    commands.add_parser(
        "verify_embeddings",
        help="Build or load the embedding and verify them for the dataset.",
    )
    embed_query = commands.add_parser("embed_query", help="Embed a user query.")
    embed_query.add_argument("query", type=str, help="The user query to embed.")

    search_command = commands.add_parser(
        "search", help="Search after movies by a query"
    )
    search_command.add_argument("query", type=str, help="The user query to search.")
    search_command.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Search limit"
    )

    chunk_command = commands.add_parser(
        "chunk", help="Chunk a document based on chunk size."
    )
    chunk_command.add_argument("text", type=str, help="The input text for chunking")
    chunk_command.add_argument(
        "--chunk-size",
        type=int,
        nargs="?",
        default=200,
        help="the chunk size. default is 200",
    )
    chunk_command.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=0,
        help="the chunk overlap size. default is 0",
    )

    args = parser.parse_args()

    cli = SemanticCliFunctions()

    match args.command:
        case "verify":
            cli.verify_model()
        case "embed_text":
            embedding = cli.embed_text(text=args.text)
            print(f"Text: {args.text}")
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Dimensions: {embedding.shape[0]}")
        case "verify_embeddings":
            doc_length, embeddings = cli.verify_embeddings()
            print(f"Number of docs:   {doc_length}")
            print(
                f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
            )
        case "embed_query":
            embedding = cli.embed_text(text=args.query)
            print(f"Query: {args.query}")
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Shape: {embedding.shape}")
        case "search":
            cli.search(query=args.query, limit=args.limit)
        case "chunk":
            res = cli.chunk(
                text=args.text, chunk_size=args.chunk_size, overlap=args.overlap
            )
            print(f"Chunking {len(args.text)} characters")
            for i in range(len(res)):
                print(f"{i+1}. {res[i]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
