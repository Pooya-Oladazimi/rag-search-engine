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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
