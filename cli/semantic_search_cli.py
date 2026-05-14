import argparse
from actions.semantic_cli_functions import SemanticCliFunctions


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    commands = parser.add_subparsers(dest="command", description="CLI commands")

    commands.add_parser("verify", help="Verify the embedding model")
    args = parser.parse_args()

    cli = SemanticCliFunctions()

    match args.command:
        case "verify":
            cli.verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
