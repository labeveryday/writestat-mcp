#!/usr/bin/env python3
"""
CLI entrypoint for the Readability MCP Server.

Usage:
    readability-mcp          # Run the MCP server (default)
    readability-mcp --help   # Show help
    readability-mcp --version  # Show version
"""

import argparse
import sys


def main() -> None:
    """Main CLI entrypoint for the readability-mcp server."""
    parser = argparse.ArgumentParser(
        prog="readability-mcp",
        description="MCP server for text readability analysis and AI content detection",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    args = parser.parse_args()

    if args.version:
        from . import __version__

        print(f"readability-mcp {__version__}")
        sys.exit(0)

    # Run the MCP server
    from .server import mcp

    mcp.run()


if __name__ == "__main__":
    main()
