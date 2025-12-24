#!/usr/bin/env python3
"""
CLI entrypoint for the WriteStat MCP Server.

Usage:
    writestat-mcp          # Run the MCP server (default)
    writestat-mcp --help   # Show help
    writestat-mcp --version  # Show version
"""

import argparse
import sys


def main() -> None:
    """Main CLI entrypoint for the writestat-mcp server."""
    parser = argparse.ArgumentParser(
        prog="writestat-mcp",
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

        print(f"writestat-mcp {__version__}")
        sys.exit(0)

    # Run the MCP server
    from .server import mcp

    mcp.run()


if __name__ == "__main__":
    main()
