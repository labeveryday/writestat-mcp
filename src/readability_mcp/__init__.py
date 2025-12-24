"""
Readability MCP Server
A Model Context Protocol server for text analysis, readability scoring, and AI content detection.
"""

__version__ = "0.1.0"

from .analyzers import AIPatternDetector, ReadabilityAnalyzer, SentenceAnalyzer
from .server import mcp

__all__ = [
    "__version__",
    "mcp",
    "ReadabilityAnalyzer",
    "SentenceAnalyzer",
    "AIPatternDetector",
]
