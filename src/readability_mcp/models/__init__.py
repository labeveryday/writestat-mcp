"""
Data models and structures for the readability server
"""

from .results import AIDetectionResult, DifficultSentence, ReadabilityResult

__all__ = ["ReadabilityResult", "DifficultSentence", "AIDetectionResult"]
