#!/usr/bin/env python3
"""
Readability MCP Server
Main server module that exposes MCP tools for text analysis
"""

import asyncio
import logging
from typing import Any

import nltk
import textstat

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise RuntimeError("Missing MCP package. Install with: pip install 'mcp>=1.0.0'")

from .analyzers import AIPatternDetector, MLDetector, ReadabilityAnalyzer, SentenceAnalyzer
from .validation import (
    ValidationError,
    create_error_response,
    validate_count,
    validate_metrics,
    validate_sensitivity,
    validate_text,
    validate_threshold,
)

# Initialize FastMCP server
mcp = FastMCP("readability-analyzer")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize analyzers
readability_analyzer = ReadabilityAnalyzer()
sentence_analyzer = SentenceAnalyzer()
ai_detector = AIPatternDetector()
ml_detector: MLDetector | None = None  # Lazy loaded to avoid startup cost


def get_ml_detector() -> MLDetector:
    """Get or create the ML detector (lazy loaded)."""
    global ml_detector
    if ml_detector is None:
        ml_detector = MLDetector()
    return ml_detector


@mcp.tool()
async def analyze_text(text: str, metrics: list[str] | None = None) -> dict[str, Any]:
    """
    Analyze text readability using multiple metrics

    Args:
        text: The text to analyze (1-500,000 characters)
        metrics: Optional list of specific metrics to calculate.
                Options: flesch_kincaid, flesch_ease, smog, ari, coleman_liau,
                        linsear, gunning_fog, dale_chall

    Returns:
        Dictionary containing:
        - flesch_kincaid_grade: Grade level (0-18+)
        - flesch_reading_ease: Ease score (0-100, higher is easier)
        - interpretation: Human-readable interpretation
        - Additional metrics based on request
        - Text statistics (word count, sentences, etc.)

    Example:
        {
            "flesch_kincaid_grade": 8.2,
            "flesch_reading_ease": 65.3,
            "interpretation": "Standard - 8th & 9th grade level",
            "statistics": {...}
        }
    """
    try:
        # Validate inputs
        text = validate_text(text)
        if metrics is not None:
            metrics = validate_metrics(metrics)

        result = readability_analyzer.analyze(text, metrics)
        logger.info(f"Analyzed text with {result['statistics']['word_count']} words")
        return result
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return create_error_response(e)
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return create_error_response(e)


@mcp.tool()
async def find_hard_sentences(text: str, count: int = 5, threshold: float = 10.0) -> dict[str, Any]:
    """
    Find the most difficult sentences in the text

    Args:
        text: The text to analyze (1-500,000 characters)
        count: Number of difficult sentences to return (1-100, default: 5)
        threshold: Minimum grade level to be considered difficult (0-30, default: 10.0)

    Returns:
        Dictionary containing:
        - difficult_sentences: List of sentences with analysis
        - total_sentences: Total number of sentences analyzed
        - average_grade_level: Average grade level of all sentences

    Each sentence includes:
        - sentence: The actual text
        - grade_level: Readability grade level
        - position: Sentence number in original text
        - issues: Specific problems identified
        - word_count: Number of words

    Example:
        {
            "difficult_sentences": [
                {
                    "sentence": "The complex...",
                    "grade_level": 16.3,
                    "position": 3,
                    "issues": ["Very long sentence (35 words)", "Multiple clauses"],
                    "word_count": 35
                }
            ],
            "total_sentences": 10,
            "average_grade_level": 9.5
        }
    """
    try:
        # Validate inputs
        text = validate_text(text)
        count = validate_count(count)
        threshold = validate_threshold(threshold)

        difficult_sentences = sentence_analyzer.find_difficult_sentences(text, count, threshold)

        # Calculate average grade level for context
        sentences = nltk.sent_tokenize(text)
        total_grade = sum(
            textstat.flesch_kincaid_grade(s) for s in sentences if len(s.strip()) > 10
        )
        avg_grade = total_grade / len(sentences) if sentences else 0

        result = {
            "difficult_sentences": difficult_sentences,
            "total_sentences": len(sentences),
            "average_grade_level": round(avg_grade, 1),
            "threshold_used": threshold,
        }

        logger.info(f"Found {len(difficult_sentences)} difficult sentences out of {len(sentences)}")
        return result

    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return create_error_response(e)
    except Exception as e:
        logger.error(f"Error finding difficult sentences: {str(e)}")
        return create_error_response(e)


@mcp.tool()
async def check_ai_phrases(text: str, sensitivity: str = "medium") -> dict[str, Any]:
    """
    Check text for common AI-generated writing patterns

    Args:
        text: The text to analyze (1-500,000 characters)
        sensitivity: Detection sensitivity - "low", "medium", or "high" (default: "medium")
                    Higher sensitivity catches more patterns but may have false positives

    Returns:
        Dictionary containing:
        - ai_likelihood_score: Score from 0-100 (higher = more AI-like)
        - interpretation: Human-readable interpretation
        - patterns_detected: Detailed list of patterns found
        - recommendations: Specific suggestions for more natural writing

    Pattern Categories:
        - dead_giveaways: Phrases almost exclusively used by AI
        - high_probability: Strong indicators of AI writing
        - moderate_indicators: Common in AI but also in formal writing
        - structural_patterns: Formatting patterns typical of AI

    Example:
        {
            "ai_likelihood_score": 45.2,
            "interpretation": "Medium - Noticeable AI patterns present",
            "patterns_detected": [...],
            "recommendations": ["Replace 'delve into' with 'explore'", ...]
        }
    """
    try:
        # Validate inputs
        text = validate_text(text)
        sensitivity = validate_sensitivity(sensitivity)

        result = ai_detector.analyze(text, sensitivity)

        logger.info(
            f"AI detection complete: score={result['ai_likelihood_score']}, "
            f"patterns={result['pattern_summary']['total_patterns']}"
        )

        return result

    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return create_error_response(e)
    except Exception as e:
        logger.error(f"Error checking AI patterns: {str(e)}")
        return create_error_response(e)


@mcp.tool()
async def detect_ai_ml(text: str) -> dict[str, Any]:
    """
    Detect AI-generated content using ML-based analysis (GPT-2 perplexity).

    This tool uses machine learning to analyze text for AI characteristics:
    - Perplexity scoring using GPT-2 (AI text is more predictable)
    - Burstiness analysis (AI text has uniform sentence patterns)
    - Vocabulary diversity (AI text often has repetitive word choices)
    - Repetition pattern detection

    Note: First call may take 10-30 seconds to load the model.

    Args:
        text: The text to analyze (1-500,000 characters)

    Returns:
        Dictionary containing:
        - ai_probability: 0-100 probability the text is AI-generated
        - interpretation: Human-readable assessment
        - confidence: Low/Medium/High based on text length
        - metrics: Detailed breakdown of each signal
        - component_scores: How each metric contributed to the score

    Example:
        {
            "ai_probability": 72.5,
            "interpretation": "Likely AI-generated - Significant AI patterns present",
            "confidence": "High",
            "metrics": {
                "perplexity": 35.2,
                "burstiness": 0.25,
                "vocabulary_diversity": 0.45
            }
        }
    """
    try:
        text = validate_text(text)

        detector = get_ml_detector()
        result = detector.analyze(text)

        logger.info(
            f"ML AI detection complete: probability={result['ai_probability']}%, "
            f"confidence={result['confidence']}"
        )

        return result

    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return create_error_response(e)
    except Exception as e:
        logger.error(f"Error in ML AI detection: {str(e)}")
        return create_error_response(e)


@mcp.tool()
async def batch_analyze(
    texts: list[str], analysis_types: list[str] | None = None
) -> dict[str, Any]:
    """
    Analyze multiple texts at once for efficient batch processing

    Args:
        texts: List of texts to analyze (1-20 texts)
        analysis_types: Types of analysis to perform on each text.
                       Options: "readability", "sentences", "ai_patterns", "ai_ml", "all"
                       Default: ["all"]

    Returns:
        Dictionary containing:
        - results: List of analysis results for each text
        - summary: Aggregate statistics across all texts
        - total_texts: Number of texts analyzed

    Example:
        {
            "results": [
                {
                    "text_index": 0,
                    "readability": {...},
                    "ai_detection": {...}
                },
                ...
            ],
            "summary": {
                "avg_grade_level": 10.5,
                "avg_ai_score": 35.2
            }
        }
    """
    try:
        # Validate inputs
        if not isinstance(texts, list):
            raise ValidationError("texts must be a list")

        if len(texts) == 0:
            raise ValidationError("texts list cannot be empty")

        if len(texts) > 20:
            raise ValidationError("Cannot analyze more than 20 texts at once")

        # Default to all analysis types (excluding ai_ml by default due to cost)
        if analysis_types is None:
            analysis_types = ["all"]

        valid_types = {"readability", "sentences", "ai_patterns", "ai_ml", "all"}
        analysis_set = set()
        for atype in analysis_types:
            if atype not in valid_types:
                raise ValidationError(
                    f"Invalid analysis type '{atype}'. Valid types: {', '.join(sorted(valid_types))}"
                )
            if atype == "all":
                analysis_set = {"readability", "sentences", "ai_patterns"}
                break
            analysis_set.add(atype)

        # Process texts in parallel using asyncio
        async def analyze_single_text(idx: int, text: str) -> dict[str, Any]:
            text = validate_text(text)
            result: dict[str, Any] = {"text_index": idx}

            if "readability" in analysis_set:
                result["readability"] = readability_analyzer.analyze(text)

            if "sentences" in analysis_set:
                difficult = sentence_analyzer.find_difficult_sentences(text, count=3)
                result["difficult_sentences"] = {"count": len(difficult), "sentences": difficult}

            if "ai_patterns" in analysis_set:
                result["ai_detection"] = ai_detector.analyze(text)

            if "ai_ml" in analysis_set:
                detector = get_ml_detector()
                result["ai_ml_detection"] = detector.analyze(text)

            return result

        # Run all analyses in parallel
        results = await asyncio.gather(
            *[analyze_single_text(idx, text) for idx, text in enumerate(texts)]
        )

        # Calculate summary statistics
        summary: dict[str, Any] = {}
        if "readability" in analysis_set:
            avg_grade = sum(r["readability"]["flesch_kincaid_grade"] for r in results) / len(
                results
            )
            avg_ease = sum(r["readability"]["flesch_reading_ease"] for r in results) / len(results)
            summary["avg_grade_level"] = round(avg_grade, 1)
            summary["avg_reading_ease"] = round(avg_ease, 1)

        if "ai_patterns" in analysis_set:
            avg_ai = sum(r["ai_detection"]["ai_likelihood_score"] for r in results) / len(results)
            summary["avg_ai_pattern_score"] = round(avg_ai, 1)

        if "ai_ml" in analysis_set:
            avg_ml = sum(r["ai_ml_detection"]["ai_probability"] for r in results) / len(results)
            summary["avg_ai_ml_probability"] = round(avg_ml, 1)

        logger.info(f"Batch analyzed {len(texts)} texts")

        return {"results": results, "summary": summary, "total_texts": len(texts)}

    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return create_error_response(e)
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        return create_error_response(e)


@mcp.tool()
async def compare_texts(
    original_text: str, revised_text: str, comparison_aspects: list[str] | None = None
) -> dict[str, Any]:
    """
    Compare two versions of text (before/after editing) to see improvements

    Args:
        original_text: The original version of the text
        revised_text: The revised/edited version of the text
        comparison_aspects: Aspects to compare (default: all)
                           Options: "readability", "ai_patterns", "sentence_complexity"

    Returns:
        Dictionary containing:
        - improvements: Summary of what improved
        - regressions: Summary of what got worse
        - detailed_comparison: Side-by-side metrics
        - recommendations: Suggestions for further improvements

    Example:
        {
            "improvements": ["Grade level decreased by 2.3", "AI score improved by 25 points"],
            "regressions": ["Text became slightly longer"],
            "detailed_comparison": {...}
        }
    """
    try:
        # Validate inputs
        original_text = validate_text(original_text)
        revised_text = validate_text(revised_text)

        if comparison_aspects is None:
            comparison_aspects = ["readability", "ai_patterns", "sentence_complexity"]

        # Analyze both versions
        orig_readability = readability_analyzer.analyze(original_text)
        rev_readability = readability_analyzer.analyze(revised_text)

        orig_ai = ai_detector.analyze(original_text)
        rev_ai = ai_detector.analyze(revised_text)

        # Calculate improvements and regressions
        improvements = []
        regressions = []

        # Grade level comparison
        grade_diff = (
            rev_readability["flesch_kincaid_grade"] - orig_readability["flesch_kincaid_grade"]
        )
        if grade_diff < -0.5:
            improvements.append(
                f"Reading level improved (decreased by {abs(grade_diff):.1f} grade levels)"
            )
        elif grade_diff > 0.5:
            regressions.append(f"Reading level increased by {grade_diff:.1f} grade levels")

        # Reading ease comparison
        ease_diff = rev_readability["flesch_reading_ease"] - orig_readability["flesch_reading_ease"]
        if ease_diff > 5:
            improvements.append(f"Reading ease improved by {ease_diff:.1f} points")
        elif ease_diff < -5:
            regressions.append(f"Reading ease decreased by {abs(ease_diff):.1f} points")

        # AI score comparison
        ai_diff = orig_ai["ai_likelihood_score"] - rev_ai["ai_likelihood_score"]
        if ai_diff > 10:
            improvements.append(
                f"AI-likeness reduced by {ai_diff:.1f} points (sounds more natural)"
            )
        elif ai_diff < -10:
            regressions.append(f"AI-likeness increased by {abs(ai_diff):.1f} points")

        # Word count comparison
        word_diff = (
            rev_readability["statistics"]["word_count"]
            - orig_readability["statistics"]["word_count"]
        )
        if abs(word_diff) > 10:
            if word_diff < 0:
                improvements.append(f"Text became more concise ({abs(word_diff)} fewer words)")
            else:
                improvements.append(f"Text expanded with more detail ({word_diff} more words)")

        # Sentence complexity
        orig_avg_words = orig_readability["statistics"]["avg_words_per_sentence"]
        rev_avg_words = rev_readability["statistics"]["avg_words_per_sentence"]
        sentence_diff = rev_avg_words - orig_avg_words
        if sentence_diff < -2:
            improvements.append(
                f"Sentences became shorter/simpler (avg {abs(sentence_diff):.1f} fewer words/sentence)"
            )
        elif sentence_diff > 2:
            regressions.append(
                f"Sentences became longer (avg {sentence_diff:.1f} more words/sentence)"
            )

        # Overall assessment
        overall_score = len(improvements) - len(regressions)
        n_imp, n_reg = len(improvements), len(regressions)
        if overall_score > 0:
            overall = f"✅ Overall improvement! {n_imp} improvements vs {n_reg} regressions"
        elif overall_score < 0:
            overall = f"⚠️ Some aspects regressed. {n_imp} improvements vs {n_reg} regressions"
        else:
            overall = f"↔️ Mixed changes. {n_imp} improvements and {n_reg} regressions"

        logger.info(
            f"Compared texts: {len(improvements)} improvements, {len(regressions)} regressions"
        )

        return {
            "overall_assessment": overall,
            "improvements": (
                improvements if improvements else ["No significant improvements detected"]
            ),
            "regressions": regressions if regressions else ["No regressions detected"],
            "detailed_comparison": {
                "original": {
                    "grade_level": orig_readability["flesch_kincaid_grade"],
                    "reading_ease": orig_readability["flesch_reading_ease"],
                    "ai_score": orig_ai["ai_likelihood_score"],
                    "word_count": orig_readability["statistics"]["word_count"],
                    "avg_words_per_sentence": orig_avg_words,
                },
                "revised": {
                    "grade_level": rev_readability["flesch_kincaid_grade"],
                    "reading_ease": rev_readability["flesch_reading_ease"],
                    "ai_score": rev_ai["ai_likelihood_score"],
                    "word_count": rev_readability["statistics"]["word_count"],
                    "avg_words_per_sentence": rev_avg_words,
                },
            },
            "recommendations": (
                rev_ai["recommendations"]
                if rev_ai["ai_likelihood_score"] > 30
                else ["Text looks good! Check difficult sentences for further improvements."]
            ),
        }

    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return create_error_response(e)
    except Exception as e:
        logger.error(f"Error comparing texts: {str(e)}")
        return create_error_response(e)


@mcp.tool()
async def health_check() -> dict[str, str]:
    """
    Check if the server is running and all dependencies are loaded

    Returns:
        Dictionary with server status and version information
    """
    import textstat

    return {
        "status": "healthy",
        "version": "0.1.0",
        "textstat_version": textstat.__version__,
        "ml_model_loaded": ml_detector is not None,
        "tools_available": [
            "analyze_text",
            "find_hard_sentences",
            "check_ai_phrases",
            "detect_ai_ml",
            "batch_analyze",
            "compare_texts",
            "health_check",
        ],
    }


# Run the server
if __name__ == "__main__":
    mcp.run()
