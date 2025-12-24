"""Tests for readability analysis."""

import pytest
import sys
sys.path.insert(0, 'src')

from writestat_mcp.analyzers import ReadabilityAnalyzer


class TestReadabilityAnalyzer:
    """Tests for ReadabilityAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        return ReadabilityAnalyzer()

    def test_analyze_simple_text(self, analyzer):
        """Test analysis of simple text."""
        text = "The cat sat on the mat. It was a sunny day."
        result = analyzer.analyze(text)

        assert "flesch_kincaid_grade" in result
        assert "flesch_reading_ease" in result
        assert "interpretation" in result
        assert "statistics" in result
        assert result["statistics"]["word_count"] > 0
        assert result["statistics"]["sentence_count"] == 2

    def test_analyze_complex_text(self, analyzer):
        """Test analysis of complex text."""
        text = """
        The implementation of sophisticated algorithmic methodologies
        necessitates comprehensive understanding of computational paradigms
        and their multifaceted implications within contemporary technological frameworks.
        """
        result = analyzer.analyze(text)

        # Complex text should have higher grade level
        assert result["flesch_kincaid_grade"] > 10

    def test_analyze_with_metrics_filter(self, analyzer):
        """Test analysis with specific metrics requested."""
        text = "This is a test sentence for analysis."
        result = analyzer.analyze(text, metrics=["smog", "ari"])

        assert "smog_index" in result
        assert "automated_readability_index" in result

    def test_interpret_reading_ease(self, analyzer):
        """Test reading ease interpretation."""
        # Very easy text
        assert "easy" in analyzer.interpret_reading_ease(95).lower()
        # Very difficult text
        assert "difficult" in analyzer.interpret_reading_ease(20).lower()

    def test_interpret_grade_level(self, analyzer):
        """Test grade level interpretation."""
        assert "elementary" in analyzer.interpret_grade_level(4).lower()
        assert "college" in analyzer.interpret_grade_level(14).lower()
        assert "graduate" in analyzer.interpret_grade_level(18).lower()

    def test_statistics_accuracy(self, analyzer):
        """Test that statistics are accurately calculated."""
        text = "One two three. Four five six seven."
        result = analyzer.analyze(text)

        assert result["statistics"]["sentence_count"] == 2
        assert result["statistics"]["word_count"] == 7

    def test_reading_time_estimate(self, analyzer):
        """Test reading time estimation."""
        # 200 words should be about 1 minute
        text = " ".join(["word"] * 200)
        result = analyzer.analyze(text)

        assert "1.0 minutes" in result["estimated_reading_time"]
