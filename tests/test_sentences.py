"""Tests for sentence difficulty analysis."""

import pytest
import sys
sys.path.insert(0, 'src')

from writestat_mcp.analyzers import SentenceAnalyzer


class TestSentenceAnalyzer:
    """Tests for SentenceAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        return SentenceAnalyzer()

    def test_find_difficult_sentences(self, analyzer):
        """Test finding difficult sentences."""
        text = """
        This is easy. The implementation of sophisticated algorithmic methodologies
        necessitates comprehensive understanding of computational paradigms
        and their multifaceted implications within contemporary technological frameworks.
        Simple again.
        """
        result = analyzer.find_difficult_sentences(text, count=1, threshold=5.0)

        assert len(result) >= 1
        # The complex sentence should be found
        assert any("implementation" in s["sentence"].lower() for s in result)

    def test_sentence_position_tracking(self, analyzer):
        """Test that sentence positions are tracked correctly."""
        text = "First. Second. Third. Fourth. Fifth."
        result = analyzer.find_difficult_sentences(text, count=5, threshold=0.0)

        positions = [s["position"] for s in result]
        # Positions should be unique and valid
        assert len(positions) == len(set(positions))

    def test_analyze_sentence_difficulty(self, analyzer):
        """Test individual sentence analysis."""
        simple = "The cat sat."
        complex_sent = "The implementation of sophisticated methodologies necessitates comprehensive understanding."

        simple_grade, simple_issues = analyzer.analyze_sentence_difficulty(simple)
        complex_grade, complex_issues = analyzer.analyze_sentence_difficulty(complex_sent)

        assert complex_grade > simple_grade
        assert len(complex_issues) > 0

    def test_detects_long_sentences(self, analyzer):
        """Test detection of long sentences."""
        long_sentence = " ".join(["word"] * 30) + "."
        _, issues = analyzer.analyze_sentence_difficulty(long_sentence)

        assert any("long" in issue.lower() for issue in issues)

    def test_detects_complex_vocabulary(self, analyzer):
        """Test detection of complex vocabulary."""
        complex_sentence = "The implementation necessitates comprehensive understanding."
        _, issues = analyzer.analyze_sentence_difficulty(complex_sentence)

        # Should detect syllable complexity
        assert any("syllable" in issue.lower() or "vocabulary" in issue.lower() for issue in issues)

    def test_detects_multiple_clauses(self, analyzer):
        """Test detection of multiple clauses."""
        multi_clause = "The cat, which was black, sat on the mat, where it slept, while the dog watched."
        _, issues = analyzer.analyze_sentence_difficulty(multi_clause)

        assert any("clause" in issue.lower() for issue in issues)

    def test_threshold_filtering(self, analyzer):
        """Test that threshold filtering works."""
        text = "Easy sentence. Another easy one. Complex multisyllabic vocabulary utilized."

        high_threshold = analyzer.find_difficult_sentences(text, count=10, threshold=15.0)
        low_threshold = analyzer.find_difficult_sentences(text, count=10, threshold=0.0)

        assert len(low_threshold) >= len(high_threshold)

    def test_count_limiting(self, analyzer):
        """Test that count parameter limits results."""
        text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten."
        result = analyzer.find_difficult_sentences(text, count=3, threshold=0.0)

        assert len(result) <= 3

    def test_skip_short_sentences(self, analyzer):
        """Test that very short sentences are skipped."""
        text = "Hi. Yes. No. OK. This is a longer sentence to analyze."
        result = analyzer.find_difficult_sentences(text, count=10, threshold=0.0)

        # Only the longer sentence should be analyzed
        assert len(result) == 1

    def test_result_structure(self, analyzer):
        """Test that results have the expected structure."""
        text = "This is a test sentence for structure verification."
        result = analyzer.find_difficult_sentences(text, count=1, threshold=0.0)

        if result:
            sentence = result[0]
            assert "sentence" in sentence
            assert "grade_level" in sentence
            assert "position" in sentence
            assert "issues" in sentence
            assert "word_count" in sentence
            assert "syllable_count" in sentence
