"""Tests for ML-based AI detection.

Note: These tests may be slow due to model loading.
Run with: pytest tests/test_ml_detector.py -v
"""

import pytest
import sys
sys.path.insert(0, 'src')

from writestat_mcp.analyzers.ml_detector import MLDetector


class TestMLDetectorStatisticalMethods:
    """Tests for statistical methods that don't require model loading."""

    @pytest.fixture
    def detector(self):
        return MLDetector()

    def test_burstiness_uniform_text(self, detector):
        """Test burstiness for uniform sentence lengths (AI-like)."""
        # Sentences all about the same length
        text = "This is five words. Here are five words. Another five word sentence."
        burstiness = detector.calculate_burstiness(text)

        # Uniform text should have low burstiness
        assert burstiness < 0.3

    def test_burstiness_varied_text(self, detector):
        """Test burstiness for varied sentence lengths (human-like)."""
        text = "Hi. This is a medium length sentence with some variation. Now here is an extremely long and elaborate sentence that goes on and on with many words and clauses included within it."
        burstiness = detector.calculate_burstiness(text)

        # Varied text should have higher burstiness
        assert burstiness > 0.3

    def test_vocabulary_diversity_low(self, detector):
        """Test vocabulary diversity for repetitive text."""
        text = "The cat sat. The cat ran. The cat jumped. The cat played. The cat slept."
        diversity = detector.calculate_vocabulary_diversity(text)

        assert diversity < 0.5

    def test_vocabulary_diversity_high(self, detector):
        """Test vocabulary diversity for varied text."""
        text = "Dogs bark loudly. Cats meow softly. Birds chirp melodiously. Fish swim gracefully. Horses gallop swiftly."
        diversity = detector.calculate_vocabulary_diversity(text)

        assert diversity > 0.4

    def test_repetition_score_high(self, detector):
        """Test repetition score for text with repeated phrases."""
        text = "It is important to note that this is important to note. We should note that it is important to note this matter."
        repetition = detector.calculate_repetition_score(text)

        assert repetition > 0.3

    def test_repetition_score_low(self, detector):
        """Test repetition score for text without repeated phrases."""
        # Need at least 20 words for meaningful analysis
        text = "The quick brown fox jumps over the lazy dog near the old barn. A wise man once said that knowledge is power and wisdom brings peace."
        repetition = detector.calculate_repetition_score(text)

        assert repetition < 0.4

    def test_short_text_handling(self, detector):
        """Test that short texts return neutral scores."""
        text = "Hello."

        burstiness = detector.calculate_burstiness(text)
        diversity = detector.calculate_vocabulary_diversity(text)

        assert burstiness == 0.5  # Neutral for single sentence
        assert diversity == 0.5  # Neutral for very short text


@pytest.mark.slow
class TestMLDetectorWithModel:
    """Tests that require loading the ML model.

    These tests are marked as slow and can be skipped with: pytest -m "not slow"
    """

    @pytest.fixture(scope="class")
    def detector(self):
        """Create detector once for all tests in class."""
        d = MLDetector()
        # Force model loading
        d._load_model()
        return d

    def test_perplexity_calculation(self, detector):
        """Test that perplexity is calculated."""
        text = "This is a simple test sentence."
        perplexity = detector.calculate_perplexity(text)

        assert perplexity > 0
        assert perplexity < 10000

    def test_full_analysis(self, detector):
        """Test complete analysis output structure."""
        text = "This is a test paragraph for analysis. It contains multiple sentences with varying content."
        result = detector.analyze(text)

        # Check required fields
        assert "ai_probability" in result
        assert "interpretation" in result
        assert "confidence" in result
        assert "metrics" in result
        assert "component_scores" in result

        # Check metrics
        assert "perplexity" in result["metrics"]
        assert "burstiness" in result["metrics"]
        assert "vocabulary_diversity" in result["metrics"]

        # Check probability is in valid range
        assert 0 <= result["ai_probability"] <= 100

    def test_confidence_levels(self, detector):
        """Test that confidence reflects text length."""
        short_text = "Short."
        long_text = "This is a much longer text with more words. " * 25  # 225 words

        short_result = detector.analyze(short_text)
        long_result = detector.analyze(long_text)

        assert short_result["confidence"] == "Low"
        assert long_result["confidence"] == "High"

    def test_interpretation_strings(self, detector):
        """Test that interpretations are generated."""
        text = "Test text for interpretation verification purposes."
        result = detector.analyze(text)

        assert result["interpretation"]  # Not empty
        assert result["metrics"]["perplexity_interpretation"]
        assert result["metrics"]["burstiness_interpretation"]
        assert result["metrics"]["diversity_interpretation"]
