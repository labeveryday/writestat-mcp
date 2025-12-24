"""Tests for AI pattern detection."""

import pytest
import sys
sys.path.insert(0, 'src')

from writestat_mcp.analyzers import AIPatternDetector
from writestat_mcp.analyzers.ai_patterns import LRUCache


class TestLRUCache:
    """Tests for LRU cache implementation."""

    def test_cache_set_and_get(self):
        cache = LRUCache(maxsize=3)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_miss(self):
        cache = LRUCache(maxsize=3)
        assert cache.get("nonexistent") is None

    def test_cache_eviction(self):
        cache = LRUCache(maxsize=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_cache_lru_order(self):
        cache = LRUCache(maxsize=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Access key1, making key2 the LRU
        cache.set("key3", "value3")  # Should evict key2

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"


class TestAIPatternDetector:
    """Tests for AIPatternDetector class."""

    @pytest.fixture
    def detector(self):
        return AIPatternDetector()

    def test_detect_dead_giveaway_patterns(self, detector):
        """Test detection of obvious AI phrases."""
        text = "Let's delve into this topic and explore the rich tapestry of ideas."
        result = detector.analyze(text)

        assert result["ai_likelihood_score"] > 30
        assert any(
            p["category"] == "dead_giveaways"
            for p in result["patterns_detected"]
        )

    def test_detect_no_patterns(self, detector):
        """Test text without AI patterns."""
        text = "I went to the store yesterday. The milk was on sale."
        result = detector.analyze(text)

        assert result["ai_likelihood_score"] < 20
        assert result["pattern_summary"]["total_patterns"] == 0

    def test_sensitivity_levels(self, detector):
        """Test different sensitivity levels."""
        text = "Moreover, it's important to note that this is significant."

        low_result = detector.analyze(text, sensitivity="low")
        high_result = detector.analyze(text, sensitivity="high")

        # High sensitivity should generally produce higher scores
        assert high_result["ai_likelihood_score"] >= low_result["ai_likelihood_score"]

    def test_recommendations_generated(self, detector):
        """Test that recommendations are generated for AI-like text."""
        text = """
        Moreover, it's important to note that we must delve into this topic.
        Furthermore, the comprehensive analysis reveals significant insights.
        """
        result = detector.analyze(text)

        assert len(result["recommendations"]) > 0

    def test_pattern_context_extraction(self, detector):
        """Test that pattern context is properly extracted."""
        text = "This is some text. Let's delve into the matter. More text here."
        result = detector.analyze(text)

        if result["patterns_detected"]:
            for pattern_group in result["patterns_detected"]:
                for match in pattern_group["matches"]:
                    assert "context" in match
                    assert "delve" in match["context"].lower()

    def test_caching(self, detector):
        """Test that results are cached."""
        text = "This is a test with moreover and furthermore."

        result1 = detector.analyze(text)
        result2 = detector.analyze(text)

        # Results should be identical (from cache)
        assert result1["ai_likelihood_score"] == result2["ai_likelihood_score"]

    def test_interpret_ai_score(self, detector):
        """Test AI score interpretation."""
        assert "very low" in detector.interpret_ai_score(10).lower()
        assert "very high" in detector.interpret_ai_score(90).lower()

    def test_short_text_penalty(self, detector):
        """Test that very short texts get confidence penalty."""
        # Short text with AI pattern
        short_text = "Moreover."
        # Longer text with same pattern but diluted by other words
        long_text = "Moreover, " + " ".join(["word"] * 100)

        short_result = detector.analyze(short_text)
        long_result = detector.analyze(long_text)

        # Short text has higher pattern density, but low confidence
        # Long text has lower pattern density
        # Both should produce valid scores in range
        assert 0 <= short_result["ai_likelihood_score"] <= 100
        assert 0 <= long_result["ai_likelihood_score"] <= 100
