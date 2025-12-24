"""
ML-based AI content detection module.

Uses GPT-2 perplexity scoring, burstiness analysis, and vocabulary diversity
to detect AI-generated text with higher accuracy than pattern matching alone.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import nltk

# Lazy imports for torch and transformers to avoid startup cost
torch = None
transformers = None


@dataclass
class MLDetectionResult:
    """Result from ML-based AI detection."""

    ai_probability: float  # 0-100, probability text is AI-generated
    perplexity: float  # Lower = more AI-like
    burstiness: float  # Lower = more AI-like (uniform)
    vocabulary_diversity: float  # Higher = more human-like
    interpretation: str
    confidence: str  # Low/Medium/High based on text length


class MLDetector:
    """
    ML-based AI content detector using GPT-2 perplexity and statistical analysis.

    This detector combines multiple signals:
    1. Perplexity - AI text has lower perplexity (more predictable)
    2. Burstiness - AI text has uniform sentence lengths
    3. Vocabulary diversity - AI text often has lower diversity
    """

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the ML detector.

        Args:
            model_name: HuggingFace model name (default: gpt2)
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = None

        # Ensure NLTK data is available
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

    def _load_model(self) -> None:
        """Lazy load the model and tokenizer."""
        if self._model is not None:
            return

        global torch, transformers

        try:
            import torch as _torch
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast

            torch = _torch
        except ImportError as e:
            raise ImportError(
                "ML detection requires torch and transformers. "
                "Install with: pip install 'writestat-mcp[ml]' "
                "or: pip install torch transformers"
            ) from e

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Pin to specific revision for reproducibility and security
        revision = "607a30d783dfa663caf39e06633721c8d4cfcd7e"  # gpt2 stable
        self._tokenizer = GPT2TokenizerFast.from_pretrained(  # nosec B615
            self.model_name, revision=revision
        )
        self._model = GPT2LMHeadModel.from_pretrained(  # nosec B615
            self.model_name, revision=revision
        ).to(self._device)
        self._model.eval()

    def calculate_perplexity(self, text: str, max_length: int = 1024) -> float:
        """
        Calculate the perplexity of text using GPT-2.

        Lower perplexity indicates more predictable (AI-like) text.

        Args:
            text: Text to analyze
            max_length: Maximum token length to process

        Returns:
            Perplexity score (lower = more AI-like)
        """
        self._load_model()

        # Tokenize the text
        encodings = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )

        input_ids = encodings.input_ids.to(self._device)

        # Handle empty or very short text
        if input_ids.size(1) < 2:
            return 100.0  # Return neutral perplexity for very short text

        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            loss = outputs.loss

        perplexity = math.exp(loss.item())
        return min(perplexity, 10000.0)  # Cap at 10000 to avoid extreme values

    def calculate_burstiness(self, text: str) -> float:
        """
        Calculate burstiness (variance in sentence complexity).

        Human writing tends to have more variation in sentence length and complexity.
        AI text tends to be more uniform.

        Args:
            text: Text to analyze

        Returns:
            Burstiness score (0-1, higher = more varied = more human-like)
        """
        sentences = nltk.sent_tokenize(text)

        if len(sentences) < 2:
            return 0.5  # Neutral for single sentence

        # Calculate sentence lengths
        lengths = [len(s.split()) for s in sentences]

        if not lengths:
            return 0.5

        mean_length = sum(lengths) / len(lengths)

        if mean_length == 0:
            return 0.5

        # Calculate coefficient of variation (normalized standard deviation)
        variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        coefficient_of_variation = std_dev / mean_length

        # Normalize to 0-1 range (typical CV ranges from 0 to ~1.5)
        burstiness = min(coefficient_of_variation / 1.5, 1.0)

        return burstiness

    def calculate_vocabulary_diversity(self, text: str) -> float:
        """
        Calculate vocabulary diversity using type-token ratio and hapax legomena.

        Human text often has richer vocabulary with more unique words.

        Args:
            text: Text to analyze

        Returns:
            Diversity score (0-1, higher = more diverse = more human-like)
        """
        # Tokenize into words, lowercase, remove punctuation
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        if len(words) < 10:
            return 0.5  # Neutral for very short text

        total_words = len(words)
        unique_words = len(set(words))

        # Type-Token Ratio (TTR)
        ttr = unique_words / total_words

        # Count hapax legomena (words appearing only once)
        word_counts = Counter(words)
        hapax = sum(1 for count in word_counts.values() if count == 1)
        hapax_ratio = hapax / total_words

        # Combine TTR and hapax ratio
        # Both are indicators of vocabulary richness
        diversity = ttr * 0.6 + hapax_ratio * 0.4

        # Normalize - typical values range from 0.3 to 0.8
        normalized = (diversity - 0.2) / 0.6
        normalized = max(0.0, min(1.0, normalized))

        return normalized

    def calculate_repetition_score(self, text: str) -> float:
        """
        Detect repetitive patterns common in AI text.

        Args:
            text: Text to analyze

        Returns:
            Repetition score (0-1, higher = more repetition = more AI-like)
        """
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        if len(words) < 20:
            return 0.5

        # Check for repeated n-grams
        def get_ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
            return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

        # Analyze bigrams and trigrams
        bigrams = get_ngrams(words, 2)
        trigrams = get_ngrams(words, 3)

        # Calculate repetition ratios
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)

        # Count repeated n-grams (appearing more than once)
        repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
        repeated_trigrams = sum(1 for count in trigram_counts.values() if count > 1)

        bigram_ratio = repeated_bigrams / len(bigrams) if bigrams else 0
        trigram_ratio = repeated_trigrams / len(trigrams) if trigrams else 0

        # Combine scores
        repetition = bigram_ratio * 0.4 + trigram_ratio * 0.6

        # Normalize to 0-1 (typical repetition rates)
        normalized = min(repetition * 3, 1.0)

        return normalized

    def analyze(self, text: str) -> dict[str, Any]:
        """
        Perform complete ML-based AI detection analysis.

        Args:
            text: Text to analyze

        Returns:
            Dictionary containing detection results and metrics
        """
        word_count = len(text.split())

        # Calculate individual metrics
        perplexity = self.calculate_perplexity(text)
        burstiness = self.calculate_burstiness(text)
        vocabulary_diversity = self.calculate_vocabulary_diversity(text)
        repetition = self.calculate_repetition_score(text)

        # Convert perplexity to a 0-1 score (lower perplexity = higher AI probability)
        # Typical perplexity ranges: AI ~20-50, Human ~50-200+
        perplexity_score = 1.0 - min(max((perplexity - 20) / 180, 0), 1)

        # Invert burstiness and diversity (higher values = more human)
        burstiness_score = 1.0 - burstiness
        diversity_score = 1.0 - vocabulary_diversity

        # Combine scores with weights
        # Perplexity is the strongest signal
        weights = {
            "perplexity": 0.45,
            "burstiness": 0.20,
            "diversity": 0.15,
            "repetition": 0.20,
        }

        combined_score = (
            perplexity_score * weights["perplexity"]
            + burstiness_score * weights["burstiness"]
            + diversity_score * weights["diversity"]
            + repetition * weights["repetition"]
        )

        # Convert to 0-100 probability
        ai_probability = round(combined_score * 100, 1)

        # Determine confidence based on text length
        if word_count < 50:
            confidence = "Low"
            confidence_note = "Text is short; results may be less reliable"
        elif word_count < 200:
            confidence = "Medium"
            confidence_note = "Moderate text length provides reasonable accuracy"
        else:
            confidence = "High"
            confidence_note = "Sufficient text for reliable analysis"

        # Generate interpretation
        if ai_probability >= 80:
            interpretation = "Very likely AI-generated - Multiple strong indicators detected"
        elif ai_probability >= 60:
            interpretation = "Likely AI-generated - Significant AI patterns present"
        elif ai_probability >= 40:
            interpretation = "Possibly AI-generated - Some AI characteristics detected"
        elif ai_probability >= 20:
            interpretation = "Likely human-written - Few AI indicators"
        else:
            interpretation = "Very likely human-written - Natural writing patterns"

        return {
            "ai_probability": ai_probability,
            "interpretation": interpretation,
            "confidence": confidence,
            "confidence_note": confidence_note,
            "metrics": {
                "perplexity": round(perplexity, 2),
                "perplexity_interpretation": self._interpret_perplexity(perplexity),
                "burstiness": round(burstiness, 3),
                "burstiness_interpretation": self._interpret_burstiness(burstiness),
                "vocabulary_diversity": round(vocabulary_diversity, 3),
                "diversity_interpretation": self._interpret_diversity(vocabulary_diversity),
                "repetition_score": round(repetition, 3),
            },
            "component_scores": {
                "perplexity_contribution": round(perplexity_score * weights["perplexity"] * 100, 1),
                "burstiness_contribution": round(burstiness_score * weights["burstiness"] * 100, 1),
                "diversity_contribution": round(diversity_score * weights["diversity"] * 100, 1),
                "repetition_contribution": round(repetition * weights["repetition"] * 100, 1),
            },
            "word_count": word_count,
        }

    def _interpret_perplexity(self, perplexity: float) -> str:
        """Interpret perplexity score."""
        if perplexity < 30:
            return "Very low - Highly predictable (AI-like)"
        elif perplexity < 50:
            return "Low - Predictable patterns"
        elif perplexity < 100:
            return "Moderate - Some variation"
        elif perplexity < 200:
            return "High - Natural variation (human-like)"
        else:
            return "Very high - Unpredictable (strongly human-like)"

    def _interpret_burstiness(self, burstiness: float) -> str:
        """Interpret burstiness score."""
        if burstiness < 0.2:
            return "Very uniform - Typical of AI"
        elif burstiness < 0.4:
            return "Somewhat uniform"
        elif burstiness < 0.6:
            return "Moderate variation"
        elif burstiness < 0.8:
            return "Good variation (human-like)"
        else:
            return "High variation - Very natural"

    def _interpret_diversity(self, diversity: float) -> str:
        """Interpret vocabulary diversity score."""
        if diversity < 0.3:
            return "Low diversity - Repetitive vocabulary"
        elif diversity < 0.5:
            return "Moderate diversity"
        elif diversity < 0.7:
            return "Good diversity"
        else:
            return "High diversity - Rich vocabulary"
