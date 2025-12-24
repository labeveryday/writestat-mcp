"""
Readability analysis module
Provides various readability metrics and scoring functions
"""

from typing import Any

import textstat

from ..models.results import ReadabilityResult


class ReadabilityAnalyzer:
    """Handles all readability analysis operations"""

    @staticmethod
    def interpret_reading_ease(score: float) -> str:
        """Interpret Flesch Reading Ease score"""
        if score < 30:
            return "Very difficult - Best understood by university graduates"
        elif score < 50:
            return "Difficult - Best understood by college students"
        elif score < 60:
            return "Fairly difficult - 10th to 12th grade level"
        elif score < 70:
            return "Standard - 8th & 9th grade level"
        elif score < 80:
            return "Fairly easy - 7th grade level"
        elif score < 90:
            return "Easy - 6th grade level"
        else:
            return "Very easy - 5th grade level or below"

    @staticmethod
    def interpret_grade_level(level: float) -> str:
        """Interpret grade level scores"""
        if level < 6:
            return "Elementary school level"
        elif level < 9:
            return "Middle school level"
        elif level < 13:
            return "High school level"
        elif level < 16:
            return "College level"
        else:
            return "Graduate level"

    def analyze(self, text: str, metrics: list[str] | None = None) -> dict[str, Any]:
        """
        Analyze text readability using multiple metrics

        Args:
            text: The text to analyze
            metrics: Optional list of specific metrics to calculate

        Returns:
            Dictionary containing readability scores and interpretation
        """
        # Basic text statistics
        word_count = textstat.lexicon_count(text, removepunct=True)
        sentence_count = textstat.sentence_count(text)
        syllable_count = textstat.syllable_count(text)

        # Calculate average words per sentence
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0

        # Core readability metrics
        flesch_kincaid = textstat.flesch_kincaid_grade(text)
        flesch_ease = textstat.flesch_reading_ease(text)

        # Create result object
        result = ReadabilityResult(
            flesch_kincaid_grade=flesch_kincaid,
            flesch_reading_ease=flesch_ease,
            interpretation=self.interpret_reading_ease(flesch_ease),
            word_count=word_count,
            sentence_count=sentence_count,
            syllable_count=syllable_count,
            avg_words_per_sentence=avg_words_per_sentence,
        )

        # Build response dictionary
        response = {
            "flesch_kincaid_grade": result.flesch_kincaid_grade,
            "flesch_reading_ease": result.flesch_reading_ease,
            "interpretation": result.interpretation,
            "grade_level_interpretation": self.interpret_grade_level(flesch_kincaid),
            "statistics": {
                "word_count": result.word_count,
                "sentence_count": result.sentence_count,
                "syllable_count": result.syllable_count,
                "avg_words_per_sentence": result.avg_words_per_sentence,
            },
        }

        # Add additional metrics if not filtered
        if metrics is None or "smog" in metrics:
            response["smog_index"] = textstat.smog_index(text)

        if metrics is None or "ari" in metrics:
            response["automated_readability_index"] = textstat.automated_readability_index(text)

        if metrics is None or "coleman_liau" in metrics:
            response["coleman_liau_index"] = textstat.coleman_liau_index(text)

        if metrics is None or "linsear" in metrics:
            response["linsear_write_formula"] = textstat.linsear_write_formula(text)

        if metrics is None or "gunning_fog" in metrics:
            response["gunning_fog"] = textstat.gunning_fog(text)

        if metrics is None or "dale_chall" in metrics:
            response["dale_chall_readability_score"] = textstat.dale_chall_readability_score(text)

        # Add reading time estimate
        reading_time_minutes = word_count / 200  # Assuming 200 words per minute
        response["estimated_reading_time"] = f"{reading_time_minutes:.1f} minutes"

        return response
