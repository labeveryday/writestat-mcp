"""
Sentence difficulty analysis module
Identifies and analyzes complex sentences in text
"""

from typing import Any

import nltk
import textstat

from ..models.results import DifficultSentence


class SentenceAnalyzer:
    """Handles sentence-level difficulty analysis"""

    def __init__(self):
        """Initialize the sentence analyzer and ensure NLTK data is available"""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    @staticmethod
    def analyze_sentence_difficulty(sentence: str) -> tuple[float, list[str]]:
        """
        Analyze why a sentence is difficult

        Args:
            sentence: The sentence to analyze

        Returns:
            Tuple of (grade_level, list_of_issues)
        """
        issues = []
        grade_level = textstat.flesch_kincaid_grade(sentence)

        # Check sentence length
        word_count = textstat.lexicon_count(sentence, removepunct=True)
        if word_count > 25:
            issues.append(f"Very long sentence ({word_count} words)")
        elif word_count > 20:
            issues.append(f"Long sentence ({word_count} words)")

        # Check syllable complexity
        syllable_count = textstat.syllable_count(sentence)
        avg_syllables = syllable_count / word_count if word_count > 0 else 0
        if avg_syllables > 2:
            issues.append(f"Complex vocabulary (avg {avg_syllables:.1f} syllables/word)")

        # Check for passive voice indicators (simple heuristic)
        passive_indicators = ["was", "were", "been", "being", "be", "is", "are", "am"]
        words = sentence.lower().split()
        for i, word in enumerate(words):
            if word in passive_indicators and i + 1 < len(words):
                if words[i + 1].endswith("ed") or words[i + 1].endswith("en"):
                    issues.append("Possible passive voice")
                    break

        # Check for multiple clauses
        clause_indicators = [",", ";", " - ", " -- ", "(", "which", "that", "who", "whom", "whose"]
        clause_count = sum(1 for indicator in clause_indicators if indicator in sentence)
        if clause_count >= 3:
            issues.append(f"Multiple clauses ({clause_count} subordinate elements)")
        elif clause_count >= 2:
            issues.append("Complex structure with multiple clauses")

        if not issues:
            if grade_level > 12:
                issues.append("High reading level vocabulary")
            else:
                issues.append("Generally clear, but could be simplified")

        return grade_level, issues

    def find_difficult_sentences(
        self, text: str, count: int = 5, threshold: float = 10.0
    ) -> list[dict[str, Any]]:
        """
        Find the most difficult sentences in the text

        Args:
            text: The text to analyze
            count: Number of difficult sentences to return
            threshold: Minimum grade level to be considered difficult

        Returns:
            List of difficult sentences with analysis
        """
        # Tokenize into sentences
        sentences = nltk.sent_tokenize(text)

        # Analyze each sentence
        sentence_data = []
        for i, sentence in enumerate(sentences):
            # Skip very short sentences
            if len(sentence.strip()) < 10:
                continue

            grade_level, issues = self.analyze_sentence_difficulty(sentence)

            # Only include sentences above threshold
            if grade_level >= threshold:
                word_count = textstat.lexicon_count(sentence, removepunct=True)
                syllable_count = textstat.syllable_count(sentence)

                difficult_sentence = DifficultSentence(
                    text=sentence.strip(),
                    grade_level=grade_level,
                    position=i + 1,
                    issues=issues,
                    word_count=word_count,
                    syllable_count=syllable_count,
                )

                sentence_data.append(
                    {
                        "sentence": difficult_sentence.text,
                        "grade_level": round(difficult_sentence.grade_level, 1),
                        "position": difficult_sentence.position,
                        "issues": difficult_sentence.issues,
                        "word_count": difficult_sentence.word_count,
                        "syllable_count": difficult_sentence.syllable_count,
                    }
                )

        # Sort by grade level (most difficult first)
        sentence_data.sort(key=lambda x: x["grade_level"], reverse=True)

        # Return top N sentences
        return sentence_data[:count]
