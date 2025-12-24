"""
AI pattern detection module
Identifies common AI-generated writing patterns and phrases
"""

import hashlib
import re
from collections import OrderedDict, defaultdict
from typing import Any


class LRUCache:
    """Simple LRU cache implementation"""

    def __init__(self, maxsize: int = 128):
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.maxsize = maxsize

    def get(self, key: str) -> Any | None:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
        self.cache[key] = value


class AIPatternDetector:
    """Detects AI-generated writing patterns in text"""

    # AI phrase patterns organized by category and weight
    AI_PATTERNS = {
        "dead_giveaways": {
            "phrases": [
                # Classic AI phrases
                "delve into",
                "delving deeper",
                "delve deeper",
                "tapestry of",
                "rich tapestry",
                "a testament to",
                "stands as a testament",
                "in today's world",
                "in today's landscape",
                "in today's society",
                "navigating the complexities",
                "navigate the complex",
                "unlock the potential",
                "unlocking insights",
                "game-changer",
                "game changer",
                "it's not just about",
                "it's about more than",
                "at its core",
                "the world of",
                "dive into",
                "diving into",
                "deep dive",
            ],
            "weight": 3.0,
        },
        "high_probability": {
            "phrases": [
                # Formal transitions AI loves
                "moreover",
                "furthermore",
                "additionally",
                "it's important to note that",
                "it's worth noting",
                "it's crucial to understand",
                "it's essential to",
                "while it's true that",
                "while it may seem",
                "on one hand",
                "on the other hand",
                "in conclusion",
                "to summarize",
                "in summary",
                # Buzzwords
                "leverage",
                "utilize",
                "paramount",
                "plethora",
                "myriad",
                "multitude",
                "a wealth of",
                # Meta-commentary
                "it bears mentioning",
                "it goes without saying",
                "needless to say",
                "as we all know",
                # Filler transitions
                "that being said",
                "with that being said",
                "that said",
                "with that said",
                "with that in mind",
                "keeping that in mind",
                "when it comes to",
                "in terms of",
                "at the end of the day",
                # AI enthusiasm
                "absolutely",
                "definitely",
                "certainly",
                "great question",
            ],
            "weight": 2.0,
        },
        "moderate_indicators": {
            "phrases": [
                # Transitions
                "however",
                "nevertheless",
                "nonetheless",
                "it should be noted",
                "bear in mind",
                # Overused adjectives
                "significant",
                "robust",
                "comprehensive",
                "various",
                "numerous",
                "multifaceted",
                "crucial",
                "vital",
                "essential",
                "pivotal",
                "key",
                "seamless",
                "cutting-edge",
                "state-of-the-art",
                # Business speak
                "synergy",
                "holistic",
                "paradigm",
                "streamline",
                "optimize",
                "enhance",
                "foster",
                "cultivate",
                "bolster",
                "elevate",
                "empower",
                "enable",
                # Domain words
                "landscape",
                "realm",
                "arena",
                "sphere",
                "ecosystem",
                # Vague intensifiers
                "incredibly",
                "extremely",
                "highly",
                "very much",
            ],
            "weight": 1.0,
        },
        "structural_patterns": {
            "phrases": [
                # Numbered structure
                "firstly",
                "secondly",
                "thirdly",
                "lastly",
                "first and foremost",
                # Hedging
                "in essence",
                "essentially",
                "fundamentally",
                "broadly speaking",
                "generally speaking",
                # Examples
                "for instance",
                "for example",
                "such as",
                "including but not limited to",
                # List intros
                "here are some",
                "here's what you need to know",
                "let's explore",
                "let's take a look",
                "consider the following",
            ],
            "weight": 0.5,
        },
        "adverb_starters": {
            "phrases": [
                # Sentence-starting adverbs (AI loves these)
                "interestingly",
                "notably",
                "importantly",
                "surprisingly",
                "remarkably",
                "undoubtedly",
                "understandably",
                "naturally",
                "obviously",
                "clearly",
                "evidently",
                "admittedly",
                "unfortunately",
                "fortunately",
                "hopefully",
                "ultimately",
                "consequently",
                "subsequently",
            ],
            "weight": 1.5,
        },
        "formatting_tells": {
            "phrases": [],  # Special handling for em dashes, colons, etc.
            "weight": 1.5,
        },
    }

    # Em dash variants to detect
    EM_DASH_PATTERN = re.compile(r"—|–|--")

    # Colon usage pattern (colon followed by explanation, not time like 10:30)
    COLON_PATTERN = re.compile(r":\s+[A-Za-z]")

    # Exclamation points
    EXCLAMATION_PATTERN = re.compile(r"[A-Za-z]+!")

    def __init__(self):
        """Initialize the AI pattern detector"""
        self.sensitivity_multipliers = {"low": 0.7, "medium": 1.0, "high": 1.3}

        # Pre-compile regex patterns for better performance
        self._compiled_patterns = {}
        for category, data in self.AI_PATTERNS.items():
            self._compiled_patterns[category] = [
                (phrase, re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE))
                for phrase in data["phrases"]
            ]

        # Initialize result cache
        self._cache = LRUCache(maxsize=128)

    def detect_patterns(
        self, text: str, sensitivity: str = "medium"
    ) -> tuple[float, list[dict[str, Any]]]:
        """
        Detect AI patterns in text and calculate AI likelihood score

        Args:
            text: The text to analyze
            sensitivity: Detection sensitivity level (low/medium/high)

        Returns:
            Tuple of (ai_score, list_of_patterns_found)
        """
        patterns_found = []
        total_score = 0

        # Get sensitivity multiplier
        multiplier = self.sensitivity_multipliers.get(sensitivity, 1.0)

        # Check each category of patterns using pre-compiled regex
        for category, data in self.AI_PATTERNS.items():
            category_matches = []

            # Skip formatting_tells - handled separately below
            if category == "formatting_tells":
                continue

            for phrase, pattern in self._compiled_patterns[category]:
                # Find all matches using pre-compiled pattern
                matches = pattern.finditer(text)

                for match in matches:
                    # Get context around the match
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()

                    # Add ellipsis if truncated
                    if start > 0:
                        context = "..." + context
                    if end < len(text):
                        context = context + "..."

                    category_matches.append(
                        {"phrase": phrase, "context": context, "position": match.start()}
                    )

                    # Add to total score
                    total_score += data["weight"] * multiplier

            if category_matches:
                patterns_found.append(
                    {
                        "category": category,
                        "confidence": self._get_confidence_level(category),
                        "matches": category_matches,
                        "count": len(category_matches),
                    }
                )

        # Detect em dashes (special handling - not word-boundary)
        em_dash_matches = []
        em_dash_sentences = []

        for match in self.EM_DASH_PATTERN.finditer(text):
            # Extract the full sentence containing the em dash
            # Find sentence boundaries (. ! ? or start/end of text)
            sent_start = text.rfind(".", 0, match.start())
            sent_start = max(sent_start, text.rfind("!", 0, match.start()))
            sent_start = max(sent_start, text.rfind("?", 0, match.start()))
            sent_start = sent_start + 1 if sent_start != -1 else 0

            sent_end = text.find(".", match.end())
            if sent_end == -1:
                sent_end = len(text)
            else:
                sent_end += 1

            # Check for ! or ? that might end sentence earlier
            excl = text.find("!", match.end())
            ques = text.find("?", match.end())
            if excl != -1 and excl < sent_end:
                sent_end = excl + 1
            if ques != -1 and ques < sent_end:
                sent_end = ques + 1

            sentence = text[sent_start:sent_end].strip()

            em_dash_matches.append(
                {
                    "phrase": f"em dash ({match.group()})",
                    "sentence": sentence,
                    "position": match.start(),
                }
            )

            if sentence not in em_dash_sentences:
                em_dash_sentences.append(sentence)

        if em_dash_matches:
            # Weight increases with count - 1-2 em dashes fine, more is suspicious
            em_dash_weight = self.AI_PATTERNS["formatting_tells"]["weight"]
            # More aggressive scoring for >2 em dashes
            if len(em_dash_matches) > 2:
                em_dash_score = em_dash_weight * (len(em_dash_matches) ** 1.5) * multiplier
            else:
                em_dash_score = em_dash_weight * len(em_dash_matches) * 0.5 * multiplier
            total_score += em_dash_score

            patterns_found.append(
                {
                    "category": "formatting_tells",
                    "confidence": self._get_em_dash_confidence(len(em_dash_matches)),
                    "matches": em_dash_matches,
                    "count": len(em_dash_matches),
                    "sentences": em_dash_sentences,
                    "type": "em_dash",
                }
            )

        # Detect colon overuse (AI uses colons to introduce explanations)
        colon_matches = list(self.COLON_PATTERN.finditer(text))
        word_count_temp = len(text.split())
        # More than 1 colon per 100 words is suspicious
        colon_density = (len(colon_matches) / max(word_count_temp, 1)) * 100
        if colon_density > 1.5 and len(colon_matches) > 2:
            colon_score = (
                self.AI_PATTERNS["formatting_tells"]["weight"]
                * (len(colon_matches) - 2)
                * multiplier
            )
            total_score += colon_score
            patterns_found.append(
                {
                    "category": "formatting_tells",
                    "confidence": "Medium" if len(colon_matches) < 5 else "High",
                    "matches": [{"phrase": "colon overuse", "count": len(colon_matches)}],
                    "count": len(colon_matches),
                    "type": "colon_overuse",
                }
            )

        # Detect exclamation points in non-casual writing
        exclamation_matches = list(self.EXCLAMATION_PATTERN.finditer(text))
        if len(exclamation_matches) > 1:
            # Multiple exclamation points in professional writing is AI-like
            excl_score = (
                self.AI_PATTERNS["formatting_tells"]["weight"]
                * len(exclamation_matches)
                * multiplier
            )
            total_score += excl_score
            excl_contexts = []
            for match in exclamation_matches[:5]:  # Limit to 5 examples
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 10)
                excl_contexts.append(text[start:end].strip())
            patterns_found.append(
                {
                    "category": "formatting_tells",
                    "confidence": "Medium" if len(exclamation_matches) < 4 else "High",
                    "matches": [
                        {"phrase": "exclamation point", "context": ctx} for ctx in excl_contexts
                    ],
                    "count": len(exclamation_matches),
                    "type": "exclamation_overuse",
                }
            )

        # Calculate final AI score (0-100)
        # Use pattern density normalized per 100 words for consistent scoring
        word_count = len(text.split())
        if word_count > 0:
            # Calculate pattern density (patterns per 100 words)
            # This makes scoring consistent regardless of text length
            pattern_density = (total_score / word_count) * 100

            # Scale to 0-100 range with a reasonable curve
            # Score of 15+ in density maps to ~100, Score of 3 maps to ~50
            ai_score = min(100, pattern_density * 6.5)

            # For very short texts (< 50 words), apply a small confidence penalty
            # to account for statistical insignificance
            if word_count < 50:
                confidence_factor = word_count / 50  # 0.2 to 1.0
                ai_score = ai_score * (0.7 + 0.3 * confidence_factor)
        else:
            ai_score = 0

        return ai_score, patterns_found

    def _get_confidence_level(self, category: str) -> str:
        """Get human-readable confidence level for a category"""
        confidence_map = {
            "dead_giveaways": "Very High",
            "high_probability": "High",
            "moderate_indicators": "Medium",
            "structural_patterns": "Low",
            "adverb_starters": "High",
            "formatting_tells": "Medium",
        }
        return confidence_map.get(category, "Unknown")

    def _get_em_dash_confidence(self, count: int) -> str:
        """Get confidence level based on em dash count"""
        if count >= 5:
            return "Very High"
        elif count > 2:
            return "High"
        else:
            return "Low"

    def interpret_ai_score(self, score: float) -> str:
        """Interpret the AI likelihood score"""
        if score < 20:
            return "Very low - Text appears naturally written"
        elif score < 40:
            return "Low - Mostly natural with minor AI indicators"
        elif score < 60:
            return "Medium - Noticeable AI patterns present"
        elif score < 80:
            return "High - Strong AI characteristics detected"
        else:
            return "Very high - Multiple strong AI patterns found"

    def get_recommendations(
        self, patterns_found: list[dict[str, Any]], ai_score: float
    ) -> list[str]:
        """
        Generate specific recommendations based on patterns found

        Args:
            patterns_found: List of detected patterns
            ai_score: Overall AI likelihood score

        Returns:
            List of specific improvement recommendations
        """
        tips = []

        # Count patterns by category
        category_counts = defaultdict(int)
        all_phrases = []

        for pattern_group in patterns_found:
            category = pattern_group["category"]
            category_counts[category] = pattern_group["count"]
            for match in pattern_group["matches"]:
                all_phrases.append(match["phrase"])

        # Specific recommendations based on patterns
        if category_counts.get("dead_giveaways", 0) > 0:
            tips.append("Replace 'delve into' with 'explore' or 'look at'")
            tips.append("Avoid 'tapestry', 'testament', 'game-changer' - use concrete descriptions")

        if category_counts.get("high_probability", 0) > 2:
            tips.append("Reduce formal transitions - start sentences directly with your point")
            tips.append("Replace 'moreover/furthermore' with 'also' or connect ideas naturally")

        if "it's important to note" in all_phrases or "it's worth noting" in all_phrases:
            tips.append(
                "Remove meta-commentary like 'it's important to note' - just state the point"
            )

        if any(
            p in all_phrases
            for p in ["that being said", "with that being said", "at the end of the day"]
        ):
            tips.append(
                "Cut filler phrases: 'that being said', 'at the end of the day' add nothing"
            )

        if category_counts.get("adverb_starters", 0) > 2:
            tips.append(
                "Reduce adverb sentence starters (Interestingly, Notably, etc.) - vary your openings"
            )

        if category_counts.get("structural_patterns", 0) > 3:
            tips.append("Vary paragraph structure - avoid rigid firstly/secondly/thirdly patterns")

        # Check for specific formatting tells by type
        formatting_types = set()
        for p in patterns_found:
            if p.get("category") == "formatting_tells" and p.get("type"):
                formatting_types.add(p["type"])

        if "em_dash" in formatting_types:
            em_count = next((p["count"] for p in patterns_found if p.get("type") == "em_dash"), 0)
            if em_count > 2:
                tips.append(
                    "Excessive em dashes (>2) - strong AI indicator. Use commas or separate sentences"
                )
            else:
                tips.append("Em dashes detected - 1-2 is normal, more than 2 is a strong AI tell")

        if "colon_overuse" in formatting_types:
            tips.append(
                "Too many colons - AI overuses them to introduce explanations. Vary your punctuation"
            )

        if "exclamation_overuse" in formatting_types:
            tips.append(
                "Multiple exclamation points detected - reduce enthusiasm in professional writing"
            )

        # Check for overuse of certain word types
        if any(p in all_phrases for p in ["leverage", "utilize", "comprehensive", "robust"]):
            tips.append("Simplify jargon: 'use' not 'utilize', 'strong' not 'robust'")

        if any(p in all_phrases for p in ["foster", "cultivate", "bolster", "elevate", "empower"]):
            tips.append("Replace growth verbs (foster, cultivate, empower) with concrete actions")

        if any(p in all_phrases for p in ["landscape", "realm", "arena", "sphere", "ecosystem"]):
            tips.append(
                "Replace vague domain words (landscape, realm, ecosystem) with specific terms"
            )

        # General recommendations based on score
        if ai_score > 60:
            tips.append(
                "Try writing in a more conversational tone - imagine explaining to a friend"
            )
            tips.append("Add personal examples or specific details to make content more authentic")
        elif ai_score > 40:
            tips.append("Consider adding more variety in sentence structure and vocabulary")
        elif ai_score < 20:
            tips.append(
                "Text appears relatively natural - minor adjustments could include varying sentence structure"
            )

        return tips

    def analyze(self, text: str, sensitivity: str = "medium") -> dict[str, Any]:
        """
        Complete AI pattern analysis with caching for identical texts

        Args:
            text: The text to analyze
            sensitivity: Detection sensitivity (low/medium/high)

        Returns:
            Dictionary containing AI score, patterns, and recommendations
        """
        # Use cached analysis for identical text+sensitivity combinations
        cache_key = f"{hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()}:{sensitivity}"
        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        ai_score, patterns_found = self.detect_patterns(text, sensitivity)

        result = {
            "ai_likelihood_score": round(ai_score, 1),
            "interpretation": self.interpret_ai_score(ai_score),
            "patterns_detected": patterns_found,
            "pattern_summary": {
                "total_patterns": sum(p["count"] for p in patterns_found),
                "categories_triggered": len(patterns_found),
                "most_common_category": (
                    max(patterns_found, key=lambda x: x["count"])["category"]
                    if patterns_found
                    else None
                ),
            },
            "recommendations": self.get_recommendations(patterns_found, ai_score),
            "sensitivity_used": sensitivity,
        }

        self._cache.set(cache_key, result)
        return result
