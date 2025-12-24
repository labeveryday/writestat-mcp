"""
Input validation utilities for text analysis
"""

from typing import Any


class ValidationError(Exception):
    """Custom exception for validation errors"""

    pass


def validate_text(text: str, min_length: int = 1, max_length: int = 500000) -> str:
    """
    Validate text input for analysis

    Args:
        text: The text to validate
        min_length: Minimum allowed length (default: 1)
        max_length: Maximum allowed length (default: 500,000 chars)

    Returns:
        Cleaned text (stripped whitespace)

    Raises:
        ValidationError: If text fails validation
    """
    if not isinstance(text, str):
        raise ValidationError("Text must be a string")

    # Strip whitespace
    cleaned_text = text.strip()

    if len(cleaned_text) < min_length:
        raise ValidationError(f"Text must be at least {min_length} characters long")

    if len(cleaned_text) > max_length:
        raise ValidationError(
            f"Text exceeds maximum length of {max_length:,} characters "
            f"(got {len(cleaned_text):,})"
        )

    # Check if text contains actual words
    word_count = len(cleaned_text.split())
    if word_count == 0:
        raise ValidationError("Text must contain at least one word")

    return cleaned_text


def validate_count(count: int, min_val: int = 1, max_val: int = 100) -> int:
    """
    Validate count parameter

    Args:
        count: The count value to validate
        min_val: Minimum allowed value (default: 1)
        max_val: Maximum allowed value (default: 100)

    Returns:
        The validated count

    Raises:
        ValidationError: If count is invalid
    """
    if not isinstance(count, int):
        raise ValidationError("Count must be an integer")

    if count < min_val:
        raise ValidationError(f"Count must be at least {min_val}")

    if count > max_val:
        raise ValidationError(f"Count cannot exceed {max_val}")

    return count


def validate_threshold(threshold: float, min_val: float = 0.0, max_val: float = 30.0) -> float:
    """
    Validate threshold parameter

    Args:
        threshold: The threshold value to validate
        min_val: Minimum allowed value (default: 0.0)
        max_val: Maximum allowed value (default: 30.0)

    Returns:
        The validated threshold

    Raises:
        ValidationError: If threshold is invalid
    """
    if not isinstance(threshold, (int, float)):
        raise ValidationError("Threshold must be a number")

    threshold = float(threshold)

    if threshold < min_val:
        raise ValidationError(f"Threshold must be at least {min_val}")

    if threshold > max_val:
        raise ValidationError(f"Threshold cannot exceed {max_val}")

    return threshold


def validate_sensitivity(sensitivity: str) -> str:
    """
    Validate sensitivity parameter

    Args:
        sensitivity: The sensitivity level to validate

    Returns:
        The validated sensitivity level

    Raises:
        ValidationError: If sensitivity is invalid
    """
    valid_levels = {"low", "medium", "high"}

    if not isinstance(sensitivity, str):
        raise ValidationError("Sensitivity must be a string")

    sensitivity = sensitivity.lower().strip()

    if sensitivity not in valid_levels:
        raise ValidationError(f"Sensitivity must be one of {', '.join(sorted(valid_levels))}")

    return sensitivity


def validate_metrics(metrics: list[str] | None) -> list[str] | None:
    """
    Validate metrics parameter

    Args:
        metrics: The metrics list to validate

    Returns:
        The validated metrics list

    Raises:
        ValidationError: If metrics are invalid
    """
    if metrics is None:
        return None

    if not isinstance(metrics, list):
        raise ValidationError("Metrics must be a list of strings")

    valid_metrics = {
        "flesch_kincaid",
        "flesch_ease",
        "smog",
        "ari",
        "coleman_liau",
        "linsear",
        "gunning_fog",
        "dale_chall",
    }

    validated = []
    for metric in metrics:
        if not isinstance(metric, str):
            raise ValidationError("Each metric must be a string")

        metric = metric.lower().strip()
        if metric not in valid_metrics:
            raise ValidationError(
                f"Invalid metric '{metric}'. Valid metrics: {', '.join(sorted(valid_metrics))}"
            )

        validated.append(metric)

    if not validated:
        raise ValidationError("Metrics list cannot be empty")

    return validated


def create_error_response(error: Exception) -> dict[str, Any]:
    """
    Create a standardized error response

    Args:
        error: The exception that occurred

    Returns:
        Error response dictionary
    """
    if isinstance(error, ValidationError):
        return {"error": "Validation error", "message": str(error), "type": "validation_error"}
    else:
        return {"error": "Processing error", "message": str(error), "type": "processing_error"}
