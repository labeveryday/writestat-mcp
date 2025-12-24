#!/usr/bin/env python3
"""
Evaluation script for writestat-mcp AI detection.

Usage:
    python scripts/evaluate.py --ai-samples data/ai_samples.jsonl --human-samples data/human_samples.jsonl

Sample JSONL format:
    {"text": "...", "source": "chatgpt-4", "topic": "technology"}
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from writestat_mcp.analyzers import AIPatternDetector


@dataclass
class EvalResult:
    actual: str  # "ai" or "human"
    predicted_score: float
    source: str
    topic: str
    word_count: int
    patterns_found: int


def load_samples(filepath: Path, label: str) -> list[dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                sample["label"] = label
                samples.append(sample)
    return samples


def evaluate_samples(samples: list[dict], detector: AIPatternDetector) -> list[EvalResult]:
    """Run detection on all samples."""
    results = []
    for sample in samples:
        text = sample["text"]
        result = detector.analyze(text)

        results.append(EvalResult(
            actual=sample["label"],
            predicted_score=result["ai_likelihood_score"],
            source=sample.get("source", "unknown"),
            topic=sample.get("topic", "unknown"),
            word_count=len(text.split()),
            patterns_found=result["pattern_summary"]["total_patterns"]
        ))
    return results


def calculate_metrics(results: list[EvalResult], threshold: float = 50.0) -> dict:
    """Calculate precision, recall, F1, accuracy."""
    tp = sum(1 for r in results if r.actual == "ai" and r.predicted_score >= threshold)
    tn = sum(1 for r in results if r.actual == "human" and r.predicted_score < threshold)
    fp = sum(1 for r in results if r.actual == "human" and r.predicted_score >= threshold)
    fn = sum(1 for r in results if r.actual == "ai" and r.predicted_score < threshold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(results) if results else 0

    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        "threshold": threshold,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate
    }


def print_report(results: list[EvalResult], thresholds: list[float] = None):
    """Print evaluation report."""
    if thresholds is None:
        thresholds = [30, 40, 50, 60, 70]

    ai_results = [r for r in results if r.actual == "ai"]
    human_results = [r for r in results if r.actual == "human"]

    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print()

    # Dataset summary
    print("DATASET SUMMARY")
    print("-" * 40)
    print(f"Total samples: {len(results)}")
    print(f"  AI samples: {len(ai_results)}")
    print(f"  Human samples: {len(human_results)}")
    print()

    # Score distributions
    print("SCORE DISTRIBUTIONS")
    print("-" * 40)
    ai_scores = [r.predicted_score for r in ai_results]
    human_scores = [r.predicted_score for r in human_results]

    if ai_scores:
        print(f"AI text avg score: {sum(ai_scores)/len(ai_scores):.1f}%")
        print(f"AI text min/max: {min(ai_scores):.1f}% / {max(ai_scores):.1f}%")
    if human_scores:
        print(f"Human text avg score: {sum(human_scores)/len(human_scores):.1f}%")
        print(f"Human text min/max: {min(human_scores):.1f}% / {max(human_scores):.1f}%")
    print()

    # Metrics by threshold
    print("METRICS BY THRESHOLD")
    print("-" * 40)
    print(f"{'Thresh':>6} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Acc':>6} {'FP%':>6} {'FN%':>6}")
    print("-" * 42)

    best_f1 = 0
    best_threshold = 50

    for threshold in thresholds:
        m = calculate_metrics(results, threshold)
        print(f"{threshold:>6.0f} {m['precision']:>6.2f} {m['recall']:>6.2f} {m['f1']:>6.2f} "
              f"{m['accuracy']:>6.2f} {m['fp_rate']*100:>5.1f}% {m['fn_rate']*100:>5.1f}%")

        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_threshold = threshold

    print()
    print(f"Best threshold: {best_threshold} (F1: {best_f1:.2f})")
    print()

    # Results by source
    print("RESULTS BY SOURCE")
    print("-" * 40)
    by_source = defaultdict(list)
    for r in results:
        by_source[r.source].append(r)

    for source, source_results in sorted(by_source.items()):
        avg_score = sum(r.predicted_score for r in source_results) / len(source_results)
        actual = source_results[0].actual
        print(f"{source}: {len(source_results)} samples, avg score: {avg_score:.1f}% ({actual})")
    print()

    # False positives
    fps = [r for r in human_results if r.predicted_score >= 50]
    if fps:
        print("FALSE POSITIVES (Human flagged as AI, threshold 50)")
        print("-" * 40)
        for r in sorted(fps, key=lambda x: -x.predicted_score)[:5]:
            print(f"  Score: {r.predicted_score:.1f}%, Source: {r.source}, Patterns: {r.patterns_found}")
    print()

    # False negatives
    fns = [r for r in ai_results if r.predicted_score < 50]
    if fns:
        print("FALSE NEGATIVES (AI missed, threshold 50)")
        print("-" * 40)
        for r in sorted(fns, key=lambda x: x.predicted_score)[:5]:
            print(f"  Score: {r.predicted_score:.1f}%, Source: {r.source}, Patterns: {r.patterns_found}")
    print()

    # Grade
    m = calculate_metrics(results, 50)
    if m['f1'] > 0.85 and m['fp_rate'] < 0.10:
        grade = "A"
    elif m['f1'] > 0.75 and m['fp_rate'] < 0.15:
        grade = "B"
    elif m['f1'] > 0.65 and m['fp_rate'] < 0.20:
        grade = "C"
    elif m['f1'] > 0.50 and m['fp_rate'] < 0.30:
        grade = "D"
    else:
        grade = "F"

    print("=" * 60)
    print(f"OVERALL GRADE: {grade}")
    print(f"F1 Score: {m['f1']:.2f}, FP Rate: {m['fp_rate']*100:.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI detection accuracy")
    parser.add_argument("--ai-samples", type=Path, required=True, help="JSONL file with AI text samples")
    parser.add_argument(
        "--human-samples", type=Path, required=True, help="JSONL file with human text samples"
    )
    parser.add_argument("--output", type=Path, help="Output JSON file for detailed results")
    args = parser.parse_args()

    # Load samples
    print("Loading samples...")
    ai_samples = load_samples(args.ai_samples, "ai")
    human_samples = load_samples(args.human_samples, "human")
    all_samples = ai_samples + human_samples

    print(f"Loaded {len(ai_samples)} AI samples, {len(human_samples)} human samples")

    # Run evaluation
    print("Running detection...")
    detector = AIPatternDetector()
    results = evaluate_samples(all_samples, detector)

    # Print report
    print_report(results)

    # Save detailed results
    if args.output:
        output_data = {
            "results": [
                {
                    "actual": r.actual,
                    "predicted_score": r.predicted_score,
                    "source": r.source,
                    "topic": r.topic,
                    "word_count": r.word_count,
                    "patterns_found": r.patterns_found
                }
                for r in results
            ],
            "metrics": {
                str(t): calculate_metrics(results, t)
                for t in [30, 40, 50, 60, 70]
            }
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
