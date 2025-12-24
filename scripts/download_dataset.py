#!/usr/bin/env python3
"""
Download and prepare evaluation datasets.

Usage:
    pip install datasets
    python scripts/download_dataset.py --dataset hc3 --output scripts/sample_data/
"""

import argparse
import json
import random
from pathlib import Path


def download_hc3(output_dir: Path, samples_per_class: int = 100):
    """Download HC3 (Human ChatGPT Comparison) dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return

    print("Downloading HC3 dataset...")
    dataset = load_dataset("Hello-SimpleAI/HC3", "all")

    ai_samples = []
    human_samples = []

    for split in ["train"]:
        for item in dataset[split]:
            question = item["question"]
            human_answers = item["human_answers"]
            chatgpt_answers = item["chatgpt_answers"]

            # Collect human answers
            for answer in human_answers:
                if len(answer.split()) >= 30:  # Min 30 words
                    human_samples.append({
                        "text": answer,
                        "source": "hc3_human",
                        "topic": "qa",
                        "question": question[:100]
                    })

            # Collect ChatGPT answers
            for answer in chatgpt_answers:
                if len(answer.split()) >= 30:
                    ai_samples.append({
                        "text": answer,
                        "source": "hc3_chatgpt",
                        "topic": "qa",
                        "question": question[:100]
                    })

    # Sample randomly
    random.shuffle(ai_samples)
    random.shuffle(human_samples)

    ai_samples = ai_samples[:samples_per_class]
    human_samples = human_samples[:samples_per_class]

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "ai_samples_hc3.jsonl", "w") as f:
        for sample in ai_samples:
            f.write(json.dumps(sample) + "\n")

    with open(output_dir / "human_samples_hc3.jsonl", "w") as f:
        for sample in human_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(ai_samples)} AI samples to {output_dir}/ai_samples_hc3.jsonl")
    print(f"Saved {len(human_samples)} human samples to {output_dir}/human_samples_hc3.jsonl")


def download_ai_detection_pile(output_dir: Path, samples_per_class: int = 100):
    """Download AI Text Detection Pile dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return

    print("Downloading AI Text Detection Pile...")
    dataset = load_dataset("artem9k/ai-text-detection-pile", split="train")

    ai_samples = []
    human_samples = []

    for item in dataset:
        text = item["text"]
        source = item["source"]  # "human" or AI model name

        if len(text.split()) < 30:
            continue

        sample = {
            "text": text[:2000],  # Truncate very long texts
            "source": f"pile_{source}",
            "topic": "essay"
        }

        if source == "human":
            human_samples.append(sample)
        else:
            ai_samples.append(sample)

    # Sample randomly
    random.shuffle(ai_samples)
    random.shuffle(human_samples)

    ai_samples = ai_samples[:samples_per_class]
    human_samples = human_samples[:samples_per_class]

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "ai_samples_pile.jsonl", "w") as f:
        for sample in ai_samples:
            f.write(json.dumps(sample) + "\n")

    with open(output_dir / "human_samples_pile.jsonl", "w") as f:
        for sample in human_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(ai_samples)} AI samples to {output_dir}/ai_samples_pile.jsonl")
    print(f"Saved {len(human_samples)} human samples to {output_dir}/human_samples_pile.jsonl")


def download_raid(output_dir: Path, samples_per_class: int = 100):
    """Download RAID benchmark dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return

    print("Downloading RAID dataset...")
    try:
        dataset = load_dataset("liamdugan/raid", split="train")
    except Exception as e:
        print(f"Could not load RAID: {e}")
        print("Try: pip install datasets --upgrade")
        return

    ai_samples = []
    human_samples = []

    for item in dataset:
        text = item.get("generation", item.get("text", ""))
        label = item.get("label", item.get("model", ""))

        if len(text.split()) < 30:
            continue

        sample = {
            "text": text[:2000],
            "source": f"raid_{label}" if label else "raid_unknown",
            "topic": item.get("domain", "unknown")
        }

        # In RAID, "human" label means human-written
        if label == "human":
            human_samples.append(sample)
        else:
            ai_samples.append(sample)

    random.shuffle(ai_samples)
    random.shuffle(human_samples)

    ai_samples = ai_samples[:samples_per_class]
    human_samples = human_samples[:samples_per_class]

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "ai_samples_raid.jsonl", "w") as f:
        for sample in ai_samples:
            f.write(json.dumps(sample) + "\n")

    with open(output_dir / "human_samples_raid.jsonl", "w") as f:
        for sample in human_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(ai_samples)} AI samples to {output_dir}/ai_samples_raid.jsonl")
    print(f"Saved {len(human_samples)} human samples to {output_dir}/human_samples_raid.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Download evaluation datasets")
    parser.add_argument("--dataset", choices=["hc3", "pile", "raid", "all"], default="hc3",
                        help="Dataset to download")
    parser.add_argument("--output", type=Path, default=Path("scripts/sample_data"),
                        help="Output directory")
    parser.add_argument("--samples", type=int, default=100,
                        help="Samples per class")
    args = parser.parse_args()

    if args.dataset == "hc3" or args.dataset == "all":
        download_hc3(args.output, args.samples)

    if args.dataset == "pile" or args.dataset == "all":
        download_ai_detection_pile(args.output, args.samples)

    if args.dataset == "raid" or args.dataset == "all":
        download_raid(args.output, args.samples)

    print("\nTo run evaluation:")
    print(f"  python scripts/evaluate.py --ai-samples {args.output}/ai_samples_*.jsonl --human-samples {args.output}/human_samples_*.jsonl")


if __name__ == "__main__":
    main()
