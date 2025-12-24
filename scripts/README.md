# Evaluation Scripts

Scripts for evaluating AI detection accuracy.

## Quick Start

```bash
pip install datasets

# Download test data
python scripts/download_dataset.py --dataset pile --samples 100

# Run evaluation
python scripts/evaluate.py \
    --ai-samples scripts/sample_data/ai_samples_pile.jsonl \
    --human-samples scripts/sample_data/human_samples_pile.jsonl
```

## Scripts

### `download_dataset.py`

Downloads AI detection datasets from HuggingFace.

```bash
python scripts/download_dataset.py --dataset <name> --samples <n>
```

| Dataset | Description | Source |
|---------|-------------|--------|
| `pile` | AI Text Detection Pile (GPT-2/3/4 + human) | [HuggingFace](https://huggingface.co/datasets/artem9k/ai-text-detection-pile) |
| `hc3` | Human ChatGPT Comparison Corpus | [HuggingFace](https://huggingface.co/datasets/Hello-SimpleAI/HC3) / [Paper](https://arxiv.org/abs/2301.07597) |
| `raid` | RAID benchmark | [HuggingFace](https://huggingface.co/datasets/liamdugan/raid) |
| `all` | Download all datasets | |

### `evaluate.py`

Runs detection on samples and reports metrics.

```bash
python scripts/evaluate.py \
    --ai-samples <file.jsonl> \
    --human-samples <file.jsonl> \
    --output results.json  # optional
```

**Output includes**:
- Precision, Recall, F1 at multiple thresholds
- Score distributions
- False positive/negative analysis
- Overall grade (A-F)

## Sample Data Format

JSONL with one sample per line:

```json
{"text": "...", "source": "chatgpt-4", "topic": "technology"}
```

## Grading

| Grade | F1 | FP Rate |
|-------|-----|---------|
| A | >0.85 | <10% |
| B | >0.75 | <15% |
| C | >0.65 | <20% |
| D | >0.50 | <30% |
| F | below D | |

## Known Limitations

- Patterns optimized for ChatGPT/Claude (2022+)
- Older datasets (GPT-2/GPT-3) may not trigger patterns
- For best results, use recent ChatGPT-4/Claude samples
