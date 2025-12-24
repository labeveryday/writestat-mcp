# WriteStat MCP Server

![WriteStat MCP Banner](assets/banner.png)

[![PyPI version](https://img.shields.io/pypi/v/writestat-mcp.svg)](https://pypi.org/project/writestat-mcp/)
[![Tests](https://github.com/labeveryday/writestat-mcp/actions/workflows/tests.yml/badge.svg)](https://github.com/labeveryday/writestat-mcp/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MCP server for text readability analysis and AI pattern detection. Helps writers identify AI-like patterns and improve readability.

Created by [Du'An Lightfoot](https://duanlightfoot.com) | [@labeveryday](https://github.com/labeveryday)

## Installation

```bash
pip install writestat-mcp

# Optional: ML-based detection (~500MB for torch/transformers)
pip install writestat-mcp[ml]

# Required: NLTK data
python -c "import nltk; nltk.download('punkt_tab')"
```

## Tools

| Tool | Description |
|------|-------------|
| `analyze_text` | Readability metrics (Flesch-Kincaid, SMOG, etc.) |
| `find_hard_sentences` | Complex sentences with explanations |
| `check_ai_phrases` | Pattern-based AI detection (60+ patterns) |
| `detect_ai_ml` | ML detection via GPT-2 perplexity (optional) |
| `batch_analyze` | Process multiple texts in parallel |
| `compare_texts` | Before/after comparison |

## Claude Desktop Setup

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%/Claude/claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "readability": {
      "command": "writestat-mcp"
    }
  }
}
```

## With Claude Code

```bash
# After PyPI publish
# Pattern detection only (lightweight)
claude mcp add writestat-mcp -- uvx writestat-mcp

# Or with ML detection (~500MB download)
claude mcp add writestat-mcp -- uvx "writestat-mcp[ml]"

# From local source
cd /path/to/writestat-mcp
pip install -e .
claude mcp add writestat-mcp -- writestat-mcp
```

## Example Prompts

**Full analysis workflow:**
> I just wrote this blog post. Check the readability, find any difficult sentences, and flag anything that sounds too AI-generated. Then suggest improvements: 

**Editing pass:**
> This is my draft and my revised version. Compare them and tell me if the readability improved and if I removed the AI-sounding phrases. {First_draft} vs {second_draft}

**Quick AI check:**
> Does this paragraph have any AI tells? Be specific about which phrases to fix: 

**Target audience check:**
> I'm writing for high school students. Is this text at the right reading level? Which sentences are too complex: 

## AI Detection: What to Expect

This tool uses **heuristic pattern matching** and **zero-shot perplexity scoring**—not a fine-tuned classifier.

### How It Works
- **Pattern detection**: Catches stylistic markers (em dashes, filler phrases, buzzwords)
- **ML detection**: Measures perplexity, vocabulary diversity, burstiness

### Accuracy Context
Research shows fine-tuned RoBERTa models achieve ~99% F1 on ChatGPT detection ([Guo et al., 2023](https://arxiv.org/abs/2301.07597)). Our lightweight approach won't match that. It's designed for:

- Quick pattern screening
- Catching obvious AI tells
- Educational awareness about AI writing patterns

**Not suitable for**: Academic integrity decisions, high-stakes verification

### What the Research Found
The [HC3 paper](https://arxiv.org/abs/2301.07597) identified key ChatGPT markers we detect:
- Lower perplexity (more predictable) ✓
- Lower vocabulary diversity ✓
- Formal conjunctions ("Furthermore", "It's important to note") ✓
- Organized structure with clear transitions ✓

## Score Interpretation

### Readability (Flesch-Kincaid Grade)
| Grade | Audience |
|-------|----------|
| 5- | Elementary |
| 6-8 | Middle school |
| 9-12 | High school |
| 13+ | College |

### AI Probability (ML)
| Score | Interpretation |
|-------|----------------|
| 0-30 | Likely human |
| 30-60 | Uncertain |
| 60-100 | Likely AI |

## Requirements

- Python 3.10+
- Core: fastmcp, textstat, nltk
- Optional `[ml]`: torch, transformers

## License

MIT

## References

- [HC3: How Close is ChatGPT to Human Experts?](https://arxiv.org/abs/2301.07597) - Guo et al., 2023
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastMCP](https://github.com/jlowin/fastmcp)
