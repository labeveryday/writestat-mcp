# Changelog

All notable changes to the WriteStat MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-24

### Added
- **Core Analysis Tools**
  - `analyze_text` - Readability scoring with 8 metrics (Flesch-Kincaid, SMOG, etc.)
  - `find_hard_sentences` - Complex sentence detection with explanations
  - `check_ai_phrases` - Pattern-based AI detection with 60+ patterns
  - `detect_ai_ml` - ML-based AI detection using GPT-2 perplexity (optional)
  - `batch_analyze` - Parallel processing for multiple texts
  - `compare_texts` - Before/after text comparison

- **AI Detection Features**
  - 60+ AI phrase patterns across categories (dead giveaways, high probability, etc.)
  - Em dash detection (>2 likely AI)
  - Colon overuse detection
  - Exclamation point analysis
  - ML-based perplexity scoring
  - Burstiness analysis (sentence length variance)
  - Vocabulary diversity metrics

- **Package Structure**
  - PyPI-ready packaging with `pyproject.toml`
  - CLI entrypoint: `writestat-mcp`
  - Optional ML dependencies: `pip install writestat-mcp[ml]`
  - Modular analyzer architecture

- **Evaluation Framework**
  - `scripts/evaluate.py` - Automated evaluation with metrics
  - `scripts/download_dataset.py` - Dataset downloader (HC3, Pile, RAID)
  - `EVALUATION.md` - Methodology documentation
  - `EVALUATION_REPORT.md` - Report template

### Technical Details
- Python 3.10+ required
- FastMCP 2.0+ for MCP protocol
- Core deps: textstat, nltk (~3.5 MB)
- Optional ML deps: torch, transformers (~1 GB)
