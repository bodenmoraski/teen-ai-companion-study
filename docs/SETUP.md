# Setup Instructions

This document provides step-by-step setup instructions for the Teen-AI Companion Research Project.

## Prerequisites

- Python 3.10+
- pip
- OpenAI API key (for LLM-based age classification)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Step 2: Configure Environment Variables

Create a `.env` file in the project root with the following content:

```bash
# OpenAI API Key (required for LLM-based age classification)
OPENAI_API_KEY=sk-your-key-here

# Optional: Anthropic API Key (for Claude-based entity resolution)
# ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Replace `sk-your-key-here` with your actual OpenAI API key.

## Step 3: Verify Setup

Run the following to verify all components are in place:

```python
from pathlib import Path

base = Path('.')
checks = {
    'data/raw': base.joinpath('data/raw').exists(),
    'src/anthroscore': base.joinpath('src/anthroscore').exists(),
    'src/chew': base.joinpath('src/chew').exists(),
    'data/raw/characterai_comments.jsonl': base.joinpath('data/raw/characterai_comments.jsonl').exists(),
    '.env': base.joinpath('.env').exists(),
}

print('Setup verification:')
for name, result in checks.items():
    status = '✓' if result else '✗'
    print(f'{status} {name}')
```

## Step 4: Next Steps

1. Review `COMPREHENSIVE_RESEARCH_PLAN.md` for methodology details
2. Check `PLAN.md` for execution status
3. Check `TODO.md` for current tasks
4. Begin Phase 1: Data Collection & Preprocessing

## Notes

- The existing `characterai_comments.jsonl` file is large (>200MB), so it has been copied (not moved) to preserve the original.
- AnthroScore V2 and Chew V2 code have been copied to `src/anthroscore/` and `src/chew/` respectively.
- Chew V2 model files (if needed) can be generated or copied later as needed.

