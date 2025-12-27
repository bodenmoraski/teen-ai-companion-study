# Cursor Agent Setup: Teen-AI Companion Research Project

This document provides everything you need to set up a single Cursor agent to execute the full research pipeline continuously.

---

## Table of Contents
1. [Pre-Flight Checklist: What YOU Need to Provide](#pre-flight-checklist)
2. [Repository Setup](#repository-setup)
3. [Cursor Configuration Files](#cursor-configuration)
4. [Master Agent Prompt](#master-agent-prompt)
5. [How the Agent Should Work](#agent-workflow)

---

## Pre-Flight Checklist: What YOU Need to Provide {#pre-flight-checklist}

### ✅ REQUIRED - You Must Supply These

| Resource | Status | How to Get It | Notes |
|----------|--------|---------------|-------|
| **OpenAI API Key** | ❓ | You mentioned you have nonprofit keys | Put in `.env` file as `OPENAI_API_KEY=sk-...` |
| **Existing r/CharacterAI data** | ❓ | Your 6,570 comments JSONL | Place in `data/raw/characterai_comments.jsonl` |
| **AnthroScore V2 code** | ❓ | You have this already | Place in `src/anthroscore/` |
| **Chew V2 code** | ❓ | You have this already | Place in `src/chew/` |

### ✅ FREE / No Key Needed - Agent Can Access Directly

| Resource | Access Method | Notes |
|----------|---------------|-------|
| **Arctic Shift API** | HTTP requests to `arctic-shift.photon-reddit.com` | No auth required, rate limits apply |
| **HuggingFace Models** | `transformers` library auto-downloads | Free, ~2-4GB disk space |
| **Reddit User History** | Via Arctic Shift author search | No Reddit API key needed |
| **spaCy Models** | `python -m spacy download en_core_web_sm` | Free, ~50MB |

### ⚠️ OPTIONAL - Nice to Have

| Resource | Benefit | How to Get |
|----------|---------|------------|
| **GPU (8GB+ VRAM)** | 10-50x faster for transformers | Your gaming laptop should work |
| **Colab Pro** | Backup compute if local fails | ~$10/month |
| **Anthropic API Key** | For Claude-based entity resolution | Your existing claude.ai access |

### 📄 Papers/Methods - Agent Can Implement From Descriptions

The agent doesn't need these papers directly - the methodology is encoded in the prompts:
- Cheng et al. (2024) AnthroScore → Already have V2 code
- Chew et al. (2021) Age Classification → Already have V2 code  
- Toronto CSS Lab community embeddings → Methodology in agent prompt
- Chu et al. emotional mirroring → Simplified version in agent prompt

---

## Repository Setup {#repository-setup}

Create this structure before starting. The agent will populate it:

```
illusion-project/
├── .cursor/
│   └── rules/
│       └── research-agent.mdc       # Main rules file (create from below)
├── .env                              # API keys (YOU CREATE THIS)
├── .gitignore
├── TODO.md                           # Agent maintains this
├── PLAN.md                           # Agent creates initial plan here
├── COMPREHENSIVE_RESEARCH_PLAN.md    # Copy the plan I created
├── requirements.txt
├── data/
│   ├── raw/                          # Put your existing data here
│   ├── processed/
│   ├── features/
│   └── annotations/
├── src/
│   ├── anthroscore/                  # Your existing AnthroScore V2
│   ├── chew/                         # Your existing Chew V2
│   ├── data_collection/
│   ├── demographics/
│   ├── analysis/
│   └── utils/
├── notebooks/
├── results/
│   ├── figures/
│   ├── tables/
│   └── models/
└── tests/
```

### Create `.env` file:
```bash
# .env
OPENAI_API_KEY=sk-your-key-here
# Optional:
# ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Create `.gitignore`:
```gitignore
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
data/raw/*.jsonl
data/processed/*.parquet
*.pkl
.cursor/
node_modules/
```

### Create initial `requirements.txt`:
```
# Core data science
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# NLP & ML
transformers>=4.30.0
torch>=2.0.0
sentence-transformers>=2.2.0
bertopic>=0.15.0
spacy>=3.5.0
vaderSentiment>=3.3.2
emoji>=2.0.0

# Reddit data
requests>=2.28.0
aiohttp>=3.8.0

# Statistics
statsmodels>=0.14.0
scikit-learn>=1.2.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.65.0
pyarrow>=12.0.0  # for parquet files

# OpenAI
openai>=1.0.0
```

---

## Cursor Configuration {#cursor-configuration}

### File: `.cursor/rules/research-agent.mdc`

Create this file in your project:

```markdown
---
description: Research agent rules for Teen-AI Companion study
globs: ["**/*.py", "**/*.md", "**/*.ipynb"]
alwaysApply: true
---

# Research Agent Rules

You are a computational social science research agent executing a study on teen-AI companion relationships on Reddit.

## Core Principles

1. **Plan First, Execute Second**: Before any implementation, update PLAN.md and TODO.md
2. **Incremental Progress**: Complete one task fully before starting another
3. **Document Everything**: Every function needs docstrings, every decision needs comments
4. **Validate Constantly**: Test each component before moving to the next
5. **Git Discipline**: Suggest commits after each completed task

## File Management

- **TODO.md**: Your task tracker. Check items off as completed. Add new tasks as discovered.
- **PLAN.md**: High-level execution plan. Update status as you progress.
- **COMPREHENSIVE_RESEARCH_PLAN.md**: Reference document. Do not modify.

## Code Style

- Python 3.10+ features allowed
- Type hints required for all functions
- Use pathlib for file paths
- Prefer pandas over raw Python for data manipulation
- Use logging instead of print statements
- PEP 8 compliant (max line length: 100)

## Error Handling

- Wrap API calls in try/except with retries
- Log errors with full context
- Never silently fail - raise or log
- Create checkpoint files for long-running operations

## Data Handling

- All raw data stays in `data/raw/`
- Processed data goes to `data/processed/`
- Features go to `data/features/`
- Use parquet format for large dataframes (faster than CSV)
- Include data validation checks

## When Stuck

1. Re-read the relevant section of COMPREHENSIVE_RESEARCH_PLAN.md
2. Check if dependencies are installed
3. Look for similar implementations in existing code
4. If truly blocked, add a `# TODO: BLOCKED - [reason]` comment and move to next task

## Prohibited Actions

- Never delete data files without explicit confirmation
- Never push to git without user approval
- Never make API calls without rate limiting
- Never store API keys in code files
```

### Cursor Settings Recommendations

In Cursor Settings → Rules → User Rules, add:
```
- Be thorough but concise
- Show actual code, not explanations of code
- When editing files, show the complete function/class being modified
- Prefer fixing errors over explaining them
- Use TODO comments for incomplete work
- Commit suggestions should be atomic and descriptive
```

### YOLO Mode Settings (Optional but Recommended)

In Cursor Settings → Features → YOLO Mode:
```
Enable: Yes
Allow list:
- pytest
- python -m pytest
- pip install
- pip list
- python -c "import X"
- ls
- cat
- head
- wc -l

Deny list:
- rm -rf
- git push
- curl (without --dry-run)
- Any command with API keys
```

---

## Master Agent Prompt {#master-agent-prompt}

**Copy this entire prompt to start the agent in Cursor's Agent Mode:**

```markdown
# Research Agent: Teen-AI Companion Relationships on Reddit

You are executing a computational social science research project. Your job is to build and run a complete analysis pipeline.

## Your First Actions

1. **Read the plan**: Open and carefully read `COMPREHENSIVE_RESEARCH_PLAN.md`
2. **Create TODO.md**: Initialize your task tracker (template below)
3. **Create PLAN.md**: Create your execution plan with status tracking
4. **Verify setup**: Check that all dependencies are available
5. **Begin Phase 1**: Start with data collection/verification

## TODO.md Template

Create this file and maintain it throughout:

```markdown
# TODO: Teen-AI Companion Research

## Current Phase: [Phase 1 - Data Collection]
## Current Task: [Verify existing data]
## Blocked: [None]

---

## Phase 1: Data Collection & Preprocessing
- [ ] Verify existing r/CharacterAI data (6,570 comments)
- [ ] Collect r/Replika data via Arctic Shift
- [ ] Collect secondary subreddits (r/replika_ai, r/AICompanions)
- [ ] Standardize all data to common schema
- [ ] Run preprocessing pipeline (dedup, filter bots, clean text)
- [ ] Generate collection statistics report
- [ ] CHECKPOINT: Save processed data

## Phase 2: Demographics Extraction
- [ ] Implement self-declaration regex extraction
- [ ] Collect user subreddit participation data
- [ ] Build subreddit co-occurrence matrix
- [ ] Create community embeddings (word2vec on subreddits)
- [ ] Build age dimension from seed pairs
- [ ] Build gender dimension from seed pairs
- [ ] Implement LLM age classification for uncertain users
- [ ] Create ensemble classifier
- [ ] Run on all users
- [ ] CHECKPOINT: Save demographic features

## Phase 3: Core Analysis
- [ ] Run AnthroScore V2 on all comments
- [ ] Run BERTopic clustering
- [ ] Run emotion classification (distilroberta)
- [ ] Aggregate features to user level
- [ ] Merge all feature sets
- [ ] CHECKPOINT: Save merged dataset

## Phase 4: Statistical Analysis
- [ ] Generate descriptive statistics tables
- [ ] Run regression models (RQ2)
- [ ] Run emotional mirroring analysis (RQ3)
- [ ] Calculate effect sizes and confidence intervals
- [ ] Generate all figures
- [ ] CHECKPOINT: Save results

## Phase 5: Validation & Output
- [ ] Create annotation sample (50 users)
- [ ] Generate annotation interface/spreadsheet
- [ ] Calculate inter-method agreement
- [ ] Write results summary
- [ ] Create final figures (publication-ready)
- [ ] Package all outputs

---

## Discovered Tasks
(Add tasks here as you discover them)

## Completed Tasks
(Move completed tasks here with completion date)

## Notes & Decisions
(Document important decisions here)
```

## How to Execute Tasks

For each task:
1. Update TODO.md to show current task
2. Plan the implementation (comment in code or think step-by-step)
3. Write the code
4. Test the code
5. Check off the task in TODO.md
6. Suggest a git commit message
7. Move to next task

## Key Implementation Details

### Arctic Shift API Usage
```python
import requests
import time

BASE_URL = "https://arctic-shift.photon-reddit.com/api"

def fetch_comments(subreddit, after_utc, before_utc, limit=1000):
    """Fetch comments from Arctic Shift API."""
    params = {
        "subreddit": subreddit,
        "after": after_utc,
        "before": before_utc, 
        "limit": limit,
        "sort": "created_utc",
        "sort_type": "asc"
    }
    
    response = requests.get(f"{BASE_URL}/comments/search", params=params)
    response.raise_for_status()
    
    time.sleep(1)  # Rate limiting
    return response.json()["data"]
```

### Community Embedding Seed Pairs
```python
AGE_SEED_PAIRS = [
    ("teenagers", "RedditForGrownups"),
    ("teenrelationships", "relationship_advice"),
    ("highschool", "college"),
    ("GenZ", "GenX"),
]

GENDER_SEED_PAIRS = [
    ("AskWomen", "AskMen"),
    ("TwoXChromosomes", "MensRights"),
    ("TheGirlSurvivalGuide", "everyman"),
]
```

### LLM Classification (GPT-4o-mini)
```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()  # Uses OPENAI_API_KEY from .env

def classify_age_llm(comments: list[str]) -> dict:
    """Classify user age bucket using GPT-4o-mini."""
    sample = comments[:20]  # Limit context
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Analyze these Reddit comments and estimate age bucket.
            
Comments:
{chr(10).join(sample)}

Respond with JSON only:
{{"age_bucket": "13-18|19-25|26-40|41-60|61-80", "confidence": 0.0-1.0}}"""
        }],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)
```

## Reference Files

- `@COMPREHENSIVE_RESEARCH_PLAN.md` - Full methodology details
- `@src/anthroscore/` - AnthroScore V2 implementation
- `@src/chew/` - Age classifier implementation
- `@data/raw/` - Source data location

## Success Criteria

You're done when:
1. All TODO items are checked off
2. `data/features/full_merged_dataset.parquet` exists
3. `results/tables/` contains all statistical outputs
4. `results/figures/` contains all visualizations
5. Each phase has a checkpoint file

Begin by reading COMPREHENSIVE_RESEARCH_PLAN.md, then create TODO.md and PLAN.md.
```

---

## How the Agent Should Work {#agent-workflow}

### Starting a Session

1. Open Cursor in the project directory
2. Open Agent mode (Cmd+I or click Composer → Agent)
3. Paste the Master Agent Prompt above
4. The agent will read the plan and create TODO.md

### Continuing Work

When you return to the project:
```
Continue working on the research project. Check TODO.md for current status and pick up where we left off.
```

### If Agent Gets Stuck

```
The current task is blocked. Please:
1. Document the blocker in TODO.md
2. Move to the next unblocked task
3. We'll resolve the blocker later
```

### Requesting Specific Work

```
Focus on [specific task]. Update TODO.md when complete.
```

### Checkpointing

After each phase, the agent should:
1. Save intermediate data files
2. Update TODO.md and PLAN.md
3. Suggest a git commit
4. Summarize what was accomplished

---

## Quick Reference: What Agent Can/Cannot Do

### ✅ Agent CAN Do Autonomously
- Read/write files in project directory
- Run Python scripts
- Install pip packages
- Make HTTP requests to Arctic Shift
- Download HuggingFace models
- Run statistical analyses
- Generate figures

### ⚠️ Agent SHOULD Ask First
- Make API calls that cost money (OpenAI)
- Delete any files
- Run long operations (>5 minutes)
- Make assumptions about methodology

### ❌ Agent CANNOT Do
- Access private APIs without keys you provide
- Run GPU code without a GPU present
- Access the internet beyond allowed domains
- Push to git repositories

---

## Troubleshooting

### "Module not found" errors
```
pip install -r requirements.txt
```

### "CUDA out of memory"
Add to the code:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
```

### "Rate limited by Arctic Shift"
The agent should already handle this, but if not:
```python
time.sleep(2)  # Increase delay between requests
```

### "OpenAI API error"
Check `.env` file has valid key:
```bash
cat .env  # Should show OPENAI_API_KEY=sk-...
```

---

## Final Checklist Before Starting

- [ ] Created project directory structure
- [ ] Placed your existing data in `data/raw/`
- [ ] Placed AnthroScore V2 code in `src/anthroscore/`
- [ ] Placed Chew V2 code in `src/chew/`
- [ ] Created `.env` with `OPENAI_API_KEY`
- [ ] Created `.cursor/rules/research-agent.mdc`
- [ ] Copied `COMPREHENSIVE_RESEARCH_PLAN.md` to project root
- [ ] Installed base requirements: `pip install -r requirements.txt`
- [ ] Opened Cursor in project directory
- [ ] Ready to paste Master Agent Prompt

**You're ready to go! 🚀**