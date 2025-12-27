# TODO: Teen-AI Companion Research

## Current Phase: [Phase 1 - Data Collection]
## Current Task: [Verify existing r/CharacterAI data]
## Blocked: [None]

---

## Phase 1: Data Collection & Preprocessing
- [x] Setup repository structure
- [x] Copy existing AnthroScore V2 code to src/anthroscore/
- [x] Copy existing Chew V2 code to src/chew/
- [x] Move existing r/CharacterAI data (6,570 comments) to data/raw/
- [x] Verify existing r/CharacterAI data (269,040 comments after preprocessing)
- [x] Attempted to collect r/Replika data via Arctic Shift (API errors - proceeding with existing data)
- [x] Standardize all data to common schema
- [x] Run preprocessing pipeline (dedup, filter bots, clean text)
- [x] Generate collection statistics report
- [x] CHECKPOINT: Save processed data (completed)

## Phase 2: Demographics Extraction
- [x] Implement self-declaration regex extraction
- [ ] Collect user subreddit participation data (skipped - requires broader Reddit data)
- [ ] Build subreddit co-occurrence matrix (skipped - requires broader Reddit data)
- [ ] Create community embeddings (word2vec on subreddits) (skipped - requires broader Reddit data)
- [ ] Build age dimension from seed pairs (skipped)
- [ ] Build gender dimension from seed pairs (skipped)
- [x] Implement LLM age classification for uncertain users
- [x] Create ensemble classifier
- [ ] Run on all users (READY - execute phase2_demographics.py)
- [ ] CHECKPOINT: Save demographic features

## Phase 3: Core Analysis
- [ ] Run AnthroScore V2 on all comments (READY - execute phase3_core_analysis.py)
- [ ] Run BERTopic clustering (READY)
- [ ] Run emotion classification (distilroberta) (READY)
- [ ] Aggregate features to user level (READY)
- [ ] Merge all feature sets (READY)
- [ ] CHECKPOINT: Save merged dataset

## Phase 4: Statistical Analysis
- [ ] Generate descriptive statistics tables (READY - execute phase4_statistical_analysis.py)
- [ ] Run regression models (RQ2) (READY)
- [ ] Run emotional mirroring analysis (RQ3) (Basic implementation ready, full analysis pending)
- [ ] Calculate effect sizes and confidence intervals (READY)
- [ ] Generate all figures (READY)
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
- [x] Repository structure created (2025-01-XX)
- [x] Configuration files created (.gitignore, requirements.txt, .cursor/rules/)
- [x] AnthroScore V2 code copied to src/anthroscore/
- [x] Chew V2 code copied to src/chew/
- [x] Existing data file copied to data/raw/

## Notes & Decisions
- Repository setup completed according to PROMPT.md specifications
- Using copy instead of move for data files to preserve originals
- Configuration files created following the template from PROMPT.md

