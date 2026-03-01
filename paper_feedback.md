# Comprehensive Audit of the Paper Draft

*Generated: February 17, 2026*
*Source: Cross-referenced against all up-to-date codebase data and analysis results*

---

## 🔴 CRITICAL ISSUE: Classifier Methodology Discrepancy

**The paper describes the age/gender classifiers as "LLM-based (GPT-4o-mini)" but the actual analysis uses ML-based (V4 stacking ensemble) predictions.**

Here's the evidence:

- The `COMPREHENSIVE_V3_ANALYSIS.py` loads from `age_predictions_v4.parquet` and `gender_predictions_v4.parquet`
- These V4 files come from `v4_advanced_models.py` — a **multi-algorithm stacking ensemble** (XGBoost + LightGBM + RF + LogReg), NOT GPT-4o-mini
- The actual GPT-4o-mini LLM classifications exist in `Data/features/llm_classifications.parquet` but only cover **5,000 users** (out of 47,062), so they were never used for the full analysis
- The validation numbers in the paper (95.0% accuracy, 97.2% teen recall, 92.3% adult recall) come from the **V3 ML model** (`FINAL_MODEL_SUMMARY.md`), not the V4 model actually used in the analysis
- The V4 model shows **100% accuracy** against ground truth users (suspicious — likely because the feature set includes self-declaration-derived features, creating circularity)

**You need to decide:** Either (a) rewrite the methodology to accurately describe the ML-based V3/V4 approach, or (b) actually run the LLM-based classifier on the full dataset and re-do the analysis. The V3 ML model is the best-validated one and its numbers are what's cited. I'd recommend describing the V3 model approach accurately.

---

## 📊 METHODOLOGY SECTION — Item-by-Item

### 1. Data Collection

**🔲 `[date range]` — FOUND:**

- **Date range: January 1, 2024 – December 26, 2025**
- You can write: *"We collected 283,895 public comments posted between January 2024 and December 2025."*

**🔲 Subreddit comment counts discrepancy:**

The paper's methodology section and `METHODOLOGY_FINAL.md` list different subreddit counts than the raw data. Here are the **raw totals** (use these for the data collection section):

| Subreddit | Raw Comments | After Cleaning |
|-----------|-------------|----------------|
| r/CharacterAI | 397,230 | 269,040 |
| r/Replika | 10,000 | 8,380 |
| r/AICompanions | 7,527 | 6,475 |
| **Total** | **414,757** | **283,895** |

The smaller numbers in `METHODOLOGY_FINAL.md` (118,686 / 3,658 / 2,196) are the counts after **demographic confidence filtering** — those belong in the analysis section, not data collection.

---

### 2. Age Classification

**🔲 TABLE 2: Age Classifier Performance (ready to use)**

These numbers come from the V3 model validated against 459 self-declared age users (254 teens, 205 adults):

| Threshold | N | Coverage | Overall Accuracy | Teen Recall | Adult Recall |
|-----------|---|----------|-----------------|-------------|--------------|
| ≥ 0.50 | 459 | 100.0% | 93.7% | 96.5% | 90.2% |
| ≥ 0.55 | 450 | 98.0% | 94.2% | 96.4% | 91.5% |
| **≥ 0.60** | **443** | **96.5%** | **95.0%** | **97.2%** | **92.3%** |
| ≥ 0.65 | 429 | 93.5% | 96.0% | 97.9% | 93.7% |
| ≥ 0.70 | 413 | 90.0% | 97.1% | 98.7% | 95.0% |
| ≥ 0.80 | 401 | 87.4% | 98.0% | 99.1% | 96.6% |
| ≥ 0.90 | 382 | 83.2% | 99.2% | 100.0% | 98.2% |

**Ground truth:** 459 users with self-declared ages extracted from comments via regex patterns (e.g., "I'm 16", "as a 35 year old", "16F").

**🔲 `[Sentence about previous work and validation]` (Commented [22]):**

You need a citation for the age inference approach. The methodology is based on [Chew et al. (2021)](https://publichealth.jmir.org/2021/3/e25807/) for the concept of comment-based age inference, extended with a stacked ensemble classifier.

---

### 3. Gender Classification

**🔲 TABLE 3: Gender Classifier Performance (ready to use)**

Validated against 4,894 self-declared gender users (3,564 male, 1,330 female):

| Threshold | N | Coverage | Overall Accuracy | Female Recall | Male Recall |
|-----------|---|----------|-----------------|---------------|-------------|
| ≥ 0.50 | 4,894 | 100.0% | 94.8% | 88.4% | 97.2% |
| ≥ 0.55 | 4,742 | 96.9% | 96.0% | 90.6% | 97.9% |
| **≥ 0.60** | **4,536** | **92.7%** | **96.9%** | **92.1%** | **98.5%** |
| ≥ 0.70 | 4,026 | 82.3% | 98.1% | 93.4% | 99.4% |
| ≥ 0.80 | 3,301 | 67.4% | 98.6% | 93.9% | 99.7% |
| ≥ 0.90 | 2,152 | 44.0% | 99.4% | 97.1% | 99.9% |

**🔲 `[sentence detailing accuracy results]`:**

*"At our primary confidence threshold (≥ 0.60), the gender classifier achieved 96.9% overall accuracy, with 92.1% recall for females and 98.5% recall for males, retaining 92.7% of users."*

---

### 4. Confidence Thresholds

**The current text says "96.5% of users" retained.** That's for age specifically. Gender retains 92.7%. You should say something like:

*"At the ≥ 0.60 threshold, 96.5% of users were retained for age classification (95.0% accuracy) and 92.7% for gender classification (96.9% accuracy)."*

---

### 5. AnthroScore V3

**🔲 `[Is it relevant to discuss how we began with MLM approach…]` (Commented [24]):**

**Yes — strongly recommended.** The evolution from MLM-based V2 to LLM-based V3 is actually one of your key methodological contributions. V2 had r = 0.11 with expert labels (essentially no correlation), while V3 has r = 0.59. This is a compelling validation story.

**🔲 `[table or figure for scale definitions]`:**

You can use this table directly from the actual prompt in `anthroscore_llm.py`:

| Score | Label | Description | Example Indicators |
|-------|-------|-------------|-------------------|
| 1 | None | AI treated as pure software/tool | "The app is buggy", "it", "the bot" |
| 2 | Minimal | Slight humanization, still clearly AI | "It's pretty smart", "the bot understood" |
| 3 | Moderate | Human pronouns, basic emotions | "She seemed confused", uses he/she |
| 4 | High | Genuine feelings/personality attributed | "He really cares", "she gets jealous" |
| 5 | Extreme | Full human-equivalent relationship | "We're in love", "they're my everything" |

**🔲 `[INSERT VALIDATION HERE WITH HUMANS, GPT-5]` & `[FIGURE 3]`:**

**Current validation status:**

GPT-5-mini as "expert" labeler vs. GPT-4.1-nano as production scorer:

| Metric | Value |
|--------|-------|
| Pearson r | **0.590** (p < 0.001) |
| Spearman ρ | 0.485 (p < 0.001) |
| Cohen's κ (quadratic) | 0.579 (moderate-to-substantial) |
| Exact accuracy | 64.0% |
| Within-1 accuracy | **96.0%** |
| Mean Absolute Error | 0.41 |
| Category accuracy (low/mid/high) | 89.0% |

Head-to-head vs MLM-based V2:

| Metric | V3 (LLM) | V2 (MLM) |
|--------|----------|----------|
| Correlation with expert | r = **0.590**\*\*\* | r = 0.107 (n.s.) |
| Head-to-head wins | **83%** | 16% |

**⚠️ Human validation is INCOMPLETE.** The `human_validation_sample.csv` has 100 stratified comments ready for annotation but the `human_score`, `human_reasoning`, and `confidence` columns are **all empty**. You still need to have humans annotate these and report the results.

**🔲 `[Commented [27-30]` — prompts, validation, etc.:]**

- The full classification prompt IS in the appendix-ready file `anthroscore_llm.py` (lines 28-57)
- You note "still need to actually validate" — correct, human validation hasn't been done yet

---

### 6. Comment-Level Score Distribution

For context, here's the actual distribution of AnthroScore V3 at comment level (useful for describing in methods/results):

| Score | Count | Percentage |
|-------|-------|------------|
| 0 (auto-filtered) | 9,855 | 3.5% |
| 1 (None) | 38,638 | 13.6% |
| 2 (Minimal) | 207,450 | 73.1% |
| 3 (Moderate) | 16,523 | 5.8% |
| 4 (High) | 10,866 | 3.8% |
| 5 (Extreme) | 563 | 0.2% |

**Key stat:** 75.7% of scored comments (excluding auto-filtered) received a score of 2, indicating heavy floor clustering. This is important for the limitations section.

---

### 7. Emotion Analysis

**🔲 `[Commented [31]: "did we allow multiple/overlap?"]`:**

Yes — the emotion classifier (`j-hartmann/emotion-english-distilroberta-base`) outputs a **probability distribution** across all 7 emotions for each comment. There's no forced single-label. You compute the **mean probability** per emotion per user for aggregation. So emotions can overlap — each comment gets probabilities for all 7 categories summing to 1.

---

## 📊 OTHER SECTIONS WITH GAPS

### Results Section

**🔲 `[Commented [33]: "stats tests"]`:**

All stats tests have been run. The key numbers from `COMPREHENSIVE_V3_ANALYSIS_RESULTS.md`:

**RQ1 Demographics:**

- 80.4% teens, 19.6% adults (at ≥0.60 confidence)
- 81.4% male, 18.6% female
- Age × Gender: χ² = 152.04, Cramér's V = 0.096 (negligible)

**RQ2 Demographics → Anthropomorphization:**

- Age: Welch's t = -19.81, Cohen's d = -0.501 (medium), p < 0.0001 → **adults higher**
- Gender: Welch's t = -12.12, Cohen's d = -0.292 (small), p < 0.0001 → **females higher**
- Two-way ANOVA: Age η² = 0.035, Gender η² = 0.009, Interaction η² = 0.0004
- Model R² = 0.049 (demographics explain ~5% of variance)

**RQ3 Emotions:**

- Joy: r = +0.115 (strongest positive)
- Neutral: r = -0.116 (strongest negative)
- Joy–Anthro relationship stronger for adults (r = 0.14 vs 0.10)
- Anger–Anthro relationship stronger for teens (r = 0.05 vs 0.02)

---

### AI Safety / Discussion Section

**🔲 Degradation and Disorientation sections say `[insert]`:**

These need to be written. You have the data to support them:

- **Degradation:** Adults show 2x more romantic content, 1.64x more loneliness language, 65% longer comments — evidence of substituting human relationships
- **Disorientation:** Joy is the strongest predictor of anthropomorphization (B = +0.461), and neutral tone is inversely related (B = -0.202) — emotional engagement narrows the frame to positive reinforcement

---

### Limitations Section

**🔲 Several limitation sections are blank stubs:**

- "Reddit-Only sample" — partially written, sentence cuts off mid-word ("our sample is o")
- "Cross-sectional design" — blank
- "Binary demographics" — blank
- "LLM-based validation" — blank

---

### Missing Sections (blank stubs)

- **Ethical Considerations** (Commented [38]) — mentions anonymization, paraphrasing, IRB
- **Future Research Directions** — blank
- **Appendix** — blank (but you have material for it)
- **Acknowledgements** (Commented [39]: "afia") — blank

---

## 📋 QUICK-REFERENCE DATA SUMMARY

| Metric | Value |
|--------|-------|
| Total comments collected | 283,895 |
| Unique users | 47,062 |
| Date range | Jan 1, 2024 – Dec 26, 2025 |
| Self-declared age users | 459 (254 teen, 205 adult) |
| Self-declared gender users | 4,894 (3,564 male, 1,330 female) |
| Analysis sample (conf ≥ 0.60) | 15,281 users |
| AnthroScore V3 expert correlation | r = 0.590 |
| Age classifier accuracy (≥0.60) | 95.0% |
| Gender classifier accuracy (≥0.60) | 96.9% |
| LLM classification model | GPT-4.1-nano (AnthroScore), GPT-4o-mini (5K users only for demographics) |
| ML classification model | V3 stacked ensemble (actual production model for demographics) |
| Emotion model | DistilRoBERTa (`j-hartmann/emotion-english-distilroberta-base`) |

---

## ✅ ACTION ITEMS (Priority Order)

1. **🔴 RESOLVE the classifier methodology description** — the paper says LLM but the actual analysis uses ML models. Either rewrite the methodology to describe the V3 ML ensemble, or re-run with actual LLM predictions.
2. **🔴 Complete human validation** of AnthroScore V3 — the 100-comment sample is ready in `human_validation_sample.csv` but unannotated.
3. **🟡 Insert Table 2** (age classifier performance) — numbers provided above.
4. **🟡 Insert Table 3** (gender classifier performance) — numbers provided above.
5. **🟡 Insert date range** — January 2024 to December 2025.
6. **🟡 Fix subreddit counts** — use raw counts (269,040 / 8,380 / 6,475) in data collection, filtered counts in analysis.
7. **🟡 Write the Degradation/Disorientation paragraphs** in the AI Safety section.
8. **🟡 Finish the Limitations subsections** (cross-sectional, binary demographics, LLM validation).
9. **🟡 Write Ethical Considerations** section.
10. **🟢 Add the AnthroScore V3 scale definition table**.
11. **🟢 Add the V3 vs V2 validation comparison figure/table**.
12. **🟢 Write Future Research Directions**.
13. **🟢 Consider Regression Model 3** — the emotion coefficients show extreme multicollinearity (all B ≈ -113,243, all p ≈ 0.21). The fixed version (dropping neutral as reference) in the Extended Analysis gives interpretable results and should be used instead.
