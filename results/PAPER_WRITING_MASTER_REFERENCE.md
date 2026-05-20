# Paper-Writing Master Reference: The Illusion Project

**Generated:** 2026-04-26 12:57
**Purpose:** One-stop, paper-facing reference file for Methods, Results, Discussion, Limitations, Appendix, and AI-assisted drafting.
**Canonical quantitative source:** this file recomputes from current parquet/JSON artifacts and should be cited internally over stale duplicate summaries.

> Important: this is a writing/reference artifact, not a journal-ready manuscript section. It includes caveats, stale-file warnings, and interpretation notes so an AI assistant or human writer does not mix incompatible analysis generations.

## 0. Executive Takeaways for the Paper

- Use **283,895 cleaned comments from 47,062 Reddit authors** as the cleaned analytic corpus. The **414,757** figure refers to pre-cleaning raw comments, not final analyzable comments.
- Improved AnthroScore V3 covers **274,191 comments** (96.6% of cleaned comments) and **45,725 users** with at least one scored comment.
- The inclusive high-confidence demographic sample is **16,347 users** at age and gender confidence >= 0.60. The canonical RQ2 report further conditions on `anthro_mean > 1`, yielding **5,160 users**.
- Inclusive estimand (all high-confidence users): adults score higher than teens (teen M=1.108, adult M=1.279, Welch t=-18.421, p=<0.001, d=-0.507); women score higher than men (male M=1.121, female M=1.225, d=-0.307).
- Canonical conditional estimand (`anthro_mean > 1`): adults score higher than teens (teen M=1.383, adult M=1.602, Welch t=-12.963, p=<0.001, d=-0.456); the gender effect is much smaller (male M=1.433, female M=1.466, d=-0.067, p=0.0402).
- Demographics alone explain about **4.1%** of variance in the emotion-complete sample; adding subreddit fixed effects and emotion proportions raises R^2 to **6.4%**.
- Emotion predictors are compositional proportions. Do not cite the saturated seven-emotion coefficients from the old Model 3. Use the drop-neutral or ALR models below.
- For draft revisions: fill Methods, Results, Analysis, Ethics, Limitations, and Appendix from this file; retire or qualify root-level duplicate/stale summaries.

## 1. Current Draft Needs This File Solves

| Draft gap or risk | What to use from this file |
| --- | --- |
| Methods is blank | Sections 2-5 give corpus, scoring, demographics, emotion, filtering, and aggregation details. |
| Results is blank | Sections 6-12 provide paper-ready results tables and statistical tests. |
| Analysis section is blank | Sections 10-13 provide corrected regressions, moderation, n-gram checks, and robustness. |
| Draft says over 414,757 comments | Use 414,757 raw pre-cleaning and 283,895 cleaned final comments; see Section 2. |
| Draft says two RQs but lists three | Use three RQs: demographics, emotions, moderation; if including gender moderation, say exploratory. |
| AnthroIndex vs AnthroScore naming | Pick one term. Code/results call it AnthroScore V3; draft calls it AnthroIndex. Define AnthroIndex as paper-facing name if desired. |
| Demographic classifier method mismatch | Current analysis reads V4 parquet files; best documented validation recommends V3. Do not call current full analysis GPT-4o-mini demographics without rerunning. |
| Emotion regression multicollinearity | Use Section 10 corrected drop-neutral/ALR models. |
| Need AI context for paper writing | Sections 15-18 give architecture, caveats, interpretation language, and exact file provenance. |

## 2. Corpus, Cleaning, and Sample Flow


### 2.1 Raw-to-cleaned corpus counts

| Subreddit | Raw comments | Cleaned comments | Retained | Unique authors |
| --- | --- | --- | --- | --- |
| r/CharacterAI | 397,230 | 269,040 | 67.7% | 43,166 |
| r/replika | 10,000 | 8,380 | 83.8% | 1,278 |
| r/AICompanions | 7,527 | 6,475 | 86.0% | 2,731 |
| Total | 414,757 | 283,895 | 68.4% | 47,062 |
- Cleaned date range from `Data/processed/all_comments.parquet`: **2024-01-01 to 2025-12-26**.
- Cleaning described in project docs: bot/deleted/removed removal, minimum text length filter, and comment ID deduplication. The exact current processed corpus is `Data/processed/all_comments.parquet`.
- Use raw counts only for data collection scope. Use cleaned counts for all analysis Ns.

### 2.2 Analysis sample flow

| Stage | N | Notes |
| --- | --- | --- |
| Cleaned comments | 283,895 | All retained comments after preprocessing |
| Cleaned unique authors | 47,062 | Authors in cleaned corpus |
| Original V3 scored comments | 283,895 | Legacy/original LLM score file |
| Improved V3 scored comments | 274,191 | 96.6% of cleaned comments |
| Improved V3 scored users | 45,725 | Users with at least one improved score |
| User emotion rows | 44,421 | User-level emotion proportions |
| Comment emotion rows | 277,420 | Comment-level emotion probabilities |
| Gender prediction rows | 47,062 | V4 parquet used by current analysis |
| Age prediction rows | 47,062 | V4 parquet used by current analysis |
| High-confidence demographic users | 16,347 | Age and gender confidence >= 0.60 with AnthroScore |
| Canonical RQ2 conditional users | 5,160 | High-confidence users with anthro_mean > 1 |
| High-confidence + emotions | 15,481 | Regression/correlation sample when all emotion proportions present |
| Partial LLM demographic classifications | 5,000 | Exists but does not cover full 47,062 users |

### 2.3 Comment volume per author

| Quantity | Value |
| --- | --- |
| Mean cleaned comments/user | 6.03 |
| Median cleaned comments/user | 2.00 |
| 10th percentile | 1.00 |
| 25th percentile | 1.00 |
| 75th percentile | 5.00 |
| 90th percentile | 12.00 |
| Max comments by one author | 1,671 |
Interpretation: user-level aggregation prevents extremely active authors from dominating primary demographic tests.

## 3. Measurement Architecture and Methodology Context


### 3.1 Anthropomorphization measure

| Score | Label | Paper description |
| --- | --- | --- |
| 1 | None | AI treated as software/tool/app; no mental states or relationship language. |
| 2 | Minimal | Light humanizing language, generic intelligence, or bot reference without strong mind attribution. |
| 3 | Moderate | Pronouns, simple agency, or basic emotional/personality attribution. |
| 4 | High | Clear feelings, care, jealousy, autonomy, attachment, or personality attributed to the AI. |
| 5 | Extreme | Human-equivalent relationship, love, dependency, or fully reciprocal social bond. |
- Current paper-facing score file: `experiments/anthroscore_v3/anthroscore_v3_improved_final.parquet`.
- The improved prompt incorporated human calibration, emotion-attribution distinctions, and overscoring correction. It produced a large 2-to-1 shift relative to the original V3 prompt.
- Validation JSON: Pearson r=0.590, Spearman rho=0.485, exact accuracy=64.0%, within-1 accuracy=96.0%, MAE=0.410, Cohen kappa=0.579.
- V3 vs V2 comparison JSON: LLM/expert r=0.590; MLM/expert r=0.107; LLM wins=83 of 100 validation cases; MLM wins=16; ties=1.
- Caveat: `validation_results.json` contains `recommendation: NEEDS REVIEW` and `passes_kappa: false`. In prose, describe the measure as substantially improved and human-calibrated, not as perfect ground truth.

### 3.2 Demographic inference

- The current comprehensive analysis reads `experiments/v2_correction/gender_predictions_v4.parquet` and `age_predictions_v4.parquet`.
- A prior audit and `FINAL_MODEL_SUMMARY.md` indicate the best documented validation narrative is for **V3 demographic models**, while the current analysis files are **V4 parquets**. Do not write that all full-sample demographics were GPT-4o-mini LLM classifications unless those are rerun over all users.
- V3 summary JSON available: gender test accuracy=82.1%; age CV accuracy=61.3%.
- V4 summary JSON flags suspiciously high threshold validation in some places; use cautiously. V4 gender CV accuracy=81.7%; V4 age CV accuracy=61.4%.
- Paper-safe wording: `We inferred binary age and gender categories using project demographic classifiers and restricted inferential analyses to users with age and gender confidence >= .60; classifier validation and limitations are reported separately.`

### 3.3 Emotion measure

- Emotion features come from `j-hartmann/emotion-english-distilroberta-base` according to project methodology, represented as user-level proportions across joy, sadness, anger, fear, surprise, disgust, and neutral.
- These proportions are compositional: when one category rises, others must fall. This matters for regression interpretation.

## 4. Confidence Thresholds and Demographic Coverage

| Threshold | Gender N | Gender coverage | Age N | Age coverage | Both N | Both coverage | Both + Anthro N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| >=0.50 | 45,725 | 100.0% | 45,725 | 100.0% | 45,725 | 100.0% | 45,725 |
| >=0.55 | 43,195 | 94.5% | 31,726 | 69.4% | 29,879 | 65.3% | 29,879 |
| >=0.60 | 40,381 | 88.3% | 18,711 | 40.9% | 16,347 | 35.8% | 16,347 |
| >=0.65 | 37,252 | 81.5% | 7,753 | 17.0% | 6,214 | 13.6% | 6,214 |
| >=0.70 | 33,556 | 73.4% | 1,165 | 2.5% | 835 | 1.8% | 835 |
| >=0.80 | 23,129 | 50.6% | 0 | 0.0% | 0 | 0.0% | 0 |
| >=0.90 | 10,036 | 21.9% | 0 | 0.0% | 0 | 0.0% | 0 |
Primary analyses use >=0.60 for both age and gender. Sensitivity analyses should report that the adult > teen direction persists across thresholds from .50 to .70, with the .70 estimate underpowered.

## 5. Descriptive Statistics


### 5.1 Improved AnthroScore comment-level distribution

| Score | Comments | Percent | 95% CI |
| --- | --- | --- | --- |
| 1 | 242,159 | 88.32% | [88.20%, 88.44%] |
| 2 | 27,657 | 10.09% | [9.97%, 10.20%] |
| 3 | 409 | 0.15% | [0.14%, 0.16%] |
| 4 | 3,904 | 1.42% | [1.38%, 1.47%] |
| 5 | 62 | 0.02% | [0.02%, 0.03%] |
Interpretation: the improved score is highly right-skewed; most Reddit comments mention AI companions without strong anthropomorphization.

### 5.2 User-level AnthroScore distribution in high-confidence sample

| Quantity | Value |
| --- | --- |
| N users | 16,347 |
| Mean | 1.140 |
| SD | 0.344 |
| Median | 1.000 |
| 10th percentile | 1.000 |
| 25th percentile | 1.000 |
| 75th percentile | 1.114 |
| 90th percentile | 1.500 |
| Min | 1.000 |
| Max | 4.500 |
| Skew | 4.429 |

### 5.3 Demographic composition at high confidence

| Dimension | Group | N | Percent | 95% CI |
| --- | --- | --- | --- | --- |
| Age | teen | 13,302 | 81.4% | [80.8%, 82.0%] |
| Age | adult | 3,045 | 18.6% | [18.0%, 19.2%] |
| Gender | male | 13,399 | 82.0% | [81.4%, 82.5%] |
| Gender | female | 2,948 | 18.0% | [17.5%, 18.6%] |

### 5.4 Primary-subreddit context

| Primary subreddit | Users | Mean Anthro | Median Anthro | % teen | % female | % any score >=3 | % any score >=4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AICompanions | 885 | 1.204 | 1.000 | 59.0% | 11.5% | 7.6% | 6.9% |
| CharacterAI | 15,447 | 1.136 | 1.000 | 82.7% | 18.4% | 7.2% | 6.6% |
| replika | 15 | 1.193 | 1.000 | 26.7% | 26.7% | 13.3% | 13.3% |
This is a user-level primary-subreddit view. A user is assigned to the subreddit where they posted most often.

## 6. RQ1/RQ2: Demographics and Anthropomorphization

**Critical estimand note:** The canonical comprehensive report tests RQ2 after filtering to users with `anthro_mean > 1` (N=5,160), which asks about differences **among users with at least some measured anthropomorphization**. The inclusive analyses below use all high-confidence users (N=16,347), including the many users whose mean score is exactly 1. These are both useful, but they answer different questions. For the main paper, state which estimand you are reporting.

### 6.0 Canonical conditional RQ2 estimates (`anthro_mean > 1`)

| Contrast | Group A | N A | Mean A | Group B | N B | Mean B | d A-B | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Age | Teen | 3,749 | 1.383 | Adult | 1,411 | 1.602 | -0.456 | <0.001 |
| Gender | Male | 3,734 | 1.433 | Female | 1,426 | 1.466 | -0.067 | 0.0402 |
These match the April 2026 canonical report: age d about -0.456; gender d about -0.067. Use these if you want consistency with `results/COMPREHENSIVE_V3_ANALYSIS_RESULTS.md`.

### 6.1 Inclusive age effect (all high-confidence users)

| Group | N | Mean | SD | Median |
| --- | --- | --- | --- | --- |
| Teen | 13,302 | 1.108 | 0.290 | 1.000 |
| Adult | 3,045 | 1.279 | 0.493 | 1.000 |
Welch t=-18.421, p=<0.001; Mann-Whitney U=15791291.5, p=<0.001; Hedges-corrected d=-0.507 (medium); mean difference teen-adult=-0.171.
Paper wording: Adults had higher mean anthropomorphization than teens. In the inclusive high-confidence sample this effect is around the small/medium boundary; in the canonical conditional sample it is small.

### 6.2 Inclusive gender effect (all high-confidence users)

| Group | N | Mean | SD | Median |
| --- | --- | --- | --- | --- |
| Male | 13,399 | 1.121 | 0.318 | 1.000 |
| Female | 2,948 | 1.225 | 0.432 | 1.000 |
Welch t=-12.424, p=<0.001; Mann-Whitney U=15558994.5, p=<0.001; Hedges-corrected d=-0.307 (small); mean difference male-female=-0.105.
Paper wording: women were higher than men in the inclusive two-group test, but the canonical conditional analysis reduces this to a negligible effect. Avoid overstating gender.

### 6.3 Simple effects by age/gender subgroup

| Contrast | N A | Mean A | N B | Mean B | Mean diff A-B | d | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gender within teen | 11,088 | 1.092 | 2,214 | 1.187 | -0.095 | -0.329 | <0.001 |
| Gender within adult | 2,311 | 1.258 | 734 | 1.343 | -0.084 | -0.171 | <0.001 |
| Age within male | 11,088 | 1.092 | 2,311 | 1.258 | -0.166 | -0.534 | <0.001 |
| Age within female | 2,214 | 1.187 | 734 | 1.343 | -0.156 | -0.365 | <0.001 |

### 6.4 Age effect within primary subreddits

| Primary subreddit | Teen N | Teen mean | Adult N | Adult mean | d teen-adult | p |
| --- | --- | --- | --- | --- | --- | --- |
| AICompanions | 522 | 1.058 | 363 | 1.414 | -0.712 | <0.001 |
| CharacterAI | 12,776 | 1.110 | 2,671 | 1.260 | -0.464 | <0.001 |
Interpretation: if this table shows adult > teen in each subreddit, the age effect is not merely a byproduct of adults being concentrated in a more anthropomorphic subreddit.

## 7. Binary/Prevalence Framing

Binary outcome here means a user had **at least one comment scored >=3** (moderate or higher anthropomorphization). This is a different estimand from mean AnthroScore.
| Quantity | Value |
| --- | --- |
| Users with any score >=3 | 1,185 |
| Prevalence | 7.2% |
| Teen prevalence | 6.2% |
| Adult prevalence | 11.7% |
| Age chi2 p | <0.001 |
| Adult vs teen OR | 1.99 |
| Male prevalence | 6.1% |
| Female prevalence | 12.3% |
| Gender chi2 p | <0.001 |
| Female vs male OR | 2.16 |
| Logistic pseudo R2 | 0.0239 |
| Term | B | SE | OR | OR 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| Intercept | -2.233 | 0.061 | 0.107 | [0.095, 0.121] | <0.001 |
| is_teen | -0.632 | 0.067 | 0.531 | [0.466, 0.606] | <0.001 |
| is_female | 0.719 | 0.067 | 2.052 | [1.800, 2.341] | <0.001 |

## 8. RQ2/RQ3: Emotions and Anthropomorphization


### 8.1 Dominant emotion distribution

| Dominant emotion | Users | Percent |
| --- | --- | --- |
| neutral | 10,246 | 66.2% |
| surprise | 1,617 | 10.4% |
| anger | 976 | 6.3% |
| sadness | 813 | 5.3% |
| disgust | 764 | 4.9% |
| joy | 715 | 4.6% |
| fear | 350 | 2.3% |

### 8.2 Correlations with user-level mean AnthroScore

| Emotion | Pearson r | Pearson p | Spearman rho | Spearman p |
| --- | --- | --- | --- | --- |
| emotion_joy | 0.100 | <0.001 | 0.248 | <0.001 |
| emotion_sadness | 0.018 | 0.0239 | 0.129 | <0.001 |
| emotion_anger | 0.060 | <0.001 | 0.207 | <0.001 |
| emotion_fear | 0.070 | <0.001 | 0.248 | <0.001 |
| emotion_disgust | 0.027 | <0.001 | 0.186 | <0.001 |
| emotion_surprise | -0.020 | 0.0114 | 0.078 | <0.001 |
| emotion_neutral | -0.129 | <0.001 | -0.105 | <0.001 |
Interpretation: neutral language has the strongest negative correlation with anthropomorphization. Joy is the clearest positive bivariate emotion signal. Pearson and Spearman signs can differ because the seven emotion features are proportions and relationships are not purely monotone.

### 8.3 Age moderation of emotion-anthropomorphization correlations

| Emotion | Teen N | Teen r | Adult N | Adult r | z | p | FDR q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| emotion_joy | 12,785 | 0.077 | 2,696 | 0.136 | -2.83 | 0.0046 | 0.0108 |
| emotion_sadness | 12,785 | 0.004 | 2,696 | 0.074 | -3.33 | <0.001 | 0.0030 |
| emotion_anger | 12,785 | 0.072 | 2,696 | 0.045 | 1.31 | 0.1903 | 0.3330 |
| emotion_fear | 12,785 | 0.061 | 2,696 | 0.083 | -1.06 | 0.2876 | 0.3528 |
| emotion_disgust | 12,785 | 0.028 | 2,696 | 0.007 | 0.98 | 0.3270 | 0.3528 |
| emotion_surprise | 12,785 | -0.020 | 2,696 | 0.000 | -0.93 | 0.3528 | 0.3528 |
| emotion_neutral | 12,785 | -0.110 | 2,696 | -0.194 | 4.09 | <0.001 | <0.001 |
Use FDR q-values for claims across the seven emotion moderation tests. If p is significant but q is not, describe as exploratory.

### 8.4 Gender moderation of emotion-anthropomorphization correlations

| Emotion | Male N | Male r | Female N | Female r | z | p | FDR q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| emotion_joy | 12,634 | 0.091 | 2,847 | 0.093 | -0.13 | 0.8941 | 0.8941 |
| emotion_sadness | 12,634 | 0.019 | 2,847 | 0.013 | 0.30 | 0.7615 | 0.8941 |
| emotion_anger | 12,634 | 0.060 | 2,847 | 0.078 | -0.87 | 0.3819 | 0.6683 |
| emotion_fear | 12,634 | 0.071 | 2,847 | 0.065 | 0.30 | 0.7669 | 0.8941 |
| emotion_disgust | 12,634 | 0.018 | 2,847 | 0.064 | -2.22 | 0.0266 | 0.1859 |
| emotion_surprise | 12,634 | -0.016 | 2,847 | -0.049 | 1.57 | 0.1175 | 0.4113 |
| emotion_neutral | 12,634 | -0.120 | 2,847 | -0.145 | 1.24 | 0.2157 | 0.5032 |
Gender moderation was missing from the canonical comprehensive report but is useful because the draft RQ3 currently mentions age and gender moderation.

## 9. Corrected Regression Models


### 9.1 Hierarchical OLS models

| Model | N | R^2 | Adj. R^2 | AIC | Model p |
| --- | --- | --- | --- | --- | --- |
| Demographics | 15,481 | 0.0415 | 0.0414 | 8908.9 | <0.001 |
| Demographics + age x gender | 15,481 | 0.0415 | 0.0414 | 8909.8 | <0.001 |
| Demographics + subreddit FE | 15,481 | 0.0416 | 0.0414 | 8910.3 | <0.001 |
| Demographics + subreddit FE + emotions | 15,481 | 0.0637 | 0.0631 | 8562.2 | <0.001 |
Increment from demographics-only to demographics + subreddit FE + emotion proportions: Delta R^2=0.0222.

### 9.2 Drop-neutral emotion model

This model includes six non-neutral emotion proportions and omits neutral, so coefficients are interpretable relative to neutral expression. This fixes the rank deficiency in the old seven-emotion model.
| Term | B | SE | t | p | 95% CI |
| --- | --- | --- | --- | --- | --- |
| Intercept | 1.152 | 0.009 | 134.236 | <0.001 | [1.136, 1.169] |
| is_teen | -0.139 | 0.007 | -20.507 | <0.001 | [-0.153, -0.126] |
| is_female | 0.081 | 0.007 | 12.137 | <0.001 | [0.068, 0.094] |
| emotion_joy | 0.273 | 0.019 | 14.406 | <0.001 | [0.236, 0.311] |
| emotion_sadness | 0.118 | 0.019 | 6.343 | <0.001 | [0.082, 0.155] |
| emotion_anger | 0.203 | 0.019 | 10.776 | <0.001 | [0.166, 0.240] |
| emotion_fear | 0.268 | 0.026 | 10.268 | <0.001 | [0.217, 0.319] |
| emotion_disgust | 0.099 | 0.020 | 4.858 | <0.001 | [0.059, 0.140] |
| emotion_surprise | 0.064 | 0.016 | 4.084 | <0.001 | [0.033, 0.095] |
| Term | VIF |
| --- | --- |
| is_teen | 1.01 |
| is_female | 1.02 |
| emotion_joy | 1.13 |
| emotion_sadness | 1.08 |
| emotion_anger | 1.11 |
| emotion_fear | 1.03 |
| emotion_disgust | 1.13 |
| emotion_surprise | 1.17 |

### 9.3 Additive log-ratio (ALR) emotion model

This compositional sensitivity model uses log(emotion / neutral) for each non-neutral emotion with a tiny pseudocount. It asks whether relative emotion composition predicts AnthroScore.
ALR model: N=15,481, R^2=0.0656, adj. R^2=0.0652, p=<0.001.
| Term | B | SE | t | p | 95% CI |
| --- | --- | --- | --- | --- | --- |
| is_teen | -0.137 | 0.007 | -20.153 | <0.001 | [-0.151, -0.124] |
| is_female | 0.074 | 0.007 | 11.090 | <0.001 | [0.061, 0.088] |
| alr_emotion_joy | 0.020 | 0.002 | 12.822 | <0.001 | [0.017, 0.023] |
| alr_emotion_sadness | -0.007 | 0.002 | -3.826 | <0.001 | [-0.011, -0.004] |
| alr_emotion_anger | 0.009 | 0.002 | 3.796 | <0.001 | [0.005, 0.014] |
| alr_emotion_fear | 0.016 | 0.002 | 7.940 | <0.001 | [0.012, 0.020] |
| alr_emotion_disgust | -0.001 | 0.002 | -0.601 | 0.5480 | [-0.006, 0.003] |
| alr_emotion_surprise | -0.008 | 0.002 | -4.245 | <0.001 | [-0.012, -0.004] |
| Term | VIF |
| --- | --- |
| is_teen | 1.02 |
| is_female | 1.03 |
| alr_emotion_joy | 1.29 |
| alr_emotion_sadness | 1.77 |
| alr_emotion_anger | 3.13 |
| alr_emotion_fear | 2.08 |
| alr_emotion_disgust | 2.76 |
| alr_emotion_surprise | 1.40 |

## 10. N-Gram and Improved-Prompt Diagnostics

The project added n-gram anthropomorphization/de-anthropomorphization signals. These should be described as prompt/context features and diagnostic covariates, not as the primary outcome.
| Feature | Comments with hit | % comments | Mean score if hit | Mean score if no hit | Difference |
| --- | --- | --- | --- | --- | --- |
| ngram_anthro_total | 10,118 | 3.6% | 1.610 | 1.132 | 0.478 |
| ngram_deanthro_total | 28,934 | 10.2% | 1.203 | 1.141 | 0.061 |
| ngram_relationship | 1,495 | 0.5% | 2.064 | 1.143 | 0.921 |
| ngram_emotion_attr | 252 | 0.1% | 1.626 | 1.147 | 0.479 |
| ngram_agency | 4,696 | 1.7% | 1.344 | 1.144 | 0.200 |
| ngram_consciousness | 851 | 0.3% | 1.620 | 1.146 | 0.474 |
| ngram_pronoun_verb | 3,587 | 1.3% | 1.892 | 1.139 | 0.753 |
| ngram_technical | 20,328 | 7.2% | 1.216 | 1.142 | 0.074 |
| ngram_tool_framing | 10,232 | 3.6% | 1.178 | 1.146 | 0.031 |
- Anthro n-gram hits should generally correspond to higher improved AnthroScore; de-anthro/technical hits should generally correspond to lower or only weakly higher scores.
- This table is useful for explaining why n-gram additions could shift older results: the improved system now distinguishes actual bot-attributed anthropomorphism from user self-expression or technical talk.

## 11. Content and Linguistic Context for Discussion

Comment-level pattern checks below use simple regex indicators. Treat them as descriptive, not as validated psychological constructs.
| Pattern | All comments | Score=1 comments | Score>=4 comments | High/low ratio | chi2 p |
| --- | --- | --- | --- | --- | --- |
| loneliness | 0.5% | 0.4% | 1.8% | 4.30 | <0.001 |
| romantic | 3.1% | 1.8% | 24.3% | 13.67 | <0.001 |
| friendship | 0.8% | 0.6% | 3.5% | 5.83 | <0.001 |
| roleplay | 8.9% | 7.7% | 11.6% | 1.50 | <0.001 |
| technical | 8.3% | 8.8% | 2.2% | 0.25 | <0.001 |
| emotional_support | 1.1% | 0.9% | 3.1% | 3.47 | <0.001 |
| creative | 7.8% | 7.0% | 8.9% | 1.28 | 0.0062 |
| Quantity | Score=1 comments | Score>=4 comments |
| --- | --- | --- |
| Mean word count | 23.01 | 36.92 |
| Median word count | 15.00 | 22.00 |
| Mean char count | 120.81 | 194.21 |
| Median char count | 73.00 | 118.00 |

## 12. Robustness and Sensitivity Results to Cite


### 12.1 Confidence threshold sensitivity

| Threshold | N | Teen mean | Adult mean | d | p | Direction |
| --- | --- | --- | --- | --- | --- | --- |
| >=0.50 | 13,897 | 1.453 | 1.583 | -0.248 | <0.001 | adults higher |
| >=0.55 | 9,206 | 1.419 | 1.600 | -0.354 | <0.001 | adults higher |
| >=0.60 | 5,160 | 1.383 | 1.602 | -0.456 | <0.001 | adults higher |
| >=0.65 | 2,096 | 1.315 | 1.539 | -0.551 | <0.001 | adults higher |
| >=0.70 | 449 | 1.239 | 1.339 | -0.314 | 0.0627 | adults higher |

### 12.2 Variance and robust alternatives

| Check | Result |
| --- | --- |
| Levene variance test | stat=109.278, p=<0.001; adult/teen variance ratio=1.711 |
| Brunner-Munzel robust test | stat=17.426, p=<0.001 |
| Robust regression is_teen coefficient | B=-0.179, p=<0.001 |

### 12.3 Confirmatory replication

| Dataset | Comments | Mean score | % score 1 | % score >=4 |
| --- | --- | --- | --- | --- |
| Main improved | 274,191 | 1.147 | 88.3% | 1.4% |
| Confirmatory | 45,704 | 1.122 | 90.7% | 1.4% |
Main vs confirmatory Cohen's d=0.056 (negligible). The confirmatory score distribution is extremely similar, supporting temporal stability of the improved scorer.

## 13. Human Calibration and Method Comparison

| Validation quantity | Value |
| --- | --- |
| Human calibration items | 151 total items; 3 annotators |
| Mean pairwise exact agreement | 43.8% in calibration report |
| Mean pairwise Cohen kappa | 0.245 in calibration report |
| Old algo vs human consensus exact | 37.6% |
| Old algo bias | +0.81 score points (overscoring) |
| Improvement summary n | 93 validation cases in final comparison table |
| Human-new algorithm exact | 47.7% |
| Human-new algorithm within +/-1 | 77.1% |
| Consensus-new weighted kappa | 0.466 |
| Method comparison paired comments | 264,654 |
| Original V3 mean | 1.997 |
| Improved V3 mean | 1.148 |
| Dominant migration | 2->1 shift for 184,663 comments (69.8% of paired comments) |
Interpretation: the improved AnthroScore is deliberately more conservative. Older reports with means near 2.0 are not directly comparable to the current improved-score analyses.

## 14. Self-Declaration and Validation Coverage

| Self-declaration field | N | Coverage of 47,062 users |
| --- | --- | --- |
| Age self-declared non-null | 459 | 1.0% |
| Age bucket self-declared non-null | 459 | 1.0% |
| Gender self-declared non-null | 4,937 | 10.5% |
Self-declared labels are sparse and should be framed as validation/calibration anchors, not the full demographic source.

## 15. Paper-Ready Claims and Suggested Wording


### 15.1 Abstract/results wording

- `We analyzed 283,895 cleaned public Reddit comments from 47,062 authors across r/CharacterAI, r/Replika, and r/AICompanions.`
- `Anthropomorphization was measured with a human-calibrated LLM-based 1-5 comment-level measure and aggregated to user-level means for demographic analyses.`
- `At the primary confidence threshold (age and gender >= .60), adults showed higher user-level anthropomorphization than teens. In the canonical conditional analysis among users with anthro_mean > 1, the teen-adult effect was d about -0.46; in the inclusive high-confidence sample it was d about -0.51.`
- `Gender effects were sensitive to the estimand: women were higher than men in inclusive analyses, but the canonical conditional analysis yielded only a negligible effect (d about -0.07).`
- `Neutral emotion was negatively associated with anthropomorphization, while joy and other non-neutral emotion proportions were positively associated relative to neutral expression in corrected emotion models.`
- `Demographic variables explain a small but non-trivial share of variance; most variance remains unexplained by age/gender, supporting future work on psychological, relational, and platform-context predictors.`

### 15.2 Claims to avoid

- Avoid: `over 414,757 comments were analyzed` unless you immediately clarify that this was the pre-cleaning raw total and 283,895 were retained.
- Avoid: `demographics were inferred by GPT-4o-mini` for the full analysis unless the full LLM classifier is rerun. The current analysis reads V4 ML prediction parquets.
- Avoid: `women strongly anthropomorphize more than men`; the estimated gender effect is negligible and model-dependent.
- Avoid: citing old root-level `COMPREHENSIVE_V3_ANALYSIS_RESULTS.md` or `FINAL_SHORT_SUMMARY.md` numbers without reconciliation.
- Avoid: interpreting seven-emotion saturated OLS coefficients from the old Model 3.

## 16. Limitations and Ethics Numbers

- Reddit-only, public-comment sample; not representative of all AI companion users.
- Comment authors are accounts, not verified individuals; multiple accounts and deleted context are possible.
- Demographics are inferred, binary, and confidence-filtered; they are not self-report labels for most users.
- Age categories collapse all adults into 19+, obscuring young adult, middle adult, and older adult differences.
- AnthroScore measures expressed linguistic anthropomorphism, not private beliefs, clinical attachment, or downstream harm.
- Emotion scores measure textual emotion probabilities, not experienced affect.
- Observational cross-sectional design: do not infer that anthropomorphism causes emotion, isolation, dependency, or harm.
- Ethical methods section should say the study used public Reddit data, minimized disclosure of identifiable examples, and analyzed aggregate patterns. If quoting comments, paraphrase or mask to reduce searchability.

## 17. File Provenance and Source-of-Truth Map

| Use for | Path | Status |
| --- | --- | --- |
| Current master reference | `results/PAPER_WRITING_MASTER_REFERENCE.md` | Generated by this script |
| Main canonical stats | `results/COMPREHENSIVE_V3_ANALYSIS_RESULTS.md` | Current, regenerated Apr 26 2026 |
| Main canonical JSON | `results/COMPREHENSIVE_V3_ANALYSIS_RESULTS.json` | Machine-readable stats |
| Improved AnthroScore comments | `experiments/anthroscore_v3/anthroscore_v3_improved_final.parquet` | Primary outcome file |
| Cleaned comments | `Data/processed/all_comments.parquet` | Corpus source |
| Enriched n-grams | `Data/features/comments_enriched.parquet` | N-gram diagnostics |
| User emotions | `Data/features/user_emotions.parquet` | User-level emotion proportions |
| Comment emotions | `Data/features/comments_with_emotions.parquet` | Comment-level emotion probabilities |
| Current age predictions | `experiments/v2_correction/age_predictions_v4.parquet` | Used by current pipeline; method caveat |
| Current gender predictions | `experiments/v2_correction/gender_predictions_v4.parquet` | Used by current pipeline; method caveat |
| Validation metrics | `experiments/anthroscore_v3/validation_results.json` | Expert/human-adjacent validation |
| V3 vs V2 metrics | `experiments/anthroscore_v3/mlm_comparison_results.json` | Method comparison |
| Stale duplicate to avoid | `COMPREHENSIVE_V3_ANALYSIS_RESULTS.md` at repo root | Older Jan 2026 numbers |

## 18. Appendix-Ready Tables


### 18.1 Full high-confidence subgroup table

| Age | Gender | N | Mean | SD | Median | % any >=3 | % any >=4 | Mean scored comments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adult | female | 734 | 1.343 | 0.520 | 1.143 | 14.7% | 12.9% | 8.99 |
| adult | male | 2,311 | 1.258 | 0.482 | 1.000 | 10.7% | 9.6% | 8.68 |
| teen | female | 2,214 | 1.187 | 0.392 | 1.000 | 11.6% | 11.0% | 9.98 |
| teen | male | 11,088 | 1.092 | 0.262 | 1.000 | 5.2% | 4.8% | 7.00 |

### 18.2 Raw subreddit date ranges

| Subreddit | Cleaned comments | Unique authors | First date | Last date |
| --- | --- | --- | --- | --- |
| r/AICompanions | 6,475 | 2,731 | 2024-04-19 | 2025-12-26 |
| r/CharacterAI | 269,040 | 43,166 | 2025-04-01 | 2025-07-31 |
| r/replika | 8,380 | 1,278 | 2024-01-01 | 2024-01-23 |

## 19. Reproducibility

Generated by `scripts/generate_paper_master_reference.py`.
Recommended reproducibility sequence:
1. `python3 scripts/COMPREHENSIVE_V3_ANALYSIS.py`
2. `python3 scripts/generate_paper_master_reference.py`
3. Recheck `results/PAPER_WRITING_MASTER_REFERENCE.md` before writing draft claims.

If n-gram prompts or scoring files are changed again, rerun both scripts and treat all older prose as stale until reconciled.
