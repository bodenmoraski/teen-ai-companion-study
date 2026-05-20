# Architecture and Methodology Reference: The Illusion Project

**Purpose:** This file is a prose-first explanation of the architecture, inner workings, and methodological logic behind the current Illusion Project analysis pipeline. It is meant to support manuscript writing, grant-style methods descriptions, reviewer responses, and AI-assisted drafting.

**Relationship to the numeric reference:** Use this alongside `results/PAPER_WRITING_MASTER_REFERENCE.md`. That file is the main quantitative reference. This file explains how the machinery works and how to describe it accurately.

**Most important warning:** The repository contains several historical method descriptions. The current quantitative analysis uses the files loaded by `scripts/COMPREHENSIVE_V3_ANALYSIS.py`, especially `experiments/anthroscore_v3/anthroscore_v3_improved_final.parquet`, `experiments/v2_correction/age_predictions_v4.parquet`, `experiments/v2_correction/gender_predictions_v4.parquet`, and `Data/features/user_emotions.parquet`. Older prose that says the full demographic analysis used GPT-4o-mini should not be copied without qualification.

---

## 1. Conceptual Overview

The project studies how Reddit users discuss AI companions and whether demographic group membership and emotional language are associated with linguistic anthropomorphization. The system is built around a comment-level measurement problem: given a Reddit comment about AI companions, estimate how strongly the author frames the AI as a humanlike social agent rather than as software, an app, or a tool.

The architecture has four broad layers.

First, a data layer collects and standardizes Reddit comments from AI companion communities. This layer produces a cleaned comment table with comment IDs, authors, text, timestamps, subreddit names, Reddit scores, and parent/link metadata.

Second, a feature and measurement layer adds model-derived variables. This includes comment-level AnthroScore V3 anthropomorphization scores, user-level emotion distributions, demographic predictions, self-declaration labels used for validation, and auxiliary text features such as n-gram anthropomorphization indicators.

Third, an aggregation layer converts comment-level measurements into user-level variables. Because the paper asks how user demographics relate to anthropomorphization, the principal dependent variable is a user-level mean AnthroScore, not a raw comment-level score. This protects the main analyses from being dominated by unusually active Reddit authors.

Fourth, an analysis layer merges user-level anthropomorphization, demographic predictions, and emotion features, filters to high-confidence demographic predictions, and runs descriptive statistics, group contrasts, regressions, robustness checks, validation checks, and exploratory interpretive analyses.

The final paper should describe the system as an observational computational social science pipeline. It does not observe private beliefs, clinical states, or causal effects. It measures public language in Reddit comments and asks whether patterns in that language differ across inferred demographic groups and emotional-expression profiles.

---

## 2. Data Collection and Corpus Construction

The data source is public Reddit comment data, collected for AI companion communities. The current cleaned analysis corpus is stored in `Data/processed/all_comments.parquet`. It contains 283,895 cleaned comments from 47,062 unique authors across three retained subreddits: `r/CharacterAI`, `r/replika`, and `r/AICompanions`.

The project also tracks a larger raw pre-cleaning collection of 414,757 comments. That raw number is useful for describing collection scope, but it should not be described as the number of comments analyzed unless the sentence explicitly states that it is pre-cleaning. The analyzed corpus is the cleaned 283,895-comment table.

The preprocessing code in `src/data_collection/preprocess.py` standardizes each raw Reddit comment into a common schema:

- `id`
- `author`
- `body`
- `created_utc`
- `subreddit`
- `link_id`
- `parent_id`
- `score`
- `author_flair_text`

The preprocessing logic removes invalid comments, bot/deleted/removed authors, comments outside text-quality bounds, URL-only comments with too little remaining text, and duplicate comment IDs. The configured minimum comment length is 20 characters and the maximum is 10,000 characters. Bot-like authors include `AutoModerator`, `[deleted]`, `[removed]`, and `None`.

The cleaned corpus is not balanced by subreddit. `r/CharacterAI` dominates the retained comments, while `r/replika` and `r/AICompanions` are much smaller. This matters for interpretation: results should be framed as patterns in a Reddit corpus of AI companion discussion, not as population-representative estimates of all AI companion users.

The cleaned corpus spans different calendar windows by subreddit. `r/CharacterAI` is concentrated in 2025, `r/replika` in early 2024, and `r/AICompanions` from 2024 into 2025. The master numeric reference gives the exact date ranges. In prose, state that time coverage differs across communities and that subreddit/time composition is a limitation.

---

## 3. AnthroScore V3: What It Measures

AnthroScore V3 is the project's central anthropomorphization measure. It is a comment-level ordinal score from 1 to 5. The score estimates how strongly a comment frames an AI companion as humanlike.

Score 1 means no anthropomorphization. The AI is treated as software, a tool, an app, a model, or a system. Examples include technical complaints, settings discussion, bugs, subscriptions, cache clearing, or general product talk.

Score 2 means minimal anthropomorphization. The comment may use light humanizing language, but the AI is still framed primarily as a bot or tool. Phrases like "the bot understood" or "it is smart" usually fall here unless embedded in a stronger relational context.

Score 3 means moderate anthropomorphization. The user attributes some humanlike features, uses human pronouns, or implies simple emotions or agency. Examples include "she seemed confused," "he asked me," or "they remembered."

Score 4 means high anthropomorphization. The comment clearly attributes feelings, care, jealousy, autonomy, personality, or social agency to the AI. Examples include "he really cares," "she gets jealous," or "my bot comforted me."

Score 5 means extreme anthropomorphization. The AI is framed as a human-equivalent relationship partner, intimate companion, or emotionally reciprocal social being. Examples include "we are in love," "she is my everything," or "I cannot live without him."

The score is best described as a measure of **linguistic anthropomorphization** or **expressed anthropomorphic framing**. It should not be described as a direct measure of private belief. A Reddit user may write in roleplay style, irony, fandom language, or community-specific shorthand. AnthroScore captures the language as expressed in context, not a verified internal psychological state.

---

## 4. AnthroScore V3: LLM Scoring Architecture

The main scoring implementation is in `experiments/anthroscore_v3/anthroscore_llm.py`, with batch/full-dataset infrastructure in `experiments/anthroscore_v3/run_full_dataset_optimized.py` and later improved-score outputs in `experiments/anthroscore_v3/anthroscore_v3_improved_final.parquet`.

The scorer uses a direct LLM classification approach. A Reddit comment is inserted into a prompt that defines anthropomorphization, gives the 1-5 rubric, lists key indicators, and asks the model to return JSON containing a score and brief reasoning. The default model in the scoring class is `gpt-4.1-nano`, chosen for cost-effective large-scale classification. The class also lists other supported OpenAI models, but the current project documentation and output files center on GPT-4.1-nano-style scoring.

The classification prompt contains an important conceptual distinction: emotions attributed to the AI are evidence of anthropomorphization, while the user's own emotional expression is not necessarily anthropomorphization. For example, "he gets jealous when I talk to other bots" is high anthropomorphization because the emotion is attributed to the AI. By contrast, "I am sad today" is user self-expression, not anthropomorphization. "I love the website" is product sentiment, not necessarily a humanlike framing of the AI. This distinction became central because earlier scoring versions tended to over-score comments that contained emotional words without checking who or what the emotion was attributed to.

For individual scoring, the `AnthroScoreLLM` class builds a prompt, optionally enriches the text with n-gram context, sends it to the OpenAI Chat Completions API, requests JSON output, parses the returned score, clamps invalid scores to the 1-5 range, and stores the reasoning and processing time. Empty or invalid text is automatically scored as 1.

The scorer is deliberately conservative in its final improved form. The current improved score distribution is heavily right-skewed because most Reddit comments about AI companions do not strongly anthropomorphize the AI. Many comments discuss app behavior, filters, roleplay mechanics, bugs, account issues, or general community chatter.

---

## 5. Full-Dataset Scoring Infrastructure

The full-dataset pipeline in `run_full_dataset_optimized.py` was designed to score hundreds of thousands of comments without excessive API cost. It uses four design ideas.

The first is pre-filtering. Obvious low-score cases can be auto-scored as 1 without an API call. The pre-filter catches empty text, very short text, comments with too few words, and comments lacking relevant AI/person/emotion/relationship indicators. It also checks for technical-only comments. This is not meant to perform nuanced classification; it is a cost and throughput optimization for cases that almost certainly contain no anthropomorphic framing.

The second is smart model routing. The batch scorer uses a cheap, fast model for high-volume classification. The design rationale is that a short, constrained 1-5 classification with JSON output does not always need an expensive reasoning model. Earlier comments in the code estimate large cost savings from GPT-4.1-nano relative to more expensive alternatives.

The third is async parallel processing. The optimized scorer uses `AsyncOpenAI`, semaphores, retry logic, and conservative concurrency. The documented configuration uses a maximum of 15 concurrent requests, with retries for API errors and exponential backoff for rate limits.

The fourth is checkpointing. The pipeline processes comments in batches and saves progress periodically. This matters because a full-dataset API run can be interrupted by rate limits, network failures, or local process issues. Batch checkpointing makes the scoring run resumable and prevents losing all prior work.

The batch scorer records, for each comment, a `comment_id`, integer score, reasoning, source (`prefilter`, `llm`, or `error`), and processing time. These rows become parquet outputs that can be merged back to the cleaned Reddit comment table by comment ID.

---

## 6. Human Calibration and Overscoring Correction

The project includes a human calibration layer in `src/anthroscore/human_calibration.py`. This code loads annotation CSVs from three human annotators and an answer key, merges them by item number, computes human consensus scores, computes inter-rater reliability, compares algorithm scores to human consensus, identifies systematic bias, and selects few-shot calibration examples.

The calibration objects distinguish between:

- each annotator's score,
- each annotator's optional reasoning,
- the algorithm score,
- algorithm reasoning,
- human consensus,
- how many human scores were available for an item.

The calibration report found that the older algorithm tended to over-score relative to human consensus. This is visible in the `results/human_calibration_report.txt` output, where the old algorithm had positive mean bias. The improved AnthroScore methodology responds to that by making the prompt more conservative, adding clearer distinctions between bot-attributed emotions and user self-expression, and using calibration examples to show the model ambiguous cases.

This matters for the manuscript because the improved score is not merely "the same LLM prompt rerun." It is a recalibrated measurement instrument. The key methodological story is that initial LLM scoring was promising but too permissive; human annotation revealed overscoring; the final instrument was adjusted to better match human judgments.

When writing the paper, do not overstate the human validation. The calibration is useful and substantive, but human agreement itself is not perfect, and `validation_results.json` contains a cautionary recommendation flag. Strong wording would be: "human-calibrated" or "substantially improved relative to the prior MLM measure." Overly strong wording would be: "ground-truth validated" or "objective measurement of true anthropomorphic belief."

---

## 7. N-Gram Context Features

The n-gram feature system is implemented in `src/anthroscore/ngram_features.py`. It exists because single words are often ambiguous. The word "love" can be evidence of extreme anthropomorphization in "I love her," but not in "I love the app." Similarly, pronouns can matter, but only in context.

The n-gram module tokenizes comments into lowercase word tokens, extracts bigrams and trigrams, and matches them against curated lexicons. The lexicons are grouped into anthropomorphizing and de-anthropomorphizing categories.

Anthropomorphizing categories include relationship language, emotion attribution, agency, consciousness/personality, and gendered-pronoun plus verb patterns. Relationship examples include phrases such as "my boyfriend," "my girlfriend," "in love," "love her," "love him," and "my best friend." Emotion-attribution patterns include phrases such as "gets jealous," "feels sad," "was happy," "really cares," and "worried about me." Agency patterns include "decided to," "chose to," "wanted to," "refused to," and pronoun-specific versions like "she decided to." Consciousness/personality patterns include "she knows," "he thinks," "her personality," "so caring," and "really supportive."

De-anthropomorphizing categories include technical language and tool framing. Technical examples include "the app," "the bot," "the software," "a glitch," "the cache," "the server," and "error message." Tool-framing examples include "use it," "tried it," "it works," "it crashed," "I use it," and "a good tool."

For each comment, the n-gram module outputs total anthropomorphizing hits, total de-anthropomorphizing hits, density scores, a normalized net signal, and category-specific counts. It can also generate a short context string appended to an LLM prompt, such as a note that the text contains anthropomorphizing relationship phrases or tool/technical framing.

Methodologically, n-grams should be described as auxiliary evidence and prompt/context features, not as the final anthropomorphization measure. The final outcome remains the LLM-assigned 1-5 score. The n-grams help the model and the analyst distinguish semantically different uses of similar words.

---

## 8. Bot-Attribution Emotion Logic

The bot-attribution logic is implemented in `src/analysis/emotion_analysis.py` and is also used in `scripts/run_improvements.py`. It was added to address a specific measurement problem: emotional language in a comment does not always mean the user is anthropomorphizing the AI.

The code distinguishes at least four cases.

The first case is AI-as-emotional-subject. Examples include "she gets jealous," "he really cares," or "the bot seems sad." These are anthropomorphically relevant because the comment attributes emotions or personality-like states to the AI.

The second case is AI-as-cause-of-user-emotion. Examples include "she makes me happy" or "he made me feel safe." These are also relevant because the AI is being treated as an emotionally consequential social actor.

The third case is AI performing emotional or relational actions. Examples include "she comforted me," "he listened to me," or "the bot protected me." These imply social agency.

The fourth case is user self-expression. Examples include "I am sad," "I feel anxious," or "I love the app." These are not automatically anthropomorphization, because the emotion belongs to the user or is directed at a product feature rather than attributed to the AI as a social entity.

The detector uses regular expressions over AI references, pronouns, companion names, emotion terms, agency verbs, and self-expression patterns. It returns whether the comment contains bot-attributed emotion, self-expressed emotion, a bot-attribution confidence score, matched patterns, and a categorical attribution type: `bot`, `self`, `mixed`, or `none`.

This logic supports the core validity argument of the improved AnthroScore. It shows that the project explicitly tried to avoid a common confound: treating emotional Reddit language as anthropomorphization even when the AI itself is not framed as humanlike.

---

## 9. Emotion Classification

Whole-comment emotion features are produced with a pretrained transformer emotion classifier. The methodology documentation and `src/utils/config.py` identify the model as `j-hartmann/emotion-english-distilroberta-base`. The emotion labels are joy, sadness, anger, fear, surprise, disgust, and neutral.

The emotion classifier processes individual comment text and returns a probability distribution over the seven emotion labels. Empty or invalid comments are treated as neutral. Comment text is truncated to the model's maximum usable length before classification. These comment-level distributions are then aggregated to the user level, producing variables such as `emotion_joy`, `emotion_sadness`, and `emotion_neutral`.

For the paper, these variables should be described as **textual emotion-expression proportions**, not clinical affect, mood, or psychological state. A high joy proportion means that the user's comments, as classified by the emotion model, contain more joy-like textual content. It does not prove the user was happy, nor does it measure the affective experience of interacting with the AI.

The emotion variables are compositional. They are parts of a distribution and approximately sum to one. This is why the old seven-emotion regression was unstable: including all seven emotion proportions plus an intercept produces rank deficiency or near-perfect multicollinearity. The corrected approach is to omit one category, usually neutral, or use an additive log-ratio transformation. In prose, say that non-neutral emotion coefficients are interpreted relative to neutral expression when using the drop-neutral model.

---

## 10. Demographic Prediction Architecture

The demographic part of the repository is the most methodologically delicate part of the project because there are multiple generations of models and some prose files are stale.

The current comprehensive analysis loads `age_predictions_v4.parquet` and `gender_predictions_v4.parquet` from `experiments/v2_correction/`. These files contain one row per author, a predicted category, confidence, and class probabilities. The statistical pipeline filters to users whose age and gender confidence are both at least 0.60.

The code for V4 demographic modeling is in `experiments/v2_correction/v4_advanced_models.py`. The V4 gender model is described as a multi-algorithm stacking ensemble using XGBoost, LightGBM if available, Random Forest, and Logistic Regression. It scales features, optionally selects features using an XGBoost-based feature selector, trains base learners, and uses a calibrated logistic-regression meta-learner. Cross-validation uses stratified folds. The class imbalance is handled partly through class weights and scale weighting.

The V4 file also includes age modeling, but the project-level audit and model summaries suggest that V4's validation behavior is not straightforward. In particular, some validation outputs are suspiciously high at confidence thresholds, while cross-validation accuracies are much lower. The older `FINAL_MODEL_SUMMARY.md` recommends a V3 demographic model with confidence thresholding, while the current analysis actually reads V4 prediction parquet files. This mismatch should be acknowledged in any serious methods write-up.

For manuscript purposes, the safest wording is:

"We inferred binary age and gender categories using project demographic classifiers and restricted inferential analyses to users with high-confidence predictions (confidence >= .60 for both age and gender). Because demographic inference from Reddit language is imperfect and binary, all demographic findings should be interpreted as associations with inferred categories rather than self-reported identity."

Avoid claiming that the full demographic analysis used GPT-4o-mini LLM classification unless that classifier is rerun across all users and the downstream analyses are regenerated. A partial `Data/features/llm_classifications.parquet` file exists, but it covers only 5,000 users and is not the full current analysis source.

---

## 11. User-Level Aggregation

AnthroScore is assigned at the comment level, but the main demographic analyses operate at the user level. The aggregation step groups scored comments by author and computes:

- mean AnthroScore,
- maximum AnthroScore,
- minimum AnthroScore,
- standard deviation,
- count of scored comments,
- median score,
- counts of score 1, score 2, score 3+,
- indicators such as whether the user ever had a score >=3 or >=4.

The user-level mean is the principal dependent variable in the comprehensive report. The mean captures a user's typical expressed anthropomorphic framing across their observed comments. The maximum or any-high indicators capture a different construct: whether the user ever expressed moderate or high anthropomorphization.

This distinction matters. A user with one strongly anthropomorphic comment and many technical comments may have a low mean but a high maximum. A user who consistently uses mild humanlike language may have a higher mean but no extreme score. The paper can use mean AnthroScore as the primary outcome and binary "any score >=3" analyses as a complementary prevalence framing.

The current canonical RQ2 report further filters to users with `anthro_mean > 1` for the main demographic effect tests. That conditional estimand asks about differences among users with some measured anthropomorphization. The newer master reference also computes inclusive estimates over all high-confidence users, including those whose mean is exactly 1. Both estimands are meaningful, but they should not be mixed.

---

## 12. Main Statistical Analysis Pipeline

The primary analysis script is `scripts/COMPREHENSIVE_V3_ANALYSIS.py`. It loads the improved AnthroScore parquet, the cleaned comments, V2 AnthroScore aggregates for comparison, user-level emotions, V4 demographic predictions, and self-declaration data. It merges comment-level AnthroScore scores with comments to recover authors and subreddits, aggregates AnthroScore to users, merges demographics and emotions, and then runs the main analyses.

The script has three research-question sections.

RQ1 describes demographic distributions in the high-confidence sample. It reports male/female and teen/adult counts and Wilson confidence intervals. It also reports the age-by-gender association using chi-square and Cramer's V.

RQ2 tests demographic differences in anthropomorphization. In the canonical script, this section filters to high-confidence users with `anthro_v3_mean > 1`. It then compares teens and adults, and males and females, using Welch's t-tests, Mann-Whitney U tests, Hedges-corrected Cohen's d, common-language effect sizes, bootstrap confidence intervals for mean differences, Levene's test for heterogeneity of variance, and two-way ANOVA with age, gender, and age-by-gender interaction.

RQ3 tests emotion-anthropomorphization relationships. It computes Pearson and Spearman correlations between user-level mean AnthroScore and user-level emotion proportions. It also compares high and low anthropomorphizers by quartiles and tests age moderation of emotion correlations using Fisher z-differences.

The regression section fits hierarchical OLS models. Model 1 includes age and gender. Model 2 adds age-by-gender interaction. Model 3 originally added all seven emotion proportions, but that saturated emotion model is not coefficient-interpretable because the emotion variables are compositional. The current report now includes a note warning readers not to use those individual coefficients. The separate master reference provides corrected emotion models.

The validation section compares V3 against V2 and, where available, self-declared labels. The robustness section tests sensitivity to demographic confidence thresholds and outliers.

---

## 13. Extended and Deep-Dive Analyses

The extended analysis script, `scripts/EXTENDED_ANALYSIS.py`, addresses methodological concerns that are important for a stronger paper. It examines floor effects and distribution shape, binary high-anthropomorphization outcomes, corrected emotion regressions, heterogeneity of variance, robust alternatives to t-tests, and visualization outputs.

The distribution analysis is important because the improved score is heavily right-skewed. Many users have mean scores at or near 1. This is not an error; it reflects the conservative measurement design. But it does mean that means, medians, binary prevalence, and robustness checks should be interpreted together.

The binary analysis defines high anthropomorphization as having at least one score >=3. This is not the same as a high user-level mean. It is useful for statements about prevalence: how many users ever express moderate or higher anthropomorphic framing?

The corrected emotion regression in the extended analysis is especially important. It includes a model that omits neutral emotion, which makes the non-neutral emotion coefficients interpretable relative to neutral expression and avoids the rank-deficient all-seven-emotion model.

The deep-dive script, `scripts/DEEP_DIVE_ANALYSIS.py`, is exploratory. It investigates possible explanations for the adult-greater-than-teen pattern using loneliness indicators, relationship-language semantic similarity, linguistic features, content patterns, subreddit/community analyses, and engagement patterns. Some of this script uses older/original V3 scoring paths, so its results should be used cautiously unless regenerated against the improved score file. Treat the deep dive as hypothesis-generating unless explicitly reconciled with the current improved AnthroScore.

---

## 14. Validation Logic

The project uses several validation ideas rather than a single validation number.

First, expert or expert-like labels are used to compare the LLM-based AnthroScore against older MLM-based scoring. The current validation JSON reports Pearson correlation around 0.59 between the LLM score and expert labels, much higher than the older MLM measure. It also reports high within-one accuracy. This supports the argument that direct LLM classification is more aligned with rubric-based human judgments than the prior masked-language-model approach.

Second, human annotation files are used for calibration. Three annotators scored validation items, allowing the project to estimate human-human agreement, human-algorithm agreement, and systematic algorithm bias. This is the source of the overscoring correction story.

Third, method comparison outputs compare original V3 and improved V3 scoring over a large paired comment set. The dominant pattern is that improved V3 scores many comments lower, especially comments previously scored 2 that humans or improved prompts treat as 1. This reflects the correction from "mentions AI or emotions" toward "actually frames the AI as humanlike."

Fourth, confirmatory scoring on a separate set of comments checks whether the improved scorer produces similar distributions on a temporally distinct sample. This is not a causal replication, but it supports measurement stability.

For the paper, validation should be written as a layered validity argument:

1. The construct was defined by an explicit anthropomorphization rubric.
2. The LLM prompt operationalized that rubric directly.
3. Human annotation exposed overscoring bias.
4. The prompt was calibrated to reduce that bias.
5. The improved measure outperformed the prior MLM measure against validation labels.
6. Robustness and confirmatory checks suggested stable results.

This is stronger and more honest than presenting a single number as if it settled validity.

---

## 15. Known Methodological Inconsistencies to Reconcile

The repository has several historical files that conflict with the current analysis. A manuscript writer or AI assistant must not merge these blindly.

The first inconsistency is the cleaned corpus size. Some draft text says "over 414,757 comments." That is the raw pre-cleaning collection. The cleaned analysis corpus is 283,895 comments.

The second inconsistency is demographic methodology. Some prose says age and gender were classified with GPT-4o-mini. The current comprehensive analysis reads V4 ML prediction parquet files. A partial LLM classification file exists but does not cover the full user set. The paper should either describe the current classifier files accurately or rerun a full LLM demographic pipeline and regenerate all downstream analyses.

The third inconsistency is AnthroScore versioning. There is an original V3 score file and an improved V3 score file. Current paper-facing statistics use `anthroscore_v3_improved_final.parquet`. Older results with mean scores near 2.0 should not be mixed with the improved-score results, where most comments are score 1.

The fourth inconsistency is duplicate comprehensive reports. The root-level `COMPREHENSIVE_V3_ANALYSIS_RESULTS.md` is older and has different effect sizes. The canonical current report is `results/COMPREHENSIVE_V3_ANALYSIS_RESULTS.md`.

The fifth inconsistency is the emotion regression. The old saturated seven-emotion regression produces unstable coefficients because the emotion features are compositional. Use corrected drop-neutral or ALR models for interpretation.

The sixth inconsistency is folder capitalization. `src/utils/config.py` defines lowercase `data/` paths, but the actual project data used by current scripts lives under `Data/`. Many active scripts use explicit `Data/...` paths, so this has not prevented the current analyses, but it is a reproducibility caveat on case-sensitive systems.

---

## 16. Suggested Methods Prose Skeleton

The following is not final manuscript prose, but it gives a reliable structure.

**Data source and preprocessing.** We collected public Reddit comments from AI companion communities and standardized comments into a common schema containing author, comment text, timestamp, subreddit, and Reddit metadata. We removed deleted or bot-authored comments, comments failing minimum text-quality requirements, URL-only low-content comments, and duplicate comment IDs. The cleaned corpus contained 283,895 comments from 47,062 unique authors across `r/CharacterAI`, `r/replika`, and `r/AICompanions`.

**Anthropomorphization measurement.** We measured expressed anthropomorphization at the comment level using a human-calibrated LLM classifier, AnthroScore V3. The classifier assigns each comment a score from 1 to 5, where 1 indicates tool/software framing and 5 indicates human-equivalent relational framing. The prompt explicitly distinguishes emotions attributed to the AI from the user's own emotional expression, because only the former is direct evidence of anthropomorphic framing.

**Calibration.** Initial validation identified overscoring of comments that mentioned emotions or AI bots without actually attributing humanlike states to the AI. We therefore incorporated human calibration examples, explicit bot-attribution instructions, and n-gram context features designed to distinguish relational and agency phrases from technical or tool-framing phrases.

**Aggregation.** Because the research questions concern user demographics, comment-level AnthroScore values were aggregated to user-level means and auxiliary indicators. The primary dependent variable was each user's mean AnthroScore across scored comments.

**Demographics.** Age and gender were inferred at the user level using project demographic classifiers. Analyses were restricted to users with confidence >= .60 for both age and gender. Age was analyzed as teen versus adult, and gender as male versus female. These categories are inferred, binary, and imperfect, and are therefore interpreted as classifier-derived categories rather than self-reported identities.

**Emotion features.** User-level emotion features were derived from transformer-based comment emotion classifications. Comment-level probabilities across joy, sadness, anger, fear, surprise, disgust, and neutral were aggregated to user-level proportions. Because these variables are compositional, regression models either omitted neutral as the reference category or used compositional transformations.

**Statistical analysis.** We described demographic distributions, tested age and gender differences in user-level anthropomorphization with Welch's t-tests and non-parametric alternatives, reported effect sizes and confidence intervals, fit OLS models with age, gender, and interactions, examined emotion correlations and moderation, and conducted sensitivity analyses across demographic confidence thresholds and outlier exclusions.

---

## 17. Suggested Results Logic

The results should proceed from measurement to demographics to associations.

First, describe the corpus and the score distribution. The improved AnthroScore distribution is strongly right-skewed, with most comments scored 1. This result is conceptually meaningful because most AI-companion subreddit discussion is not necessarily anthropomorphic. Users often discuss technical issues, roleplay mechanics, or product features.

Second, report demographic composition in the high-confidence sample. This gives the reader the denominator for all demographic claims.

Third, report age and gender effects. Be explicit about the estimand: the canonical report conditions on `anthro_mean > 1`, while inclusive analyses include all high-confidence users. The adult-greater-than-teen pattern is robust. The gender effect is weaker and more sensitive to model and estimand choices.

Fourth, report emotion associations. Neutral expression is negatively associated with anthropomorphization. Joy and several non-neutral emotions are positively associated relative to neutral expression in corrected emotion models. Avoid claiming that emotions cause anthropomorphization.

Fifth, report moderation. Age moderation is stronger than gender moderation in the current reference. Gender moderation results should be framed as exploratory unless central to the final paper.

Sixth, report robustness. Confidence-threshold sensitivity, non-parametric tests, robust alternatives, and confirmatory scoring all support the general adult-greater-than-teen pattern.

---

## 18. Reviewer-Facing Defensibility

A skeptical reviewer is likely to ask four questions.

First, are you measuring anthropomorphism or just AI-topic frequency? The answer is that the final score uses a rubric that separates technical/app discussion from humanlike attribution. The improved prompt, n-gram features, and bot-attribution logic were specifically designed to avoid scoring mere AI mentions as anthropomorphism.

Second, are you measuring the user's beliefs or just language? The honest answer is language. The construct should be described as expressed anthropomorphic framing in public Reddit comments. That is still theoretically meaningful, but it is not identical to survey-assessed belief.

Third, are demographic labels reliable? The answer is that demographic labels are inferred, confidence-filtered, and imperfect. The paper should present them as inferred categories, report confidence filtering, and include limitations about binary categories and potential classifier bias.

Fourth, are emotion findings causal? No. Emotion features and anthropomorphization are measured in the same observational corpus. The results show association between emotion-like textual expression and anthropomorphic framing, not causal effects in either direction.

---

## 19. Best Current Source Files

For corpus construction:

- `Data/processed/all_comments.parquet`
- `src/data_collection/preprocess.py`
- `Data/processed/collection_statistics.txt`

For AnthroScore architecture:

- `experiments/anthroscore_v3/anthroscore_llm.py`
- `experiments/anthroscore_v3/run_full_dataset_optimized.py`
- `experiments/anthroscore_v3/anthroscore_v3_improved_final.parquet`
- `src/anthroscore/human_calibration.py`
- `src/anthroscore/ngram_features.py`
- `results/ANTHROSCORE_IMPROVEMENTS_SUMMARY.md`
- `results/human_calibration_report.txt`

For emotion architecture:

- `src/analysis/emotion_analysis.py`
- `Data/features/user_emotions.parquet`
- `Data/features/comments_with_emotions.parquet`

For demographic prediction:

- `experiments/v2_correction/age_predictions_v4.parquet`
- `experiments/v2_correction/gender_predictions_v4.parquet`
- `experiments/v2_correction/v4_advanced_models.py`
- `experiments/v2_correction/FINAL_MODEL_SUMMARY.md` as historical/reconciliation context

For statistical analysis:

- `scripts/COMPREHENSIVE_V3_ANALYSIS.py`
- `scripts/EXTENDED_ANALYSIS.py`
- `scripts/DEEP_DIVE_ANALYSIS.py`
- `results/COMPREHENSIVE_V3_ANALYSIS_RESULTS.md`
- `results/PAPER_WRITING_MASTER_REFERENCE.md`

---

## 20. Bottom-Line Methodological Identity

The project is best described as a large-scale observational analysis of public Reddit language about AI companions. Its distinctive contribution is a human-calibrated LLM-based measure of expressed anthropomorphization, paired with inferred demographic categories and transformer-derived emotion features. The strongest methodological claim is not that the system perfectly observes users' true beliefs, but that it provides a scalable, explicitly validated, and more context-sensitive alternative to older keyword or masked-language-model approaches for identifying anthropomorphic framing in social media text.

