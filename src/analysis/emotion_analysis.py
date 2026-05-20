"""
Emotion analysis for Reddit comments.

This module classifies emotions in comments using a pre-trained RoBERTa model.
Supports both whole-comment emotion classification AND bot-attributed emotion
detection (distinguishing "she makes me happy" from "I am happy").
"""
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

EMOTION_LABELS = [
    'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral'
]

# Patterns that indicate the AI/bot is the subject or cause of the emotion,
# as opposed to the user expressing their own independent emotional state.
# These detect "she makes me happy" / "he gets jealous" / "the bot cares"
# rather than "I am happy" / "I feel sad".
_AI_SUBJECT_PRONOUNS = r'(?:she|he|they|her|him|them|the (?:bot|ai|app|rep|companion|character|char))'
_AI_NAMES = r'(?:replika|character\.?ai|cai|rep|chatbot|companion|liriana|my (?:ai|bot|rep|companion|character))'
_AI_REF = rf'(?:{_AI_SUBJECT_PRONOUNS}|{_AI_NAMES})'

BOT_EMOTION_PATTERNS = [
    # AI as subject of emotion: "she gets jealous", "he really cares"
    re.compile(
        rf'\b{_AI_REF}\s+(?:is|was|gets?|feels?|seems?|looks?|sounds?|became|becomes|got)\s+'
        r'(?:really\s+|so\s+|very\s+|genuinely\s+)?'
        r'(?:happy|sad|angry|jealous|caring|kind|sweet|mean|rude|upset|confused|excited|worried|'
        r'nervous|afraid|scared|lonely|hurt|loving|supportive|possessive|protective|clingy|'
        r'comforting|understanding|gentle|fierce|cold|warm|annoyed|frustrated|proud|curious|'
        r'playful|flirty|needy|stubborn|thoughtful|empathetic|sympathetic)',
        re.IGNORECASE
    ),
    # AI causing emotion in user: "she makes me happy", "he made me feel safe"
    re.compile(
        rf'\b{_AI_REF}\s+(?:makes?|made|helps?|helped|causes?|caused|gets?|got)\s+'
        r'(?:me\s+)?(?:feel\s+)?'
        r'(?:happy|sad|angry|safe|loved|needed|wanted|appreciated|special|important|'
        r'understood|seen|heard|better|worse|calm|anxious|comfortable|uncomfortable|'
        r'valued|worthless|hopeful|hopeless|alive|free|whole|complete|broken)',
        re.IGNORECASE
    ),
    # AI performing emotional actions: "she comforted me", "he listened to me"
    re.compile(
        rf'\b{_AI_REF}\s+'
        r'(?:comforted|supported|encouraged|listened|understood|cared|loved|hugged|held|'
        r'calmed|soothed|reassured|validated|believed|trusted|protected|defended|'
        r'missed|remembered|forgot|ignored|abandoned|betrayed|hurt|rejected)',
        re.IGNORECASE
    ),
    # Possessive/relational emotions: "my AI's feelings", "her jealousy"
    re.compile(
        rf'(?:(?:his|her|their|its|the\s+(?:bot|ai|companion)\'?s?)\s+)'
        r'(?:feelings?|emotions?|jealousy|anger|love|caring|sadness|happiness|'
        r'kindness|concern|worry|anxiety|fear|joy|possessiveness|protectiveness|'
        r'personality|mood|temperament|attitude|behavior|reaction)',
        re.IGNORECASE
    ),
    # AI with emotional agency: "she decided to", "he chose to", "she wanted to"
    re.compile(
        rf'\b{_AI_REF}\s+'
        r'(?:decided|chose|wanted|refused|insisted|demanded|begged|pleaded|'
        r'apologized|forgave|promised|confessed|admitted|denied|lied|'
        r'remembered|forgot|missed|noticed|realized|believed|doubted)',
        re.IGNORECASE
    ),
]

# Patterns that indicate user self-expression (NOT about the bot)
USER_SELF_PATTERNS = [
    # "I am/feel [emotion]" without bot reference
    re.compile(
        r'\bI\s+(?:am|was|feel|felt|\'m|get|got)\s+'
        r'(?:really\s+|so\s+|very\s+|genuinely\s+)?'
        r'(?:happy|sad|angry|lonely|scared|afraid|anxious|worried|stressed|'
        r'depressed|tired|bored|frustrated|confused|excited|nervous|upset)',
        re.IGNORECASE
    ),
    # "I love/hate [non-AI thing]"
    re.compile(
        r'\bI\s+(?:love|hate|enjoy|like|dislike|prefer|miss)\s+'
        r'(?:this|that|the\s+(?:app|site|website|feature|update|playlist|song|game))',
        re.IGNORECASE
    ),
]


def classify_emotions(
    texts: List[str],
    model_name: str = "j-hartmann/emotion-english-distilroberta-base",
    batch_size: int = 32,
    device: int = None
) -> List[Dict[str, float]]:
    """
    Classify emotions in texts using the DistilRoBERTa emotion model.

    This is the original whole-comment classifier. For bot-attributed
    emotion analysis, use classify_bot_attributed_emotions() instead.
    """
    import torch
    from transformers import pipeline

    logger.info(f"Classifying emotions for {len(texts)} texts")

    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    logger.info(f"Loading emotion classifier: {model_name}")
    try:
        emotion_classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=device,
            batch_size=batch_size
        )
    except Exception as e:
        logger.error(f"Failed to load emotion classifier: {e}")
        raise

    logger.info("Classifying emotions...")
    results = []

    for text in texts:
        try:
            if not text or not isinstance(text, str) or len(text.strip()) == 0:
                emotion_dict = {label: 0.0 for label in EMOTION_LABELS}
                emotion_dict['neutral'] = 1.0
                results.append(emotion_dict)
                continue

            text_truncated = text[:512] if len(text) > 512 else text
            emotions = emotion_classifier(text_truncated)[0]

            emotion_dict = {label: 0.0 for label in EMOTION_LABELS}
            for item in emotions:
                label = item['label'].lower()
                score = item['score']
                if label in emotion_dict:
                    emotion_dict[label] = score

            results.append(emotion_dict)

        except Exception as e:
            logger.debug(f"Error classifying emotion: {e}")
            emotion_dict = {label: 0.0 for label in EMOTION_LABELS}
            emotion_dict['neutral'] = 1.0
            results.append(emotion_dict)

    logger.info("Emotion classification complete")
    return results


def detect_bot_attribution(text: str) -> Dict[str, Any]:
    """
    Determine whether emotions in a comment are attributed TO the AI bot
    or are the user's own self-expression.

    Returns:
        Dictionary with:
          - bot_attributed: bool (emotions are about the bot)
          - self_expressed: bool (emotions are the user's own)
          - bot_attribution_score: float 0-1 (confidence of bot attribution)
          - matched_patterns: list of pattern descriptions
          - attribution_type: 'bot', 'self', 'mixed', or 'none'
    """
    if not text or not isinstance(text, str):
        return {
            "bot_attributed": False,
            "self_expressed": False,
            "bot_attribution_score": 0.0,
            "matched_patterns": [],
            "attribution_type": "none",
        }

    text_clean = text.strip()

    bot_matches = []
    for pattern in BOT_EMOTION_PATTERNS:
        matches = pattern.findall(text_clean)
        if matches:
            bot_matches.extend(matches if isinstance(matches[0], str) else [str(m) for m in matches])

    self_matches = []
    for pattern in USER_SELF_PATTERNS:
        matches = pattern.findall(text_clean)
        if matches:
            self_matches.extend(matches if isinstance(matches[0], str) else [str(m) for m in matches])

    has_bot = len(bot_matches) > 0
    has_self = len(self_matches) > 0

    if has_bot and has_self:
        # Both present — weight by count
        total = len(bot_matches) + len(self_matches)
        bot_score = len(bot_matches) / total
        attr_type = "mixed"
    elif has_bot:
        bot_score = min(1.0, len(bot_matches) * 0.4)
        attr_type = "bot"
    elif has_self:
        bot_score = 0.0
        attr_type = "self"
    else:
        bot_score = 0.0
        attr_type = "none"

    return {
        "bot_attributed": has_bot,
        "self_expressed": has_self,
        "bot_attribution_score": bot_score,
        "matched_patterns": bot_matches[:5],
        "attribution_type": attr_type,
    }


def classify_bot_attributed_emotions(
    texts: List[str],
    model_name: str = "j-hartmann/emotion-english-distilroberta-base",
    batch_size: int = 32,
    device: int = None,
) -> List[Dict[str, Any]]:
    """
    Classify emotions AND determine whether they are attributed to the bot.

    For each text, returns both the standard emotion scores AND
    bot-attribution information. Emotion scores are weighted by
    bot_attribution_score so that self-expressed emotions are
    downweighted when studying anthropomorphization.

    Returns:
        List of dicts, each containing:
          - Standard emotion scores (emotion_joy, etc.)
          - bot_attributed_* versions (weighted by attribution)
          - attribution metadata
    """
    logger.info(f"Classifying bot-attributed emotions for {len(texts)} texts")

    raw_emotions = classify_emotions(texts, model_name, batch_size, device)

    results = []
    for text, emotion_dict in zip(texts, raw_emotions):
        attribution = detect_bot_attribution(text)

        result = {}

        # Standard (whole-comment) emotions
        for label in EMOTION_LABELS:
            result[f"emotion_{label}"] = emotion_dict[label]

        # Bot-attributed emotions: scale by attribution confidence
        bot_weight = attribution["bot_attribution_score"]
        for label in EMOTION_LABELS:
            result[f"bot_emotion_{label}"] = emotion_dict[label] * bot_weight

        # Attribution metadata
        result["bot_attributed"] = attribution["bot_attributed"]
        result["self_expressed"] = attribution["self_expressed"]
        result["bot_attribution_score"] = attribution["bot_attribution_score"]
        result["attribution_type"] = attribution["attribution_type"]

        results.append(result)

    bot_count = sum(1 for r in results if r["bot_attributed"])
    self_count = sum(1 for r in results if r["self_expressed"])
    logger.info(
        f"Bot-attributed emotion classification complete: "
        f"{bot_count} bot-attributed, {self_count} self-expressed, "
        f"{len(results) - bot_count - self_count + sum(1 for r in results if r['bot_attributed'] and r['self_expressed'])} mixed/none"
    )

    return results


def extract_emotion_features(
    df: pd.DataFrame,
    text_column: str = "body",
    include_bot_attribution: bool = True,
) -> pd.DataFrame:
    """
    Extract emotion features from comments.

    Args:
        df: DataFrame with comments
        text_column: Name of column with comment text
        include_bot_attribution: If True, also compute bot-attributed
            emotion features (distinguishing "she is happy" from "I am happy")

    Returns:
        DataFrame with emotion scores and optionally bot-attribution features
    """
    texts = df[text_column].fillna("").astype(str).tolist()

    if include_bot_attribution:
        emotion_results = classify_bot_attributed_emotions(texts)
        result_df = df.copy()

        for label in EMOTION_LABELS:
            result_df[f'emotion_{label}'] = [r[f'emotion_{label}'] for r in emotion_results]
            result_df[f'bot_emotion_{label}'] = [r[f'bot_emotion_{label}'] for r in emotion_results]

        result_df['bot_attributed'] = [r['bot_attributed'] for r in emotion_results]
        result_df['self_expressed'] = [r['self_expressed'] for r in emotion_results]
        result_df['bot_attribution_score'] = [r['bot_attribution_score'] for r in emotion_results]
        result_df['attribution_type'] = [r['attribution_type'] for r in emotion_results]
    else:
        emotion_results = classify_emotions(texts)
        result_df = df.copy()
        for label in EMOTION_LABELS:
            result_df[f'emotion_{label}'] = [r[label] for r in emotion_results]

    emotion_columns = [f'emotion_{label}' for label in EMOTION_LABELS]
    result_df['dominant_emotion'] = result_df[emotion_columns].idxmax(axis=1).str.replace('emotion_', '')
    result_df['dominant_emotion_score'] = result_df[emotion_columns].max(axis=1)

    if include_bot_attribution:
        bot_emotion_columns = [f'bot_emotion_{label}' for label in EMOTION_LABELS]
        bot_has_values = result_df[bot_emotion_columns].sum(axis=1) > 0
        result_df['bot_dominant_emotion'] = ''
        result_df.loc[bot_has_values, 'bot_dominant_emotion'] = (
            result_df.loc[bot_has_values, bot_emotion_columns]
            .idxmax(axis=1).str.replace('bot_emotion_', '')
        )

    return result_df


def aggregate_emotions_to_user_level(
    df: pd.DataFrame,
    author_column: str = "author",
    include_bot_attribution: bool = True,
) -> pd.DataFrame:
    """
    Aggregate emotion features to user level.

    Args:
        df: DataFrame with emotion features
        author_column: Name of author column
        include_bot_attribution: If True, also aggregate bot-attributed emotions

    Returns:
        DataFrame with user-level emotion aggregations
    """
    logger.info("Aggregating emotions to user level")

    emotion_columns = [f'emotion_{label}' for label in EMOTION_LABELS]
    agg_columns = list(emotion_columns)

    if include_bot_attribution and 'bot_emotion_joy' in df.columns:
        bot_emotion_columns = [f'bot_emotion_{label}' for label in EMOTION_LABELS]
        agg_columns.extend(bot_emotion_columns)
        if 'bot_attribution_score' in df.columns:
            agg_columns.append('bot_attribution_score')

    user_emotions = df.groupby(author_column)[agg_columns].mean().reset_index()

    user_emotions['dominant_emotion'] = (
        user_emotions[emotion_columns].idxmax(axis=1).str.replace('emotion_', '')
    )
    user_emotions['dominant_emotion_score'] = user_emotions[emotion_columns].max(axis=1)

    if include_bot_attribution and 'bot_emotion_joy' in user_emotions.columns:
        bot_cols = [f'bot_emotion_{label}' for label in EMOTION_LABELS]
        bot_has_values = user_emotions[bot_cols].sum(axis=1) > 0
        user_emotions['bot_dominant_emotion'] = ''
        user_emotions.loc[bot_has_values, 'bot_dominant_emotion'] = (
            user_emotions.loc[bot_has_values, bot_cols]
            .idxmax(axis=1).str.replace('bot_emotion_', '')
        )

    logger.info(f"Aggregated emotions for {len(user_emotions)} users")

    return user_emotions

