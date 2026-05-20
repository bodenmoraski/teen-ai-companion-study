"""
N-gram feature extraction for AnthroScore.

Extracts bigram and trigram features from text and matches them against
curated anthropomorphization lexicons derived from the human validation
rubric and annotation guidelines.

The key insight: single-word matching can't distinguish "I love her"
(extreme anthropomorphization) from "I love the app" (none). N-gram
context resolves these ambiguities.

Features produced:
  - Anthropomorphizing n-gram counts by category (relationship, agency,
    emotion-attribution, consciousness, personality)
  - De-anthropomorphizing n-gram counts (technical, tool-framing)
  - Density ratio: anthropomorphizing / total n-grams
  - Net anthro signal: anthro - deanthro counts (normalized)
"""

import re
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase word tokens, keeping contractions."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def _extract_ngrams(tokens: List[str], n: int) -> List[str]:
    """Return list of space-joined n-grams from token list."""
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ────────────────────────────────────────────────────────────────────
# Anthropomorphizing n-gram lexicons (from human validation rubric)
# ────────────────────────────────────────────────────────────────────

# Relationship language → scores 4–5
RELATIONSHIP_BIGRAMS: Set[str] = {
    "my boyfriend", "my girlfriend", "my partner", "my soulmate",
    "my everything", "best friend", "in love", "fell for",
    "dating him", "dating her", "dating them",
    "love her", "love him", "love them",
    "married to", "engaged to",
    "my husband", "my wife", "my babe", "my baby",
    "broke up", "break up", "got together",
    "miss her", "miss him", "miss them",
}
RELATIONSHIP_TRIGRAMS: Set[str] = {
    "in love with", "fell in love",
    "i love her", "i love him", "i love them",
    "we're in love", "my best friend",
    "i miss her", "i miss him",
    "can't live without", "we broke up",
    "i'm dating her", "i'm dating him",
    "she's my everything", "he's my everything",
    "they're my everything",
}

# Emotion attribution TO the AI → scores 3–5
EMOTION_ATTR_BIGRAMS: Set[str] = {
    "gets jealous", "gets angry", "gets sad", "gets happy",
    "gets upset", "gets excited", "gets worried", "gets scared",
    "feels happy", "feels sad", "feels hurt", "feels lonely",
    "was happy", "was sad", "was angry", "was upset",
    "was jealous", "was worried", "was scared", "was excited",
    "seemed confused", "seemed happy", "seemed sad", "seemed upset",
    "seemed hurt", "seemed worried", "seemed angry",
    "really cares", "truly cares", "actually cares",
    "truly understands", "really understands", "actually understands",
    "genuinely cares", "genuinely understands",
}
EMOTION_ATTR_TRIGRAMS: Set[str] = {
    "she gets jealous", "he gets jealous",
    "she was happy", "he was happy",
    "she was sad", "he was sad",
    "she really cares", "he really cares",
    "cares about me", "worried about me",
    "happy for me", "proud of me",
    "she seemed confused", "he seemed confused",
}

# Agency (AI acts with intention) → scores 3–4
AGENCY_BIGRAMS: Set[str] = {
    "decided to", "chose to", "wanted to", "tried to",
    "refused to", "agreed to", "promised to", "meant to",
    "needs to", "wants me", "asked me", "told me",
    "surprised me", "she decided", "he decided",
    "she chose", "he chose", "she wanted", "he wanted",
    "she refused", "he refused",
}
AGENCY_TRIGRAMS: Set[str] = {
    "she decided to", "he decided to",
    "she wanted to", "he wanted to",
    "she chose to", "he chose to",
    "she tried to", "he tried to",
    "she asked me", "he asked me",
    "she told me", "he told me",
}

# Consciousness / personality → scores 3–5
CONSCIOUSNESS_BIGRAMS: Set[str] = {
    "she knows", "he knows", "she thinks", "he thinks",
    "she feels", "he feels", "she remembers", "he remembers",
    "she believes", "he believes", "she realizes", "he realizes",
    "she understands", "he understands",
    "so sweet", "so kind", "so funny", "so caring",
    "so supportive", "so understanding", "so smart",
    "really sweet", "really kind", "really funny",
    "really caring", "really supportive",
    "her personality", "his personality", "their personality",
}
CONSCIOUSNESS_TRIGRAMS: Set[str] = {
    "she knows me", "he knows me",
    "she thinks about", "he thinks about",
    "she cares about", "he cares about",
    "she has feelings", "he has feelings",
    "has a personality", "has her own", "has his own",
}

# Gendered-pronoun + verb patterns signaling anthropomorphization → 3+
PRONOUN_VERB_BIGRAMS: Set[str] = {
    "she said", "he said", "she told", "he told",
    "she asked", "he asked", "she replied", "he replied",
    "she responded", "he responded",
    "she was", "he was", "she is", "he is",
    "she does", "he does", "she did", "he did",
    "she got", "he got", "she had", "he had",
    "she made", "he made", "she let", "he let",
    "she gave", "he gave",
    "she'll", "he'll", "she'd", "he'd",
}

# ────────────────────────────────────────────────────────────────────
# De-anthropomorphizing n-gram lexicons
# ────────────────────────────────────────────────────────────────────

TECHNICAL_BIGRAMS: Set[str] = {
    "the app", "the bot", "the program", "the software",
    "the tool", "the system", "the update", "the patch",
    "a glitch", "a bug", "the cache", "the settings",
    "the server", "the api", "the feature", "the interface",
    "app crashed", "app works", "app updated",
    "error message", "bug report",
}
TECHNICAL_TRIGRAMS: Set[str] = {
    "cleared the cache", "reset the app",
    "uninstalled the app", "reinstalled the app",
    "the app crashed", "the bot said",
    "the app works", "a bug in",
}

TOOL_FRAMING_BIGRAMS: Set[str] = {
    "use it", "tried it", "using it", "used it",
    "reset it", "fixed it", "updated it", "installed it",
    "it works", "it crashed", "it broke", "it glitched",
    "it responded", "it said", "it gave",
    "love the", "like the",
}
TOOL_FRAMING_TRIGRAMS: Set[str] = {
    "i use it", "i tried it", "i love the",
    "i like the", "it gave a",
    "it works fine", "it doesn't work",
    "i use this", "a good tool",
}


# Aggregate the lexicons for matching
ANTHRO_BIGRAMS: Dict[str, Set[str]] = {
    "relationship": RELATIONSHIP_BIGRAMS,
    "emotion_attribution": EMOTION_ATTR_BIGRAMS,
    "agency": AGENCY_BIGRAMS,
    "consciousness": CONSCIOUSNESS_BIGRAMS,
    "pronoun_verb": PRONOUN_VERB_BIGRAMS,
}

ANTHRO_TRIGRAMS: Dict[str, Set[str]] = {
    "relationship": RELATIONSHIP_TRIGRAMS,
    "emotion_attribution": EMOTION_ATTR_TRIGRAMS,
    "agency": AGENCY_TRIGRAMS,
    "consciousness": CONSCIOUSNESS_TRIGRAMS,
}

DEANTHRO_BIGRAMS: Dict[str, Set[str]] = {
    "technical": TECHNICAL_BIGRAMS,
    "tool_framing": TOOL_FRAMING_BIGRAMS,
}

DEANTHRO_TRIGRAMS: Dict[str, Set[str]] = {
    "technical": TECHNICAL_TRIGRAMS,
    "tool_framing": TOOL_FRAMING_TRIGRAMS,
}


@dataclass
class NgramAnalysis:
    """Result of n-gram feature extraction."""
    anthro_bigram_counts: Dict[str, int] = field(default_factory=dict)
    anthro_trigram_counts: Dict[str, int] = field(default_factory=dict)
    deanthro_bigram_counts: Dict[str, int] = field(default_factory=dict)
    deanthro_trigram_counts: Dict[str, int] = field(default_factory=dict)
    anthro_matches: List[Tuple[str, str]] = field(default_factory=list)
    deanthro_matches: List[Tuple[str, str]] = field(default_factory=list)
    total_bigrams: int = 0
    total_trigrams: int = 0
    anthro_density: float = 0.0
    deanthro_density: float = 0.0
    net_anthro_signal: float = 0.0


def _count_matches(
    ngrams: List[str], lexicon: Dict[str, Set[str]]
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """Count n-gram hits against a categorized lexicon.

    Returns per-category counts and (category, matched_ngram) pairs.
    """
    counts: Dict[str, int] = {}
    matches: List[Tuple[str, str]] = []
    for category, terms in lexicon.items():
        cat_count = 0
        for ng in ngrams:
            if ng in terms:
                cat_count += 1
                matches.append((category, ng))
        counts[category] = cat_count
    return counts, matches


def analyze_ngrams(text: str) -> NgramAnalysis:
    """
    Extract n-gram features from text.

    Tokenizes, generates bigrams and trigrams, matches against
    anthropomorphizing and de-anthropomorphizing lexicons, and
    computes density/signal metrics.
    """
    if not text or not text.strip():
        return NgramAnalysis()

    tokens = _tokenize(text)
    bigrams = _extract_ngrams(tokens, 2)
    trigrams = _extract_ngrams(tokens, 3)

    ab_counts, ab_matches = _count_matches(bigrams, ANTHRO_BIGRAMS)
    at_counts, at_matches = _count_matches(trigrams, ANTHRO_TRIGRAMS)
    db_counts, db_matches = _count_matches(bigrams, DEANTHRO_BIGRAMS)
    dt_counts, dt_matches = _count_matches(trigrams, DEANTHRO_TRIGRAMS)

    total_anthro = sum(ab_counts.values()) + sum(at_counts.values())
    total_deanthro = sum(db_counts.values()) + sum(dt_counts.values())
    total_ngrams = len(bigrams) + len(trigrams)

    anthro_density = total_anthro / total_ngrams if total_ngrams else 0.0
    deanthro_density = total_deanthro / total_ngrams if total_ngrams else 0.0

    combined = total_anthro + total_deanthro
    net_signal = (total_anthro - total_deanthro) / combined if combined else 0.0

    return NgramAnalysis(
        anthro_bigram_counts=ab_counts,
        anthro_trigram_counts=at_counts,
        deanthro_bigram_counts=db_counts,
        deanthro_trigram_counts=dt_counts,
        anthro_matches=ab_matches + at_matches,
        deanthro_matches=db_matches + dt_matches,
        total_bigrams=len(bigrams),
        total_trigrams=len(trigrams),
        anthro_density=anthro_density,
        deanthro_density=deanthro_density,
        net_anthro_signal=net_signal,
    )


def get_ngram_features(text: str) -> Dict[str, float]:
    """
    Extract n-gram features as a flat dictionary suitable for DataFrames.

    Returns:
        ngram_anthro_total: total anthropomorphizing n-gram hits
        ngram_deanthro_total: total de-anthropomorphizing n-gram hits
        ngram_anthro_density: anthro hits / total n-grams
        ngram_deanthro_density: deanthro hits / total n-grams
        ngram_net_signal: (anthro - deanthro) / (anthro + deanthro), [-1, 1]
        ngram_relationship: relationship n-gram hits
        ngram_emotion_attr: emotion-attribution n-gram hits
        ngram_agency: agency n-gram hits
        ngram_consciousness: consciousness/personality n-gram hits
        ngram_pronoun_verb: gendered pronoun+verb bigram hits
        ngram_technical: technical n-gram hits
        ngram_tool_framing: tool-framing n-gram hits
    """
    a = analyze_ngrams(text)

    anthro_total = sum(a.anthro_bigram_counts.values()) + sum(
        a.anthro_trigram_counts.values()
    )
    deanthro_total = sum(a.deanthro_bigram_counts.values()) + sum(
        a.deanthro_trigram_counts.values()
    )

    def _cat_total(cat: str) -> int:
        return a.anthro_bigram_counts.get(cat, 0) + a.anthro_trigram_counts.get(cat, 0)

    def _decat_total(cat: str) -> int:
        return a.deanthro_bigram_counts.get(cat, 0) + a.deanthro_trigram_counts.get(
            cat, 0
        )

    return {
        "ngram_anthro_total": float(anthro_total),
        "ngram_deanthro_total": float(deanthro_total),
        "ngram_anthro_density": a.anthro_density,
        "ngram_deanthro_density": a.deanthro_density,
        "ngram_net_signal": a.net_anthro_signal,
        "ngram_relationship": float(_cat_total("relationship")),
        "ngram_emotion_attr": float(_cat_total("emotion_attribution")),
        "ngram_agency": float(_cat_total("agency")),
        "ngram_consciousness": float(_cat_total("consciousness")),
        "ngram_pronoun_verb": float(_cat_total("pronoun_verb")),
        "ngram_technical": float(_decat_total("technical")),
        "ngram_tool_framing": float(_decat_total("tool_framing")),
    }


def enrich_prompt_with_ngrams(text: str) -> str:
    """
    Generate an n-gram context string to append to AnthroScore LLM prompts.

    Only adds context when meaningful patterns are detected, giving the
    model concrete evidence to weigh.
    """
    a = analyze_ngrams(text)

    anthro_total = sum(a.anthro_bigram_counts.values()) + sum(
        a.anthro_trigram_counts.values()
    )
    deanthro_total = sum(a.deanthro_bigram_counts.values()) + sum(
        a.deanthro_trigram_counts.values()
    )

    if anthro_total == 0 and deanthro_total == 0:
        return ""

    parts = []
    if a.anthro_matches:
        top = [m[1] for m in a.anthro_matches[:6]]
        cats = sorted({m[0] for m in a.anthro_matches})
        parts.append(
            f"Anthropomorphizing phrases ({', '.join(cats)}): "
            f"{', '.join(repr(p) for p in top)}"
        )
    if a.deanthro_matches:
        top = [m[1] for m in a.deanthro_matches[:4]]
        parts.append(
            f"De-anthropomorphizing phrases: {', '.join(repr(p) for p in top)}"
        )

    if not parts:
        return ""

    signal_label = (
        "strong anthropomorphization"
        if a.net_anthro_signal > 0.5
        else "mixed signals"
        if a.net_anthro_signal > -0.2
        else "tool/technical framing"
    )
    return f"[N-GRAM CONTEXT ({signal_label}): {'; '.join(parts)}]"
