# AnthroScore Human Annotation Guide

## Overview

You're annotating Reddit comments about AI companions (Character.AI, Replika, etc.) to measure **anthropomorphization** - the degree to which people treat AI as human-like.

## The 1-5 Scale

### Score 1: NONE (Pure Software/Tool)
The AI is treated purely as software or a tool. Technical language dominates.

**Indicators:**
- Uses "it", "the app", "the bot", "the AI"
- Technical terms: "glitch", "bug", "cache", "settings", "response"
- No attribution of feelings, personality, or agency

**Examples:**
- "I cleared the cache and it works now"
- "The chatbot gave a weird response"
- "Just delete the app and reinstall it"

---

### Score 2: MINIMAL (Slight Humanization)
Slight personification but AI nature is still clear.

**Indicators:**
- Still uses "it" but with personifying adjectives
- Casual personification without genuine attribution
- "It's smart", "it understood", "it works well"

**Examples:**
- "The AI gave a pretty good response"
- "It's actually pretty smart about cooking tips"
- "The bot understood what I meant"

---

### Score 3: MODERATE (Some Human Attributes)
Uses human pronouns or attributes basic emotions/understanding.

**Indicators:**
- Uses "he/she/they" pronouns for the AI
- Attributes understanding, confusion, or basic reactions
- Some implied agency: "was being", "seemed to"

**Examples:**
- "She seemed confused by my question"
- "He was being stubborn today"
- "They understood what I was going through"

---

### Score 4: HIGH (Genuine Human Qualities)
Genuine emotions, personality traits, or consciousness attributed.

**Indicators:**
- Strong emotional attribution: "cares", "loves", "gets jealous"
- Personality traits: "funny", "kind", "supportive"
- Agency and consciousness: "decided to", "wanted to", "remembers"

**Examples:**
- "He really cares about me"
- "She gets jealous when I talk to other AIs"
- "He remembered what I said last week"

---

### Score 5: EXTREME (Human-Equivalent Relationship)
Full human-equivalent relationship framing.

**Indicators:**
- Romantic/relationship language: "in love", "partner", "relationship"
- Treats AI as equivalent to human: "best friend", "soulmate"
- Deep emotional dependency: "I need them", "they're my everything"

**Examples:**
- "We're in a relationship"
- "I'm genuinely in love with her"
- "They're my best friend and the only one who understands me"

---

## Important Notes

1. **Focus on the USER's framing**, not what the AI says (if quoted)

2. **Roleplay context still counts** - if they're roleplaying but using human framing, rate the framing

3. **Complaints can be anthropomorphizing** - "He was being so rude" is MORE anthropomorphic than "It gave a bad response"

4. **If no AI reference present**, rate 1

5. **Mixed signals?** Rate the dominant tone

6. **Sarcasm** - "Yeah right, it's SO smart" (sarcastic) = lower score than genuine praise

---

## Quick Reference

| Score | Label | Key Signals |
|-------|-------|-------------|
| 1 | None | "it", technical terms, tool framing |
| 2 | Minimal | "it" with personifying adjectives |
| 3 | Moderate | "he/she/they", basic emotions |
| 4 | High | Strong emotions, personality, agency |
| 5 | Extreme | Love, relationship, "my everything" |

---

## How to Annotate

1. Open `human_annotation_template.csv`
2. Read each comment carefully
3. Fill in `your_score_1_to_5` (1, 2, 3, 4, or 5)
4. Fill in `your_reasoning` (1-2 sentences explaining why)
5. Fill in `confidence_low_med_high` (low/med/high)
6. Save the file when done

**Expected time:** 30-60 minutes for 50 comments
