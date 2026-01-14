# Human Validation Guide for AnthroScore V3

## Instructions

Rate each comment on a 1-5 scale for anthropomorphization of AI:

| Score | Label | Description | Examples |
|-------|-------|-------------|----------|
| 1 | None | AI treated as pure software/tool | "The app is buggy", "Clear the cache" |
| 2 | Minimal | Slight humanization | "It's smart", "The bot knows stuff" |
| 3 | Moderate | Human pronouns, basic emotions | "She understands me", "He gets sad" |
| 4 | High | Strong emotional attribution | "She's my best friend", "He truly cares" |
| 5 | Extreme | Human-equivalent relationship | "I love her", "We're dating" |

## What to Rate

- Rate ONLY the anthropomorphization of the AI
- Ignore the user's emotions about themselves
- Focus on: pronouns used, emotional attribution, relationship language

## Filling Out

1. `human_score`: Your 1-5 rating
2. `human_reasoning`: Brief explanation (1 sentence)
3. `confidence`: low/medium/high

## Tips

- "It" vs "he/she" matters
- "The bot said" vs "She told me" matters
- "Helpful tool" vs "caring friend" matters
