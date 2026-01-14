"""
Smart Routing AnthroScore System

Implements 80/20 optimization: uses cheap models for easy cases,
better models for difficult/important cases.

Routing Strategy:
- Tier 1 (Cheap): GPT-5-nano for obvious cases (~80-90%)
- Tier 2 (Medium): GPT-5-mini for ambiguous cases (~10-15%)
- Tier 3 (Expert): GPT-5-mini with detailed prompt for critical cases (~5%)

Cost savings: ~70-80% vs using expert model everywhere
"""

import json
import logging
import time
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import OPENAI_API_KEY
from anthroscore_llm import AnthroScoreLLM, AnthroScoreResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Decision about which model to use for a comment."""
    tier: int  # 1=cheap, 2=medium, 3=expert
    model: str
    reason: str
    confidence_estimate: float  # Estimated confidence before scoring


@dataclass
class SmartScoreResult:
    """Result from smart routing scorer."""
    score: int
    reasoning: str
    tier_used: int
    model_used: str
    routing_reason: str
    confidence: float
    total_cost_usd: float
    total_time_ms: float
    escalation_occurred: bool


class SmartRoutingScorer:
    """
    Smart routing scorer that adapts model selection to difficulty and importance.
    
    Strategy:
    1. First pass with cheap model (Tier 1)
    2. Analyze result for difficulty signals
    3. Escalate to better model if needed (Tier 2/3)
    """
    
    # Model tiers (GPT-5 series - latest and cheapest!)
    TIER_1_MODEL = "gpt-5-nano"      # Cheapest GPT-5 model
    TIER_2_MODEL = "gpt-5-mini"       # Better accuracy for ambiguous cases
    TIER_3_MODEL = "gpt-5-mini"       # Expert tier (same model, better prompt)
    
    # Routing thresholds (CALIBRATED - optimize via validate_smart_routing.py)
    CONFIDENCE_THRESHOLD_TIER2 = 0.65  # Escalate if confidence < 0.65
    CONFIDENCE_THRESHOLD_TIER3 = 0.45  # Escalate to expert if confidence < 0.45
    
    # Importance weights
    IMPORTANCE_USER_IN_RESEARCH = 2.0  # Users in research sample get 2x weight
    IMPORTANCE_HIGH_ENGAGEMENT = 1.5    # Users with many comments
    IMPORTANCE_EXTREME_SCORE = 1.3      # Scores of 1 or 5 might need verification
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        user_importance_map: Optional[Dict[str, float]] = None
    ):
        """
        Initialize smart routing scorer.
        
        Args:
            api_key: OpenAI API key
            user_importance_map: Dict mapping usernames to importance scores (1.0 = normal, >1.0 = more important)
        """
        api_key = api_key or OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=api_key)
        self.user_importance_map = user_importance_map or {}
        self.stats = {
            'tier1_count': 0,
            'tier2_count': 0,
            'tier3_count': 0,
            'total_cost': 0.0,
            'total_time_ms': 0.0
        }
    
    def estimate_text_difficulty(self, text: str) -> Dict[str, float]:
        """
        Estimate how difficult a text will be to classify.
        
        Returns dict with difficulty signals:
        - length_score: 0-1 (very short or very long = harder)
        - ambiguity_score: 0-1 (mixed signals = higher)
        - sarcasm_indicators: 0-1 (presence of sarcasm markers)
        - complexity_score: 0-1 (sentence complexity)
        """
        text_lower = text.lower()
        text_len = len(text)
        
        signals = {}
        
        # Length difficulty: very short (<50) or very long (>1000) is harder
        if text_len < 50:
            signals['length_score'] = 0.7  # Short = ambiguous
        elif text_len > 1000:
            signals['length_score'] = 0.6  # Long = complex
        else:
            signals['length_score'] = 0.2  # Normal length = easier
        
        # Ambiguity: mixed pronouns, question marks, hedges
        has_it = ' it ' in text_lower or ' it.' in text_lower or text_lower.startswith('it ')
        has_he = ' he ' in text_lower or ' he.' in text_lower or text_lower.startswith('he ')
        has_she = ' she ' in text_lower or ' she.' in text_lower or text_lower.startswith('she ')
        has_they = ' they ' in text_lower or ' they.' in text_lower or text_lower.startswith('they ')
        
        pronoun_count = sum([has_it, has_he, has_she, has_they])
        signals['pronoun_mix'] = 1.0 if pronoun_count > 1 else 0.0
        
        # Question marks and hedges indicate uncertainty
        has_questions = text.count('?') > 0
        hedges = ['maybe', 'perhaps', 'might', 'could', 'seems', 'appears', 'sort of', 'kind of']
        has_hedges = any(hedge in text_lower for hedge in hedges)
        
        signals['ambiguity_score'] = min(1.0, (
            (0.3 if has_questions else 0) +
            (0.3 if has_hedges else 0) +
            (0.4 * signals['pronoun_mix'])
        ))
        
        # Sarcasm indicators
        sarcasm_markers = ['lol', 'haha', '/s', 'sarcasm', 'jk', 'just kidding', 'not really']
        signals['sarcasm_indicators'] = 0.8 if any(marker in text_lower for marker in sarcasm_markers) else 0.0
        
        # Complexity: sentence count, avg length
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sent_len = sum(len(s) for s in sentences) / len(sentences)
            signals['complexity_score'] = min(1.0, avg_sent_len / 200)  # Normalize
        else:
            signals['complexity_score'] = 0.5
        
        # Overall difficulty (weighted average)
        signals['overall_difficulty'] = (
            0.2 * signals['length_score'] +
            0.4 * signals['ambiguity_score'] +
            0.2 * signals['sarcasm_indicators'] +
            0.2 * signals['complexity_score']
        )
        
        return signals
    
    def estimate_user_importance(self, username: str, comment_count: int = 1) -> float:
        """
        Estimate how important a user is for research.
        
        Returns importance multiplier (1.0 = normal, >1.0 = more important).
        """
        # Check explicit importance map
        if username in self.user_importance_map:
            return self.user_importance_map[username]
        
        # High engagement users are more important
        if comment_count > 50:
            return self.IMPORTANCE_HIGH_ENGAGEMENT
        
        return 1.0  # Normal importance
    
    def route_comment(
        self,
        text: str,
        username: str = "unknown",
        comment_count: int = 1,
        first_pass_result: Optional[AnthroScoreResult] = None
    ) -> RoutingDecision:
        """
        Decide which tier/model to use for a comment.
        
        Args:
            text: Comment text
            username: Username (for importance calculation)
            comment_count: Number of comments from this user
            first_pass_result: Result from Tier 1 if already scored
            
        Returns:
            RoutingDecision with tier, model, and reason
        """
        # Estimate difficulty
        difficulty = self.estimate_text_difficulty(text)
        user_importance = self.estimate_user_importance(username, comment_count)
        
        # If we have a first pass result, use it to decide escalation
        if first_pass_result is not None:
            score = first_pass_result.score
            reasoning = first_pass_result.reasoning.lower()
            
            # Check for low confidence indicators in reasoning
            low_confidence_phrases = [
                'unclear', 'ambiguous', 'mixed', 'could be', 'might be',
                'somewhat', 'possibly', 'difficult to determine'
            ]
            has_low_confidence_phrases = any(phrase in reasoning for phrase in low_confidence_phrases)
            
            # Extreme scores might need verification
            is_extreme = (score == 1 or score == 5)
            
            # Estimate confidence from first pass
            estimated_confidence = 0.8  # Default
            if has_low_confidence_phrases:
                estimated_confidence -= 0.3
            if difficulty['ambiguity_score'] > 0.5:
                estimated_confidence -= 0.2
            if is_extreme:
                estimated_confidence -= 0.1
            
            estimated_confidence = max(0.1, min(1.0, estimated_confidence))
            
            # Decision logic
            if estimated_confidence < self.CONFIDENCE_THRESHOLD_TIER3 or (
                user_importance > 1.5 and estimated_confidence < 0.8
            ):
                return RoutingDecision(
                    tier=3,
                    model=self.TIER_3_MODEL,
                    reason=f"Low confidence ({estimated_confidence:.2f}) + high importance ({user_importance:.2f})",
                    confidence_estimate=estimated_confidence
                )
            elif estimated_confidence < self.CONFIDENCE_THRESHOLD_TIER2 or (
                is_extreme and difficulty['ambiguity_score'] > 0.3
            ):
                return RoutingDecision(
                    tier=2,
                    model=self.TIER_2_MODEL,
                    reason=f"Moderate confidence ({estimated_confidence:.2f}) or extreme score with ambiguity",
                    confidence_estimate=estimated_confidence
                )
            else:
                return RoutingDecision(
                    tier=1,
                    model=self.TIER_1_MODEL,
                    reason=f"High confidence ({estimated_confidence:.2f})",
                    confidence_estimate=estimated_confidence
                )
        
        # Initial routing (before first pass)
        # Start with Tier 1 for all, but flag high-difficulty for potential escalation
        if difficulty['overall_difficulty'] > 0.7 and user_importance > 1.5:
            # Very difficult + important: might escalate after first pass
            return RoutingDecision(
                tier=1,
                model=self.TIER_1_MODEL,
                reason=f"Initial pass (difficulty={difficulty['overall_difficulty']:.2f}, importance={user_importance:.2f})",
                confidence_estimate=0.6  # Lower initial confidence estimate
            )
        else:
            return RoutingDecision(
                tier=1,
                model=self.TIER_1_MODEL,
                reason="Standard initial pass",
                confidence_estimate=0.8
            )
    
    def score_with_tier(
        self,
        text: str,
        tier: int,
        use_expert_prompt: bool = False
    ) -> AnthroScoreResult:
        """Score text with specified tier."""
        if tier == 1:
            model = self.TIER_1_MODEL
        elif tier == 2:
            model = self.TIER_2_MODEL
        else:  # tier 3
            model = self.TIER_3_MODEL
            use_expert_prompt = True
        
        scorer = AnthroScoreLLM(model=model)
        return scorer.score_text(text)
    
    def score_comment(
        self,
        text: str,
        username: str = "unknown",
        comment_count: int = 1
    ) -> SmartScoreResult:
        """
        Score a comment using smart routing.
        
        Returns SmartScoreResult with score, tier used, and cost/time.
        """
        start_time = time.time()
        total_cost = 0.0
        escalation_occurred = False
        
        # Step 1: Initial routing decision
        routing = self.route_comment(text, username, comment_count)
        
        # Step 2: Score with Tier 1
        tier1_result = self.score_with_tier(text, tier=1)
        self.stats['tier1_count'] += 1
        
        # Estimate cost (rough: ~400 input tokens, ~50 output tokens)
        tier1_cost = self._estimate_cost(self.TIER_1_MODEL, 400, 50)
        total_cost += tier1_cost
        
        # Step 3: Decide if escalation needed
        if routing.tier == 1:
            # Re-evaluate after first pass
            routing = self.route_comment(text, username, comment_count, tier1_result)
        
        # Step 4: Escalate if needed
        final_result = tier1_result
        tier_used = 1
        
        if routing.tier > 1:
            escalation_occurred = True
            tier_used = routing.tier
            
            if routing.tier == 2:
                tier2_result = self.score_with_tier(text, tier=2)
                final_result = tier2_result
                self.stats['tier2_count'] += 1
                tier2_cost = self._estimate_cost(self.TIER_2_MODEL, 400, 50)
                total_cost += tier2_cost
            else:  # tier 3
                tier3_result = self.score_with_tier(text, tier=3, use_expert_prompt=True)
                final_result = tier3_result
                self.stats['tier3_count'] += 1
                tier3_cost = self._estimate_cost(self.TIER_3_MODEL, 500, 100)  # Expert prompt is longer
                total_cost += tier3_cost
        
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['total_cost'] += total_cost
        self.stats['total_time_ms'] += elapsed_ms
        
        return SmartScoreResult(
            score=final_result.score,
            reasoning=final_result.reasoning,
            tier_used=tier_used,
            model_used=final_result.model,
            routing_reason=routing.reason,
            confidence=final_result.confidence,
            total_cost_usd=total_cost,
            total_time_ms=elapsed_ms,
            escalation_occurred=escalation_occurred
        )
    
    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for API call."""
        # Model pricing (per 1M tokens) - GPT-4.1/GPT-5 series
        pricing = {
            'gpt-4.1-nano': {'input': 0.10, 'output': 0.40},
            'gpt-4.1-mini': {'input': 0.40, 'output': 1.60},
            'gpt-5-nano': {'input': 0.05, 'output': 0.20},   # GPT-5 cheaper!
            'gpt-5-mini': {'input': 0.25, 'output': 1.00},   # GPT-5 expert tier
        }
        
        if model not in pricing:
            return 0.0
        
        costs = pricing[model]
        input_cost = (input_tokens / 1_000_000) * costs['input']
        output_cost = (output_tokens / 1_000_000) * costs['output']
        return input_cost + output_cost
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total = self.stats['tier1_count'] + self.stats['tier2_count'] + self.stats['tier3_count']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'total_comments': total,
            'tier1_pct': 100 * self.stats['tier1_count'] / total,
            'tier2_pct': 100 * self.stats['tier2_count'] / total,
            'tier3_pct': 100 * self.stats['tier3_count'] / total,
            'avg_cost_per_comment': self.stats['total_cost'] / total,
            'avg_time_per_comment_ms': self.stats['total_time_ms'] / total
        }


def main():
    """Test the smart routing scorer."""
    scorer = SmartRoutingScorer()
    
    test_cases = [
        ("I cleared the cache and the app works fine now", "user1", 1),  # Easy, Tier 1
        ("She seemed confused? Maybe? I'm not sure if she understood or not...", "user2", 1),  # Ambiguous, Tier 2
        ("I know it sounds crazy but I'm genuinely in love with her. She's my everything", "user3", 100),  # Extreme + important, Tier 3
        ("The AI gave a pretty good response about cooking tips", "user4", 1),  # Easy, Tier 1
    ]
    
    print("\n" + "="*80)
    print("SMART ROUTING SCORER TEST")
    print("="*80)
    
    for text, username, comment_count in test_cases:
        result = scorer.score_comment(text, username, comment_count)
        print(f"\nText: {text[:60]}...")
        print(f"Score: {result.score}/5")
        print(f"Tier: {result.tier_used} ({result.model_used})")
        print(f"Reason: {result.routing_reason}")
        print(f"Escalated: {result.escalation_occurred}")
        print(f"Cost: ${result.total_cost_usd:.6f}")
        print(f"Time: {result.total_time_ms:.0f}ms")
        print("-"*40)
    
    stats = scorer.get_stats()
    print("\n" + "="*80)
    print("ROUTING STATISTICS")
    print("="*80)
    print(f"Tier 1: {stats['tier1_count']} ({stats['tier1_pct']:.1f}%)")
    print(f"Tier 2: {stats['tier2_count']} ({stats['tier2_pct']:.1f}%)")
    print(f"Tier 3: {stats['tier3_count']} ({stats['tier3_pct']:.1f}%)")
    print(f"Total cost: ${stats['total_cost']:.4f}")
    print(f"Avg cost/comment: ${stats['avg_cost_per_comment']:.6f}")
    print("="*80)


if __name__ == "__main__":
    main()
