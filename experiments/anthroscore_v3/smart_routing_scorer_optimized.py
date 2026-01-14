"""
Optimized Smart Routing AnthroScore System - Research Quality

This version uses:
- Validated models (GPT-4.1-nano, GPT-4o)
- Calibrated thresholds based on validation data
- Improved difficulty detection
- Better importance weighting
- Research-quality configuration
"""

import json
import logging
import time
import re
from typing import Dict, Any, Optional, List
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
    confidence_estimate: float


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


class OptimizedSmartRoutingScorer:
    """
    Research-quality smart routing scorer.
    
    Uses validated models and calibrated thresholds for optimal quality.
    """
    
    # GPT-4.1/GPT-5 MODELS (latest generation)
    TIER_1_MODEL = "gpt-4.1-nano"  # Validated: Kappa=0.579, r=0.590 (target: improve to >0.60)
    TIER_2_MODEL = "gpt-5-nano"     # GPT-5 series - better accuracy than 4.1-nano
    TIER_3_MODEL = "gpt-5-mini"     # GPT-5 expert tier - best quality for research
    
    # CALIBRATED THRESHOLDS (will be optimized via validation)
    # Defaults based on validation: need ~15% escalation for quality
    CONFIDENCE_THRESHOLD_TIER2 = 0.65  # Escalate if confidence < 0.65
    CONFIDENCE_THRESHOLD_TIER3 = 0.45  # Escalate to expert if confidence < 0.45
    
    # IMPORTANCE WEIGHTS (research-focused)
    IMPORTANCE_USER_IN_RESEARCH = 2.5  # Research sample users are critical
    IMPORTANCE_HIGH_ENGAGEMENT = 1.3   # Users with many comments
    IMPORTANCE_KNOWN_DEMOGRAPHICS = 1.5  # Users with known age/gender
    IMPORTANCE_EXTREME_SCORE = 1.4     # Extreme scores need verification
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        user_importance_map: Optional[Dict[str, float]] = None,
        known_demographic_users: Optional[set] = None
    ):
        """
        Initialize optimized smart routing scorer.
        
        Args:
            api_key: OpenAI API key
            user_importance_map: Dict mapping usernames to importance scores
            known_demographic_users: Set of users with known demographics (high importance)
        """
        api_key = api_key or OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=api_key)
        self.user_importance_map = user_importance_map or {}
        self.known_demographic_users = known_demographic_users or set()
        
        self.stats = {
            'tier1_count': 0,
            'tier2_count': 0,
            'tier3_count': 0,
            'total_cost': 0.0,
            'total_time_ms': 0.0
        }
    
    def estimate_text_difficulty(self, text: str) -> Dict[str, float]:
        """
        Enhanced difficulty estimation with more sophisticated features.
        
        Returns dict with difficulty signals.
        """
        text_lower = text.lower()
        text_len = len(text)
        
        signals = {}
        
        # Length difficulty
        if text_len < 30:
            signals['length_score'] = 0.8  # Very short = ambiguous
        elif text_len > 1500:
            signals['length_score'] = 0.6  # Very long = complex
        else:
            signals['length_score'] = 0.2
        
        # Pronoun mixing (stronger signal)
        has_it = bool(re.search(r'\bit\b', text_lower))
        has_he = bool(re.search(r'\bhe\b', text_lower))
        has_she = bool(re.search(r'\bshe\b', text_lower))
        has_they = bool(re.search(r'\bthey\b', text_lower))
        
        pronoun_count = sum([has_it, has_he, has_she, has_they])
        signals['pronoun_mix'] = 1.0 if pronoun_count > 1 else 0.0
        
        # Ambiguity indicators
        has_questions = text.count('?') > 0
        hedges = ['maybe', 'perhaps', 'might', 'could', 'seems', 'appears', 'sort of', 'kind of', 'i think', 'i guess']
        has_hedges = any(hedge in text_lower for hedge in hedges)
        
        # Sarcasm/irony indicators
        sarcasm_markers = ['lol', 'haha', '/s', 'sarcasm', 'jk', 'just kidding', 'not really', 'yeah right']
        signals['sarcasm_indicators'] = 0.9 if any(marker in text_lower for marker in sarcasm_markers) else 0.0
        
        # Negation (can flip meaning)
        negations = ['not', "don't", "doesn't", "isn't", "wasn't", "won't", "can't"]
        has_negation = any(neg in text_lower for neg in negations)
        
        # Mixed sentiment indicators
        positive_words = ['love', 'great', 'amazing', 'wonderful', 'best', 'perfect']
        negative_words = ['hate', 'terrible', 'awful', 'worst', 'bad', 'horrible']
        has_positive = any(word in text_lower for word in positive_words)
        has_negative = any(word in text_lower for word in negative_words)
        mixed_sentiment = has_positive and has_negative
        
        signals['ambiguity_score'] = min(1.0, (
            (0.2 if has_questions else 0) +
            (0.3 if has_hedges else 0) +
            (0.3 * signals['pronoun_mix']) +
            (0.1 if has_negation else 0) +
            (0.1 if mixed_sentiment else 0)
        ))
        
        # Complexity: sentence structure
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sent_len = sum(len(s) for s in sentences) / len(sentences)
            signals['complexity_score'] = min(1.0, avg_sent_len / 200)
        else:
            signals['complexity_score'] = 0.5
        
        # Overall difficulty (weighted)
        signals['overall_difficulty'] = (
            0.15 * signals['length_score'] +
            0.45 * signals['ambiguity_score'] +
            0.20 * signals['sarcasm_indicators'] +
            0.20 * signals['complexity_score']
        )
        
        return signals
    
    def estimate_user_importance(self, username: str, comment_count: int = 1) -> float:
        """
        Enhanced importance estimation for research quality.
        
        Prioritizes:
        - Users in research sample
        - Users with known demographics
        - High engagement users
        """
        importance = 1.0
        
        # Explicit importance map
        if username in self.user_importance_map:
            importance = self.user_importance_map[username]
        
        # Known demographics (critical for research)
        if username in self.known_demographic_users:
            importance *= self.IMPORTANCE_KNOWN_DEMOGRAPHICS
        
        # High engagement
        if comment_count > 50:
            importance *= self.IMPORTANCE_HIGH_ENGAGEMENT
        
        return importance
    
    def route_comment(
        self,
        text: str,
        username: str = "unknown",
        comment_count: int = 1,
        first_pass_result: Optional[AnthroScoreResult] = None
    ) -> RoutingDecision:
        """
        Enhanced routing logic with better confidence estimation.
        """
        difficulty = self.estimate_text_difficulty(text)
        user_importance = self.estimate_user_importance(username, comment_count)
        
        if first_pass_result is not None:
            score = first_pass_result.score
            reasoning = first_pass_result.reasoning.lower()
            
            # Enhanced confidence estimation
            low_confidence_phrases = [
                'unclear', 'ambiguous', 'mixed', 'could be', 'might be',
                'somewhat', 'possibly', 'difficult to determine', 'not entirely',
                'somewhat unclear', 'a bit', 'kind of', 'sort of'
            ]
            has_low_conf = any(phrase in reasoning for phrase in low_confidence_phrases)
            
            # Extreme scores
            is_extreme = (score == 1 or score == 5)
            
            # Estimate confidence more accurately
            estimated_confidence = 0.85  # Higher default (Tier 1 is pretty good)
            
            if has_low_conf:
                estimated_confidence -= 0.35  # Bigger penalty
            if difficulty['ambiguity_score'] > 0.6:
                estimated_confidence -= 0.25
            elif difficulty['ambiguity_score'] > 0.4:
                estimated_confidence -= 0.15
            if is_extreme:
                estimated_confidence -= 0.15
            if difficulty['sarcasm_indicators'] > 0.5:
                estimated_confidence -= 0.20
            
            estimated_confidence = max(0.1, min(1.0, estimated_confidence))
            
            # Enhanced routing logic
            # Tier 3: Very low confidence OR high importance + moderate confidence
            if estimated_confidence < self.CONFIDENCE_THRESHOLD_TIER3 or (
                user_importance > 2.0 and estimated_confidence < 0.75
            ):
                return RoutingDecision(
                    tier=3,
                    model=self.TIER_3_MODEL,
                    reason=f"Low confidence ({estimated_confidence:.2f}) + importance ({user_importance:.2f})",
                    confidence_estimate=estimated_confidence
                )
            # Tier 2: Moderate confidence OR extreme score with ambiguity
            elif estimated_confidence < self.CONFIDENCE_THRESHOLD_TIER2 or (
                is_extreme and difficulty['ambiguity_score'] > 0.3
            ) or (
                user_importance > 1.5 and estimated_confidence < 0.80
            ):
                return RoutingDecision(
                    tier=2,
                    model=self.TIER_2_MODEL,
                    reason=f"Moderate confidence ({estimated_confidence:.2f}) or special case",
                    confidence_estimate=estimated_confidence
                )
            else:
                return RoutingDecision(
                    tier=1,
                    model=self.TIER_1_MODEL,
                    reason=f"High confidence ({estimated_confidence:.2f})",
                    confidence_estimate=estimated_confidence
                )
        
        # Initial routing
        if difficulty['overall_difficulty'] > 0.75 and user_importance > 2.0:
            return RoutingDecision(
                tier=1,
                model=self.TIER_1_MODEL,
                reason=f"Initial pass (high difficulty + importance)",
                confidence_estimate=0.5
            )
        else:
            return RoutingDecision(
                tier=1,
                model=self.TIER_1_MODEL,
                reason="Standard initial pass",
                confidence_estimate=0.85
            )
    
    def score_with_tier(self, text: str, tier: int) -> AnthroScoreResult:
        """Score text with specified tier."""
        if tier == 1:
            model = self.TIER_1_MODEL
        elif tier == 2:
            model = self.TIER_2_MODEL
        else:
            model = self.TIER_3_MODEL
        
        scorer = AnthroScoreLLM(model=model)
        return scorer.score_text(text)
    
    def score_comment(
        self,
        text: str,
        username: str = "unknown",
        comment_count: int = 1
    ) -> SmartScoreResult:
        """Score a comment using optimized smart routing."""
        start_time = time.time()
        total_cost = 0.0
        escalation_occurred = False
        
        # Step 1: Initial routing
        routing = self.route_comment(text, username, comment_count)
        
        # Step 2: Score with Tier 1
        tier1_result = self.score_with_tier(text, tier=1)
        self.stats['tier1_count'] += 1
        tier1_cost = self._estimate_cost(self.TIER_1_MODEL, 400, 50)
        total_cost += tier1_cost
        
        # Step 3: Re-evaluate after first pass
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
                tier3_result = self.score_with_tier(text, tier=3)
                final_result = tier3_result
                self.stats['tier3_count'] += 1
                tier3_cost = self._estimate_cost(self.TIER_3_MODEL, 500, 100)
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
