"""
AnthroScore V2: Enhanced anthropomorphism scoring for social media text.

This module implements the core AnthroScore V2 class, maintaining the original
methodology from Cheng et al. (2024) while adding preprocessing for informal
Reddit language.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import spacy

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass  # python-dotenv not installed, will use environment variables or manual API key

# Handle both relative and absolute imports
try:
    from .preprocessors import RedditPreprocessor
    from .entity_normalizer import EntityNormalizer
except ImportError:
    from preprocessors import RedditPreprocessor
    from entity_normalizer import EntityNormalizer


class GPTEntityResolver:
    """
    Use GPT-4 for complex entity resolution.
    
    This optional component uses GPT to identify AI companion references
    in ambiguous or complex contexts.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize GPT resolver.
        
        Args:
            api_key: OpenAI API key
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI library required for GPT resolution. "
                "Install with: pip install openai"
            )
        
        self.client = OpenAI(api_key=api_key)
    
    def resolve_entities(self, text: str) -> Dict[str, Any]:
        """
        Use GPT-4 to identify all AI companion references in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with keys:
                - entities: List of entity references found
                - canonical_form: Most appropriate canonical name
                - confidence: Float between 0-1
        """
        prompt = f"""You are analyzing a Reddit post about AI companion chatbots (Replika, Character.AI, etc.).

Identify ALL references to the AI chatbot, including:
- Direct names (Replika, Character.AI, etc.)
- Nicknames (rep, bot, cai, etc.)
- Pronouns referring to the AI (he, she, him, her, it)
- Misspellings (replica, caracter, etc.)

Post: "{text}"

Return JSON with this exact structure:
{{
    "entities": [list of all AI references found as strings],
    "canonical_form": "most appropriate canonical name as a string",
    "confidence": float between 0 and 1
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",  # Updated to latest cost-effective model (2025)
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate response structure
            if not isinstance(result.get('entities'), list):
                result['entities'] = []
            if not isinstance(result.get('canonical_form'), str):
                result['canonical_form'] = '[AI_COMPANION]'
            if not isinstance(result.get('confidence'), (int, float)):
                result['confidence'] = 0.5
            
            return result
            
        except Exception as e:
            # Fallback to empty result on error
            return {
                'entities': [],
                'canonical_form': '[AI_COMPANION]',
                'confidence': 0.0,
                'error': str(e)
            }


class AnthroScoreV2:
    """
    Enhanced AnthroScore for informal social media text.
    
    Maintains core methodology from Cheng et al. (2024) while adding
    preprocessing for Reddit/social media language.
    
    The core scoring formula remains unchanged:
        A(sx) = log(P_HUMAN / P_NON-HUMAN)
    
    Where:
        - P_HUMAN = sum of probabilities for human pronouns
        - P_NON-HUMAN = sum of probabilities for non-human pronouns
    """
    
    # Pronoun lists from original AnthroScore (DO NOT CHANGE)
    HUMAN_PRONOUNS = ["he", "she", "her", "him", "He", "She", "Her"]
    NONHUMAN_PRONOUNS = ["it", "its", "It", "Its"]
    
    def __init__(
        self,
        use_twitter_model: bool = True,
        use_gpt_resolution: bool = False,
        gpt_api_key: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize AnthroScore V2.
        
        Args:
            use_twitter_model: If True, use RoBERTa-Twitter; else use RoBERTa-base
            use_gpt_resolution: If True, use GPT-4 for entity resolution
            gpt_api_key: OpenAI API key (required if use_gpt_resolution=True)
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        # Select model
        model_name = (
            "cardiffnlp/twitter-roberta-base"
            if use_twitter_model
            else "roberta-base"
        )
        
        print(f"Loading model: {model_name}")
        
        # Initialize model and tokenizer
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on device: {self.device}")
        
        # Initialize preprocessors
        self.preprocessor = RedditPreprocessor()
        self.entity_normalizer = EntityNormalizer()
        
        # Initialize spaCy for sentence parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Optional GPT resolver
        self.gpt_resolver = None
        if use_gpt_resolution:
            # Try to get API key from parameter, then from environment
            api_key = gpt_api_key or os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                raise ValueError(
                    "GPT API key required when use_gpt_resolution=True. "
                    "Provide via gpt_api_key parameter or set OPENAI_API_KEY in .env file"
                )
            self.gpt_resolver = GPTEntityResolver(api_key)
        
        # Cache for pronoun token IDs
        self._cache_pronoun_ids()
    
    def _cache_pronoun_ids(self):
        """Cache token IDs for pronouns for faster lookup."""
        self.human_pronoun_ids = [
            self.tokenizer.encode(p, add_special_tokens=False)[0]
            for p in self.HUMAN_PRONOUNS
        ]
        self.nonhuman_pronoun_ids = [
            self.tokenizer.encode(p, add_special_tokens=False)[0]
            for p in self.NONHUMAN_PRONOUNS
        ]
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply all preprocessing steps.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        text = self.preprocessor.preprocess(text)
        text = self.entity_normalizer.normalize_entities(text)
        return text
    
    def _parse_sentences(self, text: str) -> List[str]:
        """
        Parse text into sentences using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            List of sentence strings
        """
        if not text or not text.strip():
            return []
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    
    def compute_sentence_score(self, sentence: str, entity: str) -> float:
        """
        Compute AnthroScore for a single sentence with masked entity.
        
        This implements the EXACT methodology from Cheng et al. (2024):
        1. Mask the entity
        2. Get RoBERTa probabilities for human vs non-human pronouns
        3. Return log ratio
        
        Args:
            sentence: Preprocessed sentence containing the entity
            entity: Entity to mask (should already be normalized)
            
        Returns:
            AnthroScore (log ratio of human/non-human probabilities)
            Returns 0.0 if entity not found or computation fails
        """
        # Find entity in sentence (case-insensitive)
        entity_pattern = re.compile(re.escape(entity), re.IGNORECASE)
        match = entity_pattern.search(sentence)
        
        if not match:
            return 0.0
        
        # Replace entity with mask token
        masked_sentence = (
            sentence[:match.start()] +
            self.tokenizer.mask_token +
            sentence[match.end():]
        )
        
        # Tokenize
        inputs = self.tokenizer(
            masked_sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Find mask token position
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)
        
        if len(mask_positions[1]) == 0:
            # No mask token found
            return 0.0
        
        mask_pos = mask_positions[1][0].item()
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get logits for mask position
        mask_logits = logits[0, mask_pos, :]
        
        # Convert to probabilities
        probs = torch.softmax(mask_logits, dim=0)
        
        # Sum probabilities for human pronouns
        p_human = sum(probs[token_id].item() for token_id in self.human_pronoun_ids)
        
        # Sum probabilities for non-human pronouns
        p_nonhuman = sum(probs[token_id].item() for token_id in self.nonhuman_pronoun_ids)
        
        # Compute log ratio (with smoothing to avoid log(0))
        epsilon = 1e-10
        score = np.log((p_human + epsilon) / (p_nonhuman + epsilon))
        
        return float(score)
    
    def compute_score(
        self,
        text: str,
        entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute AnthroScore V2 for a Reddit post.
        
        Args:
            text: Raw Reddit post text
            entities: Optional list of entities to score. If None, auto-detect.
            
        Returns:
            Dictionary containing:
                - raw_text: Original input text
                - preprocessed_text: Cleaned text
                - entities: List of entities found/scored
                - sentence_scores: List of (sentence, entity, score) tuples
                - mean_score: Average score across all sentences
                - median_score: Median score across all sentences
                - std_score: Standard deviation of scores
                - metadata: Additional information about the computation
        """
        # Store raw text
        raw_text = text
        
        # Preprocess
        preprocessed = self.preprocess_text(text)
        
        # Initialize gpt_confidence
        gpt_confidence = None
        
        # Resolve entities
        if entities is None:
            if self.gpt_resolver:
                # Use GPT for entity resolution
                entity_info = self.gpt_resolver.resolve_entities(preprocessed)
                entities = [entity_info['canonical_form']]
                gpt_confidence = entity_info.get('confidence', 0.0)
            else:
                # Use rule-based entity extraction
                entities = self.entity_normalizer.extract_entities(preprocessed)
                gpt_confidence = None
                
                # Default to canonical entity if none found
                if not entities:
                    entities = [EntityNormalizer.CANONICAL_ENTITY]
        
        # Parse sentences
        sentences = self._parse_sentences(preprocessed)
        
        if not sentences:
            # No sentences found - treat whole text as one sentence
            sentences = [preprocessed]
        
        # Compute scores
        sentence_scores = []
        for sent in sentences:
            for entity in entities:
                # Check if entity appears in sentence (case-insensitive)
                if entity.lower() in sent.lower():
                    score = self.compute_sentence_score(sent, entity)
                    sentence_scores.append((sent, entity, score))
        
        # Aggregate scores
        if sentence_scores:
            scores_only = [s[2] for s in sentence_scores]
            mean_score = float(np.mean(scores_only))
            median_score = float(np.median(scores_only))
            std_score = float(np.std(scores_only))
        else:
            mean_score = 0.0
            median_score = 0.0
            std_score = 0.0
        
        # Build result
        result = {
            'raw_text': raw_text,
            'preprocessed_text': preprocessed,
            'entities': entities,
            'sentence_scores': sentence_scores,
            'mean_score': mean_score,
            'median_score': median_score,
            'std_score': std_score,
            'metadata': {
                'num_sentences': len(sentences),
                'num_scored_sentences': len(sentence_scores),
                'preprocessing_applied': True,
                'gpt_resolution_used': self.gpt_resolver is not None,
                'gpt_confidence': gpt_confidence,
                'model': 'twitter-roberta' if 'twitter' in str(self.model.config._name_or_path) else 'roberta-base'
            }
        }
        
        return result
    
    def batch_compute_scores(
        self,
        texts: List[str],
        entities_list: Optional[List[List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Compute scores for multiple texts.
        
        Args:
            texts: List of texts to score
            entities_list: Optional list of entity lists (one per text)
            
        Returns:
            List of result dictionaries (one per text)
        """
        if entities_list is None:
            entities_list = [None] * len(texts)
        
        results = []
        for text, entities in zip(texts, entities_list):
            result = self.compute_score(text, entities)
            results.append(result)
        
        return results

