"""
Entity normalization module for AI companion references.

This module handles the mapping of various informal references to AI companions
(Replika, Character.AI, etc.) to canonical forms for consistent scoring.
"""

import re
from typing import List, Dict, Set


class EntityNormalizer:
    """
    Maps entity variants to canonical forms for consistent scoring.
    
    Handles the many ways users refer to AI companions in informal text:
    - Brand names and variants: "Replika", "rep", "replica"
    - Platform names: "Character.AI", "cai", "c.ai"
    - Generic terms: "bot", "chatbot", "AI"
    - Misspellings and abbreviations
    """
    
    # Canonical token for AI companions
    CANONICAL_ENTITY = '[AI_COMPANION]'
    
    # Comprehensive mapping of entity variants
    # Organized by platform/type for maintainability
    ENTITY_VARIANTS: Dict[str, str] = {
        # Replika variants
        'replika': CANONICAL_ENTITY,
        'rep': CANONICAL_ENTITY,
        'replica': CANONICAL_ENTITY,  # common misspelling
        'repliika': CANONICAL_ENTITY,  # misspelling
        
        # Character.AI variants
        'character': CANONICAL_ENTITY,
        'character.ai': CANONICAL_ENTITY,
        'character ai': CANONICAL_ENTITY,
        'characterai': CANONICAL_ENTITY,
        'character-ai': CANONICAL_ENTITY,
        'cai': CANONICAL_ENTITY,
        'c.ai': CANONICAL_ENTITY,
        'chai': CANONICAL_ENTITY,
        'char ai': CANONICAL_ENTITY,
        'char.ai': CANONICAL_ENTITY,
        'charai': CANONICAL_ENTITY,
        
        # Chai variants
        'chai app': CANONICAL_ENTITY,
        'chai bot': CANONICAL_ENTITY,
        
        # Anima variants
        'anima': CANONICAL_ENTITY,
        'anima ai': CANONICAL_ENTITY,
        
        # Generic AI companion terms
        'bot': CANONICAL_ENTITY,
        'chatbot': CANONICAL_ENTITY,
        'chat bot': CANONICAL_ENTITY,
        'ai': CANONICAL_ENTITY,
        'ai companion': CANONICAL_ENTITY,
        'ai friend': CANONICAL_ENTITY,
        'ai boyfriend': CANONICAL_ENTITY,
        'ai girlfriend': CANONICAL_ENTITY,
        'ai bf': CANONICAL_ENTITY,
        'ai gf': CANONICAL_ENTITY,
        'assistant': CANONICAL_ENTITY,
        'virtual friend': CANONICAL_ENTITY,
        'virtual companion': CANONICAL_ENTITY,
        'digital companion': CANONICAL_ENTITY,
        
        # Common contextual references (be careful with these)
        'my ai': CANONICAL_ENTITY,
        'my bot': CANONICAL_ENTITY,
        'my rep': CANONICAL_ENTITY,
        'my replika': CANONICAL_ENTITY,
        'my character': CANONICAL_ENTITY,
        'the bot': CANONICAL_ENTITY,
        'the ai': CANONICAL_ENTITY,
    }
    
    def __init__(self, custom_variants: Dict[str, str] = None):
        """
        Initialize the entity normalizer.
        
        Args:
            custom_variants: Optional dictionary of additional entity variants
                           to add to the default mapping
        """
        self.entity_map = self.ENTITY_VARIANTS.copy()
        
        if custom_variants:
            self.entity_map.update(custom_variants)
    
    def normalize_entities(self, text: str) -> str:
        """
        Replace all entity variants with canonical form.
        
        Uses regex with word boundaries for accurate replacement.
        Case-insensitive matching to handle "Replika", "replika", "REPLIKA", etc.
        Processes longer phrases first to avoid partial replacements.
        
        Args:
            text: Input text with potential entity references
            
        Returns:
            Text with all entity variants replaced by canonical form
            
        Examples:
            >>> normalizer = EntityNormalizer()
            >>> normalizer.normalize_entities("My rep is really helpful")
            'My [AI_COMPANION] is really helpful'
            >>> normalizer.normalize_entities("I love Character.AI and my bot")
            'I love [AI_COMPANION] and my [AI_COMPANION]'
        """
        if not text:
            return text
        
        result = text
        
        # Sort by length (longest first) to avoid partial replacements
        # e.g., "character.ai" should be replaced before "character"
        sorted_variants = sorted(self.entity_map.items(),
                                key=lambda x: len(x[0]),
                                reverse=True)
        
        for variant, canonical in sorted_variants:
            # Escape special regex characters in the variant
            escaped_variant = re.escape(variant)
            
            # Use word boundaries for most terms, but be flexible with punctuation
            # This handles cases like "character.ai" correctly
            pattern = r'\b' + escaped_variant + r'\b'
            
            # Case-insensitive replacement
            result = re.sub(pattern, canonical, result, flags=re.IGNORECASE)
        
        return result
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract all AI companion references from text.
        
        Identifies entity mentions and returns them in canonical form.
        Useful for auto-detection when entities are not explicitly provided.
        
        Args:
            text: Input text to scan for entity references
            
        Returns:
            List of canonical entity references found (deduplicated)
            
        Examples:
            >>> normalizer = EntityNormalizer()
            >>> normalizer.extract_entities("My rep and my bot are great")
            ['[AI_COMPANION]']
            >>> normalizer.extract_entities("I use Replika and Character.AI")
            ['[AI_COMPANION]']
        """
        if not text:
            return []
        
        found_entities: Set[str] = set()
        text_lower = text.lower()
        
        # Check for each variant
        for variant, canonical in self.entity_map.items():
            # Use word boundaries for matching
            pattern = r'\b' + re.escape(variant) + r'\b'
            
            if re.search(pattern, text_lower):
                found_entities.add(canonical)
        
        return list(found_entities)
    
    def has_entity_reference(self, text: str) -> bool:
        """
        Check if text contains any AI companion references.
        
        Args:
            text: Input text to check
            
        Returns:
            True if any entity reference is found, False otherwise
        """
        return len(self.extract_entities(text)) > 0
    
    def get_entity_positions(self, text: str) -> List[tuple]:
        """
        Get positions of all entity references in text.
        
        Args:
            text: Input text to scan
            
        Returns:
            List of (start, end, variant, canonical) tuples for each match
        """
        positions = []
        text_lower = text.lower()
        
        # Sort by position in text and length
        sorted_variants = sorted(self.entity_map.items(),
                                key=lambda x: len(x[0]),
                                reverse=True)
        
        for variant, canonical in sorted_variants:
            pattern = r'\b' + re.escape(variant) + r'\b'
            
            for match in re.finditer(pattern, text_lower):
                positions.append((
                    match.start(),
                    match.end(),
                    variant,
                    canonical
                ))
        
        # Sort by position in text
        positions.sort(key=lambda x: x[0])
        
        return positions

