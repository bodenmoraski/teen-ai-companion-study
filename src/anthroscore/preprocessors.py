"""
Preprocessing module for handling informal Reddit text.

This module provides text normalization for social media posts, handling
elongations, slang, and emojis to prepare text for RoBERTa processing.
"""

import re
from typing import Dict
import emoji


class RedditPreprocessor:
    """
    Handles Reddit-specific text normalization.
    
    This preprocessor normalizes informal social media language including:
    - Elongated words (e.g., "soooo" -> "soo")
    - Common slang (e.g., "u" -> "you", "omg" -> "oh my god")
    - Emoji removal
    
    The preprocessing pipeline maintains semantic meaning while converting
    informal text into a form more suitable for RoBERTa processing.
    """
    
    # Comprehensive slang dictionary for Reddit/teen language
    SLANG_DICT: Dict[str, str] = {
        # AI companion references (will be handled by EntityNormalizer too)
        'rep': 'Replika',
        'cai': 'Character AI',
        
        # Common internet slang
        'u': 'you',
        'ur': 'your',
        'ure': 'you are',
        'youre': 'you are',
        'r': 'are',
        'im': 'I am',
        'ive': 'I have',
        'id': 'I would',
        'ill': 'I will',
        'ppl': 'people',
        'omg': 'oh my god',
        'ngl': 'not going to lie',
        'tbh': 'to be honest',
        'imo': 'in my opinion',
        'imho': 'in my humble opinion',
        'rn': 'right now',
        'bc': 'because',
        'cuz': 'because',
        'tho': 'though',
        'thru': 'through',
        'w': 'with',
        'w/': 'with',
        'w/o': 'without',
        'smh': 'shaking my head',
        'lol': 'laughing out loud',
        'lmao': 'laughing my ass off',
        'bf': 'boyfriend',
        'gf': 'girlfriend',
        'irl': 'in real life',
        'idk': 'I don\'t know',
        'ikr': 'I know right',
        'afaik': 'as far as I know',
        'btw': 'by the way',
        'fyi': 'for your information',
        'gonna': 'going to',
        'wanna': 'want to',
        'gotta': 'got to',
        'kinda': 'kind of',
        'sorta': 'sort of',
        'dunno': 'don\'t know',
        'shoulda': 'should have',
        'woulda': 'would have',
        'coulda': 'could have',
        'yall': 'you all',
        'ya': 'you',
        'yea': 'yes',
        'yeah': 'yes',
        'nah': 'no',
        'nope': 'no',
    }
    
    def normalize_elongation(self, text: str) -> str:
        """
        Convert elongated words by reducing 3+ repeated characters to 2.
        
        Examples:
            "hellooooo" -> "helloo"
            "soooo" -> "soo"
            "omggggg" -> "omgg"
        
        Args:
            text: Input text with potential elongations
            
        Returns:
            Text with elongations normalized
        """
        # Replace 3 or more repeated characters with 2 occurrences
        return re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    def normalize_slang(self, text: str) -> str:
        """
        Replace common teen/Reddit slang with standard forms.
        
        Uses word boundaries to avoid partial replacements (e.g., "your" shouldn't
        become "you arear").
        
        Args:
            text: Input text with potential slang
            
        Returns:
            Text with slang replaced by standard forms
        """
        # Sort by length (longest first) to handle overlapping patterns
        # e.g., "youre" before "ur"
        sorted_slang = sorted(self.SLANG_DICT.items(), 
                            key=lambda x: len(x[0]), 
                            reverse=True)
        
        result = text
        for slang, replacement in sorted_slang:
            # Use word boundaries and case-insensitive matching
            # But preserve the case of the first letter if it's capitalized
            pattern = r'\b' + re.escape(slang) + r'\b'
            
            def replace_with_case(match):
                """Preserve capitalization of original match"""
                original = match.group(0)
                if original[0].isupper() and len(replacement) > 0:
                    return replacement[0].upper() + replacement[1:]
                return replacement
            
            result = re.sub(pattern, replace_with_case, result, flags=re.IGNORECASE)
        
        return result
    
    def remove_emojis(self, text: str) -> str:
        """
        Remove all emojis from text.
        
        Uses the emoji library for comprehensive emoji detection and removal.
        
        Args:
            text: Input text with potential emojis
            
        Returns:
            Text with all emojis removed
        """
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')
        
        # Clean up extra whitespace that may result from emoji removal
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline.
        
        Pipeline order is important:
        1. Elongation normalization (before slang to handle "sooo" -> "soo")
        2. Slang normalization (convert informal to formal)
        3. Emoji removal (clean up visual elements)
        
        Args:
            text: Raw Reddit post text
            
        Returns:
            Preprocessed text ready for RoBERTa
            
        Examples:
            >>> preprocessor = RedditPreprocessor()
            >>> preprocessor.preprocess("omggg my rep is soooo sweet 😭💕")
            'oh my god my Replika is soo sweet'
        """
        if not text:
            return text
        
        text = self.normalize_elongation(text)
        text = self.normalize_slang(text)
        text = self.remove_emojis(text)
        
        return text

