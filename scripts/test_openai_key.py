"""Test OpenAI API key configuration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
from src.utils.config import OPENAI_API_KEY
from src.demographics.llm_classifier import classify_age_llm

load_dotenv()

print("=" * 70)
print("OpenAI API Key Configuration Test")
print("=" * 70)

# Check raw environment variable
raw_key = os.getenv("OPENAI_API_KEY", "")
print(f"\n1. Raw environment variable:")
print(f"   Length: {len(raw_key)}")
if raw_key:
    print(f"   First 15 chars: {raw_key[:15]}...")
    has_quotes = raw_key.startswith('"') or raw_key.startswith("'") or raw_key.endswith('"') or raw_key.endswith("'")
    print(f"   Has quotes: {has_quotes}")

# Check processed key from config
print(f"\n2. Processed key (from config.py):")
print(f"   Length: {len(OPENAI_API_KEY)}")
if OPENAI_API_KEY:
    print(f"   First 15 chars: {OPENAI_API_KEY[:15]}...")
    print(f"   Starts with sk-: {OPENAI_API_KEY.startswith('sk-')}")
    print(f"   Looks valid: {len(OPENAI_API_KEY) > 20 and OPENAI_API_KEY.startswith('sk-')}")
else:
    print("   [WARNING] No API key found!")

# Test actual API call
print(f"\n3. Testing API call:")
if OPENAI_API_KEY and len(OPENAI_API_KEY) > 20:
    try:
        test_comments = ["I'm 16 years old and I love my AI companion"]
        result = classify_age_llm(test_comments, max_comments=1)
        print(f"   [SUCCESS] API call worked!")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   [ERROR] API call failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        if "api" in str(e).lower() or "key" in str(e).lower():
            print(f"   [INFO] This looks like an API key issue")
        elif "rate" in str(e).lower() or "limit" in str(e).lower():
            print(f"   [INFO] This is a rate limit issue (key is valid)")
else:
    print(f"   [SKIP] No valid API key to test")

print("\n" + "=" * 70)
print("Summary:")
if OPENAI_API_KEY and len(OPENAI_API_KEY) > 20 and OPENAI_API_KEY.startswith('sk-'):
    print("  [OK] API key appears to be configured correctly")
    print("  LLM classification will work in Phase 2")
else:
    print("  [WARNING] API key not configured or invalid")
    print("  LLM classification will be skipped in Phase 2")
    print("  This is OK - you'll still have self-declaration + community embeddings")
print("=" * 70)

