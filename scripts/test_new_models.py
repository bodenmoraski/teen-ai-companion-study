"""Test newer OpenAI models for classification."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

key = os.getenv("OPENAI_API_KEY", "")
if not key or len(key) < 20:
    print("API key not configured")
    sys.exit(1)

client = OpenAI(api_key=key)

# Test models
models_to_test = [
    "gpt-4.1-nano",
    "gpt-5-nano", 
    "gpt-5-mini",
    "gpt-4o-mini"  # Current
]

test_prompt = """Analyze these Reddit comments and estimate age bucket.

Comments:
- I'm 16 and I love my AI companion
- Just graduated high school, excited for college
- My parents don't understand technology

Age buckets: 13-18, 19-25, 26-40, 41-60, 61-80

Respond with JSON: {"age_bucket": "...", "confidence": 0.0-1.0, "reasoning": "..."}"""

print("=" * 70)
print("Testing Newer OpenAI Models")
print("=" * 70)

results = {}

for model in models_to_test:
    print(f"\nTesting {model}...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
            max_tokens=200
        )
        
        result = json.loads(response.choices[0].message.content)
        results[model] = {
            "success": True,
            "result": result,
            "tokens": response.usage.total_tokens if hasattr(response, 'usage') else None
        }
        print(f"  [SUCCESS] {result.get('age_bucket')} (confidence: {result.get('confidence')})")
        
    except Exception as e:
        results[model] = {
            "success": False,
            "error": str(e)
        }
        print(f"  [ERROR] {e}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

for model, result in results.items():
    if result["success"]:
        print(f"{model}: ✓ Works")
    else:
        print(f"{model}: ✗ Failed - {result['error']}")

# Recommendation
working_models = [m for m, r in results.items() if r["success"]]
if working_models:
    print(f"\nRecommended: {working_models[0]} (newest working model)")
    print("Update config.py with this model name")

