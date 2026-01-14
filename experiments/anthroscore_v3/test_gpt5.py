"""Quick test of GPT-5 API with correct parameters."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openai import OpenAI
from src.utils.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

test_texts = [
    "I cleared the cache and the app works fine now",  # Expected: 1
    "She really gets me. I love talking to her every day.",  # Expected: 4-5
]

print("=" * 60)
print("TESTING GPT-5-NANO WITH CORRECT PARAMETERS")
print("=" * 60)

for text in test_texts:
    prompt = f'''Rate the ANTHROPOMORPHIZATION level (1-5) in this comment about an AI companion:
1 = None (software/tool), 5 = Extreme (human relationship)

COMMENT: "{text}"

Reply with JSON: {{"score": <1-5>, "reasoning": "<brief>"}}'''

    params = {
        'model': 'gpt-5-nano',
        'messages': [{'role': 'user', 'content': prompt}],
        'response_format': {'type': 'json_object'},
        'max_completion_tokens': 1000  # GPT-5 uses reasoning tokens BEFORE output
        # NO temperature - GPT-5 only supports default
    }
    
    try:
        response = client.chat.completions.create(**params)
        print(f"\nText: {text[:50]}...")
        print(f"Full response object: {response}")
        print(f"Choices: {response.choices}")
        if response.choices:
            print(f"Message: {response.choices[0].message}")
            print(f"Content: {repr(response.choices[0].message.content)}")
            print(f"Finish reason: {response.choices[0].finish_reason}")
    except Exception as e:
        print(f"\nText: {text[:50]}...")
        print(f"ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
