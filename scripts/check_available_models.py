"""Check what OpenAI models are actually available via API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print("=" * 70)
print("Checking Available OpenAI Models")
print("=" * 70)

key = os.getenv("OPENAI_API_KEY", "")
if not key or len(key) < 20:
    print("\n[ERROR] OpenAI API key not configured")
    print("Cannot check available models without API key")
    sys.exit(1)

print(f"\nAPI Key configured: {key[:15]}...")
print("\nQuerying OpenAI API for available models...\n")

try:
    client = OpenAI(api_key=key)
    models = client.models.list()
    
    # Filter for chat/completion models
    chat_models = []
    for model in models.data:
        model_id = model.id.lower()
        # Look for GPT models
        if any(x in model_id for x in ['gpt', 'o1', 'o3']):
            chat_models.append(model.id)
    
    chat_models.sort()
    
    print("Available Chat/Completion Models:")
    print("-" * 70)
    for model in chat_models:
        print(f"  {model}")
    
    print("\n" + "=" * 70)
    print("Recommended Models for Classification:")
    print("=" * 70)
    
    # Check for newer models
    has_gpt4o_mini = any('gpt-4o-mini' in m.lower() for m in chat_models)
    has_gpt41_nano = any('gpt-4.1' in m.lower() or 'gpt-4.1-nano' in m.lower() for m in chat_models)
    has_gpt5 = any('gpt-5' in m.lower() for m in chat_models)
    has_o1 = any('o1' in m.lower() for m in chat_models)
    has_o3 = any('o3' in m.lower() for m in chat_models)
    
    print(f"\nCurrent model (gpt-4o-mini): {'Available' if has_gpt4o_mini else 'NOT AVAILABLE'}")
    print(f"GPT-4.1 Nano: {'Available' if has_gpt41_nano else 'NOT AVAILABLE'}")
    print(f"GPT-5 series: {'Available' if has_gpt5 else 'NOT AVAILABLE'}")
    print(f"O1 series: {'Available' if has_o1 else 'NOT AVAILABLE'}")
    print(f"O3 series: {'Available' if has_o3 else 'NOT AVAILABLE'}")
    
    if has_gpt41_nano or has_gpt5:
        print("\n[INFO] Newer models found! Consider updating configuration.")
    elif has_o1 or has_o3:
        print("\n[INFO] O1/O3 models found, but these are for reasoning tasks, not classification.")
    else:
        print("\n[INFO] gpt-4o-mini appears to be the best available option for classification.")
    
except Exception as e:
    print(f"\n[ERROR] Failed to query API: {e}")
    print("\nThis might mean:")
    print("  1. API key is invalid")
    print("  2. Network issue")
    print("  3. API endpoint changed")
    import traceback
    traceback.print_exc()

