"""
Test: OPENROUTER_API_KEY — OpenRouter API
==========================================
Endpoint: POST https://openrouter.ai/api/v1/chat/completions
Auth:     Authorization: Bearer <OPENROUTER_API_KEY>
Format:   OpenAI-compatible
Docs:     https://openrouter.ai/docs/api-reference
"""

import os, sys, json, requests
from pathlib import Path

# Load .env manually
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

api_key = os.environ.get("OPENROUTER_API_KEY", "")
model = os.environ.get("OPEN_ROUTER_MODEL_NAME", "") or os.environ.get("OPENROUTER_MODEL", "") or "openrouter/owl-alpha"

print("=" * 60)
print("OPENROUTER_API_KEY test")
print("=" * 60)
print(f"API key set:      {'YES' if api_key else 'NO'} ({len(api_key)} chars)")
print(f"Configured model:  {model}")
print()

if not api_key:
    print("[SKIPPED] OPENROUTER_API_KEY is not set in .env")
    sys.exit(0)

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

# Try the configured model first, then fallback models
models_to_try = [model, "openai/gpt-4o-mini", "openai/gpt-3.5-turbo", "anthropic/claude-3-haiku", "meta-llama/llama-3.1-8b-instruct"]

success = False
for mdl in models_to_try:
    payload = {
        "model": mdl,
        "messages": [{"role": "user", "content": "Reply with exactly: HELLO_FROM_OPENROUTER"}],
        "max_tokens": 20,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f'Model "{mdl}": HTTP 200 OK')
            print(f'  Response: {content[:200]}')
            print()
            print('[WORKS] OpenRouter API responded')
            success = True
            break
        elif resp.status_code == 401:
            print(f'Model "{mdl}": HTTP 401 - checking next...')
            break  # If 401 on first, key is invalid
        elif resp.status_code == 402:
            print(f'Model "{mdl}": HTTP 402 (insufficient credits/limits)')
        else:
            print(f'Model "{mdl}": HTTP {resp.status_code} - trying next...')
    except Exception as e:
        print(f"Model '{mdl}': {type(e).__name__}: {e}")

if not success:
    print()
    print('[FAILED] None of the OpenRouter models responded')
    print('  Check: API key valid? Account has credits?')
