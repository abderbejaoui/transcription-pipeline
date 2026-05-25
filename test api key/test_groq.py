"""
Test: GROQ_API_KEY — Groq Cloud API
====================================
Endpoint: POST https://api.groq.com/openai/v1/chat/completions
Auth:     Authorization: Bearer <GROQ_API_KEY>
Format:   OpenAI-compatible
Docs:     https://console.groq.com/docs/api-reference
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

api_key = os.environ.get("GROQ_API_KEY", "")

print("=" * 60)
print("GROQ_API_KEY test")
print("=" * 60)
print(f"API key set: {'YES' if api_key else 'NO'} ({len(api_key)} chars)")
print()

if not api_key:
    print("[SKIPPED] GROQ_API_KEY is not set in .env")
    sys.exit(0)

# Try with known-good Groq models
models_to_try = [
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

success = False
for model in models_to_try:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: HELLO_FROM_GROQ"}],
        "max_tokens": 20,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f'Model "{model}": HTTP 200 OK')
            print(f'  Response: {content[:200]}')
            print()
            if "HELLO_FROM_GROQ" in content:
                print('[WORKS] Groq API responded correctly')
            else:
                print('[WORKS] Groq API responded (content may vary)')
            success = True
            break
        elif resp.status_code == 401:
            print(f'Model "{model}": HTTP 401 (unauthorized) - skipping remaining models')
            break
        else:
            print(f'Model "{model}": HTTP {resp.status_code} - trying next model...')
    except Exception as e:
        print(f'Model "{model}": {type(e).__name__} - trying next model...')

if not success:
    print()
    print('[FAILED] None of the Groq models responded successfully')
    print('  Check: API key valid? Account has credits/access?')
