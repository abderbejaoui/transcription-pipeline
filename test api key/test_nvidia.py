"""
Test: NVIDIA_API_KEY — NVIDIA AI Foundation / NVIDIA Cloud Endpoints API
=========================================================================
Endpoint: POST https://integrate.api.nvidia.com/v1/chat/completions
Auth:     Authorization: Bearer <NVIDIA_API_KEY>
Format:   OpenAI-compatible
Docs:     https://build.nvidia.com/ (model catalog)
          https://docs.nvidia.com/nim/large-language-models/latest/api-reference.html
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

api_key = os.environ.get("NVIDIA_API_KEY", "")

print("=" * 60)
print("NVIDIA_API_KEY test")
print("=" * 60)
print(f"API key set: {'YES' if api_key else 'NO'} ({len(api_key)} chars)")
print()

if not api_key:
    print("[SKIPPED] NVIDIA_API_KEY is not set in .env")
    sys.exit(0)

url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Common NVIDIA-hosted models
models_to_try = [
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama3-70b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "mistralai/mixtral-8x22b-instruct",
]

success = False
for model in models_to_try:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: HELLO_FROM_NVIDIA"}],
        "max_tokens": 20,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f'Model "{model}": HTTP 200 OK')
            print(f'  Response: {content[:200]}')
            print()
            print('[WORKS] NVIDIA API responded')
            success = True
            break
        elif resp.status_code == 401:
            print(f'Model "{model}": HTTP 401 (unauthorized) - stopping')
            break
        elif resp.status_code == 402:
            print(f'Model "{model}": HTTP 402 (payment required / no credits)')
        elif resp.status_code == 403:
            print(f'Model "{model}": HTTP 403 (key lacks access to this model)')
        else:
            print(f'Model "{model}": HTTP {resp.status_code} - trying next...')
    except Exception as e:
        print(f"Model '{model}': {type(e).__name__}: {e}")

if not success:
    print()
    print('[FAILED] None of the NVIDIA models responded')
    print('  Check: API key valid? Account has credits? Model name correct?')
