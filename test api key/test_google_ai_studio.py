"""
Test: GOOGLE_AI_STUDIO_API_KEY — Gemini / Google AI Studio API
===============================================================
Note: This key IS the Gemini API key. Our pipeline expects GEMINI_API_KEY,
but the .env has it as GOOGLE_AI_STUDIO_API_KEY. Same credential, different name.

Endpoint: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}
Auth:     API key passed as query parameter (?key=)
Docs:     https://ai.google.dev/gemini-api/docs
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

# Check both possible env var names
api_key = os.environ.get("GOOGLE_AI_STUDIO_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

print("=" * 60)
print("GOOGLE_AI_STUDIO_API_KEY test (Gemini API)")
print("=" * 60)
print(f"API key set: {'YES' if api_key else 'NO'} ({len(api_key)} chars)")
print()

if not api_key:
    print("❌ SKIPPED — GOOGLE_AI_STUDIO_API_KEY is not set in .env")
    sys.exit(0)

models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

success = False
for model in models_to_try:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": "Reply with exactly: HELLO_FROM_GEMINI"}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 20},
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            text = ""
            for c in data.get("candidates", []):
                for p in c.get("content", {}).get("parts", []):
                    text += p.get("text", "")
            print(f"Model '{model}': HTTP 200 ✓")
            print(f"  Response: {text[:200]}")
            print()
            print("[*] WORKS — Google AI Studio key is valid as Gemini API key")
            success = True
            break
        elif resp.status_code == 403:
            print(f"Model '{model}': HTTP 403 — key may not have access to this model")
        elif resp.status_code == 404:
            print(f"Model '{model}': HTTP 404 — model not found (try different name)")
        else:
            print(f"Model '{model}': HTTP {resp.status_code}")
            if resp.status_code == 400:
                detail = resp.json().get("error", {}).get("message", "")
                print(f"  Detail: {detail}")
    except Exception as e:
        print(f"Model '{model}': {type(e).__name__}: {e}")

if not success:
    print()
    print("❌ FAILED — None of the Gemini models responded")
    print()
    print("Possible issues:")
    print("  - Key is invalid or expired (generate new one at https://aistudio.google.com/app/apikey)")
    print("  - The variable name GOOGLE_AI_STUDIO_API_KEY is non-standard")
    print("  - Our code expects GEMINI_API_KEY — key exists but under wrong name")
