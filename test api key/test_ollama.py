"""
Test: OLLAMA_API_KEY — Ollama Cloud API
=========================================
Endpoint: POST https://ollama.com/api/chat
Auth:     Authorization: Bearer <OLLAMA_API_KEY>
Model:    glm-5.1:cloud
Docs:     https://docs.ollama.com/api
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

api_key = os.environ.get("OLLAMA_API_KEY", "")
model = os.environ.get("OLLAMA_MODEL_NAME", "") or "llama3.1"

print("=" * 60)
print("OLLAMA_API_KEY test")
print("=" * 60)
print(f"API key set:      {'YES' if api_key else 'NO'} ({len(api_key)} chars)")
print(f"Model:            {model}")
print()

if not api_key:
    print("[SKIPPED] OLLAMA_API_KEY is not set in .env")
    sys.exit(0)

url = "https://ollama.com/api/chat"
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Say exactly: HELLO_FROM_OLLAMA"}],
    "stream": False,
}
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

try:
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    print(f"HTTP {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        msg = data.get("message", {})
        content = msg.get("content", "")
        print(f"Response: {content[:200]}")
        print()
        if "HELLO_FROM_OLLAMA" in content:
            print('[WORKS] Ollama API responded correctly')
        else:
            print('[WARN] RESPONDED but content unexpected - may still work')
    else:
        print(f"Response body: {resp.text[:500]}")
        print()
        if resp.status_code == 401:
            print("❌ FAILED — HTTP 401: Invalid API key (unauthorized)")
        elif resp.status_code == 404:
            print(f"❌ FAILED — HTTP 404: Model '{model}' not found or endpoint wrong")
        else:
            print(f"❌ FAILED — HTTP {resp.status_code}")
except requests.exceptions.Timeout:
    print("❌ FAILED — Request timed out (30s)")
except requests.exceptions.ConnectionError:
    print("❌ FAILED — Could not connect to ollama.com (check internet or proxy)")
except Exception as e:
    print(f"❌ FAILED — {type(e).__name__}: {e}")
