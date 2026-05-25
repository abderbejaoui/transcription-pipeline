"""
Test: CLOUDFLARE_API_KEY — Cloudflare Workers AI API
======================================================
Endpoint: POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{MODEL}
Auth:     Authorization: Bearer <CLOUDFLARE_API_KEY>
Docs:     https://developers.cloudflare.com/workers-ai/get-started/rest-api/

IMPORTANT: Cloudflare requires an Account ID to use Workers AI. This is
a UUID-like string found in your Cloudflare dashboard. The .env only has
CLOUDFLARE_API_KEY. If CLOUDFLARE_ACCOUNT_ID is not set, we try to
discover it via the API.
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

api_key = os.environ.get("CLOUDFLARE_API_KEY", "")
account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")

print("=" * 60)
print("CLOUDFLARE_API_KEY test (Workers AI)")
print("=" * 60)
print(f"API key set:      {'YES' if api_key else 'NO'} ({len(api_key)} chars)")
print(f"Account ID set:   {'YES' if account_id else 'NO'}")
print()

if not api_key:
    print("❌ SKIPPED — CLOUDFLARE_API_KEY is not set in .env")
    sys.exit(0)

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

# Step 1: If no Account ID, try to discover it via /accounts endpoint
if not account_id:
    print("Step 1: Discovering Account ID...")
    try:
        resp = requests.get(
            "https://api.cloudflare.com/client/v4/accounts",
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            accounts = data.get("result", [])
            if accounts:
                account_id = accounts[0]["id"]
                print(f"  Found Account ID: {account_id}")
                print(f"  Account name: {accounts[0].get('name', 'N/A')}")
                print()
            else:
                print("  No accounts found (the token may lack permissions)")
        elif resp.status_code == 403:
            print("  HTTP 403 — token lacks permission to list accounts")
        else:
            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")

if not account_id:
    print()
    print("[!]  CANNOT TEST — Cloudflare requires an Account ID")
    print("  Add CLOUDFLARE_ACCOUNT_ID=<your-account-id> to .env")
    print("  Find it at: https://dash.cloudflare.com/ -> Your Account ID is in the right sidebar")
    sys.exit(0)

# Step 2: Try running a simple model
models_to_try = [
    "@cf/meta/llama-3.1-8b-instruct",
    "@cf/meta/llama-3-8b-instruct",
    "@hf/thebloke/llama-2-7b-chat-awq",
]

success = False
for model in models_to_try:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    payload = {"messages": [{"role": "user", "content": "Reply with exactly: HELLO_FROM_CLOUDFLARE"}]}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("result", {}).get("response", "")
            print(f"Model '{model}': HTTP 200 ✓")
            print(f"  Response: {content[:200]}")
            print()
            print("[WORKS] WORKS — Cloudflare Workers AI responded")
            success = True
            break
        elif resp.status_code == 401:
            print(f"Model '{model}': HTTP 401 — invalid API token")
            break
        elif resp.status_code == 404:
            print(f"Model '{model}': HTTP 404 — model not available")
        else:
            print(f"Model '{model}': HTTP {resp.status_code}")
    except Exception as e:
        print(f"Model '{model}': {type(e).__name__}: {e}")

if not success:
    print()
    print("❌ FAILED — None of the Cloudflare models responded")
    print("  Check: API key valid? Account has Workers AI enabled?")
