"""
Test: GITHUB_API_KEY — GitHub Personal Access Token
=====================================================
Endpoint: GET https://api.github.com/user
Auth:     Authorization: Bearer <GITHUB_API_KEY>
Docs:     https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api
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

api_key = os.environ.get("GITHUB_API_KEY", "")

print("=" * 60)
print("GITHUB_API_KEY test")
print("=" * 60)
print(f"API key set: {'YES' if api_key else 'NO'} ({len(api_key)} chars)")
print()

if not api_key:
    print("❌ SKIPPED — GITHUB_API_KEY is not set in .env")
    sys.exit(0)

# Test 1: Get authenticated user
print("Test 1: GET https://api.github.com/user")
try:
    resp = requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    print(f"  HTTP {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        login = data.get("login", "unknown")
        name = data.get("name", "")
        plan = data.get("plan", {})
        print(f"  User: {login} ({name})")
        print(f"  Scopes: {resp.headers.get('X-OAuth-Scopes', 'N/A')}")
        print()
        print("[WORKS] WORKS — GitHub API key is valid")
    elif resp.status_code == 401:
        print(f"  Body: {resp.text[:300]}")
        print()
        print("❌ FAILED — HTTP 401: Invalid/expired token")
    elif resp.status_code == 403:
        print(f"  Body: {resp.text[:300]}")
        print()
        print("❌ FAILED — HTTP 403: Token valid but lacks permissions")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")
    print()
    print("❌ FAILED — Could not connect to GitHub API")
