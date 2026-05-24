import sys
from pathlib import Path
import json
import urllib.request

# Ensure workspace root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

url = "http://127.0.0.1:8000/api/v2/correct"
payload = {"transcript": "This is a test transcript for scorer path check.", "interactive": False}
req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
        print(resp.status)
        print(body)
except Exception as e:
    print("ERROR", repr(e))
