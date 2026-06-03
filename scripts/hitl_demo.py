#!/usr/bin/env python3
"""Demonstrate the HITL feedback loop end-to-end.

A clinician dictates a drug ('dapagliflozin') that is NOT in the
candidate-retrieval dataset. We show:
  BEFORE  -> the term is absent from retrieval_candidates, not corrected.
  TEACH   -> clinician adds it via /api/teach (the real HITL action).
  AFTER   -> the term now appears in retrieval_candidates and is corrected.
"""
import json, sys, urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
API = "http://127.0.0.1:8000"

# Arabic mangle of "dapagliflozin" in a clinical Gulf-Arabic sentence.
TRANSCRIPT = "وصف الدكتور داباغليفلوزين للسكري من النوع الثاني"
TERM = "dapagliflozin"
MANGLE = "داباغليفلوزين"


def post(path, body):
    req = urllib.request.Request(
        f"{API}{path}", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path):
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def probe(tag):
    resp = post("/api/test-pipeline", {"transcript": TRANSCRIPT, "case_id": tag})
    cands = []
    for rc in resp.get("retrieval_candidates", []):
        cands.append((rc["span_text"], [c["term"] for c in rc["candidates"][:4]]))
    corr = [(c["span_text"], c["chosen"], c["path"]) for c in resp.get("corrections", [])]
    has_term = any(TERM.lower() in t.lower()
                   for _, terms in cands for t in terms)
    print(f"  corrected   : {resp.get('corrected')!r}")
    print(f"  corrections : {corr}")
    print(f"  retrieval   : {json.dumps(cands, ensure_ascii=False)}")
    print(f"  '{TERM}' present as a retrieval candidate? {has_term}")
    return has_term, resp.get("corrected")


print("=" * 66)
print("BEFORE — term not yet taught")
print("=" * 66)
before_has, before_corr = probe("DEMO-before")

print("\n" + "=" * 66)
print(f"TEACH — clinician adds '{TERM}' via /api/teach (alias '{MANGLE}')")
print("=" * 66)
n0 = get("/api/lexicon")["count"]
post("/api/teach", {"term": TERM, "type": "drug", "aliases": [MANGLE], "priority": 1.0})
n1 = get("/api/lexicon")["count"]
print(f"  lexicon entries: {n0} -> {n1}")

print("\n" + "=" * 66)
print("AFTER — same input, re-run (no restart)")
print("=" * 66)
after_has, after_corr = probe("DEMO-after")

print("\n" + "=" * 66)
print("VERDICT")
print("=" * 66)
print(f"  candidate '{TERM}' in retrieval:  BEFORE={before_has}  AFTER={after_has}")
print(f"  output changed by teaching:       {before_corr != after_corr}")
ok = (not before_has) and after_has
print(f"  HITL feedback loop WORKING:       {ok}")
sys.exit(0 if ok else 1)
