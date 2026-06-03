#!/usr/bin/env python3
"""End-to-end proof that the HITL feedback loop works.

Teaches one new drug, then shows BOTH effects:
  (A) the EXACT taught mangle is now auto-corrected in the output, and
  (B) a DIFFERENT, never-taught mishearing of the same drug now retrieves
      the new term as a candidate -> proof it entered the retrieval dataset.
"""
import json, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
API = "http://127.0.0.1:8000"

TERM = "dapagliflozin"
TAUGHT_MANGLE = "داباغليفلوزين"          # the form the clinician confirms
NOVEL_MANGLE = "داباجليفلوزين"           # DIFFERENT spelling (ج not غ), never taught
SENT_TAUGHT = f"وصف الدكتور {TAUGHT_MANGLE} للسكري"
SENT_NOVEL = f"المريض ياخذ {NOVEL_MANGLE} يوميا"


def post(path, body):
    req = urllib.request.Request(f"{API}{path}", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path):
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def show(sent, tag):
    resp = post("/api/test-pipeline", {"transcript": sent, "case_id": tag})
    corr = [(c["span_text"], c["chosen"], c["path"]) for c in resp.get("corrections", [])]
    cand_terms = [c["term"] for rc in resp.get("retrieval_candidates", []) for c in rc["candidates"]]
    print(f"    input     : {sent!r}")
    print(f"    corrected : {resp.get('corrected')!r}")
    print(f"    corrections: {corr}")
    in_text = TERM.lower() in (resp.get("corrected") or "").lower()
    in_cands = TERM.lower() in [t.lower() for t in cand_terms]
    return in_text, in_cands


print("=" * 70)
print("STEP 1 — BEFORE teaching")
print("=" * 70)
print("  [exact mangle]")
b_txt_a, b_cand_a = show(SENT_TAUGHT, "b-taught")
print("  [novel mangle]")
b_txt_b, b_cand_b = show(SENT_NOVEL, "b-novel")

print("\n" + "=" * 70)
print(f"STEP 2 — clinician teaches '{TERM}' (alias '{TAUGHT_MANGLE}')")
print("=" * 70)
n0 = get("/api/lexicon")["count"]
post("/api/teach", {"term": TERM, "type": "drug", "aliases": [TAUGHT_MANGLE], "priority": 1.0})
n1 = get("/api/lexicon")["count"]
print(f"  corrector lexicon: {n0} -> {n1}")

print("\n" + "=" * 70)
print("STEP 3 — AFTER teaching (no restart)")
print("=" * 70)
print("  (A) exact taught mangle -> should AUTO-CORRECT in the text")
a_txt_a, a_cand_a = show(SENT_TAUGHT, "a-taught")
print("  (B) different, never-taught mangle -> term should now be RETRIEVED")
a_txt_b, a_cand_b = show(SENT_NOVEL, "a-novel")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"  (A) exact mangle corrected in text:   BEFORE={b_txt_a}  AFTER={a_txt_a}")
print(f"  (B) novel mangle retrieves new term:  BEFORE={b_cand_b}  AFTER={a_cand_b}")
ok = (not b_txt_a) and a_txt_a and (not b_cand_b) and a_cand_b
print(f"\n  HITL LOOP FULLY WORKING: {ok}")
sys.exit(0 if ok else 1)
