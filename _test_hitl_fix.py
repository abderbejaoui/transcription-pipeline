"""Test the HITL fix: verify that /api/teach invalidates the corrector cache."""
import json, sys, urllib.request, urllib.error

# Force UTF-8 for stdout
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

API = "http://127.0.0.1:8000"

def api_post(path, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{API}{path}", data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def api_get(path):
    req = urllib.request.Request(f"{API}{path}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

passed = 0
failed = 0
errors = []

def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [+] {name}")
    else:
        failed += 1
        errors.append(f"{name}: {detail}")
        print(f"  [X] {name} -- {detail}")

# ── Step 1: Test full clinical transcript ─────────────────────────────────
print("=" * 60)
print("STEP 1: Full clinical transcript (baseline)")
print("=" * 60)

transcript = (
    "المريض يشتكي من سداع شديد و الدغط مرتفع، عنده هستوري من دايابيتس. "
    "أعطيناه اسبرين و دولبران، وعملنا فحص للقلب. "
    "يوجد شورتنس اوف بريث خفيف. "
    "التهب في الرئة مو مؤكد، نحتاج إكس ري."
)
result = api_post("/api/correct", {"text": transcript})
corrected = result["corrected_text"]
auto_corrections = result["auto_corrections"]

print(f"  Auto corrections: {len(auto_corrections)}")
for ac in auto_corrections:
    print(f"    {ac['original']!r} -> {ac['corrected']!r}")

check("سداع -> صداع", "صداع" in corrected or any("صداع" in str(ac) for ac in auto_corrections), f"corrected={corrected!r}")
check("الدغط -> الضغط", "الضغط" in corrected, f"corrected={corrected!r}")
check("هستوري -> history", "history" in corrected.lower(), f"corrected={corrected!r}")
check("دايابيتس -> diabetes", "diabetes" in corrected.lower(), f"corrected={corrected!r}")
check("اسبرين -> aspirin", "aspirin" in corrected.lower(), f"corrected={corrected!r}")
check("شورتنس اوف بريث -> shortness of breath", "shortness of breath" in corrected.lower(), f"corrected={corrected!r}")
check("التهب -> التها", "التها" in corrected, f"corrected={corrected!r}")
check("القلب preserved", "القلب" in corrected, f"corrected={corrected!r}")
check("الرئة preserved", "الرئة" in corrected, f"corrected={corrected!r}")
check("شديد preserved", "شديد" in corrected, f"corrected={corrected!r}")

# ── Step 2: Teach a new term ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Teach Doliprane via /api/teach")
print("=" * 60)

before = api_get("/api/lexicon")
n_before = before["count"]

teach_result = api_post("/api/teach", {
    "term": "Doliprane",
    "type": "drug",
    "aliases": [],
    "priority": 1.0,
})
check("Teach returned ok", teach_result.get("ok") is True, str(teach_result))

after = api_get("/api/lexicon")
match = [e for e in after["entries"] if e["term"] == "Doliprane"]
check(f"Doliprane in lexicon ({after['count']} entries)", len(match) >= 1, f"entries={[e['term'] for e in after['entries'][-5:]]}")

# ── Step 3: Re-run pipeline ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Re-run pipeline after teaching")
print("=" * 60)

result2 = api_post("/api/correct", {"text": transcript})
corrected2 = result2["corrected_text"]
auto_corrections2 = result2["auto_corrections"]

print(f"  Auto corrections: {len(auto_corrections2)}")
for ac in auto_corrections2:
    print(f"    {ac['original']!r} -> {ac['corrected']!r}")

check("Still corrects core errors", "صداع" in corrected2 or "الضغط" in corrected2, f"corrected={corrected2!r}")
check("Still التهب -> التها", "التها" in corrected2, f"corrected={corrected2!r}")

# ── Step 4: Teach another term ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Teach Metformin with alias, verify second invalidation")
print("=" * 60)

teach_result2 = api_post("/api/teach", {
    "term": "Metformin",
    "type": "drug",
    "aliases": ["metformina"],
    "priority": 1.0,
})
check("Teach Metformin ok", teach_result2.get("ok") is True, str(teach_result2))

result3 = api_post("/api/correct", {"text": "patient takes metformina 500 mg"})
corrected3 = result3["corrected_text"]
check("metformina -> Metformin via alias", 
      "Metformin" in corrected3, f"corrected={corrected3!r}")

# ── Step 5: learn_from_edit ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Test /api/learn_from_edit cache invalidation")
print("=" * 60)

learn_result = api_post("/api/learn_from_edit", {
    "raw_text": "patient has gastreitis",
    "corrected_text": "patient has Gastritis",
    "type": "term",
})
check("learn_from_edit ok", learn_result.get("ok") is True, str(learn_result)[:200])

result4 = api_post("/api/correct", {"text": "diagnosed with gastreitis"})
corrected4 = result4["corrected_text"]
check("gastreitis -> Gastritis via learn", 
      "Gastritis" in corrected4, f"corrected={corrected4!r}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed+failed}")
print("=" * 60)
if failed > 0:
    print("FAILURES:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL PASSED")
