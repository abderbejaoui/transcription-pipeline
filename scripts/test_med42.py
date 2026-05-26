"""Evaluate Llama3-Med42-70B as a medical transcript corrector.

Sends realistic doctor-patient conversations (Gulf Arabic + code-switched
English) where the ASR mangled drug / medical names into similar-sounding
Arabic gibberish. Asks Med42 to (a) flag the suspect spans and (b) propose
the correct medical term.

Run on the DGX (where the model is hosted by Ollama):
    python scripts/test_med42.py

Or remote, via SSH tunnel:
    ssh -L 11434:localhost:11434 abder@spark-a6f4
    OLLAMA_URL=http://localhost:11434/api/chat python scripts/test_med42.py

Usage flags:
    --model NAME    Ollama model tag (default: Med42-70B)
    --verbose       Print the full raw LLM response
    --case N        Run only case index N (0-based)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "hf.co/mradermacher/Llama3-Med42-70B-GGUF:Q5_K_M"
DEFAULT_OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", "http://localhost:11434/api/chat"
)


# ---------------------------------------------------------------------------
# Test cases — realistic ASR mistakes on Gulf Arabic + medical English
# ---------------------------------------------------------------------------
#
# Each case is a *transcript as the ASR produced it*. The English drug /
# medical names are mangled into Arabic gibberish — the kind of failure
# our Gulf-LoRA Qwen3-ASR actually makes.  The pipeline's job is to:
#   1. spot that something is medically suspicious in the span
#   2. propose what it probably should be
#
# `expected` lists the correct medical terms that SHOULD appear in the
# corrected transcript. We don't grade on exact full sentences, only on
# whether the correct medical terms came back.
# ---------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    transcript: str
    expected: List[str]
    notes: str = ""


CASES: List[Case] = [
    Case(
        name="paracetamol_dose",
        transcript=(
            "Doctor: شخوش حال أبو سعد، شو اشتكي اليوم؟\n"
            "Patient: عندي صداع وحرارة من امبارح.\n"
            "Doctor: طيب خود فرنسي تمان خمس مية ملليجرام كل ست ساعات لمدة ثلاث ايام."
        ),
        expected=["paracetamol"],
        notes="فرنسي تمان -> paracetamol (very common ASR mistake)",
    ),
    Case(
        name="doliprane_brand",
        transcript=(
            "Patient: الدكتور قال لي اخذ دولي فران للحمى.\n"
            "Doctor: ايوه، خذه مع المي ولا تاخذه على معدة فاضية."
        ),
        expected=["doliprane"],
        notes="دولي فران -> doliprane",
    ),
    Case(
        name="if_all_gone_efferalgan",
        transcript=(
            "Patient: اعطاني الصيدلي علاج اسمه اف اول قن او اف يور قان.\n"
            "Doctor: تقصد افرلجن؟ نعم هذا للحرارة والالم."
        ),
        expected=["efferalgan"],
        notes="if all gone -> efferalgan, the classic English mishearing",
    ),
    Case(
        name="amoxicillin_infection",
        transcript=(
            "Doctor: عندك التهاب في الحلق، لازم تاخذ اموكسي سيلين.\n"
            "Patient: امسي سيلين؟ كم حبه في اليوم؟\n"
            "Doctor: ثلاث مرات في اليوم لمدة اسبوع."
        ),
        expected=["amoxicillin"],
        notes="اموكسي سيلين / امسي سيلين -> amoxicillin",
    ),
    Case(
        name="ibuprofen_bad",
        transcript=(
            "Patient: ظهري يوجعني بعد التمرين.\n"
            "Doctor: خذ ايبو بروفان او اي بوبروفان اربع مية ملليجرام مع الاكل."
        ),
        expected=["ibuprofen"],
        notes="ايبو بروفان -> ibuprofen",
    ),
    Case(
        name="metformin_diabetes",
        transcript=(
            "Patient: سكري عالي اليوم وصل لثلاث مية.\n"
            "Doctor: لازم تنتظم على ميت فور مين خمس مية ملليجرام مرتين في اليوم."
        ),
        expected=["metformin"],
        notes="ميت فور مين -> metformin",
    ),
    Case(
        name="ventolin_inhaler",
        transcript=(
            "Patient: ابني عنده ربو وضيق نفس.\n"
            "Doctor: خليه يستخدم بخاخ فن تولين وقت النوبة."
        ),
        expected=["ventolin"],
        notes="فن تولين -> ventolin",
    ),
    Case(
        name="augmentin_strep",
        transcript=(
            "Doctor: عنده التهاب لوزتين شديد.\n"
            "Patient: العلاج؟\n"
            "Doctor: اوغ من تين شراب اطفال جرعتين في اليوم لعشر ايام."
        ),
        expected=["augmentin"],
        notes="اوغ من تين -> augmentin",
    ),
    Case(
        name="multi_drug_prescription",
        transcript=(
            "Doctor: الوصفة: فرنسي تمان لو في الم، اوميه بريزول صباحا قبل الاكل، "
            "وحبة فيتامين دي اسبوعيا. لا تنسى."
        ),
        expected=["paracetamol", "omeprazole", "vitamin d"],
        notes="three drugs in one sentence",
    ),
    Case(
        name="vital_signs_garbled",
        transcript=(
            "Nurse: الضغط مية وخمس وستين على ثمانين، النبض ثمانين، الحرارة "
            "ثمان وثلاثين، اوه اكسي ميتر تسع وتسعين."
        ),
        expected=["oximeter"],
        notes="اوه اكسي ميتر -> oximeter; vitals should stay in Arabic numerals",
    ),
]


# ---------------------------------------------------------------------------
# Med42 prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Llama3-Med42, an expert medical-conversation editor. The text "
    "you receive is an automatic-speech-recognition transcript of a Gulf "
    "Arabic doctor-patient consultation with code-switched English medical "
    "and brand-name terms. The ASR very often mangles English drug names "
    "into Arabic gibberish (e.g. 'paracetamol' becomes 'فرنسي تمان', "
    "'efferalgan' becomes 'اف اول قن'). Your job:\n"
    "\n"
    "1. Read the transcript and find every span that LOOKS like a mangled "
    "medical, pharmaceutical, or anatomical term.\n"
    "2. For each one, output the original ASR span and the most likely "
    "correct medical term (Latin spelling for drugs, English for procedures, "
    "Arabic only when the term is genuinely Arabic).\n"
    "3. Be biased toward fixing — if a span is medically plausible at all, "
    "flag it. Better to flag a near-miss than to miss a real drug name.\n"
    "4. Use your medical knowledge: think 'what drug is this dose / "
    "indication consistent with?'. Dose+indication often pins the drug "
    "uniquely.\n"
    "5. Do NOT change non-medical Arabic words. Do NOT translate the "
    "transcript. Do NOT change numbers (doses, durations, vitals).\n"
    "\n"
    "OUTPUT — strict JSON ONLY, no prose:\n"
    "{\n"
    '  "corrections": [\n'
    '    {"original": "<ASR span>", "corrected": "<medical term>", '
    '"confidence": <0.0-1.0>, "reason": "<one short clinical reason>"}\n'
    "  ],\n"
    '  "corrected_transcript": "<full transcript with corrections applied>"\n'
    "}"
)


def call_med42(transcript: str, model: str, timeout: float = 180.0) -> Dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
    }
    t0 = time.time()
    req = urllib.request.Request(
        DEFAULT_OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    content = data.get("message", {}).get("content", "")
    # Strip non-JSON wrappers if any
    text = content.strip()
    if not (text.startswith("{") and text.endswith("}")):
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = {"corrections": [], "corrected_transcript": "", "_parse_error": True}
    obj["_elapsed_seconds"] = elapsed
    obj["_raw"] = content
    return obj


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def grade(case: Case, result: Dict[str, Any]) -> Dict[str, Any]:
    """For each expected term, check if it appears (case-insensitively) in
    either the corrections list OR the corrected_transcript."""
    expected_norm = [_norm(e) for e in case.expected]
    corrected = _norm(result.get("corrected_transcript", "") or "")
    corrections = result.get("corrections", []) or []
    corrections_text = " ".join(
        _norm(c.get("corrected", "") or "") for c in corrections
    )

    found: List[str] = []
    missed: List[str] = []
    for term in expected_norm:
        if term in corrected or term in corrections_text:
            found.append(term)
        else:
            missed.append(term)

    recall = len(found) / len(expected_norm) if expected_norm else 1.0
    return {
        "case": case.name,
        "expected": case.expected,
        "found": found,
        "missed": missed,
        "recall": recall,
        "n_corrections_proposed": len(corrections),
        "elapsed_seconds": result.get("_elapsed_seconds", 0.0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--case", type=int, default=None, help="run only case N")
    args = p.parse_args()

    cases = CASES if args.case is None else [CASES[args.case]]
    print(f"Testing model: {args.model}")
    print(f"Endpoint     : {DEFAULT_OLLAMA_URL}")
    print(f"Cases        : {len(cases)}")
    print("=" * 70)

    summary: List[Dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case.name}")
        if case.notes:
            print(f"        {case.notes}")
        print("-" * 70)
        print(case.transcript)
        print("-" * 70)
        try:
            result = call_med42(case.transcript, args.model)
        except Exception as exc:
            print(f"  ERROR: {exc!r}")
            summary.append({
                "case": case.name, "expected": case.expected, "found": [],
                "missed": case.expected, "recall": 0.0, "error": repr(exc),
            })
            continue

        if args.verbose:
            print("RAW LLM RESPONSE:")
            print(result.get("_raw", ""))
            print("-" * 70)

        print("CORRECTIONS PROPOSED:")
        for c in result.get("corrections", []):
            print(f"  • {c.get('original', '?')!r:>30}  ->  "
                  f"{c.get('corrected', '?')!r:<25}  "
                  f"(conf={c.get('confidence', '?')}, "
                  f"reason={c.get('reason', '')[:60]!r})")

        print("CORRECTED TRANSCRIPT:")
        print(f"  {result.get('corrected_transcript', '')!r}")

        graded = grade(case, result)
        summary.append(graded)
        status = "✓" if not graded["missed"] else "✗"
        print(f"\nGRADE: {status} recall={graded['recall']:.2f}  "
              f"({len(graded['found'])}/{len(graded['expected'])})  "
              f"t={graded['elapsed_seconds']:.1f}s")
        if graded["missed"]:
            print(f"       missed: {graded['missed']}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n = len(summary)
    total_expected = sum(len(s["expected"]) for s in summary)
    total_found = sum(len(s["found"]) for s in summary)
    avg_recall = (sum(s["recall"] for s in summary) / n) if n else 0.0
    avg_time = (sum(s.get("elapsed_seconds", 0.0) for s in summary) / n) if n else 0.0

    print(f"  Cases tested        : {n}")
    print(f"  Medical terms found : {total_found}/{total_expected}")
    print(f"  Average recall      : {avg_recall:.3f}")
    print(f"  Average latency     : {avg_time:.1f}s per case")
    print()
    print(f"{'Case':<28} {'recall':>7} {'time':>8} {'verdict':>10}")
    for s in summary:
        verdict = "PASS" if not s.get("missed") else "FAIL"
        if "error" in s:
            verdict = "ERROR"
        print(f"  {s['case']:<26} {s['recall']:>6.2f}  "
              f"{s.get('elapsed_seconds', 0.0):>6.1f}s  {verdict:>10}")

    print()
    if avg_recall >= 0.90:
        print("VERDICT: Reliable for correction (≥90% recall). USE IT.")
    elif avg_recall >= 0.70:
        print("VERDICT: Useful but inconsistent (70-90% recall). Pair with rule-based "
              "phonetic backup.")
    elif avg_recall >= 0.40:
        print("VERDICT: Hit-and-miss. Not reliable as the only corrector.")
    else:
        print("VERDICT: Poor performance. Try a different model.")
    return 0 if avg_recall >= 0.40 else 1


if __name__ == "__main__":
    sys.exit(main())
