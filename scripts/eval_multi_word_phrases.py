"""Evaluate multi-word Arabic→English medical phrase detection.

Runs every known Arabic phrase variant through the MedicalCorrector and
reports per-case pass/fail, aggregate precision/recall, and false-positive
rate on negative controls.

Usage
-----
  python -m scripts.eval_multi_word_phrases
  python -m scripts.eval_multi_word_phrases --verbose  (show matched spans)
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force UTF-8 for stdout/stderr on Windows (cp1252 can't print Arabic)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure the app package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.correction import MedicalCorrector


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    """A single evaluation case.

    Attributes:
        transcript: Input transcript text.
        expected_phrases: Set of English phrases that MUST appear in the
            corrected text (case-insensitive check).
        disallowed_phrases: Set of English phrases that MUST NOT appear
            in the corrected text (e.g. wrong corrections).
        must_not_correct_original: If True, the original Arabic text must
            NOT be fully replaced — at least some original tokens remain.
            Used for negative/false-positive checks.
        label: Short human-readable name.
        is_positive: Whether this is a positive case (expects correction)
            or a negative case (expects no correction).
    """
    transcript: str
    expected_phrases: set = field(default_factory=set)
    disallowed_phrases: set = field(default_factory=set)
    must_not_correct_original: bool = False
    label: str = ""
    is_positive: bool = True


# --- Positive cases: phrases that SHOULD be corrected ---

POSITIVE_CASES: List[EvalCase] = [
    # === Blood sugar (بلاد شوجر) ===
    EvalCase(
        label="blood_sugar_plain",
        transcript="المريض عنده بلاد شوجر مرتفع",
        expected_phrases={"blood sugar"},
    ),
    EvalCase(
        label="blood_sugar_in_context",
        transcript="وعنده بلاد شوجر و بلد برشر من حوالي 10 سنين",
        expected_phrases={"blood sugar", "blood pressure"},
    ),
    EvalCase(
        label="blood_sugar_followup",
        transcript="تم متابعة بلاد شوجر كل 4 ساعات",
        expected_phrases={"blood sugar"},
    ),

    # === Blood pressure (بلد برشر, بلد برشر) ===
    EvalCase(
        label="blood_pressure_plain",
        transcript="بلد برشر 160 على 100",
        expected_phrases={"blood pressure"},
    ),
    EvalCase(
        label="blood_pressure_with_numbers",
        transcript="بلد برشر 160 علي 100",
        expected_phrases={"blood pressure"},
    ),
    EvalCase(
        label="blood_pressure_isolated",
        transcript="عنده بلد برشر عالي",
        expected_phrases={"blood pressure"},
    ),
    EvalCase(
        label="blood_pressure_variant_bld_prsh",
        transcript="عندي بلد برشر مرتفع جدا",
        expected_phrases={"blood pressure"},
    ),

    # === Shortness of breath (شورتنس اوف بريث) ===
    EvalCase(
        label="sob_3token_awf",
        transcript="يعاني من شورتنس اوف بريث شديد",
        expected_phrases={"shortness of breath"},
    ),
    EvalCase(
        label="sob_3token_of",
        transcript="شورتنس اوف بريث مستمر",
        expected_phrases={"shortness of breath"},
    ),
    EvalCase(
        label="sob_short_text",
        transcript="شورتنس اوف بريث",
        expected_phrases={"shortness of breath"},
    ),

    # === Chest pain (chist bain / chest bain) ===
    EvalCase(
        label="chest_pain_arabic_chist",
        transcript="يشعر المريض ب chist bain شديد",
        expected_phrases={"chest pain"},
    ),
    EvalCase(
        label="chest_pain_arabic_chest",
        transcript="يشعر ب chest bain",
        expected_phrases={"chest pain"},
    ),

    # === Antiplatelet therapy ===
    EvalCase(
        label="antiplatelet_anty",
        transcript="يحتاج anty platalet therapy",
        expected_phrases={"antiplatelet therapy"},
    ),

    # === Cardiac enzymes ===
    EvalCase(
        label="cardiac_enzymes",
        transcript="تم فحص cardiac anzymes",
        expected_phrases={"cardiac enzymes"},
    ),

    # === Oxygen saturation ===
    EvalCase(
        label="oxygen_saturation_aksyn",
        transcript="aksyn satoration 98%",
        expected_phrases={"oxygen saturation"},
    ),
    EvalCase(
        label="oxygen_saturation_full",
        transcript="oxygen satoration طبيعي",
        expected_phrases={"oxygen saturation"},
    ),

    # === Ischemic changes ===
    EvalCase(
        label="ischemic_changes",
        transcript="تظهر ischemic chenges في التخطيط",
        expected_phrases={"ischemic changes"},
    ),

    # === Short Arabic transliterations in context ===
    # Note: 'history', 'diabetes', 'hypertension' are not in the
    # medical_lexicon.jsonl, so they won't match as single-word corrections.
    # They're listed here as multi-word context tests only.
]

# --- Negative cases: phrases that should NOT be corrected ---

NEGATIVE_CASES: List[EvalCase] = [
    EvalCase(
        label="filler_only_arabic",
        transcript="السلام عليكم دكتور كيف حالك",
        must_not_correct_original=True,
        is_positive=False,
    ),
    EvalCase(
        label="short_arabic_words",
        transcript="بدا المريض يشعر بتحسن واضح",
        must_not_correct_original=True,
        is_positive=False,
    ),
    EvalCase(
        label="numbers_and_units",
        transcript="BP 120/80 HR 72 Temp 37.2",
        must_not_correct_original=True,
        is_positive=False,
    ),
    EvalCase(
        label="english_clean_text",
        transcript="The patient is stable and resting comfortably",
        must_not_correct_original=True,
        is_positive=False,
    ),
    EvalCase(
        label="short_common_english",
        transcript="I have a red car and it is fast",
        must_not_correct_original=True,
        is_positive=False,
    ),
    EvalCase(
        label="arabic_greeting",
        transcript="وعليكم السلام ورحمة الله وبركاته",
        must_not_correct_original=True,
        is_positive=False,
    ),
    # Context: 'بلد' alone should NOT match 'blood' — it needs 'برشر' with it
    EvalCase(
        label="arabic_balad_alone",
        transcript="من أي بلد أنت",
        must_not_correct_original=True,
        is_positive=False,
    ),
    # None of these tokens should fire collectively
    EvalCase(
        label="arabic_filler_mix",
        transcript="من حوالي 10 سنين",
        must_not_correct_original=True,
        is_positive=False,
    ),
]


# ---------------------------------------------------------------------------
# Full realistic transcript (the "28-correction transcript")
# ---------------------------------------------------------------------------

FULL_TRANSCRIPT = (
    "السلام عليكم دكتور، المريض عمره 56 سنة "
    "وعنده بلاد شوجر و بلد برشر من حوالي 10 سنين. "
    "اليوم جا يشتكي من شورتنس اوف بريث و chest bain شديد. "
    "تم اعطاء نيتروغلسرين و أسبرين، مع متابعة بلاد شوجر كل 4 ساعات. "
    "فحص cardiac anzymes طبيعي و aksyn satoration 98%.\n\n"
    "من هستوري المريض عنده دايابيتس و هايبرتنشن مزمنين. "
    "يحتاج anty platalet therapy و متابعة منتظمة."
)

FULL_EXPECTED = {
    "blood sugar",
    "blood pressure",
    "shortness of breath",
    "chest pain",
    "cardiac enzymes",
    "oxygen saturation",
    "antiplatelet therapy",
    # Note: history, diabetes, hypertension not in lexicon
    # Note: nitroglycerin, aspirin — single-word Arabic→English
    #   transliteration depends on lexicon entries
}

# Terms that are or are not in the lexicon for single-word Arabic corrections
# نیتروغلسرین → nitroglycerin
# أسبرین → aspirin
FULL_DISALLOWED = set()  # No specific disallowed terms


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _check_phrases(
    corrected: str,
    original: str,
    case: EvalCase,
) -> Tuple[bool, List[str]]:
    """Check if expected phrases appear and disallowed phrases are absent.

    Returns (pass, reasons).
    """
    reasons: List[str] = []
    ct_lower = corrected.lower()

    passed = True

    # Check expected phrases
    for phrase in case.expected_phrases:
        if phrase.lower() in ct_lower:
            reasons.append(f"  ✓ '{phrase}' found")
        else:
            reasons.append(f"  ✗ '{phrase}' MISSING")
            passed = False

    # Check disallowed phrases
    for phrase in case.disallowed_phrases:
        if phrase.lower() in ct_lower:
            reasons.append(f"  ✗ '{phrase}' found (disallowed!)")
            passed = False
        else:
            reasons.append(f"  ✓ '{phrase}' absent (good)")

    # Check must_not_correct_original
    if case.must_not_correct_original and original != corrected:
        reasons.append(f"  ✗ Text was modified but should not have been")
        passed = False
    elif case.must_not_correct_original:
        reasons.append(f"  ✓ Text unchanged (as expected)")

    return passed, reasons


def _check_full_transcript(
    corrected: str,
    original: str,
    expected: set,
    disallowed: set,
) -> Tuple[bool, Dict[str, Any]]:
    """Check a full transcript against expected/disallowed phrases.

    Returns (pass, details).
    """
    ct_lower = corrected.lower()
    details: Dict[str, Any] = {
        "found": [],
        "missing": [],
        "unexpected": [],
    }

    for phrase in sorted(expected):
        if phrase.lower() in ct_lower:
            details["found"].append(phrase)
        else:
            details["missing"].append(phrase)

    for phrase in sorted(disallowed):
        if phrase.lower() in ct_lower:
            details["unexpected"].append(phrase)

    passed = len(details["missing"]) == 0 and len(details["unexpected"]) == 0
    return passed, details


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

def run_eval(verbose: bool = False) -> Dict[str, Any]:
    """Run all evaluation cases and return results."""
    corrector = MedicalCorrector()

    results: Dict[str, Any] = {
        "positive": {"total": 0, "passed": 0, "cases": []},
        "negative": {"total": 0, "passed": 0, "cases": []},
        "full_transcript": {},
        "summary": {},
    }

    # --- Positive cases ---
    for case in POSITIVE_CASES:
        results["positive"]["total"] += 1
        result = corrector.correct_transcript(case.transcript)
        corrected = result["corrected_text"]
        passed, reasons = _check_phrases(corrected, case.transcript, case)

        if passed:
            results["positive"]["passed"] += 1

        case_result = {
            "label": case.label,
            "passed": passed,
            "transcript": case.transcript,
            "corrected": corrected,
            "expected": list(case.expected_phrases),
            "details": reasons,
            "spans": [
                {
                    "original": s["original_text"],
                    "correction": s["possible_correction"],
                    "score": s["score"],
                    "type": s["issue_type"],
                }
                for s in result["suspicious_spans"]
            ],
        }
        results["positive"]["cases"].append(case_result)

        if verbose or not passed:
            status = "PASS" if passed else "FAIL"
            print(f"\n[{status}] {case.label}")
            print(f"  Input:    {case.transcript}")
            print(f"  Corrected: {corrected}")
            for r in reasons:
                print(r)
            if result["suspicious_spans"]:
                print(f"  Spans ({len(result['suspicious_spans'])}):")
                for s in result["suspicious_spans"]:
                    print(f"    {s['original_text']!r} → {s['possible_correction']!r} "
                          f"(score={s['score']}, type={s['issue_type']})")

    # --- Negative cases ---
    for case in NEGATIVE_CASES:
        results["negative"]["total"] += 1
        result = corrector.correct_transcript(case.transcript)
        corrected = result["corrected_text"]
        passed, reasons = _check_phrases(corrected, case.transcript, case)

        if passed:
            results["negative"]["passed"] += 1

        case_result = {
            "label": case.label,
            "passed": passed,
            "transcript": case.transcript,
            "corrected": corrected,
            "details": reasons,
            "spans": [
                {
                    "original": s["original_text"],
                    "correction": s["possible_correction"],
                    "score": s["score"],
                    "type": s["issue_type"],
                }
                for s in result["suspicious_spans"]
            ],
        }
        results["negative"]["cases"].append(case_result)

        if verbose or not passed:
            status = "PASS" if passed else "FAIL"
            print(f"\n[{status}] {case.label} (negative)")
            print(f"  Input:    {case.transcript}")
            print(f"  Corrected: {corrected}")
            for r in reasons:
                print(r)
            if result["suspicious_spans"]:
                print(f"  Spans ({len(result['suspicious_spans'])}):")
                for s in result["suspicious_spans"]:
                    print(f"    {s['original_text']!r} → {s['possible_correction']!r} "
                          f"(score={s['score']}, type={s['issue_type']})")

    # --- Full transcript ---
    print(f"\n{'='*72}")
    print("FULL TRANSCRIPT EVALUATION")
    print("=" * 72)
    result = corrector.correct_transcript(FULL_TRANSCRIPT)
    corrected = result["corrected_text"]
    passed, details = _check_full_transcript(
        corrected, FULL_TRANSCRIPT, FULL_EXPECTED, FULL_DISALLOWED,
    )

    print(f"\nInput:\n  {FULL_TRANSCRIPT[:200]}...")
    print(f"Corrected:\n  {corrected}")
    print(f"\nExpected phrases ({len(FULL_EXPECTED)}):")
    for p in sorted(FULL_EXPECTED):
        status = "✓" if p in details["found"] else "✗"
        print(f"  {status} {p}")
    if details["unexpected"]:
        print(f"\nUnexpected corrections:")
        for p in details["unexpected"]:
            print(f"  ✗ {p}")
    print(f"\nCorrections applied:")
    for s in result["suspicious_spans"]:
        score_str = f"score={s['score']}"
        print(f"  {s['original_text']!r} → {s['possible_correction']!r} ({score_str}, {s['issue_type']})")

    results["full_transcript"] = {
        "passed": passed,
        "transcript": FULL_TRANSCRIPT[:500],
        "corrected": corrected,
        "found": details["found"],
        "missing": details["missing"],
        "unexpected": details["unexpected"],
        "corrections_applied": len(result["suspicious_spans"]),
        "corrections": [
            {
                "original": s["original_text"],
                "correction": s["possible_correction"],
                "score": s["score"],
                "type": s["issue_type"],
            }
            for s in result["suspicious_spans"]
        ],
    }

    # --- Summary ---
    pos_total = results["positive"]["total"]
    pos_passed = results["positive"]["passed"]
    neg_total = results["negative"]["total"]
    neg_passed = results["negative"]["passed"]

    pos_recall = pos_passed / pos_total if pos_total > 0 else 0.0
    neg_specificity = neg_passed / neg_total if neg_total > 0 else 0.0

    summary = {
        "positive_cases": pos_total,
        "positive_passed": pos_passed,
        "positive_recall": round(pos_recall * 100, 1),
        "negative_cases": neg_total,
        "negative_passed": neg_passed,
        "negative_specificity": round(neg_specificity * 100, 1),
        "full_transcript_passed": passed,
        "full_transcript_total_expected": len(FULL_EXPECTED),
        "full_transcript_matched": len(details["found"]),
        "full_transcript_missing": details["missing"],
    }

    results["summary"] = summary

    # Print summary
    print(f"\n{'='*72}")
    print("SUMMARY")
    print("=" * 72)
    print(f"  Positive cases:   {pos_passed}/{pos_total} passed "
          f"({summary['positive_recall']}% recall)")
    print(f"  Negative cases:   {neg_passed}/{neg_total} passed "
          f"({summary['negative_specificity']}% specificity)")
    print(f"  Full transcript:  {'PASS' if passed else 'FAIL'} "
          f"({summary['full_transcript_matched']}/{summary['full_transcript_total_expected']} "
          f"expected phrases found)")
    if details["missing"]:
        print(f"  Missing phrases: {details['missing']}")
    if details["unexpected"]:
        print(f"  Unexpected:      {details['unexpected']}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed output for every case (default: only failures)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    results = run_eval(verbose=args.verbose)

    if args.json:
        # Strip the full transcript text for cleaner JSON
        results["full_transcript"]["transcript"] = results["full_transcript"]["transcript"][:200]
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*72}")
        print(f"Done. {results['summary']['positive_passed']}/{results['summary']['positive_cases']} "
              f"positive, {results['summary']['negative_passed']}/{results['summary']['negative_cases']} "
              f"negative.")

    # Return exit code based on success
    if not args.json:
        all_pass = (
            results["summary"]["positive_passed"] == results["summary"]["positive_cases"]
            and results["summary"]["negative_passed"] == results["summary"]["negative_cases"]
            and results["summary"]["full_transcript_passed"]
        )
        sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
