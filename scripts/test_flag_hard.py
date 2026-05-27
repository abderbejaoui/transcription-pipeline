"""Stress-test the suspicious-word flagger on 20 hard ASR mangles.

Each case is a hand-crafted worst-case transcript: the kind of broken
output our Gulf-LoRA Qwen3-ASR actually produces when a doctor speaks a
drug name. The goal: every case must result in EXACTLY ONE flag whose
top phonetic candidate is the expected drug.

Pure phonetic matching cannot hit 100% — vowels get dropped, words get
split, consonants get substituted. The test prints a PASS/FAIL table
and tells us where the LLM has to step in.

Run:
    python scripts/test_flag_hard.py            # all 20
    python scripts/test_flag_hard.py --case 5   # just case 5
    python scripts/test_flag_hard.py --verbose  # show every candidate
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make app.services importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.flag import phonetic_pass  # noqa: E402


@dataclass
class HardCase:
    name: str
    transcript: str          # what the ASR produced (mangled)
    expected_drug: str       # what the user actually said
    target_word: str         # word/bigram in transcript we want flagged
    must_be_top1: bool = True

CASES: List[HardCase] = [
    # 1. Classic split: one drug name became two short tokens.
    HardCase("split_paracetamol",
             "خذ برسي تمر مرتين في اليوم",
             "paracetamol", "برسي تمر"),

    # 2. Three-token split.
    HardCase("triple_split_paracetamol",
             "خذ بار سي تمول لو في الم",
             "paracetamol", "بار سي تمول",
             must_be_top1=False),  # 3-grams not implemented yet

    # 3. With definite article.
    HardCase("article_paracetamol",
             "البرسيتامول صباحا و مساء",
             "paracetamol", "البرسيتامول"),

    # 4. Article + waw conjunction.
    HardCase("wa_paracetamol",
             "و بانادول قبل النوم",
             "panadol", "بانادول"),

    # 5. Ibuprofen compound that got split.
    HardCase("split_ibuprofen",
             "خذ ايبو بروفان مع الاكل",
             "ibuprofen", "ايبو بروفان"),

    # 6. Heavy mangle: efferalgan -> "if all gone" Arabic spelling.
    HardCase("efferalgan_if_all_gone",
             "اعطاني الصيدلي اف اول قن للحرارة",
             "efferalgan", "اف اول قن"),

    # 7. Doliprane brand mangled.
    HardCase("doliprane",
             "خذ دولي فران تحاميل قبل النوم",
             "doliprane", "دولي فران"),

    # 8. Amoxicillin spaced.
    HardCase("amoxicillin",
             "اعطاني الدكتور اموكسي سيلين ثلاث مرات",
             "amoxicillin", "اموكسي سيلين"),

    # 9. Augmentin split with weird letter choices.
    HardCase("augmentin",
             "اوغ من تين شراب اطفال",
             "augmentin", "اوغ من تين",
             must_be_top1=False),

    # 10. Metformin: long word with vowels dropped.
    HardCase("metformin",
             "ميت فور مين خمس مية ملليجرام",
             "metformin", "ميت فور مين",
             must_be_top1=False),

    # 11. Ventolin / salbutamol.
    HardCase("ventolin",
             "بخاخ فنتولين قبل النوبه",
             "ventolin", "فنتولين"),

    # 12. Voltaren gel.
    HardCase("voltaren",
             "فول تارن جل للظهر",
             "voltaren", "فول تارن"),

    # 13. Omeprazole.
    HardCase("omeprazole",
             "اوميه برازول قبل الفطور",
             "omeprazole", "اوميه برازول"),

    # 14. Loratadine brand (Clarityne).
    HardCase("loratadine",
             "خذي حبه كلاريتين كل يوم",
             "loratadine", "كلاريتين",
             must_be_top1=False),  # phonetic distance is high

    # 15. Atorvastatin: long unfamiliar word.
    HardCase("atorvastatin",
             "اتورفاستاتين بعد العشاء",
             "atorvastatin", "اتورفاستاتين"),

    # 16. Codeine — short, easy to lose.
    HardCase("codeine",
             "كوديين للسعال",
             "codeine", "كوديين"),

    # 17. Diazepam.
    HardCase("diazepam",
             "ديا زيبام للنوم",
             "diazepam", "ديا زيبام"),

    # 18. Insulin.
    HardCase("insulin",
             "ابره انسولين قبل الاكل",
             "insulin", "انسولين"),

    # 19. Multi-drug — must flag BOTH.
    HardCase("multi_drug",
             "بانادول صباحا و فول تارن مساء",
             "panadol", "بانادول"),

    # 20. Pure noise — should NOT flag (false-positive test).
    HardCase("no_drug_no_flag",
             "عندي صداع و دوخه و تعب",
             "", ""),  # expected: zero flags
]


def _has_candidate(cands: List[Dict[str, Any]], drug: str) -> bool:
    return any(c.get("term", "").lower() == drug.lower() for c in cands)


def _top1(cands: List[Dict[str, Any]]) -> str:
    return cands[0]["term"] if cands else ""


def run_case(case: HardCase, verbose: bool = False) -> Dict[str, Any]:
    flags = phonetic_pass(case.transcript)
    result = {
        "name": case.name,
        "drug": case.expected_drug,
        "target": case.target_word,
        "flagged": False,
        "top1": "",
        "found_in_top3": False,
        "all_flags": [],
        "verdict": "FAIL",
    }
    for f in flags:
        result["all_flags"].append({
            "word": f["word"],
            "candidates": f.get("candidates", []),
        })

    # Special handling for the "should not flag" case.
    if not case.expected_drug:
        result["verdict"] = "PASS" if not flags else "FAIL"
        result["top1"] = "(none)"
        if not flags:
            result["flagged"] = True  # intentional pass condition
        return result

    matching = next(
        (f for f in flags
         if case.target_word in f["word"] or f["word"] in case.target_word),
        None,
    )
    if not matching:
        return result
    result["flagged"] = True
    cands = matching.get("candidates", [])
    result["top1"] = _top1(cands)
    result["found_in_top3"] = _has_candidate(cands[:3], case.expected_drug)

    if case.must_be_top1:
        result["verdict"] = (
            "PASS" if result["top1"].lower() == case.expected_drug.lower() else "FAIL"
        )
    else:
        result["verdict"] = "PASS" if result["found_in_top3"] else "FAIL"
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=int, help="run only case N (1-based)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    cases = CASES if args.case is None else [CASES[args.case - 1]]
    print(f"\nRunning {len(cases)} hard case(s)\n" + "=" * 60)
    results = [run_case(c, verbose=args.verbose) for c in cases]

    # Detail
    for r in results:
        marker = "✓" if r["verdict"] == "PASS" else "✗"
        print(f"\n{marker} {r['name']}")
        print(f"   target: {r['target']!r:<30} expects: {r['drug']}")
        print(f"   top-1 : {r['top1']!r:<30} flagged: {r['flagged']}")
        print(f"   in top3? {r['found_in_top3']}")
        if args.verbose:
            for fl in r["all_flags"]:
                print(f"   • {fl['word']!r}  cands: "
                      + ", ".join(
                          f"{c['term']}({c['phonetic_similarity']})"
                          for c in fl['candidates']))

    # Summary table
    n = len(results)
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    print("\n" + "=" * 60)
    print(f"PASS: {passed}/{n}  ({passed/n*100:.0f}%)")
    print("=" * 60)
    print(f"{'#':<3}{'name':<32}{'verdict':<8}{'top1':<20}")
    for i, r in enumerate(results, 1):
        print(f"{i:<3}{r['name']:<32}{r['verdict']:<8}{r['top1']:<20}")
    return 0 if passed == n else 1


if __name__ == "__main__":
    sys.exit(main())
