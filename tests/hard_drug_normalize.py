"""HARD adversarial test set for the Arabic-script -> Latin drug normalizer.

This is deliberately tougher than tests/eval_drug_normalize.py. It probes the
phonetic matcher's real weak spots:

  * heavy unseen mangles (letters dropped/swapped, no listed variant)
  * dialect / colloquial spellings
  * affixed drug tokens (plural, definite article, prepositions glued on)
  * multi-drug sentences and dosage noise
  * NEAR-MISS negatives: ordinary Arabic words that share consonants with a
    drug skeleton and MUST stay untouched (these are the dangerous ones)

For every case it prints the input, the output, the verdict, and the TOP-3
phonetic candidates with similarity scores, so you can see *why* the matcher
decided what it did.

Run:
    python -m tests.hard_drug_normalize
    python -m tests.hard_drug_normalize --scores   # show candidate scores
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import List

from app.services.drug_normalize import (
    normalize_drugs,
    _ar_skeleton,
    _phonetic_similarity,
    _INDEX,
)


@dataclass
class Case:
    text: str
    expect: List[str]          # canonical Latin names that MUST appear
    forbid: List[str] = field(default_factory=list)  # Latin that must NOT appear
    must_keep: List[str] = field(default_factory=list)  # Arabic that must survive
    note: str = ""


def _contains_arabic(s: str) -> bool:
    return any("\u0600" <= ch <= "\u06ff" for ch in s)


# ---------------------------------------------------------------------------
# HARD POSITIVES: unseen mangles / affixes / dialect. Should still recover.
# ---------------------------------------------------------------------------
HARD_POSITIVE: List[Case] = [
    Case("بنادولات للأطفال", ["panadol"], note="panadol + plural ات"),
    Case("البنادول انتهى", ["panadol"], note="panadol + definite article ال"),
    Case("بناضول حبتين", ["panadol"], note="panadol with ض (emphatic d)"),
    Case("دولبران فقط", ["doliprane"], note="doliprane missing internal vowel"),
    Case("الدوليبران ضروري", ["doliprane"], note="doliprane + ال"),
    Case("دوليپران مرتين", ["doliprane"], note="doliprane with پ (Persian p)"),
    Case("نوفادل للحرارة", ["novadol"], note="novadol short form نوفادل"),
    Case("اموكسيلين شراب", ["amoxicillin"], note="amoxicillin dropped syllable"),
    Case("الفولتارين جل", ["voltaren"], note="voltaren + ال, no internal vowel"),
    Case("اوغمنتين حبوب", ["augmentin"], note="augmentin with غ"),
    Case("زيترومكس مرة", ["zithromax"], note="zithromax compressed"),
    Case("ترمادول للألم", ["tramadol"], note="tramadol no first vowel"),
    Case("ميتفورمن للسكر", ["metformin"], note="metformin no last vowel"),
    Case("اوميبرازل صباحا", ["omeprazole"], note="omeprazole compressed"),
    Case("الأسبرين يوميا", ["aspirin"], note="aspirin + ال"),
    Case("افرالجان شراب", ["efferalgan"], note="efferalgan alt spelling"),
]

# ---------------------------------------------------------------------------
# MULTI-DRUG / NOISY sentences.
# ---------------------------------------------------------------------------
MULTI: List[Case] = [
    Case("أعطني بنادول و دوليبران و فينتولين",
         ["panadol", "doliprane", "ventolin"], note="three drugs"),
    Case("خذ بنادول صباحا و فولتارين مساء",
         ["panadol", "voltaren"], must_keep=["صباحا", "مساء"], note="drugs + time words"),
    Case("الوصفة فيها اموكسيسيلين و اوميبرازول",
         ["amoxicillin", "omeprazole"], must_keep=["الوصفة"], note="two long drugs in sentence"),
    Case("بنادولات و دوليبرانات معا",
         ["panadol", "doliprane"], note="two pluralized drugs"),
]

# ---------------------------------------------------------------------------
# NEAR-MISS NEGATIVES: ordinary Arabic sharing consonants with a drug. The
# matcher MUST leave every one of these untouched. A single hit here is a fail.
# ---------------------------------------------------------------------------
HARD_NEGATIVE: List[Case] = [
    Case("بدائل أخرى متاحة", [], forbid=["panadol", "doliprane"],
         must_keep=["بدائل", "متاحة"], note="badaa'il ~ panadol onset b-d-l"),
    Case("الفلفل الأحمر حار", [], forbid=["flagyl"],
         must_keep=["الفلفل", "حار"], note="pepper f-l-f-l ~ flagyl f-l-j-l"),
    Case("متروك للطبيب القرار", [], forbid=["tramadol"],
         must_keep=["متروك", "القرار"], note="matrouk ~ tramadol t-r-m"),
    Case("سيروا في طريقكم بسرعة", [], forbid=["ciprofloxacin"],
         must_keep=["سيروا", "طريقكم"], note="siru ~ cipro onset"),
    Case("الفنان رسم لوحة جميلة", [], forbid=["ventolin"],
         must_keep=["الفنان", "لوحة"], note="al-fannan ~ ventolin f-n"),
    Case("التمرين مفيد للجسم", [], forbid=["tramadol", "tamiflu"],
         must_keep=["التمرين", "للجسم"], note="tamrin ~ tamiflu/tramadol t-m-r"),
    Case("النوم العميق مهم", [], forbid=["novadol"],
         must_keep=["النوم", "مهم"], note="nawm ~ novadol n-w"),
    Case("الأمل موجود دائما", [], forbid=["amoxicillin"],
         must_keep=["الأمل", "موجود"], note="al-amal ~ amoxi onset"),
    Case("الصبر مفتاح الفرج", [], forbid=["aspirin"],
         must_keep=["الصبر", "الفرج"], note="as-sabr ~ aspirin s-b-r"),
    Case("انسجام الفريق رائع", [], forbid=["insulin"],
         must_keep=["انسجام", "الفريق"], note="insijam ~ insulin n-s"),
    Case("درجة الحرارة مرتفعة جدا", [], forbid=["doliprane"],
         must_keep=["درجة", "الحرارة"], note="daraja ~ doliprane d-r"),
    Case("ودول كثيرة شاركت", [], forbid=["panadol", "novadol"],
         must_keep=["ودول", "كثيرة"], note="wa-duwal (and states) ~ ...dol suffix"),
]


def _top_candidates(token: str, k: int = 3):
    sk = _ar_skeleton(token)
    if len(sk) < 3:
        return []
    scored = []
    seen = set()
    for cand_sk, canonical in _INDEX:
        if abs(len(cand_sk) - len(sk)) > 2:
            continue
        sim = _phonetic_similarity(sk, cand_sk)
        key = canonical
        # keep best score per canonical
        prev = next((i for i, (c, _) in enumerate(scored) if c == key), None)
        if prev is None:
            scored.append((canonical, sim))
        elif sim > scored[prev][1]:
            scored[prev] = (canonical, sim)
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


@dataclass
class Result:
    case: Case
    output: str
    ok: bool
    reason: str
    fatal: bool   # false positive / forbidden hit


def run_case(case: Case) -> Result:
    out, reps = normalize_drugs(case.text)

    for keep in case.must_keep:
        if keep not in out:
            return Result(case, out, False, f"corrupted Arabic '{keep}'", True)

    for bad in case.forbid:
        if bad in out:
            return Result(case, out, False, f"FORBIDDEN match '{bad}' appeared", True)

    if not case.expect:
        in_latin = {w for w in case.text.split() if not _contains_arabic(w)}
        out_latin = {w for w in out.split() if not _contains_arabic(w)}
        introduced = out_latin - in_latin
        if introduced:
            return Result(case, out, False, f"introduced Latin {sorted(introduced)}", True)
        return Result(case, out, True, "untouched", False)

    missing = [d for d in case.expect if d not in out]
    if missing:
        return Result(case, out, False, f"missed {missing}", False)
    return Result(case, out, True, "recovered", False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="store_true", help="show top-3 candidate scores per token")
    args = ap.parse_args()

    groups = [
        ("HARD POSITIVE (unseen mangles/affixes)", HARD_POSITIVE),
        ("MULTI-DRUG / NOISY", MULTI),
        ("HARD NEGATIVE (near-miss, must stay untouched)", HARD_NEGATIVE),
    ]

    all_results: List[Result] = []
    for title, cases in groups:
        results = [run_case(c) for c in cases]
        all_results.extend(results)
        passed = sum(r.ok for r in results)
        print(f"\n== {title}  ({passed}/{len(results)}) ==")
        for r in results:
            flag = "PASS " if r.ok else ("FATAL" if r.fatal else "MISS ")
            print(f"  [{flag}] {r.case.note}")
            print(f"          in : {r.case.text}")
            print(f"          out: {r.output}")
            if not r.ok:
                print(f"          why: {r.reason}")
            if args.scores:
                for tok in r.case.text.split():
                    if not _contains_arabic(tok):
                        continue
                    cands = _top_candidates(tok)
                    if cands:
                        pretty = ", ".join(f"{c}={s:.2f}" for c, s in cands)
                        print(f"          {tok}: {pretty}")

    total = len(all_results)
    passed = sum(r.ok for r in all_results)
    fatals = [r for r in all_results if r.fatal]
    misses = [r for r in all_results if not r.ok and not r.fatal]

    print("\n" + "=" * 64)
    print(f"PASSED          : {passed}/{total}")
    print(f"MISSES (recall) : {len(misses)}  (drug not recovered — not dangerous)")
    print(f"FATAL (safety)  : {len(fatals)}  (false positive / forbidden — MUST be 0)")
    print("=" * 64)
    if fatals:
        print("\nFATAL cases:")
        for r in fatals:
            print(f"  - {r.case.note}: {r.reason}")
    if misses:
        print("\nMISS cases:")
        for r in misses:
            print(f"  - {r.case.note}: {r.reason}")

    return 1 if fatals else 0


if __name__ == "__main__":
    sys.exit(main())
