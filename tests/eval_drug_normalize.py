"""Evaluation harness for the Arabic-script -> Latin drug normalizer.

Goal: prove that any change to ``app.services.drug_normalize`` improves drug
recovery WITHOUT introducing false positives (corrupting ordinary Arabic).

Run:
    python -m tests.eval_drug_normalize
    python -m tests.eval_drug_normalize --verbose

Exit code is non-zero if there is ANY false positive, so this can gate CI.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional

from app.services.drug_normalize import normalize_drugs


@dataclass
class Case:
    """A single labeled test case.

    text:     input transcript (as the ASR would emit it, Arabic script)
    expect:   canonical Latin drug names that MUST appear in the output, in
              order. Empty list => nothing should be normalized.
    must_keep: Arabic substrings that MUST survive untouched in the output
              (guards against false positives corrupting ordinary words).
    note:     human description of what the case probes.
    """

    text: str
    expect: List[str]
    must_keep: List[str]
    note: str


# ---------------------------------------------------------------------------
# POSITIVE cases: a known drug, written/mangled in Arabic, must be recovered.
# ---------------------------------------------------------------------------
POSITIVE: List[Case] = [
    Case("أريد بنادول", ["panadol"], ["أريد"], "clean panadol"),
    Case("أريد بنادل", ["panadol"], ["أريد"], "mangled panadol (missing و)"),
    Case("خذ دوليبران بعد الأكل", ["doliprane"], ["خذ", "بعد", "الأكل"], "clean doliprane in sentence"),
    Case("دوريبران مرتين", ["doliprane"], ["مرتين"], "mangled doliprane d->? "),
    Case("أعطني بنادول و دوليبران", ["panadol", "doliprane"], ["أعطني"], "two drugs"),
    Case("ودوليبران ضروري", ["doliprane"], ["ضروري"], "leading conjunction و glued"),
    Case("نوفادول للصداع", ["novadol"], ["للصداع"], "novadol"),
    Case("باراسيتامول جرعة", ["paracetamol"], ["جرعة"], "paracetamol long name"),
    Case("ايبوبروفين للالتهاب", ["ibuprofen"], ["للالتهاب"], "ibuprofen"),
    Case("اسبرين يوميا", ["aspirin"], ["يوميا"], "aspirin"),
    Case("اموكسيسيلين ثلاث مرات", ["amoxicillin"], ["ثلاث", "مرات"], "amoxicillin"),
    Case("اوجمنتين قرص", ["augmentin"], ["قرص"], "augmentin"),
    Case("فينتولين بخاخ", ["ventolin"], ["بخاخ"], "ventolin"),
    Case("فولتارين جل", ["voltaren"], ["جل"], "voltaren"),
    Case("افرلجان شراب", ["efferalgan"], ["شراب"], "efferalgan"),
    Case("فلاجيل للمعدة", ["flagyl"], ["للمعدة"], "flagyl"),
    Case("زيثروماكس مضاد حيوي", ["zithromax"], ["مضاد", "حيوي"], "zithromax"),
    Case("تاميفلو للانفلونزا", ["tamiflu"], ["للانفلونزا"], "tamiflu"),
    Case("ترامادول للألم", ["tramadol"], ["للألم"], "tramadol"),
    Case("ميتفورمين للسكري", ["metformin"], ["للسكري"], "metformin"),
    Case("انسولين حقنة", ["insulin"], ["حقنة"], "insulin"),
    Case("اوميبرازول للحموضة", ["omeprazole"], ["للحموضة"], "omeprazole"),
    Case("سيبروفلوكساسين", ["ciprofloxacin"], [], "ciprofloxacin"),
    Case("ازيثرومايسين", ["azithromycin"], [], "azithromycin"),
]

# ---------------------------------------------------------------------------
# GENERALIZATION cases: Arabic spellings of KNOWN drugs that are deliberately
# NOT listed in _DRUG_VARIANTS. A closed-dictionary matcher tends to MISS these;
# a true phonetic engine should recover them. These are the cases that justify
# moving from fuzzy-string matching to a phonetic (CEQ/Editex) core.
# ---------------------------------------------------------------------------
GENERALIZATION: List[Case] = [
    Case("بنادولات كثيرة", ["panadol"], [], "panadol + plural suffix"),
    Case("بناضول", ["panadol"], [], "panadol with ض instead of د"),
    Case("دولبران", ["doliprane"], [], "doliprane missing internal vowel"),
    Case("دوليپران", ["doliprane"], [], "doliprane with پ"),
    Case("اموكسيلين", ["amoxicillin"], [], "amoxicillin dropped syllable"),
    Case("فولتارن", ["voltaren"], [], "voltaren no internal vowel"),
    Case("اوغمنتين", ["augmentin"], [], "augmentin with غ"),
    Case("زيترومكس", ["zithromax"], [], "zithromax compressed"),
    Case("ترمادول", ["tramadol"], [], "tramadol no first vowel"),
    Case("ميتفورمن", ["metformin"], [], "metformin no last vowel"),
    Case("اوميبرازل", ["omeprazole"], [], "omeprazole compressed"),
]

# ---------------------------------------------------------------------------
# NEGATIVE cases: ordinary Arabic that MUST NOT be touched. These are the most
# important: a single false positive here fails the whole run.
# ---------------------------------------------------------------------------
NEGATIVE: List[Case] = [
    Case("أريد أن أذهب إلى المستشفى", [], ["أريد", "أذهب", "المستشفى"], "plain sentence"),
    Case("الدواء موجود في الصيدلية", [], ["الدواء", "الصيدلية"], "pharmacy sentence"),
    Case("عندي صداع وألم في المعدة", [], ["صداع", "المعدة"], "symptoms"),
    Case("الطبيب قال لي خذ راحة", [], ["الطبيب", "راحة"], "doctor said rest"),
    Case("ودول العالم تجتمع اليوم", [], ["ودول", "العالم"], "and the states (looks like panadol-ish)"),
    Case("بدأت أشعر بتحسن كبير", [], ["بدأت", "بتحسن"], "feeling better"),
    Case("هذا الدواء مفيد جدا", [], ["مفيد", "جدا"], "useful medicine"),
    Case("درجة الحرارة مرتفعة", [], ["درجة", "الحرارة"], "temperature high"),
    Case("الوصفة الطبية جاهزة", [], ["الوصفة", "الطبية"], "prescription ready"),
    Case("نوم جيد يساعد على الشفاء", [], ["نوم", "الشفاء"], "sleep helps"),
    Case("المريض يحتاج إلى فحص", [], ["المريض", "فحص"], "patient needs exam"),
    Case("اشرب ماء كثيرا", [], ["اشرب", "ماء"], "drink water"),
    Case("الضغط مرتفع قليلا", [], ["الضغط", "مرتفع"], "blood pressure"),
    Case("فيتامين سي مهم", [], ["فيتامين", "مهم"], "vitamin C (not in dict)"),
    Case("والدتي مريضة منذ يومين", [], ["والدتي", "مريضة"], "mother sick (leading و)"),
    Case("نتيجة التحليل سليمة", [], ["نتيجة", "التحليل"], "analysis result fine"),
    # Hard negatives: ordinary words that are phonetically near a drug skeleton.
    Case("بدائل أخرى متاحة", [], ["بدائل", "متاحة"], "badaa'il (alternatives) ~ panadol-ish"),
    Case("الفلفل حار جدا", [], ["الفلفل", "حار"], "pepper ~ flagyl-ish f-l-f-l"),
    Case("متروك للطبيب القرار", [], ["متروك", "القرار"], "matrouk ~ tramadol-ish"),
    Case("سيروا في طريقكم", [], ["سيروا", "طريقكم"], "siru (walk) ~ cipro-ish onset"),
    Case("انتظر دقيقة من فضلك", [], ["انتظر", "دقيقة"], "wait a minute"),
]


def _contains_arabic(s: str) -> bool:
    return any("\u0600" <= ch <= "\u06ff" for ch in s)


@dataclass
class Result:
    case: Case
    output: str
    ok: bool
    reason: str
    is_false_positive: bool


def run_case(case: Case) -> Result:
    out, _ = normalize_drugs(case.text)

    # must_keep first: any corrupted ordinary token is a false positive.
    for keep in case.must_keep:
        if keep not in out:
            return Result(case, out, False, f"corrupted/removed expected Arabic '{keep}'", True)

    # For pure-negative cases, ALSO ensure no Arabic word got swapped for Latin.
    if not case.expect:
        # Count Latin words that appeared but were not in the input.
        in_latin = {w for w in case.text.split() if not _contains_arabic(w)}
        out_latin = {w for w in out.split() if not _contains_arabic(w)}
        introduced = out_latin - in_latin
        if introduced:
            return Result(case, out, False, f"introduced Latin token(s) {sorted(introduced)}", True)
        return Result(case, out, True, "ok (untouched)", False)

    # Positive case: every expected canonical must be present.
    missing = [d for d in case.expect if d not in out]
    if missing:
        return Result(case, out, False, f"missed drug(s) {missing}", False)
    return Result(case, out, True, "ok (recovered)", False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    all_cases = POSITIVE + GENERALIZATION + NEGATIVE
    results = [run_case(c) for c in all_cases]

    n_pos, n_gen, n_neg = len(POSITIVE), len(GENERALIZATION), len(NEGATIVE)
    pos_results = results[:n_pos]
    gen_results = results[n_pos:n_pos + n_gen]
    neg_results = results[n_pos + n_gen:]

    pos_pass = sum(r.ok for r in pos_results)
    gen_pass = sum(r.ok for r in gen_results)
    neg_pass = sum(r.ok for r in neg_results)
    false_positives = [r for r in results if r.is_false_positive]

    def _print_group(title: str, group: List[Result]) -> None:
        print(f"\n== {title} ({sum(r.ok for r in group)}/{len(group)}) ==")
        for r in group:
            if r.ok and not args.verbose:
                continue
            flag = "PASS" if r.ok else ("FALSE-POS" if r.is_false_positive else "FAIL")
            print(f"  [{flag}] {r.case.note}")
            print(f"        in : {r.case.text}")
            print(f"        out: {r.output}")
            print(f"        why: {r.reason}")

    _print_group("POSITIVE (drug recovery)", pos_results)
    _print_group("GENERALIZATION (unseen mangles)", gen_results)
    _print_group("NEGATIVE (must not corrupt Arabic)", neg_results)

    total = len(results)
    passed = pos_pass + gen_pass + neg_pass
    print("\n" + "=" * 60)
    print(f"POSITIVE recall      : {pos_pass}/{n_pos}")
    print(f"GENERALIZATION recall: {gen_pass}/{n_gen}  (closed-dict tends to miss)")
    print(f"NEGATIVE safety      : {neg_pass}/{n_neg}")
    print(f"FALSE POSITIVES      : {len(false_positives)}  (must be 0)")
    print(f"TOTAL                : {passed}/{total}")
    print("=" * 60)

    # Fail the run on ANY false positive; that is the non-negotiable invariant.
    return 1 if false_positives else 0


if __name__ == "__main__":
    sys.exit(main())
