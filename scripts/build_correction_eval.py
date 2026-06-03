"""Build a comprehensive correction evaluation dataset.

Combines:
  1. Existing records from eval/medical_transcript_eval.jsonl (20 records)
  2. Seed corrections from data/user_corrections.jsonl (12 records)
  3. Arabic transliteration cases (30+)
  4. Arabic spelling correction cases (15+)
  5. Clean English inputs that must NOT change (40+)
  6. Clean Arabic inputs that must NOT change (40+)
  7. Mixed Arabic-English cases (15+)
  8. Additional English misspelling cases (20+)

Each record includes the PROMPT.md-specified fields:
  raw, gold, lang, notes  (plus internal fields: transcript, gold_spans, split, difficulty, contains_error)

Total target: 200+ records.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "eval" / "correction_eval.jsonl"

RNG = random.Random(42)

_ERROR_CLASS_NOTES = {
    "arabic_transliteration": "Arabic→English medical transliteration used in Gulf clinical speech",
    "arabic_spelling": "Arabic phonetic misspelling (e.g. سداع→صداع)",
    "english_misspelling": "English medical term misspelled by ASR",
    "split_phrase_should_merge": "ASR split a single term across multiple words",
    "wrong_medical_term": "ASR produced a plausible but wrong medical term",
}


def _reconstruct_gold(record: Dict[str, Any]) -> str:
    """Reconstruct the gold text by applying gold_spans to the transcript."""
    transcript = record["transcript"]
    gold = transcript
    for gs in record.get("gold_spans", []):
        orig = gs.get("original_text", "")
        corr = gs.get("possible_correction", "")
        if orig and corr:
            gold = gold.replace(orig, corr, 1)
    return gold


def _make_notes(record: Dict[str, Any]) -> str:
    """Generate a descriptive notes field for a record."""
    if record.get("contains_error"):
        issue_types = set()
        for gs in record.get("gold_spans", []):
            issue = gs.get("issue_type", "")
            note = _ERROR_CLASS_NOTES.get(issue, issue)
            if note:
                issue_types.add(note)
        return "; ".join(sorted(issue_types)) if issue_types else "contains error"
    else:
        lang = record.get("lang", "en")
        labels = {"en": "clean English clinical input — must not change",
                  "ar": "clean Arabic clinical input — must not change",
                  "mixed": "clean mixed Arabic-English input — must not change"}
        return labels.get(lang, "clean input — must not change")


def load_existing() -> List[Dict[str, Any]]:
    """Load existing records from eval/medical_transcript_eval.jsonl."""
    path = PROJECT_ROOT / "eval" / "medical_transcript_eval.jsonl"
    records: List[Dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_user_corrections() -> List[Dict[str, Any]]:
    """Convert user_corrections.jsonl to eval format."""
    path = PROJECT_ROOT / "data" / "user_corrections.jsonl"
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            original = rec.get("original", "").strip()
            corrected = rec.get("corrected", "").strip()
            if not original or not corrected:
                continue
            has_arabic = bool(set(original) & set("ابتثجحخدذرزسشصضطظعغفقكلمنهويءؤئآأإة"))
            if has_arabic:
                if set(corrected).isdisjoint(set("ابتثجحخدذرزسشصضطظعغفقكلمنهويءؤئآأإة")):
                    issue_type = "arabic_transliteration"
                else:
                    issue_type = "arabic_spelling"
            else:
                issue_type = "english_misspelling"

            gold_spans = [{
                "original_text": original,
                "possible_correction": corrected,
                "issue_type": issue_type,
            }]
            split = "dev" if i < 8 else "test"
            records.append({
                "id": f"user_corr_{i:03d}",
                "split": split,
                "difficulty": "easy" if " " not in original else "medium",
                "contains_error": True,
                "transcript": original,
                "gold_spans": gold_spans,
                "lang": "ar" if has_arabic else "en",
            })
    return records


def synthesize_arabic_transliterations() -> List[Dict[str, Any]]:
    """Generate Arabic→English transliteration test cases."""
    cases = [
        ("هستوري", "history", "easy"),
        ("دايابيتس", "diabetes", "easy"),
        ("هايبرتنشن", "hypertension", "easy"),
        ("شورتنس", "shortness", "easy"),
        ("بريث", "breath", "easy"),
        ("نيتروغلسرين", "nitroglycerin", "medium"),
        ("انتيبلاتلت", "antiplatelet", "medium"),
        ("السمتمز", "symptoms", "easy"),
        ("شوجر", "sugar", "easy"),
        ("دايابيتس تايب 2", "diabetes type 2", "medium"),
        ("هايبرتنشن شديد", "hypertension severe", "medium"),
        ("هستوري مرض", "history of disease", "medium"),
        ("الانسولين", "insulin", "medium"),
        ("الكارداك انزايمز", "cardiac enzymes", "hard"),
        ("بلاد شوجر", "blood sugar", "medium"),
        ("بلد برشر", "blood pressure", "medium"),
        ("هارت ريت", "heart rate", "medium"),
        ("اكسجن ساتوريشن", "oxygen saturation", "hard"),
        ("دزي نس", "dizziness", "medium"),
        ("ناوسيا", "nausea", "medium"),
        ("توبونين", "troponin", "medium"),
        ("هايپرگلايسيميا", "hyperglycemia", "medium"),
        ("نيتروغلسرين واسبرين", "nitroglycerin and aspirin", "hard"),
        ("بلاد شوجر مرتفع جدا", "blood sugar very high", "medium"),
        ("يعاني من شورتنس اوف بريث", "suffers from shortness of breath", "hard"),
        ("عنده هستوري دايابيتس و هايبرتنشن", "has history of diabetes and hypertension", "hard"),
        ("يحتاج انتيبلاتلت ثيرابي", "needs antiplatelet therapy", "hard"),
        ("الكارداك انزايمز مرتفعة", "cardiac enzymes are elevated", "hard"),
        ("تم اعطاء نيتروغلسرين", "given nitroglycerin", "hard"),
        ("متابعة بلاد شوجر كل 4 ساعات", "monitor blood sugar every 4 hours", "hard"),
        ("دكتور عنده هستوري اوف دايابيتس", "doctor has history of diabetes", "hard"),
        ("يعاني من شورتنس اوف بريث و دزي نس", "suffers from shortness of breath and dizziness", "hard"),
        ("الفيتل ساينز", "vital signs", "medium"),
        ("بلَد برشر", "blood pressure", "medium"),
        ("تمبرتشر", "temperature", "medium"),
        ("الاكسجن ساتوريشن", "oxygen saturation", "hard"),
    ]
    records = []
    for i, (ar, en, diff) in enumerate(cases):
        split = "dev" if i < 22 else "test"
        records.append({
            "id": f"ar_translit_{i:03d}",
            "split": split,
            "difficulty": diff,
            "contains_error": True,
            "transcript": ar,
            "gold_spans": [{
                "original_text": ar,
                "possible_correction": en,
                "issue_type": "arabic_transliteration",
            }],
            "lang": "ar",
        })
    return records


def synthesize_arabic_spelling() -> List[Dict[str, Any]]:
    """Generate Arabic spelling correction cases (سداع→صداع, etc.)."""
    cases = [
        ("سداع", "صداع", "easy"),
        ("الدغط", "الضغط", "easy"),
        ("ارتفاع الدغط", "ارتفاع الضغط", "medium"),
        ("طعم", "تعب", "easy"),
        ("التهب", "التهاب", "medium"),
        ("التهبات", "التهابات", "medium"),
        ("انيميا", "فقر دم", "medium"),
        ("الم في الصدر", "ألم في الصدر", "easy"),
        ("ارتفاح الضغط", "ارتفاع الضغط", "medium"),
        ("اضظراب", "اضطراب", "easy"),
        ("المرض السكري", "مرض السكري", "easy"),
        ("انتضام", "انتظام", "easy"),
        ("برستاتا", "بروستاتا", "medium"),
        ("حساسيه", "حساسية", "easy"),
        ("تعبان جدا", "تعبان جدا", "easy"),  # already correct
        ("النبض منتضام", "النبض منتظم", "medium"),
        ("اضظرابات النوم", "اضطرابات النوم", "medium"),
    ]
    records = []
    for i, (raw, gold, diff) in enumerate(cases):
        has_error = raw != gold
        split = "dev" if i < 10 else "test"
        gold_spans = [{
            "original_text": raw,
            "possible_correction": gold,
            "issue_type": "arabic_spelling",
        }] if has_error else []
        records.append({
            "id": f"ar_spell_{i:03d}",
            "split": split,
            "difficulty": diff,
            "contains_error": has_error,
            "transcript": raw,
            "gold_spans": gold_spans,
            "lang": "ar",
        })
    return records


def synthesize_clean_english() -> List[Dict[str, Any]]:
    """Generate clean English inputs that must NOT change."""
    cases = [
        "The patient is stable and resting comfortably.",
        "Patient is well and has no complaints today.",
        "Vital signs are within normal limits.",
        "The patient is a 45-year-old male with no significant medical history.",
        "BP 120/80 HR 72 Temp 37.2 RR 16",
        "The patient was discharged home in stable condition.",
        "Please follow up in the clinic in 2 weeks.",
        "The patient reports feeling much better today.",
        "No acute distress noted on examination.",
        "The surgical wound is clean and dry with no signs of infection.",
        "Patient is alert and oriented to person, place, and time.",
        "The patient's medications have been reviewed and are appropriate.",
        "Physical examination reveals no abnormalities.",
        "The patient is able to ambulate independently.",
        "Diet and activity as tolerated.",
        "Continue current medications as prescribed.",
        "The patient is scheduled for a follow-up appointment next week.",
        "Lab results are within normal range.",
        "The patient denies any chest pain or shortness of breath.",
        "The patient has a history of well-controlled hypertension.",
        "I saw the patient in the clinic today for a routine check-up.",
        "The patient is a 32-year-old female with a benign past medical history.",
        "Review of systems is negative for fever, chills, or night sweats.",
        "The patient is on a regular diet and tolerating meals well.",
        "Immunizations are up to date.",
        "The patient has no known drug allergies.",
        "Social history: the patient does not smoke or drink alcohol.",
        "The physical exam is unremarkable.",
        "The patient was advised to increase fluid intake.",
        "The patient will continue with the current treatment plan.",
        "The patient was seen in the emergency department for evaluation.",
        "The wound is healing well with no signs of infection.",
        "The patient is afebrile and hemodynamically stable.",
        "The patient's mental status is at baseline.",
        "The patient tolerates the procedure well without complications.",
        "The patient is to follow up as an outpatient.",
        "All medications have been reconciled.",
        "The patient has a follow-up appointment scheduled.",
        "Discharge instructions were reviewed with the patient.",
        "The patient's condition is improving with the current management.",
        "The patient reports compliance with all medications.",
        "No new symptoms were reported at today's visit.",
        "The patient is a 58-year-old male here for a routine physical.",
        "The patient is a 28-year-old female with no complaints.",
        "The patient has a good appetite and is eating well.",
        "The patient is sleeping well at night.",
    ]
    records = []
    for i, text in enumerate(cases):
        split = "dev" if i < 25 else "test"
        records.append({
            "id": f"clean_en_{i:03d}",
            "split": split,
            "difficulty": "easy",
            "contains_error": False,
            "transcript": text,
            "gold_spans": [],
            "lang": "en",
        })
    return records


def synthesize_clean_arabic() -> List[Dict[str, Any]]:
    """Generate clean Arabic inputs that must NOT change."""
    cases = [
        "السلام عليكم دكتور",
        "كيف حالك اليوم",
        "المريض في حالة مستقرة",
        "شكرا جزيلا",
        "الحمد لله على السلامة",
        "أشعر بتحسن اليوم",
        "سوف نتابع الحالة غدا إن شاء الله",
        "الرجاء العودة للعيادة بعد أسبوعين",
        "لا يوجد ألم في الصدر",
        "الفحص السريري طبيعي",
        "العلامات الحيوية ضمن الحدود الطبيعية",
        "المريض واع ومتجاوب",
        "تم شرح خطة العلاج للمريض",
        "سيتم متابعة المريض في العيادة الخارجية",
        "الجرح نظيف وجاف بدون علامات التهاب",
        "تم إعطاء التعليمات للمريض",
        "المريض يتحمل الطعام جيدا",
        "سيتم صرف الدواء من الصيدلية",
        "لا يوجد حساسية معروفة للأدوية",
        "يرجى مراجعة الطبيب في حال ازدياد الألم",
        "الضغط طبيعي والحمد لله",
        "السكر منخفض قليلا",
        "نبضات القلب منتظمة",
        "درجة الحرارة طبيعية",
        "نسبة الأكسجين ممتازة",
        "المريض يمارس رياضة المشي يوميا",
        "العملية تمت بنجاح",
        "فترة النقاهة تتطلب الراحة",
        "لا توجد مضاعفات بعد العملية",
        "التحاليل المخبرية ضمن المعدل الطبيعي",
        "تم أخذ التاريخ المرضي الكامل",
        "المريض لا يدخن ولا يشرب الكحول",
        "الوزن مستقر",
        "الشهية جيدة",
        "النوم منتظم",
        "تم صرف المضاد الحيوي",
        "الجرعة مرتين يوميا",
        "الدواء بعد الأكل",
        "يرجى الالتزام بالمواعيد المحددة",
        "الفحص الإشعاعي طبيعي",
        "لا توجد كسور في العظام",
        "القلب سليم",
        "الكبد والكلى بحالة جيدة",
        "المفاصل لا يوجد فيها التهاب",
        "يحتاج المريض إلى راحة تامة",
        "تم عمل الفحوصات اللازمة",
    ]
    records = []
    for i, text in enumerate(cases):
        split = "dev" if i < 25 else "test"
        records.append({
            "id": f"clean_ar_{i:03d}",
            "split": split,
            "difficulty": "easy",
            "contains_error": False,
            "transcript": text,
            "gold_spans": [],
            "lang": "ar",
        })
    return records


def synthesize_mixed_arabic_english() -> List[Dict[str, Any]]:
    """Generate mixed Arabic-English cases."""
    cases = [
        ("المريض عنده هستوري of دايابيتس", "المريض عنده history of diabetes",
         "arabic_transliteration", "hard"),
        ("يحتاج clopidogr 75 mg", "يحتاج clopidogrel 75 mg",
         "english_misspelling", "medium"),
        ("السلام عليكم دكتور patient is stable", "السلام عليكم دكتور patient is stable",
         "none", "easy"),
        ("BP 160 over 100 وعنده بلاد شوجر", "BP 160 over 100 وعنده blood sugar",
         "arabic_transliteration", "hard"),
        ("الفحص أظهر mild wheezeng", "الفحص أظهر mild wheezing",
         "english_misspelling", "medium"),
        ("تم إجراء ECG وطلع possble ischemic chenges",
         "تم إجراء ECG وطلع possible ischemic changes",
         "english_misspelling", "medium"),
        ("اللاب ريزلتس بينت elevated troponen",
         "اللاب ريزلتس بينت elevated troponin",
         "english_misspelling", "hard"),
        ("يعاني من شيفر chest bain", "يعاني من شيفر chest pain",
         "english_misspelling", "medium"),
        ("مريض السكر يحتاج insulin", "مريض السكر يحتاج insulin",
         "none", "easy"),
        ("الثلاث ايام الماضية كان عنده هستوري نزيف",
         "الثلاث ايام الماضية كان عنده history of bleeding",
         "arabic_transliteration", "hard"),
        ("يعطي المريض أسبرين يوميا", "يعطي المريض aspirin يوميا",
         "arabic_transliteration", "medium"),
        ("يحتاج المريض دوليبران للالم", "يحتاج المريض doliprane للالم",
         "arabic_transliteration", "medium"),
        ("ياخذ أسبرين and clopidogr", "ياخذ aspirin and clopidogrel",
         "mixed", "hard"),
        ("CT scan أظهر pneumonia في الرئة اليمنى", "CT scan أظهر pneumonia في الرئة اليمنى",
         "none", "easy"),
        ("Hart rate 112 and بلد برشر 160/100", "Heart rate 112 and blood pressure 160/100",
         "arabic_transliteration", "hard"),
    ]
    records = []
    for i, (raw, gold, err_type, diff) in enumerate(cases):
        has_error = err_type != "none"
        gold_spans = []
        if has_error:
            raw_words = raw.split()
            gold_words = gold.split()
            for rw, gw in zip(raw_words, gold_words):
                if rw != gw:
                    gold_spans.append({
                        "original_text": rw,
                        "possible_correction": gw,
                        "issue_type": err_type,
                    })
        split = "dev" if i < 8 else "test"
        records.append({
            "id": f"mixed_{i:03d}",
            "split": split,
            "difficulty": diff,
            "contains_error": has_error,
            "transcript": raw,
            "gold_spans": gold_spans,
            "lang": "mixed",
        })
    return records


def synthesize_english_misspellings() -> List[Dict[str, Any]]:
    """Generate English misspelling cases beyond the existing ones."""
    cases = [
        ("Patient needs clopidogr 75 mg daily", "Patient needs clopidogrel 75 mg daily", "medium"),
        ("Take amoxicilin 500 mg", "Take amoxicillin 500 mg", "easy"),
        ("hyperglacymia in a diabetic patient", "hyperglycemia in a diabetic patient", "medium"),
        ("wheezeng and shortnes of breath", "wheezing and shortness of breath", "medium"),
        ("bilateral creptations in the lungs", "bilateral crepitations in the lungs", "medium"),
        ("possble ischemic chenges on ECG", "possible ischemic changes on ECG", "medium"),
        ("elevated troponen levels", "elevated troponin levels", "easy"),
        ("start antiplatelet theraphy", "start antiplatelet therapy", "medium"),
        ("patient has hypertention and diabites", "patient has hypertension and diabetes", "medium"),
        ("needs insolin 20 units", "needs insulin 20 units", "easy"),
        ("mild anemis detected", "mild anemia detected", "easy"),
        ("acute pancriatitis suspected", "acute pancreatitis suspected", "medium"),
        ("start antibiotic therpy", "start antibiotic therapy", "easy"),
        ("liver cirhosis diagnosed", "liver cirrhosis diagnosed", "medium"),
        ("chronic obstruktive pulmonary disease", "chronic obstructive pulmonary disease", "hard"),
        ("myokardial infarcton ruled out", "myocardial infarction ruled out", "hard"),
        ("gastrointeritis for 3 days", "gastroenteritis for 3 days", "medium"),
        ("hemorrage from the wound", "hemorrhage from the wound", "medium"),
        ("thyroid function test shows hypothiroidism", "thyroid function test shows hypothyroidism", "medium"),
        ("patient with athma exacerbation", "patient with asthma exacerbation", "easy"),
        ("acute diarhea for 2 days", "acute diarrhea for 2 days", "easy"),
        ("needs dapto and mycin for infection", "needs daptomycin for infection", "hard"),
    ]
    records = []
    for i, (raw, gold, diff) in enumerate(cases):
        gold_spans = []
        raw_words = raw.split()
        gold_words = gold.split()
        for rw, gw in zip(raw_words, gold_words):
            if rw.lower() != gw.lower():
                gold_spans.append({
                    "original_text": rw,
                    "possible_correction": gw,
                    "issue_type": "english_misspelling",
                })
        split = "dev" if i < 12 else "test"
        records.append({
            "id": f"en_misspell_{i:03d}",
            "split": split,
            "difficulty": diff,
            "contains_error": True,
            "transcript": raw,
            "gold_spans": gold_spans,
            "lang": "en",
        })
    return records


def main() -> None:
    records: List[Dict[str, Any]] = []

    # 1. Existing records
    existing = load_existing()
    print(f"Loaded {len(existing)} existing records")
    records.extend(existing)

    # 2. User corrections
    user_corr = load_user_corrections()
    print(f"Loaded {len(user_corr)} user correction records")
    records.extend(user_corr)

    # 3. Arabic transliterations
    ar_translit = synthesize_arabic_transliterations()
    print(f"Synthesized {len(ar_translit)} Arabic transliteration records")
    records.extend(ar_translit)

    # 4. Arabic spelling
    ar_spell = synthesize_arabic_spelling()
    print(f"Synthesized {len(ar_spell)} Arabic spelling records")
    records.extend(ar_spell)

    # 5. Clean English
    clean_en = synthesize_clean_english()
    print(f"Synthesized {len(clean_en)} clean English records")
    records.extend(clean_en)

    # 6. Clean Arabic
    clean_ar = synthesize_clean_arabic()
    print(f"Synthesized {len(clean_ar)} clean Arabic records")
    records.extend(clean_ar)

    # 7. Mixed Arabic-English
    mixed = synthesize_mixed_arabic_english()
    print(f"Synthesized {len(mixed)} mixed records")
    records.extend(mixed)

    # 8. English misspellings
    en_misspell = synthesize_english_misspellings()
    print(f"Synthesized {len(en_misspell)} English misspelling records")
    records.extend(en_misspell)

    # Deduplicate by id
    seen_ids: set = set()
    deduped = []
    for r in records:
        rid = r["id"]
        if rid in seen_ids:
            print(f"  WARNING: duplicate id {rid}, skipping")
            continue
        seen_ids.add(rid)
        deduped.append(r)
    records = deduped

    # Assign unique ids and add raw/gold/notes fields per PROMPT.md schema
    for i, r in enumerate(records):
        r["id"] = f"eval_{i:04d}"
        r["raw"] = r["transcript"]
        r["gold"] = _reconstruct_gold(r)
        r["notes"] = _make_notes(r)

    # Shuffle
    RNG.shuffle(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    n_with_errors = sum(1 for r in records if r["contains_error"])
    n_clean = sum(1 for r in records if not r["contains_error"])
    n_en = sum(1 for r in records if r.get("lang") == "en")
    n_ar = sum(1 for r in records if r.get("lang") == "ar")
    n_mixed = sum(1 for r in records if r.get("lang") == "mixed")
    n_dev = sum(1 for r in records if r["split"] == "dev")
    n_test = sum(1 for r in records if r["split"] == "test")
    n_easy = sum(1 for r in records if r["difficulty"] == "easy")
    n_medium = sum(1 for r in records if r["difficulty"] == "medium")
    n_hard = sum(1 for r in records if r["difficulty"] == "hard")

    print(f"\n{'='*50}")
    print(f"Total records: {len(records)}")
    print(f"  With errors: {n_with_errors}")
    print(f"  Clean:       {n_clean}")
    print(f"  English:     {n_en}")
    print(f"  Arabic:      {n_ar}")
    print(f"  Mixed:       {n_mixed}")
    print(f"  Dev:         {n_dev}")
    print(f"  Test:        {n_test}")
    print(f"  Easy:        {n_easy}")
    print(f"  Medium:      {n_medium}")
    print(f"  Hard:        {n_hard}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
