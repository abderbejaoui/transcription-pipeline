"""Generate training data for the n-gram language model.

Produces a single text file with one sentence per line, combining:
  1. Clean English medical transcripts from eval sets
  2. Synthetic Gulf Arabic + English code-switched sentences from lexicon terms
  3. Clean Arabic medical text from Gulf evaluation data

Usage:
    python scripts/generate_lm_training_data.py > data/lm_training_corpus.txt
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# 1. Clean medical English sentences from eval data
# --------------------------------------------------------------------------

def extract_eval_sentences() -> list[str]:
    """Extract clean English sentences from medical_transcript_eval.jsonl."""
    sentences: list[str] = []
    eval_path = PROJECT_ROOT / "eval" / "medical_transcript_eval.jsonl"
    if not eval_path.exists():
        return sentences
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        transcript = rec.get("transcript", "").strip()
        if not transcript:
            continue
        # Split into sentences (split on . ! ?)
        parts = re.split(r'(?<=[.!?])\s+', transcript)
        for part in parts:
            part = part.strip()
            if part and len(part) >= 10:
                sentences.append(part)
    return sentences


# --------------------------------------------------------------------------
# 2. Synthetic sentences from medical lexicon
# --------------------------------------------------------------------------

# Sentence templates for medical terms (English only)
_EN_TEMPLATES: list[str] = [
    "The patient has {term}.",
    "We diagnosed {term}.",
    "The patient was found to have {term}.",
    "Start the patient on {term}.",
    "Administer {term} intravenously.",
    "The lab results are consistent with {term}.",
    "We are concerned about {term}.",
    "The imaging findings suggest {term}.",
    "Begin treatment with {term} immediately.",
    "The biopsy confirmed {term}.",
    "The patient's history is significant for {term}.",
    "Continue {term} for the next seven days.",
    "The prescribed medication is {term}.",
    "Monitor the patient's response to {term}.",
    "The symptoms are consistent with {term}.",
    "We will start {term} and monitor closely.",
    "The culture grew {term}.",
    "The echocardiogram revealed {term}.",
    "The patient has a history of {term}.",
    "Due to the severity of {term}, we are changing the treatment plan.",
    "Stop {term} if side effects develop.",
    "The recommended dosage of {term} is 500 mg twice daily.",
    "We will continue {term} for the next two weeks.",
    "The differential diagnosis includes {term}.",
    "The patient requires {term} for symptom control.",
]

# Sentence templates for Gulf Arabic + English code-switched sentences
_AR_TEMPLATES: list[str] = [
    "الدكتور قال عندي {term}",
    "المريض عنده {term} من فترة",
    "نبدأ {term} اليوم",
    "نعطي {term} وريديا",
    "نحتاج {term} للحالة هذي",
    "المريض ياخذ {term} كل يوم",
    "الدكتور وصف {term}",
    "نتائج التحاليل تظهر {term}",
    "الأشعة تبين {term}",
    "نستمر على {term} لمدة أسبوع",
    "المريض عنده تاريخ مع {term}",
    "نبدأ {term} ونراقب الوضع",
    "عنده هستوري اوف {term}",
    "ياخذ {term} مرتين في اليوم",
    "الدكتور يقول عندي {term}",
    "الصيدلي اعطاني {term}",
    "عندي الم شديد من {term}",
    "نحتاج نغير العلاج بسبب {term}",
    "اعراض {term} بدت تظهر",
]

# Common clinical phrases (not from lexicon, just general medical text)
_CLINICAL_PHRASES: list[str] = [
    "The patient is a 45-year-old male with no significant past medical history.",
    "Vital signs are stable with blood pressure 120 over 80 and heart rate 72.",
    "The patient reports chest pain that started two hours ago.",
    "Review of systems is negative for fever, chills, or night sweats.",
    "The patient was admitted for further management and monitoring.",
    "We will continue current medications and monitor renal function.",
    "The patient was referred to cardiology for further evaluation.",
    "Discharge instructions include follow-up in one week.",
    "The patient should return to the emergency department if symptoms worsen.",
    "The computed tomography scan of the chest shows no acute findings.",
    "Laboratory studies are notable for elevated white blood cell count.",
    "The patient's oxygen saturation is 98 percent on room air.",
    "We discussed the risks and benefits of the procedure with the patient.",
    "Informed consent was obtained prior to the procedure.",
    "The patient tolerated the procedure well without complications.",
    "Diagnostic imaging was obtained and reviewed with radiology.",
    "The patient will be discharged home with close follow-up.",
    "We will repeat the lab studies in the morning.",
    "Allergies include penicillin and sulfa drugs.",
    "The patient is currently hemodynamically stable.",
    "Past surgical history is significant for appendectomy five years ago.",
    "Family history is positive for coronary artery disease and diabetes.",
    "The patient denies any recent travel or sick contacts.",
    "Physical examination reveals clear lungs and regular heart rate and rhythm.",
    "The abdomen is soft, non-tender, and non-distended.",
    "Neurological examination is grossly intact.",
    "A 12-lead electrocardiogram was obtained and shows normal sinus rhythm.",
    "The chest x-ray shows no evidence of pneumonia or effusion.",
    "Blood cultures were drawn prior to antibiotic administration.",
    "The patient was started on broad-spectrum antibiotics empirically.",
    "The infectious disease team was consulted for guidance.",
    "We will narrow antibiotic coverage once culture results return.",
    "The patient was started on intravenous fluids for rehydration.",
    "Urine output has been adequate over the past 24 hours.",
    "Pain is well controlled with oral analgesics.",
    "The patient is able to tolerate a regular diet.",
    "Bowel sounds are normal in all four quadrants.",
    "The wound appears clean with no signs of infection.",
    "Follow up with primary care in one to two weeks.",
    "The patient should call the clinic if they develop a fever greater than 101.",
    "The patient has Type 2 diabetes mellitus with good glycemic control.",
    "Hypertension is well managed with current oral medications.",
    "The patient has hyperlipidemia and is on a statin.",
    "The latest hemoglobin A1C was 7.1 percent.",
    "The patient's lipid panel shows well-controlled cholesterol levels.",
    "We will adjust the insulin dosage based on blood glucose readings.",
    "The patient has chronic kidney disease stage three.",
    "Thyroid function tests are within normal limits.",
    "Liver enzymes are mildly elevated but improving.",
    "Complete blood count shows mild anemia.",
    "Basic metabolic panel is significant for low sodium.",
    "Influenza and COVID-19 testing was negative.",
    "Urinalysis shows no evidence of infection.",
    "We will obtain an echocardiogram to evaluate cardiac function.",
    "A stress test was ordered to evaluate for ischemic heart disease.",
]

# Clinical Arabic phrases (clean medical Arabic)
_AR_CLINICAL: list[str] = [
    "المريض يعاني من ألم في الصدر منذ ساعتين",
    "العلامات الحيوية مستقرة والحالة العامة جيدة",
    "تم إدخال المريض إلى المستشفى للمتابعة والعلاج",
    "الفحص السريري يظهر رئتين نظيفتين",
    "صورة الصدر الشعاعية لا تظهر أي التهاب رئوي",
    "تم سحب مزارع الدم قبل البدء بالمضادات الحيوية",
    "المريض يتحمل العلاج بشكل جيد",
    "نحتاج متابعة وظائف الكلى أثناء العلاج",
    "سيتم إخراج المريض غدا بعد التأكد من استقرار حالته",
    "جرعة الدواء 500 ملغ مرتين في اليوم",
    "يجب مراجعة الطبيب إذا ساءت الأعراض",
    "المريض لا يعاني من أي حساسية معروفة",
    "التحاليل المخبرية تظهر ارتفاع في كريات الدم البيضاء",
    "نسبة الأكسجين في الدم 98 بالمائة",
    "تم أخذ موافقة المريض على الإجراء الطبي",
    "سيتم تكرار التحاليل في الصباح",
    "التصوير المقطعي للصدر لا يظهر أي نتائج حادة",
    "المريض يعاني من السكري من النوع الثاني",
    "ضغط الدم مرتفع ويحتاج متابعة",
    "نحتاج تقييم وظائف الغدة الدرقية",
    "المريض يعاني من فقر دم خفيف",
    "الدواء يسبب بعض الأعراض الجانبية الخفيفة",
    "يرجى متابعة المريض في العيادة بعد أسبوعين",
    "الجرح نظيف ولا توجد علامات التهاب",
    "المريض يستطيع تناول الطعام بشكل طبيعي",
]


def generate_synthetic_sentences(lexicon_entries: list[dict]) -> list[str]:
    """Generate synthetic medical sentences from lexicon terms."""
    rng = random.Random(42)
    sentences: list[str] = []

    for entry in lexicon_entries:
        term = entry.get("term", "")
        term_type = entry.get("type", "term")
        if not term:
            continue

        # English templates
        for template in rng.sample(_EN_TEMPLATES, min(5, len(_EN_TEMPLATES))):
            sentence = template.format(term=term)
            sentences.append(sentence)

        # Arabic + English code-switched templates
        for template in rng.sample(_AR_TEMPLATES, min(3, len(_AR_TEMPLATES))):
            sentence = template.format(term=term)
            sentences.append(sentence)

    # Add clinical phrases multiple times for higher weight
    sentences.extend(_CLINICAL_PHRASES * 5)
    sentences.extend(_AR_CLINICAL * 3)

    rng.shuffle(sentences)
    return sentences


def main() -> None:
    # 1. Load lexicon
    lexicon_path = PROJECT_ROOT / "data" / "medical_lexicon.jsonl"
    entries: list[dict] = []
    if lexicon_path.exists():
        for line in lexicon_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"[lm_data] Loaded {len(entries)} lexicon entries", file=sys.stderr)

    # 2. Extract eval sentences
    eval_sentences = extract_eval_sentences()
    print(f"[lm_data] Extracted {len(eval_sentences)} eval sentences", file=sys.stderr)

    # 3. Generate synthetic sentences
    synthetic = generate_synthetic_sentences(entries)
    print(f"[lm_data] Generated {len(synthetic)} synthetic sentences", file=sys.stderr)

    # 4. Combine and write
    rng = random.Random(123)
    all_sentences = eval_sentences + synthetic
    rng.shuffle(all_sentences)

    out_path = PROJECT_ROOT / "data" / "lm_training_corpus.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for sent in all_sentences:
            f.write(sent.strip() + "\n")

    print(f"[lm_data] Wrote {len(all_sentences)} sentences to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
