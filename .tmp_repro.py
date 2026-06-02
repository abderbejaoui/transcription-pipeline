"""Reproduce the false-positive bug on the user's test sentence.

Runs the text-only correction path components directly and prints scores
so we can see WHY normal Arabic words get force-matched.
"""
import os
os.environ.setdefault("USE_LLM", "0")          # rule-based only, no network
os.environ.setdefault("USE_LLM_CORRECTOR", "0")
os.environ.setdefault("USE_API_FALLBACK", "0")

TEXT = ("المريض عنده اتريل فيبريليشن ومسجل عنده من زمان، وياخذ bisoprlol و "
        "digoxine بشكل يومي. اليوم جاء يشتكي من تشيست بين وصعال جاف من ثلاث "
        "ايام. الفحوصات بينت عنده hypokalimia خفيف. وصفت له esomeprazol "
        "للمعده وطلبت اعادة تخطيط القلب بعد اسبوع.")

print("=" * 70)
print("INPUT:")
print(TEXT)
print("=" * 70)

# --- Stage 1: MedicalCorrector ---
from app.main import _get_text_corrector
corrector = _get_text_corrector()
result = corrector.correct_transcript(TEXT)

print("\n--- STAGE 1: MedicalCorrector ---")
print("corrected_text:")
print(result["corrected_text"])
print("\nsuspicious_spans:")
for s in result["suspicious_spans"]:
    print(f"  {s.get('original_text')!r:40} -> {s.get('possible_correction')!r:30} "
          f"score={s.get('score'):.1f}  type={s.get('issue_type')}")

# --- Stage 2: Hybrid matcher on missed Arabic words ---
print("\n--- STAGE 2: HybridMatcher per Arabic word (score detail) ---")
import re
from app.main import _get_hybrid_matcher, _has_arabic, _is_arabic_filler
hybrid = _get_hybrid_matcher()
for w in re.split(r"\s+", TEXT.strip()):
    clean = re.sub(r"^[\s،؛؟!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]+", "", w)
    clean = re.sub(r"[\s،؛؟!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]+$", "", clean)
    if not clean or not _has_arabic(clean) or len(clean) < 3:
        continue
    filler = _is_arabic_filler(clean)
    cands = hybrid.match(clean, top_k=3, context=TEXT)
    top = cands[0] if cands else None
    flag = ""
    if top and top["score"] >= 80.0 and not filler:
        flag = "  <== AUTO-APPLIED (>=80, not filler)"
    cand_str = ", ".join(f"{c['term']}={c['score']:.1f}" for c in cands) if cands else "(none)"
    print(f"  {clean!r:18} filler={str(filler):5} cands=[{cand_str}]{flag}")
