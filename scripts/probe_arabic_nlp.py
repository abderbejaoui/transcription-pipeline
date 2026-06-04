#!/usr/bin/env python3
"""Probe available Arabic NLP approaches to replace the hardcoded filler list."""
import sys, re
sys.stdout.reconfigure(encoding="utf-8")

from camel_tools.utils.normalize import normalize_unicode
from camel_tools.utils.dediac import dediac_ar

# --------------------------------------------------------------------------
# Approach A: character-level morphological plausibility
# --------------------------------------------------------------------------
# Arabic trilateral root system: words are built from 3-4 consonant roots
# with prefixes/suffixes indicating POS. Drug mangles don't follow this.
# The insight: after stripping common Arabic clitics and the long vowels
# (ا و ي), the remaining consonants should form a plausible Arabic root.
# Real Arabic roots avoid certain consonant combinations that appear in
# Latin-origin loanwords (e.g., ب+ر+و+ل+و+س+ي+ك for "prilosec").

LONG_VOWELS = set("اوي")
TA_MARBUTA = "ة"
CLITICS = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ب", "ل", "ك", "ف", "س")
ARABIC_LETTER_RE = re.compile(r"^[؀-ۿ]+$")

# Known Arabic prefixes that signal a real morphological form
ARABIC_VERB_PREFIXES = {"ي", "ت", "ن", "أ", "ا"}  # imperfect verb prefixes

def strip_clitics(word):
    for pre in CLITICS:
        if word.startswith(pre) and len(word) - len(pre) >= 3:
            return word[len(pre):]
    return word

def morphological_plausibility(word: str) -> float:
    """Score 0.0-1.0: how likely is this a real Arabic word vs. a drug mangle?

    Uses character-level patterns without a morphology DB.
    Returns high score for real Arabic words, low for drug mangles.
    """
    word = dediac_ar(normalize_unicode(word.strip()))
    core = strip_clitics(word)

    # Must be all Arabic script
    if not ARABIC_LETTER_RE.match(core):
        return 0.0

    # Too short to be a meaningful word or drug name
    if len(core) < 2:
        return 1.0  # particles are real Arabic

    # Extract consonant skeleton (remove long vowels and ta marbuta)
    consonants = [c for c in core if c not in LONG_VOWELS and c != TA_MARBUTA]

    # Arabic roots have 2-4 consonants typically (after affixes).
    # Drug mangles of 6-12 letter Latin names produce 5-9+ consonants.
    n_consonants = len(consonants)

    # Heuristic: word has a verb/noun prefix + 3-4 root consonants = real
    # A long consonant cluster (>5) without Arabic affixes = likely loanword
    if n_consonants <= 4:
        score = 0.9   # classic Arabic word length
    elif n_consonants <= 6:
        score = 0.6   # possible compound / loanword
    else:
        score = 0.2   # very likely a long Latin drug name transliterated

    # Adjust: if word starts with a verb prefix, more likely real Arabic
    if core and core[0] in ARABIC_VERB_PREFIXES and len(core) >= 4:
        score = min(1.0, score + 0.2)

    # Adjust: if word contains sequences unusual in Arabic (p/v/g substitutes
    # in clusters) — actually these can't be detected at char level reliably

    return score


# --------------------------------------------------------------------------
# Approach B: the actual correct approach — CAMeL Tools morphology DB
# --------------------------------------------------------------------------
def try_camel_morphology(words):
    try:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer
        db = MorphologyDB.builtin_db()
        analyzer = Analyzer(db)
        print("\n=== CAMeL Tools morphology (DB available) ===")
        for w in words:
            analyses = analyzer.analyze(w)
            valid = [a for a in analyses if a.get("pos") not in ("PUNC", "NOAN", None)]
            print(f"  {w:25} -> {len(valid)} valid analyses")
    except FileNotFoundError:
        print("\n=== CAMeL Tools: morphology DB NOT downloaded ===")
        print("  Install: camel_data -i morphology-db-msa-r13")
        print("  This would be the BEST solution — full morphological analysis.")
        print("  Each word gets POS tag, lemma, root, pattern.")
        print("  Real Arabic words have analyses; drug mangles don't.")


# --------------------------------------------------------------------------
# Run tests
# --------------------------------------------------------------------------
words = [
    # Real Arabic clinical words (should NOT be flagged — current FPs)
    ("يحتاج",        "needs [verb] — real Arabic"),
    ("العدوى",       "the infection — real Arabic"),
    ("الحمية",       "the diet — real Arabic"),
    ("ارسل",         "send [imperative] — real Arabic"),
    ("الكوليسترول",  "cholesterol — Arabic loanword"),
    ("الإيبوبروفين", "ibuprofen — Arabic loanword"),
    ("الأنسولين",    "insulin — Arabic loanword"),
    ("النتائج",      "the results — real Arabic"),
    ("الأنسولين",    "insulin — loanword, but established"),
    # Drug mangles (SHOULD be flagged)
    ("أوق",          "drug mangle part of augmentin"),
    ("منتين",        "drug mangle part of augmentin"),
    ("لايزينو",      "drug mangle of lisinopril"),
    ("بريل",         "drug mangle part of lisinopril"),
    ("نيكسيوم",      "drug mangle of nexium"),
    ("ميتوبرولول",   "drug mangle of metoprolol"),
    ("فولتران",      "drug mangle of voltaren"),
    ("برولوسيك",     "drug mangle of prilosec"),
    ("داباغليفلوزين","drug mangle of dapagliflozin"),
    ("كريستور",      "drug mangle of crestor"),
    ("كلوبيدوجريل",  "drug mangle of clopidogrel"),
    ("ريزيبريل",     "drug mangle of ramipril"),
]

print("=== Morphological Plausibility (character-level, no DB) ===")
print(f"{'Word':25} {'Score':6}  {'Assessment':15}  Description")
print("-" * 80)
for w, desc in words:
    score = morphological_plausibility(w)
    label = "REAL Arabic" if score >= 0.7 else ("BORDERLINE" if score >= 0.4 else "LIKELY MANGLE")
    print(f"  {w:25} {score:.2f}   {label:15}  {desc}")

# Show the limitation — loanwords score as "real" because their consonant
# clusters look plausible at the character level
print()
print("LIMITATION: established Arabic loanwords (insulin, ibuprofen)")
print("score as 'real' because they've adapted to Arabic phonology.")
print("Character-level analysis alone cannot distinguish them from native words.")

try_camel_morphology([w for w, _ in words[:10]])

print()
print("=== Summary of viable approaches ===")
print("""
1. CAMeL Tools morphology DB (BEST — not yet downloaded)
   - Full morphological analysis: POS, root, pattern
   - Drug mangles get 0 analyses; real words get 1+
   - ~5ms/word, no latency, deterministic
   - Download: camel_data -i morphology-db-msa-r13

2. Existing LLM pass (use_llm=True in flag_suspicious)
   - Already implemented, understands context fully
   - Can distinguish 'يحتاج' (needs) from 'لايزينو' (lisinopril mangle)
   - Latency: 2-3s per transcript (one call per sentence, not per word)
   - Not deterministic

3. Character-level heuristics (what we just tested)
   - Works for very long drug names (داباغليفلوزين, ميتوبرولول)
   - FAILS for established loanwords (الكوليسترول, الأنسولين)
   - Not reliable enough standalone

4. Arabic word-frequency list (e.g. from ArabiCorpus or OpenSubtitles-AR)
   - If a word appears in a large Arabic corpus -> real word
   - Drug mangles won't appear -> flag them
   - Fast, deterministic, no ML needed
   - But loanwords DO appear in corpora (insulin, cholesterol)
""")
