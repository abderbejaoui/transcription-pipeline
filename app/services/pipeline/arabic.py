"""pipeline/arabic.py — Arabic text processing utilities.

Owns:
  - Arabic → Latin transliteration
  - Phonetic skeleton extraction (Arabic and Latin sides)
  - Edit-distance helpers
  - Arabic filler word detection (via CAMeL Tools morphology)
"""

from __future__ import annotations

import functools
import re
import unicodedata
from typing import List

# ---------------------------------------------------------------------------
# Transliteration table
# ---------------------------------------------------------------------------

_AR2LAT = {
    "ا": "a", "أ": "a", "إ": "a", "آ": "a", "ٱ": "a",
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h",
    "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z",
    "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t",
    "ظ": "z", "ع": "a", "غ": "gh", "ف": "f", "ق": "q",
    "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h",
    "و": "w", "ي": "y", "ى": "a", "ة": "h", "ء": "",
    "ؤ": "w", "ئ": "y",
}
_TASHKEEL_RE = re.compile(r"[ً-ْٰـ]")


# ---------------------------------------------------------------------------
# CAMeL Tools morphological analyzers (Gulf Arabic + MSA)
# ---------------------------------------------------------------------------

def _load_morph_analyzers():
    glf, msa = None, None
    try:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer
        try:
            glf = Analyzer(MorphologyDB.builtin_db("calima-glf-01"))
        except Exception:
            pass
        try:
            msa = Analyzer(MorphologyDB.builtin_db("calima-msa-r13"))
        except Exception:
            pass
    except ImportError:
        pass
    return glf, msa


_GLF_ANALYZER, _MSA_ANALYZER = _load_morph_analyzers()


@functools.lru_cache(maxsize=8192)
def _morph_is_real_arabic(word: str) -> bool:
    """Return True if `word` has at least one valid morphological analysis.

    Drug mangles (Arabic-script transliterations of drug names) produce zero
    analyses; genuine Arabic words always produce at least one.
    """
    for analyzer in (_GLF_ANALYZER, _MSA_ANALYZER):
        if analyzer is None:
            continue
        try:
            if any(a.get("pos") not in ("PUNC", None) for a in analyzer.analyze(word)):
                return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Transliteration
# ---------------------------------------------------------------------------

def _strip_arabic_clitics(word: str) -> str:
    """Drop common attached Arabic morphemes before phonetic matching.

    Strips 'al-' (the), 'wa-' (and), 'bi-' (with), 'li-' (to), 'fa-' (so).
    Conservative: only strip when the remainder is at least 4 characters.
    """
    PREFIXES = ("ال", "وال", "بال", "كال", "فال", "لل",
                "و", "ف", "ب", "ل", "ك", "س")
    for pre in PREFIXES:
        if word.startswith(pre) and len(word) - len(pre) >= 4:
            return word[len(pre):]
    return word


def translit(word: str, *, strip_clitics: bool = True) -> str:
    """Transliterate an Arabic word to a Latin approximation."""
    s = unicodedata.normalize("NFKC", word)
    s = _TASHKEEL_RE.sub("", s)
    if strip_clitics:
        s = _strip_arabic_clitics(s)
    out: List[str] = []
    for ch in s:
        if ch in _AR2LAT:
            out.append(_AR2LAT[ch])
        elif ch.isascii() and ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


# ---------------------------------------------------------------------------
# Phonetic skeleton extraction
# ---------------------------------------------------------------------------

_PHONETIC_CLASS = {
    "p": "b", "v": "f", "c": "k",
    "a": "@", "e": "@", "i": "@", "o": "@", "u": "@", "y": "@", "w": "@",
    "h": "",
}


def phonetic_skeleton(s: str) -> str:
    """Collapse a Latin string to its consonant skeleton (drops vowels and
    maps phonetic-class equivalents: p→b, v→f, c→k)."""
    out = []
    for ch in s.lower():
        out.append(_PHONETIC_CLASS.get(ch, ch))
    result = []
    prev = None
    for ch in out:
        if ch != prev:
            result.append(ch)
        prev = ch
    return "".join(c for c in result if c != "@")


def consonant_skeleton_ar(s: str) -> str:
    """Strip vowels from an already-transliterated Arabic word.

    Arabic drops short vowels; long vowels come out as a/w/y in
    transliteration. Drop those and 'h' (silent ta-marbuta) so comparison
    hits consonants only.
    """
    VOWELS = set("aeiouy w h".replace(" ", ""))
    return "".join(c for c in s.lower() if c not in VOWELS)


def consonant_skeleton_latin(s: str) -> str:
    """Strip vowels from a Latin drug name + map phonetic classes Arabic loses.

    'paracetamol' → 'brktml'   (p→b, c→k, vowels dropped)
    'efferalgan'  → 'ffrlkn'   (g→k)
    'ibuprofen'   → 'bbrfn'    (p→b)
    """
    VOWELS = set("aeiouy")
    SUBST = {"p": "b", "v": "f", "c": "k", "g": "k", "q": "k", "x": "ks"}
    out = []
    for ch in s.lower():
        if ch in VOWELS:
            continue
        out.append(SUBST.get(ch, ch))
    return "".join(out)


# ---------------------------------------------------------------------------
# Edit distance and string helpers
# ---------------------------------------------------------------------------

def lev_sim(a: str, b: str) -> float:
    """Normalised Levenshtein similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return 1.0 - prev[m] / max(n, m)


def length_ratio_ok(needle: str, term: str, *, tolerance: float = 0.5) -> bool:
    """Reject candidates whose length differs too much from the needle.

    Short needles (≤6 chars) use a tighter tolerance (0.65) because
    edit distance is too forgiving on short strings.
    """
    if not needle or not term:
        return False
    if len(needle) <= 6 or len(term) <= 6:
        tolerance = max(tolerance, 0.65)
    ratio = min(len(needle), len(term)) / max(len(needle), len(term))
    return ratio >= tolerance


def longest_common_substring(a: str, b: str) -> int:
    """Length of the longest contiguous substring shared between a and b.

    Used as a precision check: coincidental letter overlap (scattered matches)
    produces low LCS even when edit-distance similarity is deceptively high.
    """
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


# ---------------------------------------------------------------------------
# Arabic filler word detection
# ---------------------------------------------------------------------------

_ARABIC_SHORT_PARTICLES = {
    "و", "أو", "او", "إذ",
    "في", "من", "إلى", "الى", "على", "عن", "مع", "لو",
    "لا", "ما", "لم", "لن", "قد", "ثم", "هو", "هي",
    "له", "لها", "لهم", "لنا", "به", "بها", "بهم",
    "ان", "إن", "أن",
    "ملغ", "مجم",
    "ف", "ب", "ل", "ك",
}

_ARABIC_FILLER_FALLBACK = {
    "صداع", "دوخه", "دوار", "تعب", "حرارة", "الم", "وجع",
    "نفس", "ربو", "سكر", "ضغط", "التهاب", "مستشفى",
    "يحتاج", "عشان", "الحين", "استمر", "النتائج", "نتائج",
    "عدوى", "حمية", "كوليسترول", "الكوليسترول",
    "الدكتور", "الطبيب", "علاج", "دواء", "تحليل",
    "اليوم", "ساعه", "يوم", "شهر", "مرات", "حبوب",
}


def is_arabic_filler(word: str) -> bool:
    """Return True if `word` is a real Arabic word that should not be flagged.

    Strategy:
    1. Empty → True (skip).
    2. Explicit short particle set → True.
    3. ≤3 chars not in particle set → False (may be drug fragment).
    4. Longer: morphological check via CAMeL Tools Gulf + MSA analyzers.
    5. Fallback if analyzers unavailable: compact hardcoded set.
    """
    w = _TASHKEEL_RE.sub("", unicodedata.normalize("NFKC", word))
    if not w:
        return True
    if w in _ARABIC_SHORT_PARTICLES:
        return True
    if len(w) <= 3:
        return False
    if _GLF_ANALYZER is not None or _MSA_ANALYZER is not None:
        return _morph_is_real_arabic(w)
    if w in _ARABIC_FILLER_FALLBACK:
        return True
    for pre in ("ال", "وال", "بال", "كال", "فال", "لل"):
        if w.startswith(pre) and len(w) > len(pre):
            if w[len(pre):] in _ARABIC_FILLER_FALLBACK:
                return True
    return False


_ARABIC_DIGIT_RE = re.compile(r"^[0-9٠-٩۰-۹]+$")
_ARABIC_LETTER_RE = re.compile(r"[؀-ۿ]")


def is_pure_latin_or_digit(word: str) -> bool:
    """True for Latin-only words and pure-digit tokens (Arabic-Indic included).

    These should be skipped by the n-gram pass to avoid mixing scripts.
    """
    if not word:
        return True
    if _ARABIC_DIGIT_RE.match(word):
        return True
    return not _ARABIC_LETTER_RE.search(word)
