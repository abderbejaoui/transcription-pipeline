"""Suspicious-word flagger for plain-text ASR transcripts.

Combines two passes:
  1. Phonetic pass — for every transcript word, transliterate to Latin and
     compare against `medical_terms.txt` via normalized edit distance.
     Any word within `phonetic_threshold` of a known term is flagged.
  2. LLM pass — ask the chat model to flag any other words that LOOK or
     SOUND medical but didn't pass the dictionary check (rare disease
     names, brand names not in the file).

The two outputs are merged and deduplicated. Each flag includes the
word, the index in the whitespace-tokenized transcript, and the
candidate medical terms phonetically closest to it.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .llm_config import (
    get_llm_headers,
    get_llm_model,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)
from . import llm_scorer

# Used by _is_arabic_normalcy() for fuzzy skeleton comparison
from rapidfuzz import fuzz as _rapidfuzz

# ---------------------------------------------------------------------------
# Character-class regexes — used by many functions throughout this module.
# Placed here (near imports) so they are available before any function
# that references them, even functions defined earlier in the file.
# ---------------------------------------------------------------------------

_ARABIC_DIGIT_RE = re.compile(r"^[0-9\u0660-\u0669\u06f0-\u06f9]+$")
_ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06ff]")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDICAL_TERMS_PATH = PROJECT_ROOT / "medical_terms.txt"


# ---------------------------------------------------------------------------
# Cheap Arabic -> Latin transliteration (for phonetic comparison only).
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
_TASHKEEL_RE = re.compile(r"[\u064b-\u0652\u0670\u0640]")


def _strip_arabic_clitics(word: str) -> str:
    """Drop common attached morphemes before phonetic matching.

    Arabic glues 'al-' (the), 'wa-' (and), 'bi-' (with), 'li-' (to) and
    'fa-' (so) onto the next word. Without stripping these, words like
    'البرسيتامول' (= 'the paracetamol') score badly against
    'paracetamol' because of the extra 'al' prefix.

    We're conservative: we only strip when the remainder is at least
    3 characters, so we don't decapitate very short words.
    """
    PREFIXES = ("ال", "وال", "بال", "كال", "فال", "لل",
                "و", "ف", "ب", "ل", "ك")
    for pre in PREFIXES:
        if word.startswith(pre) and len(word) - len(pre) >= 3:
            return word[len(pre):]
    return word


def _translit(word: str, *, strip_clitics: bool = True) -> str:
    import unicodedata
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
# Latin phonetic-class collapsing
#
# Arabic transliteration substitutes phonetic-class consonants:
#   p -> b   (Arabic has no /p/)
#   v -> f   (Arabic has no /v/)
#   c -> k or s (depending on position)
#   g -> q or gh
# Plus vowels are unreliable in both directions. To make Levenshtein
# meaningful across these substitutions, we collapse each Latin string
# to a coarse phonetic skeleton before comparing.
# ---------------------------------------------------------------------------

_PHONETIC_CLASS = {
    "p": "b", "v": "f", "c": "k",
    # vowel collapse
    "a": "@", "e": "@", "i": "@", "o": "@", "u": "@", "y": "@", "w": "@",
    # silent / interchangeable
    "h": "",
}


def _phonetic_skeleton(s: str) -> str:
    out = []
    for ch in s.lower():
        out.append(_PHONETIC_CLASS.get(ch, ch))
    # Collapse runs of identical chars (paracetamol -> parsetmel, then
    # consecutive duplicates collapsed if any).
    result = []
    prev = None
    for ch in out:
        if ch != prev:
            result.append(ch)
        prev = ch
    # Drop the vowel placeholder when comparing (it's used as a
    # separator; final compare strips it out entirely).
    return "".join(c for c in result if c != "@")


# ---------------------------------------------------------------------------
# Medical lexicon
# ---------------------------------------------------------------------------

_lex_cache: Optional[List[str]] = None


def load_medical_lexicon() -> List[str]:
    global _lex_cache
    if _lex_cache is not None:
        return _lex_cache
    if not MEDICAL_TERMS_PATH.exists():
        _lex_cache = []
        return _lex_cache
    terms: List[str] = []
    for line in MEDICAL_TERMS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    _lex_cache = terms
    return _lex_cache


def _lev_sim(a: str, b: str) -> float:
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


# ---------------------------------------------------------------------------
# Phonetic pass
# ---------------------------------------------------------------------------


def _length_ratio_ok(needle: str, term: str, *, tolerance: float = 0.5) -> bool:
    """Reject candidates whose length is wildly different.

    A 5-char Arabic word like 'الاكل' (translit 'alakl') matching the
    6-char 'flagyl' at sim 0.5 is meaningful only if the lengths are
    close. Tolerance 0.5 means lengths must be within 50% of each other.

    For SHORT needles (<= 6 chars) the bar tightens to 0.65 because the
    edit-distance scale is too forgiving on short strings: a 4-char
    needle vs a 7-char term can hit sim 0.85 by accident. Example:
    'انسولين' -> 'nsln' (4 chars) vs 'prednisolone' (8 chars
    skeleton 'brdnsln' 7 chars) scored 0.86 with the default
    tolerance — knocked out by the stricter short-needle bar.
    """
    if not needle or not term:
        return False
    short_threshold = 6
    if len(needle) <= short_threshold or len(term) <= short_threshold:
        tolerance = max(tolerance, 0.65)
    ratio = min(len(needle), len(term)) / max(len(needle), len(term))
    return ratio >= tolerance


def _consonant_skeleton_ar(s: str) -> str:
    """Strip short vowels + long vowel markers from an already-transliterated
    Arabic word, but KEEP 'h' which is a real consonant in Arabic.

    Arabic doesn't write short vowels — when we transliterate, the
    long vowels 'ا'/'و'/'ي' come out as 'a'/'w'/'y'. Drop 'w' and 'y'
    (long vowel markers) but KEEP 'h' because it represents the consonant
    ه (hā') in transliterations like هستوري (hstwry) and هارت (hart).
    The ONLY case where 'h' is silent is ة → 'h' (tā' marbūṭa), but
    dropping ALL 'h's destroys critical skeleton matches (e.g. هستوري
    → skeleton 'str' instead of 'hstr', which fails to match 'history'
    at skeleton 'hstr').

    IMPORTANT: 'h' from Arabic DIGRAPHS (غ→gh, ش→sh, ث→th, خ→kh, ذ→dh)
    is NOT a real consonant — it's part of a two-letter representation of
    a single Arabic letter. We drop 'h' when it follows g, s, t, k, or d
    to handle this. Examples:
      - نيتروغلسرين (nytrwghlsryn): the 'h' after 'g' is from غ→gh,
        not a real consonant. Keeping it would produce skeleton 'ntrghlsrn'
        which mismatches 'nitroglycerin's Latin skeleton 'ntrglcrn'.
      - بريث (bryth): the 'h' after 't' is from ث→th. Keeping it would
        produce 'brth' instead of 'brt', breaking the match with 'breath'.
    """
    VOWELS = set("aeiouy w")  # Keep 'h' as consonant — it's a real sound in Arabic
    result = "".join(c for c in s.lower() if c not in VOWELS)
    # Drop 'h' from digraphs: gh→g, sh→s, th→t, kh→k, dh→d
    # (re is already imported at module level)
    result = re.sub(r'([gstkd])h', r'\1', result)
    return result


def _consonant_skeleton_latin(s: str) -> str:
    """Strip vowels from a Latin drug name + map phonetic classes that
    Arabic transliteration loses: p->b, v->f, c->k, g->k, q->k.

    'paracetamol' -> 'brktml'   (p->b, c->k, vowels dropped)
    'efferalgan'  -> 'ffrlkn'   (g->k)
    'ibuprofen'   -> 'bbrfn'    (p->b)
    'augmentin'   -> 'kmntn'    (g->k, second part)
    'quetiapine'  -> 'ktpn'     (q->k)  -- so Arabic 'كويتيابين'
                                        (skeleton 'ktbyn') matches it
    """
    VOWELS = set("aeiouy")
    SUBST = {"p": "b", "v": "f", "c": "k", "g": "k", "q": "k", "x": "ks"}
    out = []
    for ch in s.lower():
        if ch in VOWELS:
            continue
        sub = SUBST.get(ch, ch)
        out.append(sub)
    return "".join(out)


def _longest_common_substring(a: str, b: str) -> int:
    """Length of the longest CONTIGUOUS substring shared between a and b.

    Used as a secondary precision check for n-grams: when two n-grams
    score similarly under edit distance, the one with a longer contiguous
    shared run is almost always the right drug. Edit distance alone
    can be fooled by scattered letter overlap (e.g. a person's name
    'فواد علي النزار' shares 'f', 'l', 'n', 'z' with 'fluconazole' but
    no run longer than 2 chars, which is coincidental).
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
# Drug vs. disease classification
#
# Our medical_terms.txt mixes ~100 diseases (bursitis, asthma...) with
# ~50 drug names (paracetamol, insulin...). When an Arabic ASR mangle
# ties between a disease and a drug at the same similarity, the drug
# is almost always the correct answer — the speaker actually said a
# drug name and the ASR was just unable to spell it. We use a small
# tie-breaker to nudge drugs above diseases at equal score.
# ---------------------------------------------------------------------------

_DRUG_HINT_SUFFIXES = (
    "in", "ol", "ide", "ine", "ate", "ium", "one", "an", "el",
    "il", "etamol", "ofen", "azole", "cillin", "prazole",
    "statin", "sartan", "ipine", "formin", "tolin", "tarn", "tic",
    "amol", "ralgan",
)
_DRUG_HINT_TERMS = {
    "insulin", "panadol", "codeine", "morphine", "doliprane",
    "voltaren", "ventolin", "augmentin", "efferalgan", "flagyl",
    "warfarin", "heparin", "zithromax", "tramadol",
}

# Common English-pronunciation mishearings produced by an Arabic-trained
# ASR. Each entry maps a frequently-seen Arabic ASR fragment to the brand
# / drug it's most likely a mishearing of. When the FULL flagged span
# (after stripping spaces and clitics) matches one of these keys, we
# bump the corresponding drug to the top of the candidate list.
#
# These are *only* used as a tiebreaker for hard cases that pure
# phonetic similarity can't recover (e.g. 'اف اول قن' -> 'efferalgan',
# the classic "if all gone" mishearing). Each entry is well-known in the
# Gulf clinical-ASR literature.
_PHONETIC_ALIAS: Dict[str, str] = {
    # =================================================================
    # Efferalgan family: English "if all gone" / "if your gan" / "ever again"
    # =================================================================
    "afawlqn": "efferalgan",         # اف اول قن (if all gone)
    "afaqln": "efferalgan",           # اف قالن  (if gallen)
    "afawlqln": "efferalgan",         # اف اول قالن
    "afywrqan": "efferalgan",         # اف يور قان (if your gan)
    "afywrqn": "efferalgan",          # اف يور قن
    "afyrqn": "efferalgan",           # اف ير قن  (ef-er-gan)
    "afrqan": "efferalgan",           # افر قان
    "afiragn": "efferalgan",          # افرجن
    # =================================================================
    # Augmentin: "aught men tin" / "og mentin" / "oagmentin"
    # =================================================================
    "awqmntyn": "augmentin",          # اوغ من تين (aught men teen)
    "awgmnty": "augmentin",           # اوغمنتي
    "awqmntn": "augmentin",           # اوغ متن (augment-n)
    "awgmntyn": "augmentin",          # اومنتين
    "awqmn": "augmentin",             # اوغمن (aug-men)
    "awqmt": "augmentin",             # اوغمت (aug-met)
    # =================================================================
    # Amoxicillin: "amoxy cillin" / "mix a cillin"
    # =================================================================
    "amksylyn": "amoxicillin",        # اموكسي سيلين
    "amwksylyn": "amoxicillin",       # اموكسيسيلين
    "mksylyn": "amoxicillin",         # مكسيلين
    "amksy": "amoxicillin",           # اموكسي
    "amksylynb": "amoxicillin",       # اموكسيلينب
    # =================================================================
    # Paracetamol: "burse tamr" / "parassi tamol" / "barasetamol"
    # =================================================================
    "brsytml": "paracetamol",         # برسيتامل
    "brastml": "paracetamol",         # براستامل
    "brastamwl": "paracetamol",       # براستامول
    "parastml": "paracetamol",        # پاراستامل
    "pyrsytml": "paracetamol",        # پيرسيتامل
    # =================================================================
    # Ciprofloxacin: "cipro floxacin" / "siprofloxacin"
    # =================================================================
    "sybrwflwksasyn": "ciprofloxacin",  # سيبروفلوكساسين
    "sprwflwks": "ciprofloxacin",     # سبروفلوكس
    "sybrwflwks": "ciprofloxacin",    # سيبروفلوكس
    # =================================================================
    # Atorvastatin: "ator vastatin" / "atorvasta"
    # =================================================================
    "atwrfstasyn": "atorvastatin",    # اتورفاستاتين
    "atwrfstatyn": "atorvastatin",    # اتورفاستاتين
    "atwrfsta": "atorvastatin",       # اتورفاستا
    # =================================================================
    # Metformin: "met formin" / "mitformin" / "metaformin"
    # =================================================================
    "mitfwrmin": "metformin",         # ميتفورمين
    "mtafwrmyn": "metformin",         # متافورمين
    "matfwrmyn": "metformin",         # ميتفورمين
    # =================================================================
    # Nitroglycerin: "nitro glycerin" / "nitroglycerin"
    # =================================================================
    "nytrwghlsryn": "nitroglycerin",   # نيتروغلسرين
    "nytrwglycryn": "nitroglycerin",  # نيتروغليسرين
    "nytrwlscryn": "nitroglycerin",   # نيترولسيرين
    # =================================================================
    # Omeprazole: "ome prazole" / "omaprazole" / "lumiprazole"
    # =================================================================
    "awmbrazwl": "omeprazole",        # ومبرازول
    "awmbraz": "omeprazole",          # ومبراز
    "awmbrazawl": "omeprazole",        # ومبرازول
    # =================================================================
    # Prednisolone / Prednisone: "prednisolone" / "prednison"
    # =================================================================
    "bridnyswlyn": "prednisolone",    # بريدنيسولون
    "bridnyswn": "prednisone",        # بريدنيسون
    "bridnyslyn": "prednisolone",     # بريدنيسلون
    "bridnyzwn": "prednisone",        # بريدنيزون
    # =================================================================
    # Clopidogrel: "clopi dogrel" / "clopidogrel" / "cloba dogrel"
    # =================================================================
    "klwbyjdwgrl": "clopidogrel",     # كلوبيدوجرل
    "klwbydwgrl": "clopidogrel",      # كلوبيدوجرل
    "klwbajwgrl": "clopidogrel",      # كلوباجوجرل
    # =================================================================
    # Warfarin: "war farin" / "warfarin" / "walfarin"
    # =================================================================
    "wrfaryn": "warfarin",            # ورفارين
    "walfaryn": "warfarin",           # والفارين
    "wrfar": "warfarin",              # ورفار
    # =================================================================
    # Heparin: "hep arin" / "heparin" / "ebarin"
    # =================================================================
    "hybaryn": "heparin",             # هيبارين
    "hibrin": "heparin",              # هيبرين
    "hbarn": "heparin",               # هبارن
    # =================================================================
    # Insulin: "in sulin" / "insulin" / "ansulin"
    # =================================================================
    "answlyn": "insulin",             # انسولين
    "anslyn": "insulin",              # انسلين
    "ynslyn": "insulin",              # ينسلين
    # =================================================================
    # Tramadol: "tra ma dol" / "tramadol" / "tramadol"
    # =================================================================
    "tramadwl": "tramadol",           # ترامادول
    "tramdwla": "tramadol",           # ترامدولا
    "trmdwl": "tramadol",             # ترامدول
    # =================================================================
    # Voltaren: "vol taren" / "voltaren" / "foltaren"
    # =================================================================
    "fwltrn": "voltaren",             # فولترن
    "fwltr": "voltaren",              # فولتر
    "fwltrn": "voltaren",             # فولترن
    # =================================================================
    # Azithromycin: "azithro mycin" / "azithromycin" / "azetromycin"
    # =================================================================
    "azythrwmysyn": "azithromycin",   # ازيثروميسين
    "azythrmysyn": "azithromycin",    # ازيثروميسين
    "aztrmysyn": "azithromycin",      # ازيتروميسين
    # =================================================================
    # Vancomycin: "vanco mycin" / "vancomycin" / "vankomycin"
    # =================================================================
    "fankwmaysyn": "vancomycin",      # فانكومايسين
    "fankwmwmysyn": "vancomycin",     # فانكوموميسين
    "wankwmysyn": "vancomycin",       # وانكوميسين
    # =================================================================
    # Levofloxacin: "levo floxacin" / "levofloxacin"
    # =================================================================
    "lyfwflwksasyn": "levofloxacin",  # ليفوفلوكساسين
    "lyfwflwks": "levofloxacin",      # ليفوفلوكس
    # =================================================================
    # Metoprolol: "meto prolol" / "metopro" / "metobrolol"
    # =================================================================
    "mitwbrwlwl": "metoprolol",       # ميتوبرولول
    "mitwprwlwl": "metoprolol",       # ميتوبرولول
    # =================================================================
    # Amlodipine: "amlo dipine" / "amladipine" / "amlodobene"
    # =================================================================
    "amlwdybyn": "amlodipine",        # املوديبين
    "amlydbyn": "amlodipine",         # امليديبين
    # =================================================================
    # Ativan / Lorazepam: "ati van" -> "lorazepam"
    # =================================================================
    "atyfan": "lorazepam",            # اتيفان (ativan brand → lorazepam)
    "lwrazbam": "lorazepam",          # لورازبام
    # =================================================================
    # Xanax / Alprazolam: "xanax" / "zanax" / "sanas"
    # =================================================================
    "zanaks": "alprazolam",           # زاناكس (xanax → alprazolam)
    "snaks": "alprazolam",             # سناكص
    # =================================================================
    # Morphine: "mor phine" / "morphine" / "murfin"
    # =================================================================
    "mwrfyn": "morphine",             # مورفين
    "murfin": "morphine",             # مورفين
    # =================================================================
    # Codeine: "co deine" / "codeine" / "kuwaitin"
    # =================================================================
    "kwdyn": "codeine",               # كودين
    "kwtyn": "codeine",               # كوتين (kuwait-teen)
    "kwdyyn": "codeine",              # كوديين
    # =================================================================
    # Diazepam / Valium: "valium" / "falium" / "balium"
    # =================================================================
    "falywm": "diazepam",             # فاليوم (valium → diazepam)
    "balywm": "diazepam",             # باليوم
    # =================================================================
    # Fluconazole: "flu conazole" / "fluconazole" / "fulcanazole"
    # =================================================================
    "flwknazwl": "fluconazole",       # فلوكنازول
    "flkwnazwl": "fluconazole",       # فلكونازول
    # =================================================================
    # Pantoprazole: "panto prazole" / "pantozole" / "bantoprazole"
    # =================================================================
    "fantwbrazwl": "pantoprazole",    # فانتب رازول
    "bantwbrazwl": "pantoprazole",    # بانتب رازول
    # =================================================================
    # Misoprostol: "miso prostol" / "mesoprostol"
    # =================================================================
    "myswbrwstwl": "misoprostol",     # ميسوبروستول
    "myzbrstwl": "misoprostol",       # ميزبروستول
    # =================================================================
    # Aspirin: "as prin" / "aspirin" / "asbarin" / "estern"
    # =================================================================
    "asbryn": "aspirin",              # اسبرين (NO DUPLICATE — kept intentionally for Estren mishearing)
    "asbarn": "aspirin",              # اسبرن
    # =================================================================
    # Ceftriaxone: "cef triaxone" / "ceftriaxone" / "seftriakson"
    # =================================================================
    "syftryakswn": "ceftriaxone",     # سيفترياكسون
    "sftryakswn": "ceftriaxone",      # سفترياكسون
    # =================================================================
    # Flagyl: "flag yl" / "flagil" / "felagyl"
    # =================================================================
    "flajyl": "flagyl",               # فلاجيل
    # =================================================================
    # Ibuprofen: "ibu pro fen" / "ibuprofen" / "iboprufin"
    # =================================================================
    "aybwbrwfyn": "ibuprofen",        # ايبوبروفين
    "ybwbrwfyn": "ibuprofen",         # يبوبروفين
}


def _phonetic_alias_lookup(needle_translits: List[str]) -> Optional[str]:
    """Return the drug name if any of the translit variants of the
    flagged span is a known English-mishearing alias. Used as a final
    rescue when standard phonetic matching fails."""
    for n in needle_translits:
        if n in _PHONETIC_ALIAS:
            return _PHONETIC_ALIAS[n]
        # Also try the consonant skeleton in case 'awqmntyn' came in with
        # different vowel placement.
        sk = _consonant_skeleton_ar(n)
        for key, drug in _PHONETIC_ALIAS.items():
            if _consonant_skeleton_ar(key) == sk and len(sk) >= 3:
                return drug
    return None


def _is_likely_drug(term: str) -> bool:
    term = term.lower().strip()
    if term in _DRUG_HINT_TERMS:
        return True
    return any(term.endswith(suf) for suf in _DRUG_HINT_SUFFIXES)


def _phonetic_candidates(
    word: str, lexicon: List[str], k: int = 3,
    *, threshold: float = 0.45,
    min_skeleton_len: int = 3,
) -> List[Dict[str, Any]]:
    """Find up to `k` lexicon entries phonetically similar to `word`.

    Strategy: compare CONSONANT SKELETONS, not full strings.
      - Arabic 'برسيتامول' -> 'brsytamwl' -> consonant 'brstml'
      - Latin 'paracetamol' -> consonant skeleton 'brktml'
        (p->b, c->k, vowels dropped) -> sim ~0.67-0.83

    Ranking tiebreaker: when two candidates tie on similarity, the
    one classified as a DRUG (suffix -in/-ol/-ine/...) wins over a
    disease. This breaks 'برسي تمر' ties where bursitis and
    paracetamol both score 0.667 and we want paracetamol.
    """
    if len(word) < 2:
        return []
    needles = list({_translit(word, strip_clitics=True),
                    _translit(word, strip_clitics=False)})
    needles = [n for n in needles if len(n) >= 2]
    if not needles:
        return []
    needle_sks = [_consonant_skeleton_ar(n) for n in needles]
    scored = []
    for term in lexicon:
        term_lat = re.sub(r"[^a-z]", "", term.lower())
        if not term_lat:
            continue
        term_sk = _consonant_skeleton_latin(term_lat)
        if not term_sk:
            continue
        best = 0.0
        for n, n_sk in zip(needles, needle_sks):
            # Raw string compare (catches close matches).
            if _length_ratio_ok(n, term_lat):
                best = max(best, _lev_sim(n, term_lat))
            # Consonant-skeleton compare.
            if (len(n_sk) >= min_skeleton_len
                    and len(term_sk) >= min_skeleton_len
                    and _length_ratio_ok(n_sk, term_sk)):
                best = max(best, _lev_sim(n_sk, term_sk))
        if best < threshold:
            continue
        scored.append({
            "term": term,
            "phonetic_similarity": round(best, 3),
            "_is_drug": _is_likely_drug(term),
        })
    # Phonetic-alias rescue: if the flagged span literally matches a
    # known English-mishearing pattern (e.g. 'اف اول قن' = 'if all gone'
    # -> efferalgan), promote that drug to the top with similarity 0.95.
    alias_drug = _phonetic_alias_lookup(needles)
    if alias_drug:
        # Find or inject it as the top candidate.
        alias_idx = next(
            (i for i, c in enumerate(scored) if c["term"].lower() == alias_drug),
            None,
        )
        if alias_idx is not None:
            scored[alias_idx]["phonetic_similarity"] = max(
                scored[alias_idx]["phonetic_similarity"], 0.95
            )
        else:
            # Only add it if it's in the lexicon — keep the contract that
            # candidates come from the user's medical_terms.txt.
            if any(t.lower() == alias_drug for t in lexicon):
                scored.insert(0, {
                    "term": alias_drug,
                    "phonetic_similarity": 0.95,
                    "_is_drug": True,
                })

    # Sort:
    #   1. similarity DESC
    #   2. drugs before non-drugs at the same score
    #   3. smaller |len(needle) - len(term)| first  (insulin vs amoxicillin)
    #   4. number of matching letters at the SAME positions in needle DESC
    #      (tiebreaks 'هيبارين' = 'hybaryn' between aspirin and heparin —
    #      'h' aligns with 'heparin' but not 'aspirin', so heparin wins).
    needle_skel = needles[0] if needles else ""
    needle_len = len(needle_skel)

    def _longest_common_substring_len(term: str) -> int:
        """Length of the longest contiguous substring shared between
        `needle_skel` and `term`. Strong signal that the candidate is
        the right drug even when overall edit distance is mediocre
        ('ليفوثيروكسين' -> needle 'yfwthyrwksyn' shares 'thyr' with
        levothyroxine — 4 chars — but only 'f' with ceftriaxone)."""
        t = re.sub(r"[^a-z]", "", term.lower())
        if not t or not needle_skel:
            return 0
        # DP: O(n*m) — fine for short strings.
        n, m = len(needle_skel), len(t)
        prev = [0] * (m + 1)
        best = 0
        for i in range(1, n + 1):
            cur = [0] * (m + 1)
            ni = needle_skel[i - 1]
            for j in range(1, m + 1):
                if ni == t[j - 1]:
                    cur[j] = prev[j - 1] + 1
                    if cur[j] > best:
                        best = cur[j]
            prev = cur
        return best

    scored.sort(key=lambda d: (
        -d["phonetic_similarity"],
        not d["_is_drug"],
        abs(len(re.sub(r"[^a-z]", "", d["term"].lower())) - needle_len),
        -_longest_common_substring_len(d["term"]),
    ))
    # Drop the internal flag before returning.
    for s in scored:
        s.pop("_is_drug", None)
    return scored[:k]


def phonetic_pass(transcript: str) -> List[Dict[str, Any]]:
    """For each word in `transcript`, return flag records with phonetic
    candidates from the medical lexicon.

    Also tries pair-of-words (bigrams) against the lexicon — Gulf-LoRA
    Qwen3 frequently splits a single mangled drug name into two short
    tokens (e.g. 'paracetamol' -> 'برسي تمر') which match nothing on
    their own but match well when joined.
    """
    lexicon = load_medical_lexicon()
    if not lexicon:
        return []
    words = [w for w in re.split(r"\s+", transcript.strip()) if w]
    flags: List[Dict[str, Any]] = []
    consumed: set = set()

    # Stage A: Compute suspicion scores as a pre-filter BEFORE any phonetic matching.
    # Words scoring below SUSPICION_THRESHOLD are safe — skip entirely.
    # Suspicious words go to Stage B for phonetic candidate generation.
    # Context windows (2 words each side) are passed to score_suspicion()
    # so the LM perplexity and semantic coherence signals can fire.

    # --- LLM suspicion scorer (Stage A+): call once for the whole transcript ---
    # This makes a single API call — cache handles dedup within the 5-min TTL.
    # On failure (timeout/rate-limit), llm_scores is None and we fall through
    # to the algorithmic signals cleanly.
    llm_scores = llm_scorer.score_words(words, timeout=20.0)

    single_results: List[Optional[List[Dict[str, Any]]]] = []
    for i, word in enumerate(words):
        left_ctx = words[max(0, i - 2):i]
        right_ctx = words[i + 1:i + 3]
        llm_val = llm_scores.get(i, 0.0) if llm_scores is not None else None
        susp = score_suspicion(word, left_context=left_ctx, right_context=right_ctx,
                               llm_suspicion=llm_val)
        if susp < SUSPICION_THRESHOLD:
            single_results.append(None)
            continue
        single_results.append(_phonetic_candidates(word, lexicon, k=3))

    # --- Try n-grams first when there's a strong potential match. This
    # gives split drug names ('برسي تمر' -> paracetamol) priority over
    # any single component matching a different lexicon entry coincidentally
    # ('برسي' alone matches pleurisy 0.75, but joined with تمر it matches
    # paracetamol with even higher confidence in the consonant skeleton).

    # --- N-gram pass for the remaining (weak / unmatched) words: try
    # 3-grams then 2-grams. Drug names sometimes split into 2-3 tokens
    # ('paracetamol' -> برسي تمر, 'augmentin' -> اوغ من تين). Only
    # consider windows where ALL words are still unconsumed AND at
    # least one of them had a weak single-word match (>=0.45) — otherwise
    # we'd merge random non-medical words.
    def _try_ngram(n: int, threshold: float, filler_threshold: float) -> None:
        """N-gram pass. Two thresholds:
          - `threshold`: minimum similarity for an n-gram with NO filler
            words. Lower because the window is more likely to be a real
            mangled drug span.
          - `filler_threshold`: higher minimum for n-grams that include
            a filler word (article/preposition). Drug names sometimes
            need to bridge a filler to be recognised ('اوغ من تين' has
            'من' which is filler, but joined to neighbours it spells
            'augmentin'). Requires a stronger match to fire.
        """
        for i in range(len(words) - n + 1):
            if any((i + off) in consumed for off in range(n)):
                continue
            window = words[i:i + n]
            # Reject if any word in the window is pure-Latin or a digit:
            # combining a Latin drug with surrounding Arabic words via the
            # n-gram pass produces nonsense matches ('ventolin ٢' bigram
            # spuriously matches 'ventolin' again with the digit attached).
            if any(_is_pure_latin_or_digit(w) for w in window):
                continue
            # Reject if the conjunction 'و' (and) is the bridging word
            # for n >= 2 — it almost always separates two distinct drugs.
            # Without this 'سيلين و اوغ' from 'اموكسي سيلين و اوغمنتين'
            # joins across the conjunction and matches 'saline'.
            if n >= 2 and "و" in window[1:-1] if n >= 3 else False:
                pass  # placeholder
            if n >= 2 and any(w == "و" for w in window):
                continue
            filler_count = sum(1 for w in window if _is_arabic_filler(w))
            # Reject if more than half the window is filler (or for n=2,
            # if BOTH are filler — protects 'مع الاكل' false positive).
            if n == 2 and filler_count >= 2:
                continue
            if n == 3 and filler_count >= 2:
                continue
            has_filler = filler_count > 0
            joined = "".join(window)
            candidates = _phonetic_candidates(
                joined, lexicon, k=3, threshold=threshold,
            )
            if not candidates:
                continue
            top = candidates[0]
            min_score = filler_threshold if has_filler else threshold
            if top["phonetic_similarity"] < min_score:
                continue
            # Precision check for borderline n-grams: when the
            # similarity is in the noisy 0.55-0.65 range AND the
            # contiguous shared skeleton substring is only 2 chars or
            # less, drop the match. This kills false positives like a
            # person's name 'فواد علي النزار' which matches fluconazole
            # at 0.57 with LCS=2 — pure scattered-letter coincidence.
            # Real mangled drugs almost always have EITHER a higher
            # similarity (≥0.65) OR a 3+ contiguous shared substring.
            joined_skel = _consonant_skeleton_ar(_translit(joined))
            term_skel = _consonant_skeleton_latin(top["term"])
            lcs_len = _longest_common_substring(joined_skel, term_skel)
            if top["phonetic_similarity"] < 0.65 and lcs_len < 3:
                continue
            # Don't hijack a window when one of its component words has a
            # near-perfect single-drug match on its own. Example:
            # 'ابره انسولين' bigram matches prednisolone (sim 0.857), but
            # 'انسولين' alone matches insulin (sim 1.0). Prefer the single
            # only when it's a strong, well-matched single (>=0.85) AND its
            # term is similar length to the single word (not the joined
            # window). Otherwise (e.g. 'برسي' weakly matching pleurisy
            # 0.75) keep the bigram match for paracetamol.
            should_skip_bigram = False
            for off in range(n):
                sc = single_results[i + off]
                if not sc:
                    continue
                single_top = sc[0]
                single_sim = single_top["phonetic_similarity"]
                # 0.80 (was 0.85): 'البرسيتامول' alone matches
                # paracetamol at 0.833, but the bigram البرسيتامول+لمدة
                # also passes the n-gram threshold and was hijacking it.
                if single_sim < 0.80:
                    continue
                # Single must be a credible standalone match: needle
                # length ~ term length.
                from_word = words[i + off]
                from_translit = _translit(from_word)
                term = single_top["term"]
                ratio = (min(len(from_translit), len(term)) /
                         max(len(from_translit), len(term)))
                if ratio < 0.65:
                    continue
                # Don't let the single block the bigram when the bigram
                # is materially longer than the single AND covers it.
                # Example: 'سيلين' alone matches 'saline' (1.0), but the
                # bigram 'اموكسي سيلين' is the real drug 'amoxicillin'
                # and the single is just half of it. Joined-len > 1.7x
                # single-len is a strong signal of a split-drug.
                joined_translit = _translit(joined)
                if len(joined_translit) > 1.7 * len(from_translit):
                    continue
                # Single beats this bigram only if its score is genuinely
                # higher (not equal — n-gram wins ties since it's more
                # context-aware).
                if single_sim > top["phonetic_similarity"]:
                    should_skip_bigram = True
                    break
            if should_skip_bigram:
                continue
            flags.append({
                "index": i,
                "word": " ".join(window),
                "reason": f"phonetic_near_medical_{n}gram",
                "candidates": candidates,
                "span_indices": list(range(i, i + n)),
            })
            for off in range(n):
                consumed.add(i + off)

    _try_ngram(3, threshold=0.55, filler_threshold=0.75)
    _try_ngram(2, threshold=0.50, filler_threshold=0.70)

    # --- Single-word pass for words not absorbed by an n-gram. Catches
    # both already-correct drug spellings (panadol -> sim 1.0 same word,
    # skipped) and genuine single-token mangles (kuwaiteen -> codeine).
    for i, cands in enumerate(single_results):
        if i in consumed or not cands:
            continue
        top = cands[0]
        if top["phonetic_similarity"] < 0.55:
            continue
        # Skip only when the literal word IS already the Latin term.
        if words[i].lower() == top["term"].lower():
            continue
        # Precision check for borderline single matches: when sim is
        # only 0.55-0.65 AND the contiguous shared skeleton is only
        # 2 chars or less, drop it. Example: 'النزار' (skel 'nzr')
        # matches 'olanzapine' (skel 'lnzpn') at 0.6 by scattered-letter
        # coincidence (LCS=2). A real drug match usually scores ≥0.65
        # or has LCS≥3.
        if top["phonetic_similarity"] < 0.65:
            word_skel = _consonant_skeleton_ar(_translit(words[i]))
            term_skel = _consonant_skeleton_latin(top["term"])
            lcs = _longest_common_substring(word_skel, term_skel)
            if lcs < 3:
                continue
        flags.append({
            "index": i,
            "word": words[i],
            "reason": "phonetic_near_medical",
            "candidates": cands,
        })

    flags.sort(key=lambda f: f["index"])
    return flags


# ---------------------------------------------------------------------------
# Auto-detection: Arabic normalcy via lexicon skeleton matching
# ---------------------------------------------------------------------------

# Cache of Latin consonant skeletons from the medical lexicon (term + aliases).
# Precomputed once so _is_arabic_normalcy() can check efficiently whether an
# Arabic word could possibly be a medical transliteration.
_LEXICON_SKELETONS: List[str] = []
_LEXICON_SKELETONS_LOADED = False
# Path to the primary medical lexicon (JSONL with term/type/aliases/priority)
_LEXICON_MEDICAL_PATH = PROJECT_ROOT / "data" / "medical_lexicon.jsonl"


def _load_lexicon_terms() -> List[str]:
    """Load all terms + aliases from the medical_lexicon.jsonl file."""
    terms: List[str] = []
    if not _LEXICON_MEDICAL_PATH.exists():
        return terms
    with _LEXICON_MEDICAL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                term = entry.get("term", "")
                if term:
                    terms.append(term)
                for alias in entry.get("aliases", []):
                    if alias:
                        terms.append(alias)
            except json.JSONDecodeError:
                continue
    return terms


def _ensure_lexicon_skeletons() -> None:
    """Precompute Latin consonant skeletons for all lexicon terms + aliases.

    This is called once (lazily) and cached.  Each term is converted to a
    Latin consonant skeleton via _consonant_skeleton_latin(), which strips
    vowels and maps p→b, v→f, c→k, etc.  The resulting set of skeletons is
    used by _is_arabic_normalcy() to check if an Arabic word's skeleton
    could possibly match any known medical term.
    """
    global _LEXICON_SKELETONS, _LEXICON_SKELETONS_LOADED
    if _LEXICON_SKELETONS_LOADED:
        return
    terms = _load_lexicon_terms()
    seen: set = set()
    for term in terms:
        lat = re.sub(r"[^a-z]", "", term.lower())
        # Skip very short terms — they produce trivial skeletons
        if not lat or len(lat) < 4:
            continue
        sk = _consonant_skeleton_latin(lat)
        if sk and len(sk) >= 3:
            seen.add(sk)
    _LEXICON_SKELETONS = list(seen)
    _LEXICON_SKELETONS_LOADED = True


def _clear_lexicon_skeleton_cache() -> None:
    """Invalidate the skeleton cache so new lexicon terms are picked up.

    Called when the user teaches a new term via /api/teach.
    """
    global _LEXICON_SKELETONS, _LEXICON_SKELETONS_LOADED
    _LEXICON_SKELETONS = []
    _LEXICON_SKELETONS_LOADED = False


def _is_arabic_normalcy(word: str) -> bool:
    """Check if an Arabic word is almost certainly normal Arabic (not a
    medical transliteration), by verifying that its consonant skeleton does
    NOT match any term in the medical lexicon above a low threshold.

    How it works:
      1. Transliterate the Arabic word to Latin.
      2. Compute the Arabic consonant skeleton (strips vowels, 'w', 'y'
         that represent long vowel markers, and digraph 'h').
      3. If the skeleton is too short (< 3 chars), it cannot be a meaningful
         medical transliteration → return True (normal Arabic).
      4. Compare against ALL precomputed Latin lexicon skeletons using
         fuzzy Levenshtein ratio.
      5. If ANY lexicon skeleton scores >= 40% similarity, the word COULD
         be a medical transliteration → return False.
      6. If no match above threshold, the word is normal Arabic → return True.

    The 40% threshold is calibrated to be MORE permissive than the
    phonetic_candidates threshold (45%), ensuring that we never block a
    genuine transliteration at this gate.  Words that pass through here
    (return False) still go through the full scoring pipeline which has
    a much higher acceptance bar (~80%).

    Returns True if the word should be treated as normal Arabic (skip it).
    Returns False if the word COULD be a medical transliteration.
    """
    # Only applies to Arabic-script words
    if not _ARABIC_LETTER_RE.search(word):
        return True

    _ensure_lexicon_skeletons()
    # If the lexicon is empty, we can't check — let the word through
    # (the downstream scoring pipeline will handle it).
    if not _LEXICON_SKELETONS:
        return False

    # Transliterate and strip clitics (e.g. الـ prefix)
    translit = _translit(word, strip_clitics=True)
    if not translit or len(translit) < 3:
        return True  # Too short to be a meaningful medical transliteration

    # Compute Arabic consonant skeleton
    arabic_sk = _consonant_skeleton_ar(translit)
    if not arabic_sk or len(arabic_sk) < 3:
        return True  # Skeleton too short for meaningful comparison

    # Quick length-ratio pre-filter: if the Arabic skeleton is less than
    # 35% the length of the longest Latin skeleton, skip immediately.
    # Most Arabic transliterations have skeletons 3-7 chars (same as Latin).
    for latin_sk in _LEXICON_SKELETONS:
        if len(latin_sk) < 3:
            continue
        # Length ratio pre-filter: skeletons must be within 65% of each other
        len_ratio = len(arabic_sk) / max(1, len(latin_sk))
        if len_ratio < 0.35 or len_ratio > 3.0:
            continue

        # Fuzzy score on consonant skeletons
        sim = float(_rapidfuzz.ratio(arabic_sk, latin_sk))
        # Threshold: >= 40% similarity → potentially a transliteration
        if sim >= 40.0:
            return False  # Could be a medical transliteration — don't block

    # No match found above threshold → almost certainly normal Arabic
    return True


# ---------------------------------------------------------------------------
# Arabic filler words: pronouns, prepositions, common verbs, conjunctions.
# These can sometimes phonetic-match a disease name by accident (e.g.
# 'الاكل' -> 'flagyl'). We never flag them.
# ---------------------------------------------------------------------------

_ARABIC_FILLER = {
    # particles & prepositions
    "و", "في", "من", "الى", "على", "عن", "مع", "بعد", "قبل", "لو",
    "اذا", "ان", "انت", "انا", "هو", "هي", "هم", "هذا", "هذه", "ذلك",
    "كل", "لا", "ما", "لم", "لن", "قد", "ثم", "او", "اي", "كما", "تحت",
    "فوق", "بين", "حول", "بدون", "غير", "نفس", "بنفس",
    # greetings & polite forms
    "السلام", "سلام", "عليكم", "مرحبا", "اهلا", "أهلا", "شكرا", "عفوا",
    "معليش", "معلش", "ياليت", "ياريت", "لو", "لكن", "بس", "بس",
    # common verbs (Gulf dialect)
    "بدا", "بدأ", "يبدا", "يبدأ", "يشتكي", "اشتكى", "أخذ", "أخد", "ياخذ",
    "يأخذ", "بينت", "بين", "نبين", "تبين", "ونسوي", "نسوي", "سوينا",
    "يسوي", "تم", "لاحظ", "لاحظنا", "يلاحظ", "ظهر", "يظهر", "يقول",
    "قال", "قلت", "نقول",
    # common adjectives
    "بسيط", "خفيف", "خفيفة", "شديد", "شديدة", "مزمن", "مزمنة", "حاد", "حادة",
    # pronouns & demonstratives
    "فيه", "فيها", "فيهم", "عليه", "عليها", "عليهم",
    "هذا", "هذه", "ذلك", "هذي", "هذول", "الي", "اللي",
    "إنه", "أنه", "إنها", "أنها", "إنهم", "أنهم",
    # conjunctions & adverbs
    "كذلك", "أيضا", "ايضا", "هكذا", "كمان", "برضه", "برضو",
    "حاليا", "سابقا", "لاحقا", "اخر", "آخر", "اخرى", "أخرى",
    # anatomical / medical context words (NOT drug/disease names)
    "يمتد", "لليسار", "لليمين", "يسار", "يمين",
    "متابعة", "فحص", "نتيجة", "نتيجه", "تشخيص", "علاج",
    # common verbs (Gulf imperatives + frequent forms)
    "خذ", "خذي", "خذو", "خود", "اخذ", "اخذي", "تاخذ", "تاخذي",
    "قال", "قالت", "قلت", "اعطاني", "اعطته", "اعطيه", "اعطاء", "أعطاء",
    "استعمل",
    "استعملي", "ابي", "اروح", "احس", "تعبان", "وصف", "خليه",
    "خليني", "روح", "تعال", "اجلس", "ينفع", "يصحى", "يطلب",
    # code-switch function words (Arabic-script renditions of English words
    # that are NOT medical terms — must not be matched against the lexicon)
    "اوف", "أوف", "اف", "أن", "ال", "ذ", "ذيز", "ذات",
    "فور", "فار", "ان", "اند", "بت", "باي",
    # Arabic prepositional phrases & pronouns
    "ومعاه", "ومعها", "معاه", "معها", "عنده", "عندها", "عندي",
    "عندك", "عندكم", "اللي", "الي", "الذي", "التي",
    "بعدين", "هني", "هناك", "دائما", "احيانا", "كثير", "شوي",
    # body / symptom / anatomy words
    "صداع", "دوخه", "تعب", "حرارة", "الم", "وجع", "ضيق",
    "نفس", "ربو", "سكر", "ضغط", "ظهر", "ظهري", "حلق", "بطن",
    "كتف", "كتفي", "رقبه", "رقبتي", "راس", "راسي", "عين", "عيون",
    "اذن", "اذني", "انف", "فم", "اسنان", "يد", "يدي", "رجل", "رجلي",
    "قدم", "قدمي", "ركبه", "ركبتي", "مفاصل", "عضلات", "عظام",
    "قلب", "قلبي", "صدر", "صدري", "معده", "كبد", "كلى", "كلية",
    "دم", "بول", "براز", "شعر", "جلد", "الجلد", "النبض", "الضغط",
    "العين", "الاذن", "النوم", "النوبه", "نوبه", "السعال", "سعال",
    "العمليه", "العملية", "البلعوم", "الانف", "الاطفال", "العشاء",
    "الفطور", "الغداء",
    # time words
    "اليوم", "ساعه", "ساعات", "يوم", "اسبوع", "اسبوعي", "اسبوعيه",
    "شهر", "صباحا", "مساء", "ليل", "نهار", "السبت", "الاحد",
    "الاثنين", "الثلاثاء", "الاربعاء", "الخميس", "الجمعه",
    # dosage / form words
    "مرات", "مرتين", "مره", "حبه", "حبتين", "حبوب", "شراب", "كاسة",
    "كاسه", "ماي", "ماء", "ابره", "بخاخ", "تحاميل", "جل", "جرعتين",
    "جرعه", "كبسوله", "كبسولات", "نقطه", "نقاط", "قطره", "قطرات",
    "ملليجرام", "ميكروجرام", "جرام", "وحده", "وحدات",
    # numbers (essential to suppress vital-signs false positives)
    "مية", "مئة", "خمسماية", "خمسميه", "خمسمائه", "مئتين",
    "واحد", "اثنين", "ثلاثه", "ثلاث", "اربعه", "اربع", "خمسه", "خمس",
    "ستة", "ست", "سبعه", "سبع", "ثمانيه", "ثمان", "تسعه", "تسع",
    "عشره", "عشر", "عشرين", "ثلاثين", "اربعين", "خمسين",
    "ستين", "سبعين", "ثمانين", "تسعين", "وعشرين", "وثلاثين",
    "ومايه", "ولفين", "ثلث", "ربع", "نصف",
    # honorifics / roles
    "الدكتور", "دكتور", "الطبيب", "طبيب", "الصيدلي", "صيدلي",
    "ابني", "امي", "ابي", "اختي",
    "اخوي", "خالي", "خالتي", "عمي", "عمتي", "جدي", "جدتي",
    "المريض", "المريضه", "الوصفه", "الوصفة", "الفحص", "تحليل",
    "اشعه", "اشعة", "صوره", "صورة", "موعد", "اخصائي", "طبيب",
    # general medical context words (NOT drug names)
    "علاج", "دواء", "ادويه", "وصفة", "وصفه", "مستشفى", "صيدليه",
    "صيدلية", "عيادة", "عياده", "نتيجه", "نتيجة", "تحاليل",
    "التهاب", "التهابات", "مرض", "امراض", "اعراض", "عرض",
    "حساسيه", "حساسية",
    # food / drink (frequently appears in dosing instructions)
    "اكل", "الاكل", "طعام", "الطعام", "اكله", "اكلات", "وجبه",
    "وجبات", "افطار", "غداء", "عشاء", "سحور", "افطر", "تفطر",
    "شرب", "شراب", "عصير", "عصائر", "ماء", "ماي", "حليب",
    # common Arabic first names (suppress 'الدكتور <name>' false flags)
    "محمد", "احمد", "علي", "حسن", "حسين", "ابراهيم", "اسماعيل",
    "يوسف", "ادم", "موسى", "عيسى", "نوح", "خالد", "سعد", "سعيد",
    "سالم", "سلمان", "سليمان", "صالح", "ناصر", "فهد", "فيصل",
    "بدر", "ماجد", "طلال", "عبدالله", "عبدالرحمن", "عبدالعزيز",
    "عبدالكريم", "عبدالمجيد", "عمر", "عثمان", "ابوبكر", "بكر",
    "زيد", "ياسر", "فؤاد", "فواد", "كريم", "نبيل", "وليد", "هاني",
    "طارق", "ايمن", "سامي", "اسامه", "اسامة", "حمد", "حمدان",
    "راشد", "سيف", "زايد", "منصور", "سلطان", "حربي", "مطلق",
    "فاطمه", "فاطمة", "عائشه", "عائشة", "خديجه", "خديجة", "مريم",
    "زينب", "هدى", "نوره", "نورة", "موضي", "نوف", "ساره", "سارة",
    "هند", "ريم", "لمى", "شهد", "غلا", "العنود", "الجوهره",
    # Tribal/family-name particles (al-, ibn-, abu-, umm-)
    "ابو", "أبو", "ام", "أم", "ابن", "بنت", "بن", "بنت",
    # time / quantity words
    "سنة", "سنه", "سنين", "سنوات", "سنتين",
    "شهر", "شهرين", "اشهر", "أشهر",
    "حوالي", "تقريبا", "تقريباً",
    "ساعة", "ساعه", "ساعتين",
    # verbs / adjectives (Gulf medical context)
    "يمتد", "تمتد",
    "مجهود",
    "جا", "جات", "يجي",
    "خلص", "خلاص", "يخلص",
    "باقي",
    # nouns / qualifiers
    "الألم", "الالم",
    "الأم",
    "شي", "شى",
    "لان", "لأن", "لانه", "لأنه",
    "وهو", "وهي", "وهم",
    "كيف", "اشلون", "شلون",
    # identity / possession words
    "اسم", "اسمي", "اسمك", "اسمه", "اسمها", "عمري", "عمرك", "عمره",
    "بلدي", "بلدك", "جنسيتي", "رقمي", "هاتفي", "تلفوني",
    # Gulf Arabic clinical words (common in dictations, not drug names)
    "مريض", "مرتفع", "منخفض", "طبيعي", "ممتاز",
    "لازم", "ضروري", "ممكن", "لابد", "فقط",
    "وايد", "مره", "قليل", "كثير",
    "قسنا", "قست", "يقيس",
    "نغير", "يغير", "غيرنا",
    "نحتاج", "احتاج", "يحتاج",
    "نقدر", "يقدر", "قدرنا",
    "راح", "نروح", "يروح",
    "مال", "حساب",
    "حق", "بخصوص",
    "لذلك", "لهذا", "هكذا",
    "بعض", "نفس",
    # additional anatomy / symptom words
    "جرح", "كسر", "خلع", "ورم", "نزيف", "حروق",
    # additional time / quantity words
    "دقيقه", "دقيقة", "دقايق", "دقائق",
    # ================================================================
    # Common Gulf clinical words (MUST be here to prevent false
    # positives from English transliteration matching).
    # These are normal Arabic words — NOT medical transliterations.
    # ================================================================
    # Verbs — daily actions in clinical dictation
    "حضر", "يحضر", "حضور",
    "بدأ", "بدا", "بدأت", "بدات", "يبدأ", "يبدا",
    "صار", "صارت", "وصار", "وصارت",
    "عمل", "يعمل", "أعمل", "نعمل", "تعمل",
    "عملنا", "عملت", "عملوا",
    "أعمل", "تعمل", "نعمل",
    "شمل", "يشمل", "تشمل", "يتضمن", "تتضمن",
    "أظهر", "اظهر", "أظهرت", "اظهرت", "وأظهرت", "تظهر", "يظهر",
    "يحتاج", "تحتاج", "احتاج", "نحتاج",
    "يدخل", "ادخال", "إدخال", "يدخل", "إجراء", "اجراء", "أجرى",
    # Nouns — common in clinical reports
    "عام", "شامل",
    "خفان", "خفقان", "خفكان", "خفقان",
    "تنفس", "التنفس",
    "نبض", "نبضة", "نبضات",
    "نسبة", "نسب",
    "أكسجين", "اكسجين", "الأكسجين", "الاكسجين",
    "أوكسجين", "اوكسجين",
    "شفة", "شفتين", "الشفتين",
    "ساق", "ساقين", "الساقين",
    "ماض", "ماضية", "ماضيين", "الماضيين",
    "اضطراب", "اضطرابات", "الاضطراب",
    "ضرابات",  # common Gulf misspelling of اضطرابات
    "اضظراب", "اضظرابات",  # additional common misspellings
    "نسيان", "النسيان",
    "تورم", "انتفاخ",
    "ازرقاق", "زرقة",
    "مخبري", "مخبرية", "مخبريه",
    "ارتفاع", "ارتفاح",
    "هيموغلوبين",
    "ارتشاح", "ارتشاحات",
    "رئة", "الرئة", "رئوي",
    "تخطيط",
    "احتشاء",
    "حديث",
    "مضاد", "مضادات",
    "حيوي", "حيوية", "حيويه", "الحيوية",
    "علامة", "علامات",
    "تسارع",
    "نقص", "ناقص",
    "ألم", "الم", "الالم", "الألم",
    "إصابة", "اصابة",
    "فحوصات", "اختبار", "اختبارات",
    "جراحة", "عمليه", "عملي",
    "أشعة", "اشعة",
    "إبرة", "ابره", "ابرة",
    # Adjectives — clinical descriptions
    "أسوأ", "اسوأ", "أسوء",
    "منتظم", "منتضام",
    "تدريجي", "تدريجية", "تدريجيا", "تدريجياً",
    "واضح", "واضحة", "واضحه",
    "حالي", "حالية", "الحالي", "الحالية",
    # Prepositions / conjunctions / particles
    "بشكل",
    "انتظام",  # allows بانتظام (ب+انتظام) via clitic stripping
    "إنتظام",  # hamza variant allows بإنتظام (ب+إنتظام)
    "بسبب",
    "أكثر", "اكثر",
    "أيام", "ايام",
    "بالمئة", "بالمائة", "بالميه",
    # Numbers (verbal)
    "أربعة", "اربعة",
    "ستة",
    "سبعة",
    "ثمانية",
    "تسعة",
    # Hamza forms (أ/إ variants of existing filler words)
    # These MUST be here to prevent the Arabic spelling corrector
    # from "correcting" أ→ا (which is wrong direction in MSA).
    "أعراض",  # symptoms (with hamza above) — NOT اعراض
    "إعطاء",  # giving (with hamza below) — NOT اعطاء
    "ألم", "الالم", "الألم",  # pain (with hamza above)
    "أخذ", "يأخذ", "تأخذ",  # take/takes
    "أسبوع", "أسبوعين", "أسابيع",  # week/weeks
    "إصابة", "أصابة",  # injury
    # Additional hamza variants for common words
    "إلا", "ألا",
    "إلى",
    "أي",
    "أين",
    "أمام",
}


def _is_arabic_filler(word: str) -> bool:
    """Check if an Arabic word is a known filler/normal word that should
    NOT be treated as a potential medical transliteration.

    Single-gate design: relies entirely on _is_arabic_normalcy(), which
    compares the word's consonant skeleton against the full medical lexicon.
    If no skeleton reaches 40% similarity, the word is almost certainly
    normal Arabic → treat it as filler.

    NOTE: The _ARABIC_FILLER set (defined above) is intentionally NOT used
    as a fast path here. A manual whitelist is fragile — it must be kept in
    sync with the lexicon and clinical vocabulary. The auto-detection via
    consonant skeleton matching handles the full distribution of Arabic
    clinical words without manual maintenance. The _ARABIC_FILLER set is
    preserved only for backward compatibility (correction.py imports it
    as a vocabulary for the Arabic spelling corrector).

    Returns True if the word is normal Arabic (skip suspicion entirely).
    Returns False if the word COULD be a medical transliteration.
    """
    if _ARABIC_LETTER_RE.search(word):
        return _is_arabic_normalcy(word)
    return False


# ---------------------------------------------------------------------------
# Feedback loop: record high-confidence corrections so Stage A learns
# which Arabic transliteration skeletons are genuinely medical.
# ---------------------------------------------------------------------------

# Maps (consonant_skeleton, language_tag) -> count of confirmed corrections.
# Populated by _record_correction() calls from apply_high_confidence_corrections().
# Stage A uses this to boost suspicion scores for skeletons that have been
# corrected before, making the pipeline self-improving over time.
_CORRECTION_FEEDBACK: Dict[Tuple[str, str], int] = {}


def _record_correction(original_word: str, corrected_term: str) -> None:
    """Record a high-confidence correction so Stage A becomes more
    sensitive to the same transliteration skeleton in future transcripts.

    The key is (consonant skeleton of original, 'ar'|'en') so we learn
    specific transliteration patterns rather than generic boosts.
    """
    if not original_word or not corrected_term:
        return
    is_arabic = bool(_ARABIC_LETTER_RE.search(original_word))
    tag = 'ar' if is_arabic else 'en'
    # Compute the consonant skeleton of the original word after transliteration
    if is_arabic:
        translit = _translit(original_word, strip_clitics=True)
        sk = _consonant_skeleton_ar(translit) if len(translit) >= 3 else original_word.lower()
    else:
        sk = _consonant_skeleton_latin(original_word.lower())
        if not sk or len(sk) < 3:
            sk = original_word.lower()[:6]
    key = (sk, tag)
    _CORRECTION_FEEDBACK[key] = _CORRECTION_FEEDBACK.get(key, 0) + 1
    # Also record a more general version: first 4 chars of the skeleton
    # so 'brstml' (paracetamol) and 'brsytml' both contribute to 'brs'
    # Only if the prefix is different from the full key (avoids double-count
    # when skeleton is exactly 4 chars, e.g. 'hstr' for 'history').
    if len(sk) >= 4:
        gen_key = (sk[:4], tag)
        if gen_key != key:
            _CORRECTION_FEEDBACK[gen_key] = _CORRECTION_FEEDBACK.get(gen_key, 0) + 1


def _get_feedback_boost(word: str) -> float:
    """Check if this word's skeleton has been seen in previous high-confidence
    corrections. Returns a boost value 0.0-0.20 based on correction count."""
    if not _CORRECTION_FEEDBACK:
        return 0.0
    is_arabic = bool(_ARABIC_LETTER_RE.search(word))
    tag = 'ar' if is_arabic else 'en'
    # Compute skeleton the same way _record_correction does
    if is_arabic:
        translit = _translit(word, strip_clitics=True)
        sk = _consonant_skeleton_ar(translit) if len(translit) >= 3 else word.lower()
    else:
        sk = _consonant_skeleton_latin(word.lower())
        if not sk or len(sk) < 3:
            return 0.0
    # Check exact match
    count = _CORRECTION_FEEDBACK.get((sk, tag), 0)
    # Check partial match (first 4 chars of skeleton)
    if count == 0 and len(sk) >= 4:
        count = _CORRECTION_FEEDBACK.get((sk[:4], tag), 0)
    # Check partial match (first 3 chars)
    if count == 0 and len(sk) >= 3:
        count = _CORRECTION_FEEDBACK.get((sk[:3], tag), 0)
    if count <= 0:
        return 0.0
    # Scale: 1 correction → 0.05, 2 → 0.08, 3+ → 0.12
    # (deliberately modest so feedback doesn't dominate other signals)
    if count == 1:
        return 0.05
    elif count == 2:
        return 0.08
    elif count == 3:
        return 0.10
    else:
        return min(0.15, 0.10 + 0.02 * (count - 3))


# ---------------------------------------------------------------------------
# Stage A: Suspicion Scoring — lexicon-independent pre-filter
# ---------------------------------------------------------------------------

# Suspicion score constants (0.0 = definitely safe, higher = more suspicious)
SUSPICION_SCORE_NONE = 0.0      # Definitely not suspicious — skip entirely
SUSPICION_SCORE_LOW = 0.05      # Minimal baseline suspicion (default for Latin)
SUSPICION_SCORE_MED = 0.15      # Medium suspicion (Arabic, not a known filler)
SUSPICION_SCORE_HIGH = 0.50     # High suspicion

# Stage A fusion threshold: words scoring >= this enter Stage B
SUSPICION_THRESHOLD = 0.10

# LM perplexity scaling: typical word PPL is 0.5-5.0, errors can be 10+
# We map PPL to 0-1 via: suspicion = 1 - exp(-PPL / LM_PPL_SCALE)
LM_PPL_SCALE = 4.0

# Weights for algorithmic baseline (when LLM is not available, or as the
# pre-gate score when LLM is available). Must sum to 1.0.
WEIGHT_NO_LLM_NORMALCY    = 0.30
WEIGHT_NO_LLM_PERPLEXITY  = 0.35
WEIGHT_NO_LLM_SEMANTIC    = 0.20
WEIGHT_NO_LLM_FEEDBACK    = 0.15

# ---------------------------------------------------------------------------
# Lazy-loaded n-gram language model for context perplexity scoring
# ---------------------------------------------------------------------------

_LM_CACHE: Optional['NGramLM'] = None


def _load_lm() -> Optional['NGramLM']:
    """Load the trained n-gram LM lazily (first call caches it)."""
    global _LM_CACHE
    if _LM_CACHE is not None:
        return _LM_CACHE
    try:
        from pathlib import Path
        from .ngram_lm import NGramLM
        pkl_path = Path(__file__).resolve().parent / "medical_lm.pkl"
        if pkl_path.exists():
            _LM_CACHE = NGramLM.load(str(pkl_path))
            return _LM_CACHE
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# English words that are common in clinical transcripts — safe-list
# ---------------------------------------------------------------------------

_COMMON_ENGLISH: set = {
    "that", "this", "these", "those", "which", "what", "when", "where",
    "there", "their", "them", "they", "your", "yours", "some", "such",
    "than", "then", "also", "very",
    "been", "being", "having", "doing", "make", "made", "take", "took",
    "taken", "given", "said", "tell", "told", "know", "known", "think",
    "need", "needs", "want", "wants", "like", "come", "came", "feel",
    "felt", "keep", "keeps", "show", "start", "starts", "stop", "stops",
    "call", "calls", "use", "uses", "used", "using", "see", "saw", "seen",
    "look", "looks", "find", "finds", "found", "help", "helps", "work",
    "works", "worked",
    "about", "above", "after", "again", "against", "almost", "along",
    "always", "around", "away", "back", "before", "behind", "below",
    "between", "beyond", "close", "during", "early", "ever", "every",
    "first", "forward", "here", "high", "inside", "last", "later",
    "left", "long", "more", "much", "near", "never", "next", "often",
    "once", "only", "open", "other", "outside", "over", "past",
    "quite", "rather", "really", "right", "since", "still", "sure",
    "through", "thus", "today", "together", "tomorrow", "under",
    "until", "upon", "well", "while", "whole", "within", "without",
    "today", "yesterday", "tomorrow", "morning", "afternoon", "evening",
    "night", "week", "weeks", "month", "months", "year", "years",
    "time", "times", "hour", "hours", "minute", "minutes", "day", "days",
    "patient", "doctor", "nurse", "hospital", "clinic", "office", "home",
    "family", "people", "child", "wife", "husband", "mother", "father",
    "brother", "sister", "baby", "adult",
    "able", "better", "best", "clear", "cold", "common", "different",
    "easy", "full", "good", "great", "hard", "heavy", "hot", "important",
    "large", "little", "major", "minor", "normal", "old", "other",
    "poor", "possible", "quick", "ready", "right", "same", "second",
    "short", "simple", "small", "strong", "sure", "true", "usual",
    "warm", "weak", "wide", "young",
    "reason", "result", "results", "cause", "problem", "problems",
    "issue", "issues", "condition", "change", "changes", "check",
    "checked", "course", "plan", "plans", "care", "test", "tests",
    "tested", "report", "reports", "noted", "note", "notes", "list",
    "type", "types", "set", "group", "number", "level", "levels",
    "point", "points", "part", "parts", "way", "ways", "side", "sides",
    "step", "steps", "case", "cases", "sample", "samples", "value",
    "values", "range", "ranges", "rate", "rates", "count", "counts",
    "total", "average", "current", "previous", "initial", "final",
    "stable", "typical", "routine", "regular", "frequent", "severe",
    "mild", "moderate", "slight", "slightly",
    # Clinical context words (frequently appear in dictation, not medical terms)
    "please", "review", "department", "biopsy", "examination",
    "examine", "examined", "continuing", "continue", "continued",
    "status", "treatment", "therapy", "infection", "disease",
    "team", "clinic", "follow", "followed", "refer", "referred",
    "assessment", "evaluation", "consultation", "admission",
    "discharge", "discharged", "transfer", "transferred",
    "require", "requires", "required", "provide", "provided",
    "perform", "performed", "schedule", "scheduled",
    "complete", "completed", "consider", "considered",
    "recommend", "recommended", "monitor", "monitored",
    "remain", "remains", "remained", "started",
    "setting", "already", "promyelocytic", "leukemia",
    "hematology", "all", "trans", "retinoic", "acid",
    "coagulopathy", "risk", "clean", "physical",
    "unremarkable", "here", "place", "time",
    "vital", "signs", "stable",
    "scan", "mri", "ct", "xray", "x-ray", "ultrasound",
    "lab", "labs", "result", "results",
    "symptom", "symptoms", "complaint", "complaints",
    "chief", "pain", "fever", "cough", "nausea", "vomiting",
    "diarrhea", "headache", "fatigue", "weakness", "weight",
    "appetite", "sleep",
    "both", "upper", "lower", "left", "right", "central",
    "proximal", "distal", "medial", "lateral", "anterior",
    "posterior", "superior", "inferior",
    "blood", "pressure", "heart", "rate", "temperature",
    "respiratory", "saturation", "oxygen",
    "body", "mass", "index", "bmi",
    "the", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "hundred", "thousand",
    "single", "double", "triple",
}


def _normalize_ppl(ppl: float) -> float:
    """Map LM perplexity to a 0-1 score.
    PPL=0 → 0.0 (very expected), PPL=10 → ~0.92 (very surprising)."""
    return 1.0 - math.exp(-ppl / LM_PPL_SCALE)


def _context_perplexity(word: str, left_context: List[str]) -> float:
    """Compute LM perplexity of word given its left context.

    Returns a 0-1 suspicion score, or 0.0 if no LM is available.
    """
    lm = _load_lm()
    if lm is None:
        return 0.0
    try:
        ppl = lm.word_perplexity(word, left_context)
        return _normalize_ppl(ppl)
    except Exception:
        return 0.0


def _semantic_coherence(word: str, left_context: List[str], right_context: List[str]) -> float:
    """Compute context-aware coherence score.

    Detects script-mismatch anomalies: an English word surrounded by
    Arabic (or vice versa) is more suspicious because the ASR may have
    produced a wrong word that happens to be in the wrong script.

    Also detects when a word's character shape is unusual for its
    script (e.g., mixed Arabic-Latin letters in a single token).

    Returns 0.0-1.0 where higher = more suspicious.
    """
    score = 0.0

    # Signal 1: Script mismatch with context
    has_arabic = bool(_ARABIC_LETTER_RE.search(word))
    has_latin = bool(re.search(r'[a-zA-Z]', word))

    # Count Arabic/Latin in context
    ctx_words = left_context[-2:] + right_context[:2]
    ctx_arabic = sum(1 for w in ctx_words if _ARABIC_LETTER_RE.search(w))
    ctx_latin = sum(1 for w in ctx_words if re.search(r'[a-zA-Z]', w))
    total_ctx = len(ctx_words)

    if total_ctx >= 2:
        # If word is Arabic but most context is Latin (or vice versa): suspicious
        if has_arabic and not has_latin:
            # Arabic word in mostly Latin context → could be transliteration
            if ctx_latin >= ctx_arabic and ctx_latin >= 2:
                score = max(score, 0.15)
        elif has_latin and not has_arabic:
            # Latin word in mostly Arabic context → could be ASR hallucination
            if ctx_arabic >= ctx_latin and ctx_arabic >= 2:
                score = max(score, 0.10)

    # Signal 2: Mixed-script tokens are almost always ASR errors
    if has_arabic and has_latin:
        score = max(score, 0.30)

    # Signal 3: Pure digits surrounded by letters (likely a measurement)
    # Not suspicious — handled by _ARABIC_DIGIT_RE earlier.

    return score


def score_suspicion(
    word: str,
    left_context: Optional[List[str]] = None,
    right_context: Optional[List[str]] = None,
    llm_suspicion: Optional[float] = None,
) -> float:
    """Stage A: Return a suspicion score for a word by fusing multiple
    lexicon-independent signals.

    Returns a value 0.0–1.0 where:
      0.0 = definitely not suspicious (safe — skip correction)
      >= SUSPICION_THRESHOLD (0.10) = potentially suspicious

    Fusion method: MULTIPLICATIVE LLM GATING (not additive weights).

    First, the algorithmic baseline is computed from 4 signals:
      1. Normalcy (30%): Arabic auto-detection via _is_arabic_normalcy().
         Normal Arabic words → 0.0. Possible transliterations → 0.3+.
         Latin words not in _COMMON_ENGLISH → 0.05 baseline.
      2. LM Perplexity (35%): does this word fit its context? N-gram LM
         computes -log P(word | context). Higher = more suspicious.
      3. Semantic Coherence (20%): script-mismatch detection.
      4. Feedback Loop (15%): prior high-confidence corrections boost.

    Then, when LLM signal IS available, the algorithmic baseline is
    gated multiplicatively:
      - LLM says "not suspicious" (0.0): dampen algorithmic by 5× (×0.20)
        so normal Arabic words fall below SUSPICION_THRESHOLD.
      - LLM says "suspicious" (1.0): amplify algorithmic by 2× (×2.0).
      - Intermediate values (0.0-1.0): linearly interpolate gate factor
        between 0.20 and 2.0.

    When LLM signal is NOT available (API failure, rate-limit), pure
    algorithmic score is returned.

    This multiplicative approach was chosen because an additive LLM
    weight could not override the noisy n-gram LM perplexity signal
    (which assigns PPL~10 to all OOV Arabic words, contributing ~0.32
    to the fused score at 35% weight). With multiplicative gating,
    the LLM's "not suspicious" verdict (0.0) damps the entire
    algorithmic signal below threshold, while "suspicious" amplifies
    it strongly.

    Words scoring below SUSPICION_THRESHOLD skip Stage B entirely.
    """
    # --- Hard gates: never suspicious ---

    # Digits are never suspicious
    if _ARABIC_DIGIT_RE.match(word):
        return SUSPICION_SCORE_NONE

    # Very short words can't be meaningful ASR errors for medical terms
    if len(word) < 3:
        return SUSPICION_SCORE_NONE

    # Pure Latin, known common English → never suspicious
    if not _ARABIC_LETTER_RE.search(word):
        if word.lower() in _COMMON_ENGLISH:
            return SUSPICION_SCORE_NONE

    # --- Compute signals ---

    signal_normalcy = 0.0
    signal_perplexity = 0.0
    signal_semantic = 0.0

    # Signal 1: Arabic normalcy / Latin baseline
    if _ARABIC_LETTER_RE.search(word):
        # Arabic script: if _is_arabic_normalcy() says it's normal → 0.0
        # If it could be a transliteration → base suspicion
        if _is_arabic_normalcy(word):
            signal_normalcy = 0.0  # Normal Arabic word, skip entirely
        else:
            signal_normalcy = 0.30  # Could be a medical transliteration
    else:
        # Latin word not in common English → low baseline
        signal_normalcy = 0.05

    # Signal 2: LM perplexity (requires left context)
    if left_context is not None and len(left_context) > 0:
        signal_perplexity = _context_perplexity(word, left_context)

    # Signal 3: Semantic coherence (script-mismatch detection)
    if right_context is not None:
        signal_semantic = _semantic_coherence(word, left_context or [], right_context)

    # Signal 4: Feedback loop — has this skeleton been corrected before?
    signal_feedback = _get_feedback_boost(word)

    # --- Compute algorithmic baseline (always the same) ---
    algorithmic = (
        signal_normalcy * WEIGHT_NO_LLM_NORMALCY +
        signal_perplexity * WEIGHT_NO_LLM_PERPLEXITY +
        signal_semantic * WEIGHT_NO_LLM_SEMANTIC +
        signal_feedback * WEIGHT_NO_LLM_FEEDBACK
    )

    # --- Fuse signals ---
    if llm_suspicion is not None:
        # Multiplicative LLM gating:
        #   LLM says "not suspicious" (0.0): dampen algorithmic signals by 5×
        #     (×0.20) so normal Arabic words fall below SUSPICION_THRESHOLD.
        #     3× was tested empirically — the worst Arabic word hit 0.136
        #     (above 0.10), so 5× is needed for reliable clearance.
        #   LLM says "suspicious" (1.0): amplify by 2× so flagged words
        #     strongly exceed threshold.
        #   In between (0.0 < val < 1.0): linearly interpolate gate factor.
        gate_factor = 0.20 + (2.0 - 0.20) * llm_suspicion
        fused = algorithmic * gate_factor
    else:
        # No LLM signal → pure algorithmic score
        fused = algorithmic

    return min(1.0, fused)


def _is_pure_latin_or_digit(word: str) -> bool:
    """True for Latin-only words (e.g. 'paracetamol', 'ventolin') and
    pure-digit tokens (Arabic-Indic numerals included). These should be
    skipped by the n-gram pass to avoid mixing scripts."""
    if not word:
        return True
    if _ARABIC_DIGIT_RE.match(word):
        return True
    return not _ARABIC_LETTER_RE.search(word)


# ---------------------------------------------------------------------------
# LLM pass
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You audit ASR transcripts of Gulf Arabic doctor-patient consultations "
    "with code-switched English. Your job: flag every word that LOOKS or "
    "SOUNDS like a mishearing of a medical / pharmaceutical / brand / "
    "anatomical term. Be biased toward flagging — better to over-flag a "
    "weird word than miss a real drug.\n\n"
    "Strict rules:\n"
    "1. Output STRICT JSON only, no prose.\n"
    "2. Word indices are zero-based, computed by splitting the transcript "
    "on whitespace.\n"
    "3. Each flag entry: {\"index\": <int>, \"word\": <str>, "
    "\"reason\": <short string>, "
    "\"likely_term\": <best guess at the intended medical term, "
    "in correct Latin spelling for drug names / English for procedures / "
    "or empty string if you cannot identify>, "
    "\"confidence\": <0.0 to 1.0 — how certain you are about likely_term>}.\n"
    "4. Schema: {\"flags\": [<flag entry>, ...]}.\n"
    "5. Do NOT flag plain Arabic words that aren't medical (e.g. 'لمدة', "
    "'كل', 'اليوم'), normal English filler ('okay'), or numbers.\n"
    "6. Use confidence >= 0.90 ONLY when the audio context (drug + dose + "
    "frequency / indication) makes the term unambiguous. Use 0.5-0.85 for "
    "plausible guesses. Use 0.0 when unsure."
)


def llm_pass(transcript: str, timeout: float = 60.0) -> List[Dict[str, Any]]:
    user = json.dumps(
        {"transcript": transcript,
         "tokens": list(enumerate(re.split(r"\s+", transcript.strip())))},
        ensure_ascii=False,
    )
    payload = {
        "model": get_llm_model(get_llm_provider()),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": user},
        ],
    }
    try:
        req = urllib.request.Request(
            get_llm_url(get_llm_provider()),
            data=json.dumps(payload).encode("utf-8"),
            headers=get_llm_headers(get_llm_provider()),
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = parse_chat_content(data, get_llm_provider()).strip()
        if not (text.startswith("{") and text.endswith("}")):
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                text = m.group(0)
        obj = json.loads(text)
        return list(obj.get("flags", []))
    except Exception as exc:
        print(f"[flag] LLM pass failed: {exc!r}")
        return []


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def flag_suspicious(
    transcript: str, use_llm: bool = True
) -> List[Dict[str, Any]]:
    """Return one entry per suspicious word, with phonetic candidates and
    optionally an LLM 'likely_term' attached."""
    phon = phonetic_pass(transcript)
    phon_by_idx = {f["index"]: f for f in phon}

    if use_llm:
        for entry in llm_pass(transcript):
            try:
                idx = int(entry.get("index"))
            except (TypeError, ValueError):
                continue
            llm_conf = float(entry.get("confidence", 0.0) or 0.0)
            likely = entry.get("likely_term") or ""
            existing = phon_by_idx.get(idx)
            if existing:
                existing["llm_reason"] = entry.get("reason") or existing["reason"]
                if likely:
                    existing["llm_likely_term"] = likely
                existing["llm_confidence"] = llm_conf
            else:
                word = entry.get("word") or ""
                phon_by_idx[idx] = {
                    "index": idx,
                    "word": word,
                    "reason": entry.get("reason") or "llm_flag",
                    "candidates": _phonetic_candidates(word, load_medical_lexicon()),
                    "llm_reason": entry.get("reason"),
                    "llm_likely_term": likely,
                    "llm_confidence": llm_conf,
                }
    return sorted(phon_by_idx.values(), key=lambda f: f["index"])


# ---------------------------------------------------------------------------
# Auto-correction: build a corrected transcript using HIGH-CONFIDENCE LLM
# suggestions only. The dashboard surfaces this as a separate string so the
# user can compare it to the raw transcript without losing the original.
# ---------------------------------------------------------------------------

def apply_high_confidence_corrections(
    transcript: str,
    flags: List[Dict[str, Any]],
    *,
    confidence_threshold: float = 0.90,
    phonetic_strong_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Rewrite the transcript with high-confidence corrections.

    Two sources of corrections, in priority order:

    1. PHONETIC TOP-1 (preferred when strong): if the top phonetic
       candidate scored >= `phonetic_strong_threshold` (default 0.85),
       trust it directly. Phonetic match is deterministic and grounded
       in the actual ASR output — when it scores very high it is
       essentially certainly the right drug.

    2. LLM `likely_term` (only as fallback): used ONLY when phonetic
       is weak AND the LLM confidence is high (>= `confidence_threshold`)
       AND the LLM's proposed term EXISTS IN OUR LEXICON. This guards
       against LLM hallucinations like 'Foltranis' or 'Paracetamol'
       when the audio clearly said 'voltaren' / 'panadol'.
    """
    lexicon_lower = {t.lower() for t in load_medical_lexicon()}
    tokens = re.split(r"(\s+)", transcript)  # keep whitespace tokens
    # word-index -> token-index in the split (only non-space tokens count)
    word_to_tok: List[int] = []
    for ti, t in enumerate(tokens):
        if t.strip():
            word_to_tok.append(ti)

    applied: List[Dict[str, Any]] = []
    for f in flags:
        idx = f.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(word_to_tok):
            continue
        cands = f.get("candidates") or []
        top = cands[0] if cands else None
        top_sim = float(top["phonetic_similarity"]) if top else 0.0
        llm_conf = float(f.get("llm_confidence", 0.0) or 0.0)
        llm_term = (f.get("llm_likely_term") or "").strip()

        # --- 1. Strong phonetic match → trust it.
        chosen = None
        source = None
        chosen_conf = 0.0
        if top and top_sim >= phonetic_strong_threshold:
            chosen = top["term"]
            chosen_conf = top_sim
            source = "phonetic"

        # --- 2. Fallback to LLM IF and ONLY IF:
        #     - phonetic was weak (didn't trigger above)
        #     - LLM is confident
        #     - LLM term is in our lexicon (so it's not a hallucination)
        if chosen is None and llm_conf >= confidence_threshold and llm_term:
            if llm_term.lower() in lexicon_lower:
                chosen = llm_term
                chosen_conf = llm_conf
                source = "llm"

        if not chosen:
            continue

        # span_indices is set for bigram/trigram flags. Replace the FIRST
        # word and clear the rest so 'وفولتران مسا' → 'voltaren'.
        spans = f.get("span_indices") or [idx]
        first = spans[0]
        original_parts = []
        for off in spans:
            if 0 <= off < len(word_to_tok):
                original_parts.append(tokens[word_to_tok[off]])
        ti_first = word_to_tok[first]
        tokens[ti_first] = chosen
        # Clear later words AND their leading whitespace so we don't leave
        # 'voltaren مسا' as the output of a 2-gram replacement.
        for off in spans[1:]:
            if 0 <= off < len(word_to_tok):
                tw_idx = word_to_tok[off]
                tokens[tw_idx] = ""
                # Also blank the whitespace token just before this word.
                if tw_idx - 1 >= 0:
                    tokens[tw_idx - 1] = ""
        applied.append({
            "index": idx,
            "span_indices": spans,
            "original": " ".join(original_parts),
            "corrected": chosen,
            "confidence": chosen_conf,
            "source": source,
        })
        # Record the correction in the feedback loop so Stage A becomes
        # more sensitive to this transliteration pattern in future runs.
        _record_correction(" ".join(original_parts), chosen)

    # Collapse runs of empty tokens.
    out = "".join(tokens)
    out = re.sub(r"\s+", " ", out).strip()
    return {
        "corrected_transcript": out,
        "applied": applied,
        "threshold": confidence_threshold,
    }
