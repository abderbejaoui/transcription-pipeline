"""Deterministic Arabic-script -> Latin drug-name normalizer for the A/B view.

Why this exists
---------------
Qwen3-ASR correctly *hears* code-switched brand names ("panadol", "doliprane")
but, because the carrier sentence is Arabic, it writes them in Arabic script
("بنادول", "دوليبران") and sometimes mangles them ("بنادل", "دوريبر").

The user wants the final transcript to keep the brand names in Latin. Rather
than fight the ASR's script choice, we post-process: map any Arabic-script
token that is phonetically close to a known drug back to its canonical Latin
spelling. This is fast, deterministic, and LLM-free — ideal for the A/B tester.

It is INTENTIONALLY scoped to drug/brand names only, so it cannot corrupt
normal Arabic words: a token is only replaced when its phonetic skeleton
matches a known drug within a tight edit-distance budget.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Canonical Latin drug names we want to surface. The Arabic-script keys are the
# common (and commonly-mangled) ways the ASR writes each one. Keys are matched
# by phonetic skeleton, so near-misses ("بنادل" for "بنادول") still resolve.
# ---------------------------------------------------------------------------
_DRUG_VARIANTS: Dict[str, List[str]] = {
    "panadol": ["بنادول", "بانادول", "بنادل", "باندول", "بنادو"],
    "doliprane": [
        "دوليبران", "دوليبرين", "دوريبران", "دوليبرا", "دوريبر", "دوليبر",
    ],
    "novadol": ["نوفادول", "نوفادل", "نوفا دول", "نوفدول"],
    "paracetamol": ["باراسيتامول", "بارسيتامول", "باراستامول"],
    "ibuprofen": ["ايبوبروفين", "إيبوبروفين", "ابوبروفين"],
    "aspirin": ["اسبرين", "أسبرين", "اسبيرين", "أسبيرين"],
    "amoxicillin": ["اموكسيسيلين", "أموكسيسيلين", "اموكسسيلين"],
    "augmentin": ["اوجمنتين", "أوجمنتين", "اغمنتين"],
    "ventolin": ["فينتولين", "فنتولين"],
    "voltaren": ["فولتارين", "فولتارن"],
    "efferalgan": ["افرلجان", "إفرلجان", "افرالجان"],
    "flagyl": ["فلاجيل", "فلاجل"],
    "zithromax": ["زيثروماكس", "زثروماكس"],
    "tamiflu": ["تاميفلو", "تميفلو"],
    "ventoline": ["فنتولين"],
    "tramadol": ["ترامادول", "تramadol", "ترامادل"],
    "metformin": ["ميتفورمين", "متفورمين"],
    "insulin": ["انسولين", "إنسولين", "أنسولين"],
    "omeprazole": ["اوميبرازول", "أوميبرازول"],
    "ciprofloxacin": ["سيبروفلوكساسين", "سبروفلوكساسين"],
    "azithromycin": ["ازيثرومايسين", "أزيثرومايسين"],
}

# Phonetic folding for Arabic: collapse letters the ASR confuses and that don't
# change the perceived brand name. We map to a small Latin-ish skeleton.
_AR_FOLD = {
    "ا": "a", "أ": "a", "إ": "a", "آ": "a", "ى": "a",
    "ب": "b", "پ": "b",
    "ت": "t", "ط": "t", "ة": "t",
    "ث": "s", "س": "s", "ص": "s",
    "ج": "j", "چ": "j",
    "ح": "h", "ه": "h", "خ": "k",
    "د": "d", "ذ": "z", "ض": "d",
    "ر": "r",
    "ز": "z", "ظ": "z",
    "ش": "s",
    "ع": "a", "غ": "g",
    "ف": "f", "ڤ": "f",
    "ق": "k", "ك": "k", "گ": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "و": "w", "ؤ": "w",
    "ي": "y", "ئ": "y",
}

def _strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _strip_leading_conjunction(token: str) -> str:
    """Drop a leading 'و' (and) so 'ودوليبران' matches 'دوليبران'.

    Only stripped when the remainder is still reasonably long, so we don't
    butcher genuinely short words that merely start with و.
    """
    if token.startswith("و") and len(token) >= 6:
        return token[1:]
    return token


def _ar_skeleton(token: str) -> str:
    """Fold an Arabic token to a compact consonant-ish skeleton."""
    token = _strip_leading_conjunction(_strip_diacritics(token))
    out = []
    for ch in token:
        out.append(_AR_FOLD.get(ch, ""))
    sk = "".join(out)
    # Collapse runs of the same char and drop weak vowels for matching.
    sk = re.sub(r"(.)\1+", r"\1", sk)
    return sk


def _lat_skeleton(name: str) -> str:
    name = name.lower()
    name = name.replace("ph", "f").replace("ou", "w")
    name = name.translate(str.maketrans({"c": "k", "q": "k", "v": "f", "p": "b", "g": "j", "x": "k"}))
    name = re.sub(r"(.)\1+", r"\1", name)
    return name


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


# Pre-compute skeletons for every known variant + the canonical name itself.
def _build_index() -> List[Tuple[str, str]]:
    index: List[Tuple[str, str]] = []  # (skeleton, canonical_latin)
    for canonical, variants in _DRUG_VARIANTS.items():
        index.append((_lat_skeleton(canonical), canonical))
        for v in variants:
            index.append((_ar_skeleton(v), canonical))
    return index


_INDEX = _build_index()

# Only Arabic-script tokens are eligible for replacement (so we never touch
# normal English text the model already got right).
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def _best_match(token: str) -> Tuple[str, int, int] | None:
    """Return (canonical_latin, distance, cand_len) for the closest known drug.

    Length-guarded: a candidate is only considered if its skeleton length is
    within 2 of the token's skeleton length. This stops a short ordinary word
    (e.g. a 4-char verb) from matching a long 9-char drug skeleton.
    """
    sk = _ar_skeleton(token)
    if len(sk) < 3:
        return None
    best: Tuple[str, int, int] | None = None
    for cand_sk, canonical in _INDEX:
        if abs(len(cand_sk) - len(sk)) > 2:
            continue
        d = _levenshtein(sk, cand_sk)
        if best is None or d < best[1]:
            best = (canonical, d, len(cand_sk))
    return best


def normalize_drugs(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Replace Arabic-script drug tokens with their canonical Latin names.

    Returns (normalized_text, replacements) where each replacement is
    {"from": original_token, "to": canonical}. Tokens are only replaced when
    their phonetic skeleton is within a tight edit-distance of a known drug,
    so ordinary Arabic words are left untouched.
    """
    if not text:
        return text, []

    replacements: List[Dict[str, str]] = []

    def _sub(match: re.Match) -> str:
        token = match.group(0)
        if not _ARABIC_RE.search(token):
            return token
        best = _best_match(token)
        if best is None:
            return token
        canonical, dist, cand_len = best
        tok_len = len(_ar_skeleton(token))
        # Short Arabic tokens are too ambiguous (e.g. "ودول" = "and states"),
        # so they only resolve on an EXACT variant match (distance 0). Longer
        # tokens get a small, length-scaled fuzzy budget.
        if tok_len <= 4 or cand_len <= 4:
            budget = 0
        else:
            budget = 1 if cand_len <= 6 else 2 if cand_len <= 8 else 3
            budget = min(budget, max(1, cand_len // 3))
        if dist <= budget:
            replacements.append({"from": token, "to": canonical})
            return canonical
        return token

    # Split on whitespace but keep separators so spacing is preserved.
    normalized = re.sub(r"\S+", _sub, text)
    return normalized, replacements
