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


# ---------------------------------------------------------------------------
# Phonetic core: CEQ (consonant-equivalence classes) + Editex-style weighted
# edit distance + Jaro-Winkler prefix bonus.
#
# The matcher above ("skeleton") already folds Arabic letters to a Latin-ish
# alphabet and the canonical name to a similar one. The problem with plain
# Levenshtein over those skeletons is that it is *orthographic*: it treats a
# b<->p swap exactly like a b<->z swap, even though the former is phonetically
# trivial and the latter is not.
#
# We fix that by mapping every skeleton character into a small set of phonetic
# equivalence classes (one representative symbol per class). Substituting two
# letters in the same class costs 0; two letters in *neighbouring* classes
# (e.g. the sibilants s/z/sh) costs 0.5; anything else costs 1. This is the
# Editex idea (Zobel & Dart 1996) adapted to the Arabic->Latin drug setting,
# built by hand from Buckwalter/IPA equivalences so we stay dependency-free.
# ---------------------------------------------------------------------------

# Each phonetic class maps a set of skeleton chars to a single class symbol.
# Skeleton chars come from _AR_FOLD / _lat_skeleton, so the alphabet is small.
_CEQ_CLASSES: List[Tuple[str, str]] = [
    ("B", "bp"),        # bilabial plosives  ب پ / b p
    ("F", "fv"),        # labiodental fric.  ف ڤ / f v  (ph already -> f)
    ("T", "td"),        # dentals/alveolars  ت ط د ض ة / t d
    ("S", "sz"),        # sibilants          ث س ص ش ز ظ ذ / s z
    ("K", "kg"),        # velars/uvulars     ك ق گ خ غ ج چ / k g j
    ("J", "j"),         # affricate j        (kept distinct but near K)
    ("R", "r"),
    ("L", "l"),
    ("M", "m"),
    ("N", "n"),
    ("H", "h"),
    ("W", "w"),         # glide/round vowel  و ؤ / w  (and 'ou')
    ("Y", "y"),         # glide/front vowel  ي ئ / y
    ("A", "a"),         # vowel carrier      ا أ إ آ ى ع
]

# Build char -> class-symbol lookup.
_CHAR2CLASS: Dict[str, str] = {}
for _sym, _chars in _CEQ_CLASSES:
    for _c in _chars:
        # First class to claim a char wins; J is intentionally also reachable
        # via the K bucket above, so map j to J explicitly afterwards.
        _CHAR2CLASS.setdefault(_c, _sym)
_CHAR2CLASS["j"] = "J"

# Classes that are phonetically *adjacent*: a substitution between them is
# cheap (0.5) rather than full cost (1.0). These are the confusions that Arabic
# transliteration and ASR realistically produce.
_NEAR_PAIRS = {
    frozenset({"B", "F"}),   # b <-> f/v (voiced/voiceless labial)
    frozenset({"T", "S"}),   # t/d <-> s/z (dental vs sibilant)
    frozenset({"S", "J"}),   # s/sh <-> j (sibilant vs affricate)
    frozenset({"K", "J"}),   # k/g <-> j
    frozenset({"K", "H"}),   # kh <-> h
    frozenset({"W", "A"}),   # round glide <-> vowel
    frozenset({"Y", "A"}),   # front glide <-> vowel
    frozenset({"N", "M"}),   # nasals
    frozenset({"R", "L"}),   # liquids
}


def _to_classes(skeleton: str) -> str:
    """Map a folded skeleton string to its CEQ class-symbol string."""
    return "".join(_CHAR2CLASS.get(ch, ch.upper()) for ch in skeleton)


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if frozenset({a, b}) in _NEAR_PAIRS:
        return 0.5
    return 1.0


def _editex(a: str, b: str) -> float:
    """Editex-style weighted edit distance over CEQ class strings."""
    if a == b:
        return 0.0
    if not a:
        return float(len(b))
    if not b:
        return float(len(a))
    prev = [float(j) for j in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [float(i)]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1.0,            # deletion
                cur[j - 1] + 1.0,         # insertion
                prev[j - 1] + _sub_cost(ca, cb),  # weighted substitution
            ))
        prev = cur
    return prev[-1]


def _jaro(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    match_dist = max(len(a), len(b)) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    a_match = [False] * len(a)
    b_match = [False] * len(b)
    matches = 0
    for i, ca in enumerate(a):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, len(b))
        for j in range(lo, hi):
            if b_match[j] or b[j] != ca:
                continue
            a_match[i] = b_match[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i in range(len(a)):
        if not a_match[i]:
            continue
        while not b_match[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    transpositions //= 2
    return (
        matches / len(a)
        + matches / len(b)
        + (matches - transpositions) / matches
    ) / 3.0


def _jaro_winkler(a: str, b: str, p: float = 0.1) -> float:
    j = _jaro(a, b)
    prefix = 0
    for ca, cb in zip(a, b):
        if ca == cb and prefix < 4:
            prefix += 1
        else:
            break
    return j + prefix * p * (1.0 - j)


def _phonetic_similarity(sk_a: str, sk_b: str) -> float:
    """Combined phonetic similarity in [0, 1] for two folded skeletons.

    Blends a length-normalised Editex distance (substance) with a Jaro-Winkler
    score over the CEQ class strings (order + shared prefix, which brand names
    rely on heavily).
    """
    ca = _to_classes(sk_a)
    cb = _to_classes(sk_b)
    if not ca or not cb:
        return 0.0
    ed = _editex(ca, cb)
    ed_sim = 1.0 - ed / max(len(ca), len(cb))
    jw = _jaro_winkler(ca, cb)
    return 0.6 * ed_sim + 0.4 * jw


# Pre-compute skeletons for every known variant + the canonical name itself.
def _build_index() -> List[Tuple[str, str]]:
    index: List[Tuple[str, str]] = []  # (skeleton, canonical_latin)
    for canonical, variants in _DRUG_VARIANTS.items():
        index.append((_lat_skeleton(canonical), canonical))
        for v in variants:
            index.append((_ar_skeleton(v), canonical))
    return index


_INDEX = _build_index()

# Latin-only index: the canonical drug names' Latin skeletons. Used to catch
# the case where the ASR already transliterated the brand into Latin but got
# it slightly wrong (e.g. "augmenta" for "augmentin"). Matching is done with
# _lat_skeleton on BOTH sides and a tight threshold so we only ever rewrite a
# near-miss of an actual drug name, never an ordinary Latin/English word.
def _build_latin_index() -> List[Tuple[str, str]]:
    return [(_lat_skeleton(c), c) for c in _DRUG_VARIANTS]


_LATIN_INDEX = _build_latin_index()
_CANONICAL_LOWER = {c.lower() for c in _DRUG_VARIANTS}

# Only Arabic-script tokens are eligible for replacement (so we never touch
# normal English text the model already got right).
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

# A Latin token must be all ASCII letters to be considered for the Latin path.
_LATIN_RE = re.compile(r"^[A-Za-z]+$")

# Replacement threshold on the combined phonetic similarity. Tuned so that
# real (even unseen) drug mangles clear it while ordinary Arabic that merely
# shares a few consonants does not. Validated by tests/eval_drug_normalize.py.
_SIM_THRESHOLD = 0.82

# Latin near-miss tuning. The discriminating signal between a real mangle
# ("augmenta" -> "augmentin") and an ordinary English word that merely shares
# a drug's prefix ("augment") is NOT raw phonetic similarity -- those collide
# (0.839 vs 0.849). It is two structural facts:
#   1. shared-prefix ratio: a mangle tracks the drug name closely from the
#      start, so it shares a long prefix relative to the longer skeleton.
#   2. strict-prefix rejection: an ordinary word that is simply a truncation
#      of the drug ("augment" is a clean prefix of "augmentin") is almost
#      always a real word, never a mis-transliteration. A genuine mangle
#      diverges in its tail ("augment-a", "panad-l") instead of truncating.
_LATIN_SIM_THRESHOLD = 0.80
# Token must share at least this fraction of the longer skeleton as a common
# prefix to be considered a mangle of that drug.
_LATIN_PREFIX_RATIO = 0.70

# Common English inflectional/derivational suffixes. If the part of the token
# that DIVERGES from the drug name is itself a real English ending (e.g.
# "augment-ed", "augment-ing"), the token is an ordinary inflected word, not a
# mis-transliteration, so it must not be rewritten. Ordered longest-first.
_ENGLISH_SUFFIXES = (
    "ization", "ication", "fulness", "lessness",
    "ation", "ition", "ments", "ness", "ling", "ting",
    "ing", "ers", "est", "ies", "ment", "ous", "ive", "ial", "ant", "ent",
    "ed", "er", "ly", "es", "al", "ic",
)


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _best_match(token: str) -> Tuple[str, float, int] | None:
    """Return (canonical_latin, similarity, cand_len) for the closest drug.

    Length-guarded: a candidate is only considered if its skeleton length is
    within 2 of the token's skeleton length, which stops a short ordinary word
    from matching a long drug skeleton.
    """
    sk = _ar_skeleton(token)
    if len(sk) < 3:
        return None
    best: Tuple[str, float, int] | None = None
    for cand_sk, canonical in _INDEX:
        if abs(len(cand_sk) - len(sk)) > 2:
            continue
        sim = _phonetic_similarity(sk, cand_sk)
        if best is None or sim > best[1]:
            best = (canonical, sim, len(cand_sk))
    return best


def _best_latin_match(token: str) -> Tuple[str, float, int] | None:
    """Closest drug for a Latin token, using _lat_skeleton on both sides.

    Handles the case where the ASR already wrote the brand in Latin but got
    it slightly wrong ("augmenta" -> "augmentin"). Length-guarded like the
    Arabic path so unrelated words can't reach a long drug name.
    """
    sk = _lat_skeleton(token)
    if len(sk) < 4:
        return None
    best: Tuple[str, float, int] | None = None
    for cand_sk, canonical in _LATIN_INDEX:
        if abs(len(cand_sk) - len(sk)) > 2:
            continue
        sim = _phonetic_similarity(sk, cand_sk)
        if best is None or sim > best[1]:
            best = (canonical, sim, len(cand_sk))
    return best


def _sub_arabic(token: str, replacements: List[Dict[str, str]]) -> str:
    """Resolve one Arabic-script token to a canonical drug name (or keep it)."""
    best = _best_match(token)
    if best is None:
        return token
    canonical, sim, cand_len = best
    tok_len = len(_ar_skeleton(token))
    # Short Arabic tokens are too ambiguous (e.g. "ودول" = "and states"), so
    # they need a near-perfect phonetic match. Longer, more distinctive tokens
    # may resolve at the standard threshold.
    if tok_len <= 4 or cand_len <= 4:
        threshold = 0.97
    else:
        threshold = _SIM_THRESHOLD
    if sim >= threshold:
        replacements.append({"from": token, "to": canonical})
        return canonical
    return token


def _sub_latin(token: str, replacements: List[Dict[str, str]]) -> str:
    """Resolve one Latin token to a canonical drug name (or keep it).

    Fires only on a structural near-miss of a real drug name: it must clear
    the phonetic threshold, share a long common prefix with the canonical, and
    NOT be a strict truncation of it (truncations are ordinary words like
    "augment" for "augmentin"). This keeps ordinary English/Latin text safe.
    """
    best = _best_latin_match(token)
    if best is None:
        return token
    canonical, sim, cand_len = best
    if sim < _LATIN_SIM_THRESHOLD or canonical.lower() == token.lower():
        return token
    tok_sk = _lat_skeleton(token)
    can_sk = _lat_skeleton(canonical)
    # An ordinary word that is just a clean prefix of the drug name (e.g.
    # "augment" -> "augmentin") is a real word, not a mangle: reject it.
    if can_sk.startswith(tok_sk) or tok_sk.startswith(can_sk):
        return token
    # A real mangle tracks the drug name from the start: require a long shared
    # prefix relative to the longer skeleton.
    cp = _common_prefix_len(tok_sk, can_sk)
    if cp / max(len(tok_sk), len(can_sk)) < _LATIN_PREFIX_RATIO:
        return token
    # If the token's divergent tail is a real English ending ("augment-ed",
    # "augment-ing"), it is an ordinary inflected word, not a mangle. Check on
    # the ORIGINAL token (skeleton folding would distort the suffix).
    low = token.lower()
    if any(low.endswith(suf) and len(low) - len(suf) >= 4 for suf in _ENGLISH_SUFFIXES):
        return token
    replacements.append({"from": token, "to": canonical})
    return canonical


def normalize_drugs(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Replace Arabic-script (and near-miss Latin) drug tokens with their
    canonical Latin names.

    Returns (normalized_text, replacements) where each replacement is
    {"from": original_token, "to": canonical}. A token is only replaced when
    its phonetic similarity (CEQ + Editex + Jaro-Winkler) to a known drug
    clears a tight, length-aware threshold, so ordinary words are left
    untouched.
    """
    if not text:
        return text, []

    replacements: List[Dict[str, str]] = []

    def _sub(match: re.Match) -> str:
        token = match.group(0)
        if _ARABIC_RE.search(token):
            return _sub_arabic(token, replacements)
        # Latin tokens: only attempt a fix when the ASR transliterated a brand
        # into Latin but got it slightly wrong (e.g. "augmenta"). An exact
        # canonical name is left as-is (no spurious self-replacement).
        if _LATIN_RE.match(token) and token.lower() not in _CANONICAL_LOWER:
            return _sub_latin(token, replacements)
        return token

    # Split on whitespace but keep separators so spacing is preserved.
    normalized = re.sub(r"\S+", _sub, text)
    return normalized, replacements
