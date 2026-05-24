"""Stage 1: score words for suspicion using heuristic detection + optional BART.

The primary scorer uses a combination of:

1. Canonical term match — if the exact word is a canonical lexicon term
   (NOT an alias), it receives low suspicion and in_lexicon=True.

2. Character-level edit distance — words that are character-similar
   (>55 %) to a canonical term but not exact matches are likely
   misspellings → high suspicion.

3. Bigram phrase matching — if a two-word sequence (e.g. "dolly prahn")
   matches a multi-word lexicon term or alias exactly, both words get
   high suspicion. This handles multi-word misspellings that individual
   character similarity would miss.

4. Common English word list — words that appear in a curated medical-
   English word list and aren't similar to any canonical term receive
   low suspicion (they're likely correct dictation).

Note: This replaces an earlier BART-based masked-scoring approach that proved
unreliable (misspelled words scored LOW while common words scored HIGH). BART
remains available as a future optional enhancement if needed.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.services import lexicon

from .models import ScoredWord


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+(?:\.\d+)?")
STOP_WORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "had",
    "has", "have", "he", "her", "him", "his", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "she", "so", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "to", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your", "should",
}

# Common English words that are unlikely to be medical misspellings.
# These serve as a baseline receiving low suspicion.
COMMON_ENGLISH: Set[str] = {
    "patient", "daily", "day", "days", "week", "weeks", "month", "months",
    "year", "years", "take", "takes", "taking", "taken", "took", "dose",
    "doses", "dosage", "mg", "ml", "cc", "hour", "hours", "minute", "minutes",
    "time", "times", "once", "twice", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "every", "per", "oral", "iv", "intravenous",
    "intramuscular", "subcutaneous", "topical", "inhaled", "given", "administered",
    "prescribed", "recommended", "started", "continued", "discontinued", "stopped",
    "increased", "decreased", "adjusted", "monitored", "checked", "tested",
    "showed", "revealed", "indicated", "demonstrated", "reported", "complained",
    "presented", "admitted", "discharged", "transferred", "seen", "evaluated",
    "assessed", "examined", "measured", "observed", "noted", "noticed",
    "developed", "experienced", "suffered", "improved", "worsened", "resolved",
    "fever", "pain", "cough", "sputum", "dyspnea", "shortness", "breath",
    "wheeze", "wheezing", "crackles", "rhonchi", "chest", "lung", "lungs",
    "heart", "cardiac", "blood", "pressure", "rate", "rhythm", "pulse",
    "oxygen", "saturation", "temperature", "weight", "height", "bmi",
    "headache", "nausea", "vomiting", "diarrhea", "constipation", "abdomen",
    "abdominal", "back", "neck", "throat", "nose", "ear", "eyes", "skin",
    "rash", "lesion", "ulcer", "wound", "infection", "fracture", "trauma",
    "surgery", "surgical", "procedure", "biopsy", "scan", "xray", "x-ray",
    "mri", "ct", "ultrasound", "ekg", "ecg", "lab", "labs", "test", "tests",
    "results", "normal", "abnormal", "positive", "negative", "elevated",
    "decreased", "within", "without", "history", "past", "family", "social",
    "allergies", "medications", "treatment", "plan", "follow", "followup",
    "follow-up", "next", "return", "clinic", "primary", "care", "emergency",
    "room", "hospital", "ward", "icu", "nursing", "home", "rehabilitation",
    "physical", "therapy", "occupational", "speech", "diet", "nutrition",
    "fluid", "fluids", "electrolytes", "potassium", "sodium", "calcium",
    "magnesium", "phosphorus", "glucose", "sugar", "hemoglobin",
    "hematocrit", "platelet", "platelets", "white", "red", "cell", "cells",
    "wbc", "rbc", "hgb", "hct", "bun", "creatinine", "liver", "kidney",
    "renal", "hepatic", "cardiac", "pulmonary", "neurologic", "musculoskeletal",
    "skin", "soft", "tissue", "bone", "joint", "joints", "muscle", "muscles",
    "numbness", "tingling", "weakness", "fatigue", "dizziness", "syncope",
    "seizure", "seizures", "confusion", "disorientation", "lethargy",
    "somnolence", "coma", "unconscious", "unresponsive", "awake", "alert",
    "oriented", "person", "place", "situation",
    "secondary", "presents", "presenting", "alongside", "using",
    "attending", "attends", "attended",
    "high", "low", "range", "normal", "abnormal", "elevated",
    "mild", "moderate", "severe", "acute", "chronic", "recurrent",
}

# Minimum character similarity ratio to flag as likely misspelling
SIMILARITY_MIN = 0.55

# When a common English word is this similar to a canonical term, flag it anyway.
# This prevents substring-like matches (e.g. "infection" in "hiv infection") from
# generating false positives.
COMMON_ENGLISH_SIM_CAP = 0.90


def tokenize_transcript(transcript: str) -> List[Tuple[str, int, int]]:
    return [(match.group(), match.start(), match.end()) for match in WORD_RE.finditer(transcript)]


def _is_stop_word(token: str) -> bool:
    return token.lower() in STOP_WORDS


def _is_common_english(token: str) -> bool:
    return token.lower() in COMMON_ENGLISH


def _canonical_form(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _char_similarity(a: str, b: str) -> float:
    """Normalized character-level similarity (0-1) using difflib."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _lexicon_entry(token: str) -> Optional[Dict[str, Any]]:
    """Check if token matches a CANONICAL term (not alias) in the lexicon."""
    entry = lexicon.find_by_canonical(token)
    if entry is None:
        return None
    return {
        "term": entry.term,
        "type": entry.term_type,
        "canonical_form": _canonical_form(entry.term),
        "aliases": {_canonical_form(a) for a in entry.aliases if a.strip()},
    }


# ── canonical index helpers ─────────────────────────────────────────────

# Module-level cache that can be cleared (e.g. in tests when lexicon path changes).
_canonical_index: Dict[str, Dict[str, Any]] = {}
_bigram_map: Dict[str, List[Dict[str, Any]]] = {}
_cache_dirty: bool = True


def clear_caches() -> None:
    """Force index / bigram-map rebuild on next access.

    Call this in tests after monkeypatching the lexicon path.
    """
    global _cache_dirty, _canonical_index, _bigram_map
    _cache_dirty = True
    _canonical_index = {}
    _bigram_map = {}


def _build_canonical_index() -> Dict[str, Dict[str, Any]]:
    """Build a map: canonical_form -> {term, type, aliases set}."""
    index: Dict[str, Dict[str, Any]] = {}
    for entry in lexicon.load_lexicon():
        cf = _canonical_form(entry.term)
        if cf:
            index[cf] = {
                "term": entry.term,
                "type": entry.term_type,
                "canonical_form": cf,
                "aliases": {_canonical_form(a) for a in entry.aliases if a.strip()},
            }
    return index


def _get_canonical_index() -> Dict[str, Dict[str, Any]]:
    global _cache_dirty, _canonical_index
    if _cache_dirty or not _canonical_index:
        _canonical_index = _build_canonical_index()
        _cache_dirty = False
    return _canonical_index


def _build_bigram_map() -> Dict[str, List[Dict[str, Any]]]:
    """Build map: bigram -> list of entries with that bigram in term or alias."""
    bigram_map: Dict[str, List[Dict[str, Any]]] = {}
    index = _get_canonical_index()
    for entry in index.values():
        term_words = entry["canonical_form"].split()
        if len(term_words) >= 2:
            for i in range(len(term_words) - 1):
                bg = f"{term_words[i]} {term_words[i+1]}"
                bigram_map.setdefault(bg, []).append(entry)
        for alias_cf in entry["aliases"]:
            alias_words = alias_cf.split()
            if len(alias_words) >= 2:
                for i in range(len(alias_words) - 1):
                    bg = f"{alias_words[i]} {alias_words[i+1]}"
                    bigram_map.setdefault(bg, []).append(entry)
    return bigram_map


def _get_bigram_map() -> Dict[str, List[Dict[str, Any]]]:
    global _cache_dirty, _bigram_map
    if _cache_dirty or not _bigram_map:
        _bigram_map = _build_bigram_map()
    return _bigram_map


def _find_bigram_matches(tokens: List[Tuple[str, int, int]], index: int) -> bool:
    """Check if token at `index` is part of a bigram matching a lexicon entry."""
    bigram_map = _get_bigram_map()
    # Check bigram (tokens[index], tokens[index+1])
    if index + 1 < len(tokens):
        bg = _canonical_form(f"{tokens[index][0]} {tokens[index+1][0]}")
        if bg in bigram_map:
            return True
    # Check bigram (tokens[index-1], tokens[index])
    if index - 1 >= 0:
        bg = _canonical_form(f"{tokens[index-1][0]} {tokens[index][0]}")
        if bg in bigram_map:
            return True
    return False


# ── heuristic scoring ───────────────────────────────────────────────────

def _best_canonical_similarity(token: str) -> float:
    """Return the best character-level similarity to any canonical term or alias."""
    cf = _canonical_form(token)
    if not cf:
        return 0.0
    best_score = 0.0
    for canon, entry in _get_canonical_index().items():
        score = _char_similarity(cf, canon)
        if score > best_score:
            best_score = score
        for alias in entry["aliases"]:
            score = _char_similarity(cf, alias)
            if score > best_score:
                best_score = score
    return best_score


def _score_token(
    token: str,
    tokens: List[Tuple[str, int, int]],
    index: int,
) -> float:
    """Compute heuristic suspicion score for a single token.

    Returns a score in [0, 1]:
    - 0.0 - 0.20: definitely correct
    - 0.20 - 0.40: likely correct
    - 0.40 - 0.60: uncertain
    - 0.60 - 0.85: likely misspelling
    - 0.85 - 1.00: definite misspelling

    Priority order:
    1. Bigram phrase match → high suspicion (multi-word misspelling)
    2. Common English word (and not very similar to canonical term) → low suspicion
    3. High character similarity to canonical term → high suspicion (likely misspelling)
    4. Medium similarity → uncertain
    5. Short token → low-medium
    6. Unknown → medium
    """
    cf = _canonical_form(token)
    if not cf:
        return 0.0

    # 1. Check bigram matches first (multi-word misspellings like "dolly prahn")
    if _find_bigram_matches(tokens, index):
        return 0.85

    # 2. Character similarity to canonical terms + aliases
    sim = _best_canonical_similarity(token)

    # 3. Common English word → low suspicion (unless extremely similar to canonical term)
    if _is_common_english(token):
        if sim >= COMMON_ENGLISH_SIM_CAP:
            # Very similar to a canonical term despite being common English
            return 0.60 + (sim - COMMON_ENGLISH_SIM_CAP) / (1.0 - COMMON_ENGLISH_SIM_CAP) * 0.30
        return 0.05

    # 4. High similarity to a canonical term → likely misspelling
    if sim >= SIMILARITY_MIN:
        score = 0.60 + (sim - SIMILARITY_MIN) / (1.0 - SIMILARITY_MIN) * 0.35
        return min(0.95, max(0.60, score))

    # 5. Short token (< 4 chars) not similar to anything → medium-low
    if len(token) < 4:
        return 0.25

    # 6. Unknown token, no strong similarity → medium (uncertain)
    return 0.45


def score_transcript(transcript: str) -> List[ScoredWord]:
    """Score each word in the transcript using heuristic detection."""
    tokens = tokenize_transcript(transcript)
    if not tokens:
        return []

    scored: List[ScoredWord] = []

    for index, (text, start, end) in enumerate(tokens):
        is_stop = _is_stop_word(text)
        entry = _lexicon_entry(text)

        if is_stop:
            scored.append(ScoredWord(
                index=index, text=text, suspicion=0.0,
                in_lexicon=(entry is not None), start=start, end=end,
            ))
            continue

        if entry is not None:
            # Exact canonical match → known correct term, low suspicion
            scored.append(ScoredWord(
                index=index, text=text, suspicion=0.05,
                in_lexicon=True, start=start, end=end,
            ))
            continue

        # Compute heuristic score
        suspicion = _score_token(text, tokens, index)

        scored.append(ScoredWord(
            index=index, text=text, suspicion=suspicion,
            in_lexicon=False, start=start, end=end,
        ))

    return scored