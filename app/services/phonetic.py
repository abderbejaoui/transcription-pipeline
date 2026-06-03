"""IPA-based phonetic matching for Arabic→English medical term correction.

Uses `phonemizer` (Python package, wraps espeak-ng internally) to convert
both Arabic transliterations and English lexicon terms to IPA, then compares
via normalized edit distance on the IPA strings. Falls back to improved
consonant-skeleton matching when phonemizer is unavailable.

Multi-word Arabic phrases ("بلاد شوجر" → "blood sugar") are handled by
collapsing transliterated tokens and matching the combined IPA against
multi-word lexicon entries.
"""

from __future__ import annotations

import functools
import logging
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IPA generation via phonemizer (wraps espeak-ng)
# ---------------------------------------------------------------------------

_HAVE_PHONEMIZER = False
try:
    import phonemizer
    from phonemizer import phonemize as _phonemize_func
    _HAVE_PHONEMIZER = True
except ImportError:
    _HAVE_PHONEMIZER = False


@functools.lru_cache(maxsize=16384)
def _get_ipa(text: str, language: str = "en-us") -> Optional[str]:
    """Generate IPA for a single word/phrase using phonemizer.

    phonemizer wraps espeak-ng via Python (no subprocess), which is faster
    and more reliable than direct subprocess calls. The result is cached
    so repeated lookups for the same word are instant.

    Returns the IPA string stripped of stress markers, or None on failure.
    """
    if not _HAVE_PHONEMIZER:
        return None
    if not text or len(text) < 2:
        return None
    try:
        ipa = _phonemize_func(
            text,
            language=language,
            backend="espeak",
            preserve_punctuation=True,
            with_stress=False,
        )
        if ipa:
            return ipa.strip()
        return None
    except Exception as exc:
        logger.debug(f"[phonetic] phonemizer failed for {text!r}: {exc!r}")
        return None


# Alias for backwards compatibility (combined_phonetic_similarity uses this)
_espeak_ipa = _get_ipa


def ipa_phonemize(words: Sequence[str], language: str = "en-us") -> List[Optional[str]]:
    """Batch-convert words to IPA.

    Each word is phonemized independently, so multi-word strings encode
    their own spaces (e.g. "blood sugar" → two-word IPA sequence).
    """
    return [_get_ipa(w, language) for w in words]


# ---------------------------------------------------------------------------
# Phonetic similarity — IPA-based
# ---------------------------------------------------------------------------


def _ipa_lev_similarity(ipa_a: Optional[str], ipa_b: Optional[str]) -> float:
    """Normalized edit-distance similarity between two IPA strings (0-100)."""
    if not ipa_a or not ipa_b:
        return 0.0
    # Use rapidfuzz for fast Levenshtein
    return float(fuzz.ratio(ipa_a, ipa_b))


# ---------------------------------------------------------------------------
# Fallback: improved consonant skeleton matching
# (used when espeak-ng is unavailable)
# ---------------------------------------------------------------------------

_VOWELS = set("aeiouy")
_PHONETIC_SUBST = {
    "p": "b", "v": "f", "c": "k", "g": "k", "q": "k", "x": "ks",
    "z": "s",  # Arabic often merges z/s
    "d": "t",  # Arabic often merges d/t in final position
}

# Vowel patterns: Arabic transliterations often preserve the vowel PATTERN
# (sequence of vowel classes) even when the exact vowel changes.
# E.g., هستوري (hstwry) → vowels: a, u, i → pattern: short, long, short
#       history (hstr) → vowels: i, o → pattern: short, short
# The vowel pattern is a weaker signal but helps disambiguate.
_SHORT_VOWELS = set("aiou")
_LONG_VOWELS = set("āīūēōā")


def _consonant_skeleton(text: str) -> str:
    """Extract consonant skeleton from a Latin/transliterated string.

    Strips vowels, collapses phonetic substitutions, and returns the
    remaining consonant sequence.  This is the same logic as flag.py's
    _consonant_skeleton_latin but centralised here.
    """
    out: List[str] = []
    for ch in text.lower():
        if ch in _VOWELS:
            continue
        sub = _PHONETIC_SUBST.get(ch, ch)
        out.append(sub)
    return "".join(out)


def _vowel_pattern(text: str) -> str:
    """Extract a vowel-pattern signature: 's' for short, 'l' for long, '-' for none.

    E.g., 'history' → vowels i,o → 'ss'
          'historie' → vowels i,o,i,e → 'ssss' (for 'history' comparison, this is a weaker signal)
    """
    pattern: List[str] = []
    for ch in text.lower():
        if ch in _SHORT_VOWELS:
            pattern.append("s")
        elif ch in _LONG_VOWELS:
            pattern.append("l")
    return "".join(pattern)


def skeleton_similarity(span_text: str, variant_text: str) -> float:
    """Consonant-skeleton similarity with vowel-pattern tiebreaker.

    Returns a score 0-100.
    """
    s_skel = _consonant_skeleton(span_text)
    v_skel = _consonant_skeleton(variant_text)
    if not s_skel or not v_skel:
        return 0.0

    skel_score = float(fuzz.ratio(s_skel, v_skel))

    # Vowel pattern bonus: if skeletons are very similar (>80%) but not
    # identical, check if vowel patterns also align.
    if 80.0 <= skel_score < 100.0:
        s_vowels = _vowel_pattern(span_text)
        v_vowels = _vowel_pattern(variant_text)
        if s_vowels and v_vowels:
            vowel_sim = float(fuzz.ratio(s_vowels, v_vowels))
            # Small boost (up to 5 points) if vowels agree
            if vowel_sim >= 75.0:
                skel_score = min(100.0, skel_score + (vowel_sim - 75.0) * 0.2)

    return skel_score


# ---------------------------------------------------------------------------
# Main similarity API
# ---------------------------------------------------------------------------


def combined_phonetic_similarity(
    span_text: str,
    variant_text: str,
    prefer_ipa: bool = True,
) -> Dict[str, float]:
    """Compute phonetic similarity between a span and a lexicon variant.

    Strategy (in order of preference):
      1. IPA edit distance (if espeak-ng available)
      2. Consonant skeleton similarity (always available, serves as baseline)

    Returns a dict with keys:
      - "ipa": 0-100 IPA-based score (0 if unavailable)
      - "skeleton": 0-100 consonant skeleton score
      - "score": combined best score (max of IPA and skeleton)
      - "method": "ipa", "skeleton", or "none"
    """
    ipa_score = 0.0
    method = "skeleton"

    if prefer_ipa:
        ipa_a = _espeak_ipa(span_text)
        ipa_b = _espeak_ipa(variant_text)
        if ipa_a and ipa_b:
            ipa_score = _ipa_lev_similarity(ipa_a, ipa_b)
            if ipa_score > 0.0:
                method = "ipa"

    skel_score = skeleton_similarity(span_text, variant_text)

    return {
        "ipa": round(ipa_score, 2),
        "skeleton": round(skel_score, 2),
        "score": round(max(ipa_score, skel_score), 2),
        "method": method if max(ipa_score, skel_score) > 0 else "none",
    }


# ---------------------------------------------------------------------------
# Multi-word Arabic phrase support
# ---------------------------------------------------------------------------

# Multi-word English medical phrases against which Arabic transliterations
# are matched.  Each entry is (collapsed_transliteration, english_phrase).
# These are terms ASR commonly produces as separate Arabic words that should
# map to a single English multi-word term.
#
# Format: (
#   "arabic transliteration (space-separated, collapsed)",
#   "intended english canonical"
# )
_MULTI_WORD_PHRASES: List[Tuple[str, str, str]] = [
    # Vital signs
    ("blad shwjr", "blood sugar", "blood sugar"),
    ("blad brshr", "blood pressure", "blood pressure"),
    ("blad pre sh", "blood pressure", "blood pressure"),
    # Symptoms
    ("shortness of brith", "shortness of breath", "shortness of breath"),
    ("shortness awf brith", "shortness of breath", "shortness of breath"),
    ("shwrtns of brith", "shortness of breath", "shortness of breath"),
    ("shwrtns awf brith", "shortness of breath", "shortness of breath"),
    ("shwrtns of broath", "shortness of breath", "shortness of breath"),
    ("shwrtns awf broath", "shortness of breath", "shortness of breath"),
    ("shortnis of brith", "shortness of breath", "shortness of breath"),
    ("shortnis awf brith", "shortness of breath", "shortness of breath"),
    ("shortnis of braith", "shortness of breath", "shortness of breath"),
    ("shortnis awf braith", "shortness of breath", "shortness of breath"),
    ("shwrtns of braith", "shortness of breath", "shortness of breath"),
    ("shwrtns awf braith", "shortness of breath", "shortness of breath"),
    # Pain descriptions
    ("chist bain", "chest pain", "chest pain"),
    ("chest bain", "chest pain", "chest pain"),
    ("chist bayn", "chest pain", "chest pain"),
    # Therapies
    ("anty blatalet therapy", "antiplatelet therapy", "antiplatelet therapy"),
    ("anty blatalet theraby", "antiplatelet therapy", "antiplatelet therapy"),
    ("anty playtalet therapy", "antiplatelet therapy", "antiplatelet therapy"),
    ("inty platalet theraby", "antiplatelet therapy", "antiplatelet therapy"),
    # Heart-related
    ("cardiac anzymes", "cardiac enzymes", "cardiac enzymes"),
    ("cardiac enza yms", "cardiac enzymes", "cardiac enzymes"),
    ("cardiac ensymes", "cardiac enzymes", "cardiac enzymes"),
    # Oxygen
    ("aksyn satoration", "oxygen saturation", "oxygen saturation"),
    ("aksygen satoration", "oxygen saturation", "oxygen saturation"),
    ("oxygen satoration", "oxygen saturation", "oxygen saturation"),
    ("akssjen satoration", "oxygen saturation", "oxygen saturation"),
    # Ischemic
    ("ischemic changes", "ischemic changes", "ischemic changes"),
    ("ischemic chenges", "ischemic changes", "ischemic changes"),
    ("ischemic shanges", "ischemic changes", "ischemic changes"),
    ("iskemic changes", "ischemic changes", "ischemic changes"),
]


def match_multi_word_arabic(
    transliterated_tokens: Sequence[str],
) -> List[Dict[str, any]]:
    """Try to match a sequence of transliterated Arabic tokens against known
    multi-word English medical phrases.

    Returns a list of (start_idx, end_idx, english_phrase, score) matches.
    The input `transliterated_tokens` is the list of transliterated tokens
    from the transcript.
    """
    matches: List[Dict[str, any]] = []
    if not transliterated_tokens:
        return matches

    # Try sliding windows of length 2-4
    for window_size in range(4, 1, -1):  # 4, 3, 2
        for i in range(len(transliterated_tokens) - window_size + 1):
            window = " ".join(transliterated_tokens[i:i + window_size])
            window_lower = window.lower()

            for translit, english, _ in _MULTI_WORD_PHRASES:
                if window_lower == translit:
                    matches.append({
                        "start": i,
                        "end": i + window_size,
                        "english": english,
                        "score": 100.0,
                        "phrase_type": "multi_word_arabic",
                    })
                    break
                # Fuzzy match for minor spelling variations.
                # IMPORTANT: only allow fuzzy skeleton match when the number
                # of tokens (words) in the phrase matches the window size.
                # Otherwise, a 3-token window crossing a phrase boundary
                # (e.g., "blad shwjr bald") could incorrectly match a
                # 2-token phrase ("blad shwjr") via substring skeleton.
                n_phrase_words = translit.count(' ') + 1
                if n_phrase_words == window_size:
                    skel_sim = skeleton_similarity(window_lower, translit)
                    if skel_sim >= 80.0:
                        matches.append({
                            "start": i,
                            "end": i + window_size,
                            "english": english,
                            "score": round(skel_sim, 2),
                            "phrase_type": "multi_word_arabic_fuzzy",
                        })
                        break

    # Dedup: keep highest score per start position
    seen_start: Dict[int, Dict[str, any]] = {}
    for m in matches:
        if m["start"] not in seen_start or m["score"] > seen_start[m["start"]]["score"]:
            seen_start[m["start"]] = m
    return sorted(seen_start.values(), key=lambda x: -x["score"])


# ---------------------------------------------------------------------------
# Convenience: score a span against all lexicon variants
# ---------------------------------------------------------------------------


def score_span_against_lexicon(
    span_text: str,
    lexicon_entries: Sequence[Dict[str, any]],
    is_arabic: bool = False,
) -> List[Dict[str, any]]:
    """Score a single span text against all lexicon variants.

    Returns scored candidates sorted by descending score.
    """
    from .correction import normalize_text

    candidates: List[Dict[str, any]] = []
    span_norm = normalize_text(span_text)

    for entry in lexicon_entries:
        term = entry.get("term", "")
        if not term:
            continue
        for variant in [term] + entry.get("aliases", []):
            if not variant:
                continue
            phon = combined_phonetic_similarity(span_text, variant)
            if phon["score"] > 0:
                candidates.append({
                    "term": term,
                    "variant": variant,
                    "score": phon["score"],
                    "ipa": phon["ipa"],
                    "skeleton": phon["skeleton"],
                    "method": phon["method"],
                })

    candidates.sort(key=lambda c: -c["score"])
    return candidates
