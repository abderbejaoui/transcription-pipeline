"""Arabic phonetic spelling correction for Gulf clinical transcripts.

Gulf Arabic has well-known phonetic mergers where certain letters are
substituted for one another in casual speech or by ASR. This module
corrects these Arabic→Arabic spelling errors BEFORE the pipeline falls
through to English medical transliteration matching.

Approach:
  1. Check explicit known misspelling map (for multi-change corrections)
  2. Generate single-substitution variants via phonetic merger map
  3. If any variant is a known correct Arabic word (from filler set), apply it
  4. Otherwise return None — fall through to English transliteration matching

This prevents false positives where Arabic misspellings like 'سداع'
coincidentally match English medical terms via consonant skeleton.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple


# Clitic prefixes for Arabic: definite article, prepositions, conjunctions
_ARABIC_CLITICS = ("ال", "وال", "بال", "كال", "فال", "لل",
                   "و", "ف", "ب", "ل", "ك")


def _strip_clitics(word: str) -> str:
    """Strip common Arabic clitic prefixes for vocabulary lookup."""
    for pre in _ARABIC_CLITICS:
        if word.startswith(pre) and len(word) - len(pre) >= 3:
            return word[len(pre):]
    return word


def _in_vocabulary(word: str, vocabulary: Set[str]) -> bool:
    """Check if a word is in the vocabulary, accounting for clitic prefixes."""
    if word in vocabulary:
        return True
    stripped = _strip_clitics(word)
    return stripped != word and stripped in vocabulary


# ---------------------------------------------------------------------------
# Gulf Arabic Phonetic Merger Map
# ---------------------------------------------------------------------------
# Maps each Arabic letter to the set of letters it is commonly substituted
# WITH (either direction) in Gulf Arabic speech/ASR output.
# The goal: given a misspelled word, try swapping a letter for its
# phonetic-class alternative and check if the result is a known Arabic word.
_ARABIC_MERGER: Dict[str, Tuple[str, ...]] = {
    # Sibilants: س ↔ ص ↔ ث
    "س": ("ص", "ث"),
    "ص": ("س", "ض"),
    "ث": ("س", "ت", "ذ"),
    # Emphatic ↔ non-emphatic: ط ↔ ت
    "ط": ("ت", "ظ"),
    "ت": ("ط", "ث", "د"),
    # Dental mergers: د ↔ ض ↔ ظ ↔ ذ ↔ ز
    "د": ("ض", "ذ", "ظ", "ت"),
    "ض": ("د", "ظ", "ز"),
    "ذ": ("ز", "ظ", "د"),
    "ظ": ("ز", "ذ", "ض"),
    "ز": ("ذ", "ض", "ظ"),
    # Gutturals: ع ↔ أ ↔ إ ↔ ء ↔ ا
    "ع": ("ا", "أ", "إ", "ء"),
    "ا": ("ع", "أ", "إ", "ى"),
    "أ": ("ا", "ع", "إ"),
    "إ": ("ا", "أ", "ع"),
    "ء": ("ا", "ع", "أ"),
    # Ghayn ↔ Qaf ↔ Kaf
    "غ": ("ق", "ع"),
    "ق": ("غ", "ك"),
    "ك": ("ق",),
    # Yaa ↔ Alif maqsura
    "ى": ("ا", "ي"),
    "ي": ("ى",),
    # Ta marbuta ↔ Ha
    "ة": ("ه",),
    "ه": ("ة",),
    # Hamza carriers
    "ؤ": ("و", "ء"),
    "ئ": ("ي", "ء"),
}

# ---------------------------------------------------------------------------
# Explicit misspellings that need more than single-letter substitution
# ---------------------------------------------------------------------------
# These require insertions/deletions or multiple substitutions that the
# single-substitution generator cannot produce.
_ARABIC_MISSPELLING: Dict[str, str] = {
    # Missing ي (long vowel) + د→ض
    "مرد": "مريض",
    # Missing ت (core consonant in هستوري)
    "هسري": "هستوري",
    # هـ→ب substitution (التهب→التهاب — common Gulf misspelling)
    "تهب": "تهاب",
}


def _generate_single_substitutions(word: str) -> Set[str]:
    """Generate all possible single-letter substitution variants using the
    phonetic merger map. Only phonetic-class substitutions are tried (not
    arbitrary deletions/insertions), which keeps precision high.

    Each position in the word, if the character has known phonetic
    alternatives, produces one variant with that character swapped.
    """
    variants: Set[str] = set()
    for i, ch in enumerate(word):
        if ch in _ARABIC_MERGER:
            for alt in _ARABIC_MERGER[ch]:
                variant = word[:i] + alt + word[i + 1 :]
                variants.add(variant)
    return variants


def correct_arabic_spelling(word: str, vocabulary: Set[str]) -> Optional[Tuple[str, float]]:
    """Try to correct an Arabic word's spelling using known phonetic mergers.

    Args:
        word: The Arabic-script word to check (raw, without clitic stripping)
        vocabulary: Set of known-correct Arabic words (e.g., the filler set)

    Returns:
        Tuple of (corrected_word, confidence_0_to_1) if a correction is found
        and confident enough. None if the word appears correct or uncorrectable.
    """
    # Already correct — don't touch it
    if _in_vocabulary(word, vocabulary):
        return None
    if len(word) < 3:
        return None

    # Extract clitic prefix so we can re-attach it after correction.
    # E.g. المرد → prefix='ال', stem='مرد' → map lookup 'مرد' → 'مريض' → 'المريض'
    clitic_pre: str = ""
    stem: str = word
    for pre in _ARABIC_CLITICS:
        if word.startswith(pre) and len(word) - len(pre) >= 3:
            clitic_pre = pre
            stem = word[len(pre):]
            break

    # 1. Explicit misspelling map (multi-change corrections)
    # Check the stripped stem, then re-attach the clitic prefix.
    # This handles المرد → stem=مرد → map → مريض → result=المريض
    if stem in _ARABIC_MISSPELLING:
        corrected = _ARABIC_MISSPELLING[stem]
        if corrected != stem:
            return (clitic_pre + corrected, 0.85)

    # 2. Single phonetic merger substitutions
    best: Optional[Tuple[str, float]] = None
    for variant in _generate_single_substitutions(word):
        if _in_vocabulary(variant, vocabulary):
            # Score: 1 change via phonetic merger costs 0.5 out of max(len)
            n_changes = sum(1 for a, b in zip(word, variant) if a != b)
            max_len = max(len(word), len(variant))
            score = 1.0 - (n_changes * 0.5) / max_len
            if best is None or score > best[1]:
                best = (variant, score)

    if best and best[1] >= 0.60:
        return best

    return None
