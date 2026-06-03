"""Weighted Damerau-Levenshtein Arabic spelling corrector with SymSpell indexing.

Replaces the single-substitution generator in arabic_spelling.py with a
proper weighted edit distance that encodes the Gulf Arabic phonetic mergers
as substitution costs.

Architecture:
  1. Build a SymSpell index from the Arabic filler vocabulary + medical lexicon.
  2. Given a misspelled word, query SymSpell for near matches.
  3. Re-rank candidates using weighted edit distance where the cost matrix
     encodes known Gulf mergers (س↔ص↔ث cost=1, ط↔ت cost=1, unrelated pairs cost=3).
  4. Return the best correction if confidence exceeds threshold.

The weighted edit distance ensures that phonetic-class substitutions cost
less than unrelated substitutions, making the corrector more permissive
for real Gulf Arabic misspellings while still rejecting random noise.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from symspellpy import SymSpell, Verbosity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gulf Arabic Phonetic Merger Cost Matrix
# ---------------------------------------------------------------------------
# Maps each Arabic letter to a dict of (alternative_letter -> substitution_cost).
# Cost 1 = very common merger (same phonetic class, e.g. س↔ص)
# Cost 2 = less common but still seen in Gulf speech/dialect
# Cost 3 = rare or dubious (only when there's weak acoustic evidence)
# No entry = effectively infinite cost (never substitute)

_MERGER_COST: Dict[str, Dict[str, int]] = {
    # Sibilants: very high frequency mergers
    "س": {"ص": 1, "ث": 1},
    "ص": {"س": 1, "ض": 2},
    "ث": {"س": 1, "ت": 2, "ذ": 2},
    # Emphatic ↔ non-emphatic
    "ط": {"ت": 1, "ظ": 2},
    "ت": {"ط": 1, "ث": 2, "د": 2},
    # Dental mergers
    "د": {"ض": 1, "ذ": 2, "ظ": 2, "ت": 2},
    "ض": {"د": 1, "ظ": 2, "ز": 2},
    "ذ": {"ز": 1, "ظ": 2, "د": 2},
    "ظ": {"ز": 2, "ذ": 2, "ض": 1},
    "ز": {"ذ": 1, "ض": 2, "ظ": 2},
    # Gutturals/hamza variants (frequent in dictation)
    "ع": {"ا": 2, "أ": 2, "إ": 2, "ء": 2},
    "ا": {"ع": 2, "أ": 1, "إ": 1, "ى": 2},
    "أ": {"ا": 1, "إ": 1, "ع": 2, "ء": 2},
    "إ": {"ا": 1, "أ": 1, "ع": 2},
    "ء": {"ا": 2, "ع": 2, "أ": 2},
    # Ghayn-Qaf-Kaf
    "غ": {"ق": 1, "ع": 2},
    "ق": {"غ": 1, "ك": 2},
    "ك": {"ق": 2},
    # Yaa ↔ Alif maqsura
    "ى": {"ا": 2, "ي": 1},
    "ي": {"ى": 1},
    # Ta marbuta ↔ Ha
    "ة": {"ه": 1},
    "ه": {"ة": 1},
    # Hamza carriers
    "ؤ": {"و": 2, "ء": 3},
    "ئ": {"ي": 2, "ء": 3},
}

# Precompute: for each character, the set of alternatives and minimum cost
_CHAR_ALTERNATIVES: Dict[str, List[Tuple[str, int]]] = {}
for ch, alts in _MERGER_COST.items():
    _CHAR_ALTERNATIVES[ch] = [(alt, cost) for alt, cost in alts.items()]

# ---------------------------------------------------------------------------
# Weighted Damerau-Levenshtein distance
# ---------------------------------------------------------------------------


def _weighted_damerau_levenshtein(a: str, b: str) -> int:
    """Compute weighted Damerau-Levenshtein distance between two Arabic strings.

    Substitution cost depends on the phonetic class of the characters being
    swapped (from _MERGER_COST). Insertion, deletion, and transposition all
    cost 3 (higher than any substitution), biasing the algorithm toward
    substitutions over other edit operations.

    Returns integer distance (lower = more similar).
    """
    n, m = len(a), len(b)
    if n == 0:
        return m * 3  # all insertions
    if m == 0:
        return n * 3  # all deletions

    # Use a 2-row DP to save memory
    prev_row = list(range(0, (m + 1) * 3, 3))  # row for i-1
    cur_row = [0] * (m + 1)

    for i in range(1, n + 1):
        cur_row[0] = i * 3  # deletion cost
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                # Look up substitution cost
                alts = _CHAR_ALTERNATIVES.get(a[i - 1], {})
                sub_cost = 3  # default: high cost
                for alt_char, cost_val in alts:
                    if alt_char == b[j - 1]:
                        sub_cost = cost_val
                        break

                cost = sub_cost

            cur_row[j] = min(
                prev_row[j] + 3,       # deletion
                cur_row[j - 1] + 3,    # insertion
                prev_row[j - 1] + cost,  # substitution
            )

            # Transposition
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur_row[j] = min(cur_row[j], prev_row[j - 2] + 3)

        prev_row, cur_row = cur_row, prev_row

    return prev_row[m]


def _weighted_similarity(a: str, b: str) -> float:
    """Convert weighted edit distance to a 0-1 similarity score.

    0 = completely different, 1 = identical.
    Score = 1 - (distance / max_possible_distance)
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    dist = _weighted_damerau_levenshtein(a, b)
    max_len = max(len(a), len(b))
    max_possible = max_len * 3  # worst case: all operations cost 3
    if max_possible == 0:
        return 0.0
    return max(0.0, 1.0 - dist / max_possible)


# ---------------------------------------------------------------------------
# SymSpell index
# ---------------------------------------------------------------------------

# Clitic prefixes for Arabic
_ARABIC_CLITICS = ("ال", "وال", "بال", "كال", "فال", "لل",
                   "و", "ف", "ب", "ل", "ك")


def _strip_clitics(word: str) -> str:
    for pre in _ARABIC_CLITICS:
        if word.startswith(pre) and len(word) - len(pre) >= 3:
            return word[len(pre):]
    return word


class WeightedArabicCorrector:
    """SymSpell-based Arabic spelling corrector with weighted edit distance.

    Builds an index from a vocabulary of known-correct Arabic words, then
    queries it for misspelled words. Candidates are re-ranked using the
    weighted Damerau-Levenshtein distance that encodes Gulf phonetic mergers.
    """

    def __init__(
        self,
        vocabulary: Optional[Set[str]] = None,
        max_edit_distance: int = 2,
        prefix_length: int = 3,  # shorter prefix catches early-position substitutions
    ):
        self._max_edit_distance = max_edit_distance
        self._prefix_length = prefix_length
        self._vocabulary: Set[str] = set()
        self._built = False

        if vocabulary:
            self.build(vocabulary)

    def build(self, vocabulary: Set[str]) -> None:
        """Build the SymSpell index from a vocabulary of known-correct Arabic words.

        Args:
            vocabulary: Set of Arabic words (with or without clitic prefixes).
        """
        self._vocabulary = vocabulary
        self._symspell = SymSpell(
            max_dictionary_edit_distance=self._max_edit_distance,
            prefix_length=self._prefix_length,
        )

        # Add all vocabulary words and their clitic-stripped forms
        words_to_index: Set[str] = set()
        for word in vocabulary:
            if not word or len(word) < 2:
                continue
            words_to_index.add(word)
            # Add clitic-prefixed forms: the corrector needs to know
            # that 'بالضغط' is valid even though the root 'ضغط' is in vocab
            for pre in _ARABIC_CLITICS:
                prefixed = pre + word
                if len(prefixed) >= 3:
                    words_to_index.add(prefixed)

        for word in sorted(words_to_index):
            # SymSpell expects (term, frequency) pairs
            # We use a base frequency of 1 for all terms
            self._symspell.create_dictionary_entry(word, 1)

        self._built = True
        logger.info(
            "[weighted_arabic] SymSpell index built: %d vocabulary entries → %d indexed forms",
            len(vocabulary), len(words_to_index),
        )

    def correct(
        self,
        word: str,
        threshold: float = 0.75,
        max_candidates: int = 3,
    ) -> Optional[Tuple[str, float]]:
        """Find the best weighted-edit-distance correction for an Arabic word.

        Conservative approach: only correct when the original word is NOT a
        known vocabulary word AND the candidate IS a known vocabulary word.
        This prevents changing one real Arabic word into another (e.g.
        نظيف→نزيف, سليم→سلام) while still catching true misspellings
        (سداع→صداع, الدغط→الضغط).

        Architecture:
          1. Vocabulary check: if word is already known-correct → skip.
          2. SymSpell lookup (edit distance 1) for near matches.
          3. Candidate must be a known vocabulary word.
          4. Weighted edit distance re-ranking with Gulf merger costs.

        Unlike the legacy ``arabic_correction`` module, this corrector does
        NOT use ``_is_arabic_filler()`` as a hard pre-filter. That function
        classifies short Arabic words (3-5 chars) with no skeleton match
        in the medical lexicon as "normal Arabic" and skips them — but it
        also blocks genuine misspellings like ``سداع→صداع`` (skeleton ``sd``)
        which happen to be too short to match any lexicon term. Instead, we
        rely on:
          - Vocabulary membership (word NOT in vocab → may try correction)
          - SymSpell edit distance 1 (only single-edit changes allowed)
          - Candidate must be in vocabulary (only known-correct words as output)
          - Length-ratio check (prevents prefix dropping e.g. ضمن→من)
          - Adaptive threshold: short words (3-5 chars) require higher
            confidence (0.75) than longer words (6+ chars, threshold 0.60)
            to prevent short-word coincidental matches.

        Args:
            word: The Arabic-script word to check.
            threshold: Minimum similarity for a candidate to be accepted.
            max_candidates: Maximum candidates to consider from SymSpell.

        Returns:
            Tuple of (corrected_word, confidence_0_to_1) if a good correction
            is found, or None if the word appears correct or uncorrectable.
        """
        if not self._built or not word or len(word) < 3:
            return None

        # First, check if the word (or its clitic-stripped stem) is ALREADY
        # a known vocabulary word. If so, it's already correct — don't touch it.
        stripped = _strip_clitics(word)
        if word in self._vocabulary or stripped in self._vocabulary:
            return None  # already correct

        # --- Phase 1: SymSpell lookup with Verbosity.ALL ---
        # Verbosity.ALL returns ALL matches within max_edit_distance, including
        # distance-0 (exact) and distance-1 (correctable). This is critical
        # because the query word may be in the index (as a clitic-prefixed form
        # of a vocab word), and Verbosity.CLOSEST would only return the exact
        # match, missing the intended correction at distance 1.
        suggestions = self._symspell.lookup(
            word,
            Verbosity.ALL,
            max_edit_distance=1,
            transfer_casing=True,
        )

        # Adaptive threshold: short words (3-5 chars) need higher confidence
        # to prevent coincidental matches (e.g. سليم→سلام at edit distance 1).
        word_len = len(word)
        if word_len <= 5:
            effective_threshold = max(threshold, 0.75)
        else:
            effective_threshold = max(threshold, 0.60)

        best_candidate: Optional[Tuple[str, float]] = None

        # Phase 1a: Score SymSpell candidates
        for suggestion in suggestions[:max_candidates * 2]:
            candidate = suggestion.term
            if candidate == word:
                continue

            # CRITICAL GUARD: only accept corrections where the CANDIDATE
            # is a known vocabulary word. This prevents changing a real word
            # to another real word (e.g. سليم→سلام, نظيف→نزيف).
            cand_stripped = _strip_clitics(candidate)
            if cand_stripped not in self._vocabulary and candidate not in self._vocabulary:
                continue

            # Length-ratio check: candidates must have similar length to
            # the original word.
            len_ratio = len(candidate) / max(1, word_len)
            if len_ratio < 0.65 or len_ratio > 1.35:
                continue

            sim = _weighted_similarity(word, candidate)
            if sim >= effective_threshold:
                confidence = min(0.95, sim)
                if best_candidate is None or confidence > best_candidate[1]:
                    best_candidate = (candidate, confidence)

        # --- Phase 2: Single-substitution generator (fallback) ---
        # For cases SymSpell misses (when the query word is in the index
        # as a clitic-prefixed form, SymSpell returns only the exact match
        # at distance 0 even with Verbosity.ALL — the bucket search skips
        # the correct form because the exact match is prioritized).
        #
        # The legacy generator creates substitution variants for every
        # position using the phonetic merger map, which handles ALL
        # substitutions regardless of position in the word.
        try:
            from .arabic_spelling import _generate_single_substitutions
            for variant in _generate_single_substitutions(word):
                if variant == word:
                    continue
                # Check if variant (or its clitic-stripped form) is known
                v_stripped = _strip_clitics(variant)
                if v_stripped not in self._vocabulary and variant not in self._vocabulary:
                    continue
                # Length-ratio check
                len_ratio = len(variant) / max(1, word_len)
                if len_ratio < 0.65 or len_ratio > 1.35:
                    continue
                # Compute weighted similarity
                sim = _weighted_similarity(word, variant)
                if sim >= effective_threshold:
                    confidence = min(0.95, sim)
                    if best_candidate is None or confidence > best_candidate[1]:
                        best_candidate = (variant, confidence)
        except ImportError:
            pass

        return best_candidate


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-built)
# ---------------------------------------------------------------------------

_INSTANCE: Optional[WeightedArabicCorrector] = None


def get_weighted_arabic_corrector(
    vocabulary: Optional[Set[str]] = None,
) -> WeightedArabicCorrector:
    """Get (or create) the singleton WeightedArabicCorrector."""
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE

    if vocabulary is not None:
        _INSTANCE = WeightedArabicCorrector(vocabulary=vocabulary)
    else:
        # Build from default vocabulary
        try:
            from .flag import _ARABIC_FILLER
            _INSTANCE = WeightedArabicCorrector(vocabulary=set(_ARABIC_FILLER))
        except ImportError:
            _INSTANCE = WeightedArabicCorrector(vocabulary=set())

    return _INSTANCE
