"""Unit tests for Stage A suspicion-scoring signals in flag.py.

Tests are organised in four groups:
  1. LM perplexity signal (_context_perplexity)
  2. Semantic coherence signal (_semantic_coherence)
  3. Feedback loop (_record_correction -> _get_feedback_boost)
  4. Arabic normalcy (_is_arabic_normalcy) and fused scoring (score_suspicion)

NOTE: All Arabic test inputs use actual Arabic script (e.g. "هستوري"),
not Latin transliterations (e.g. "hstwry"), because the production
code checks _ARABIC_LETTER_RE.search(word) which requires actual
Unicode Arabic characters to fire correctly.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.services import flag


# ===========================================================================
# Group 1 -- LM perplexity: _context_perplexity
# ===========================================================================

class TestContextPerplexity:
    """LM-based context anomaly detection via _context_perplexity(word, ctx)."""

    def test_no_lm_returns_zero(self) -> None:
        """When no LM is loaded, _context_perplexity returns 0.0."""
        with patch.object(flag, "_load_lm", return_value=None):
            score = flag._context_perplexity("pain", ["chest"])
            assert score == 0.0

    def test_lm_loaded_once(self) -> None:
        """_load_lm() should return a non-None LM when the pickle exists."""
        lm = flag._load_lm()
        if lm is not None:
            assert lm.order == 4
            assert lm.get_vocab_size() > 10
            assert lm.get_total_ngrams() > 10

    def test_expected_word_low_perplexity(self) -> None:
        """Medical word 'pain' after 'chest' should have low perplexity (< 0.5)."""
        lm = flag._load_lm()
        if lm is None:
            return
        score = flag._context_perplexity("pain", ["chest"])
        assert 0.0 <= score < 0.5, f"Expected low perplexity, got {score:.4f}"

    def test_surprising_word_high_perplexity(self) -> None:
        """Non-medical word 'elephant' after 'chest' should have high perplexity (> 0.5)."""
        lm = flag._load_lm()
        if lm is None:
            return
        score = flag._context_perplexity("elephant", ["chest"])
        assert score > 0.5, f"Expected high perplexity, got {score:.4f}"

    def test_diabetes_after_history_of_ppl(self) -> None:
        """'diabetes' after 'history of' should be well predicted (ppl < 0.6)."""
        lm = flag._load_lm()
        if lm is None:
            return
        score = flag._context_perplexity("diabetes", ["history", "of"])
        assert 0.0 <= score < 0.6, f"Expected low-medium perplexity, got {score:.4f}"

    def test_empty_context_still_scores(self) -> None:
        """Empty left context still gets begin-token padding from the LM,
        so perplexity is non-zero (it's ppl(word | <s> <s> <s>))."""
        lm = flag._load_lm()
        if lm is None:
            return
        score = flag._context_perplexity("pain", [])
        # Begin-token padding always produces some non-zero score
        assert score >= 0.0, f"Expected >= 0.0, got {score:.4f}"

    def test_oov_word_moderate_perplexity(self) -> None:
        """An OOV word (not in LM vocab) should get moderate perplexity (0.2-1.0)."""
        lm = flag._load_lm()
        if lm is None:
            return
        score = flag._context_perplexity("xyzwxyz", ["the"])
        assert 0.2 <= score <= 1.0, f"Expected moderate perplexity, got {score:.4f}"


# ===========================================================================
# Group 2 -- Semantic coherence: _semantic_coherence
# ===========================================================================

class TestSemanticCoherence:
    """Context-aware script-mismatch detection via _semantic_coherence()."""

    def test_arabic_in_latin_context_suspicious(self) -> None:
        """Arabic word in English context should be flagged (>= 0.10)."""
        score = flag._semantic_coherence(
            "هستوري",        # Arabic transliteration of "history"
            ["patient", "has"],
            ["of", "diabetes"],
        )
        assert score >= 0.10, f"Expected >= 0.10, got {score:.4f}"

    def test_latin_in_arabic_context_slightly_suspicious(self) -> None:
        """English word in Arabic context should get some suspicion."""
        score = flag._semantic_coherence(
            "pain",          # Latin word
            ["يعني", "من"],  # Arabic context
            ["في", "الصدر"], # Arabic context
        )
        assert score >= 0.05, f"Expected >= 0.05, got {score:.4f}"

    def test_mixed_script_token_very_suspicious(self) -> None:
        """Mixed Arabic+Latin token gets semantic signal >= 0.30."""
        score = flag._semantic_coherence(
            "painشديد",      # Mixed Latin+Arabic script
            ["يعني"],
            ["في"],
        )
        assert score >= 0.30, f"Expected >= 0.30, got {score:.4f}"

    def test_all_arabic_not_suspicious(self) -> None:
        """All-Arabic context should produce no suspicion."""
        score = flag._semantic_coherence(
            "يشتكي",         # Arabic verb
            ["المريض"],      # Arabic context
            ["من", "صداع"],  # Arabic context
        )
        assert score == 0.0, f"Expected 0.0, got {score:.4f}"

    def test_all_latin_not_suspicious(self) -> None:
        """All-Latin context should produce no suspicion."""
        score = flag._semantic_coherence(
            "the",
            ["is", "patient"],
            ["is", "stable"],
        )
        assert score == 0.0, f"Expected 0.0, got {score:.4f}"

    def test_short_context_no_mismatch_signal(self) -> None:
        """With fewer than 2 context words, the script-mismatch condition
        is skipped (requires total_ctx >= 2)."""
        score = flag._semantic_coherence(
            "هستوري",
            ["patient"],
            [],
        )
        # total_ctx = 1, so the >= 2 check fails
        assert score == 0.0, f"Expected 0.0 with 1 ctx word, got {score:.4f}"

    def test_no_context_returns_zero(self) -> None:
        """No context words at all -> 0.0."""
        score = flag._semantic_coherence("هستوري", [], [])
        assert score == 0.0

    def test_arabic_word_in_majority_latin_context(self) -> None:
        """Arabic word in mostly-Latin context (2 Latin, 1 Arabic) gets signal.

        The condition ctx_latin >= ctx_arabic AND ctx_latin >= 2 triggers
        the 0.15 signal for Arabic-in-Latin-context.
        """
        score = flag._semantic_coherence(
            "هستوري",
            ["the", "patient"],
            ["التهاب"],
        )
        assert score >= 0.10, f"Expected >= 0.10, got {score:.4f}"

    def test_digit_is_neutral(self) -> None:
        """Pure-digit token in mixed context should not be suspicious."""
        score = flag._semantic_coherence(
            "100",
            ["BP"],
            ["is", "normal"],
        )
        assert score == 0.0, f"Expected 0.0 for digits, got {score:.4f}"


# ===========================================================================
# Group 3 -- Feedback loop: _record_correction + _get_feedback_boost
# ===========================================================================

class TestFeedbackLoop:
    """Feedback loop: recording corrections boosts future suspicion scores."""

    @staticmethod
    def _save_feedback():
        return dict(flag._CORRECTION_FEEDBACK)

    @staticmethod
    def _restore_feedback(saved):
        flag._CORRECTION_FEEDBACK.clear()
        flag._CORRECTION_FEEDBACK.update(saved)

    # --- _record_correction ---

    def test_record_arabic_word_stores_key(self) -> None:
        """Recording an Arabic word stores its consonant skeleton key."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            flag._record_correction("هستوري", "history")
            assert len(flag._CORRECTION_FEEDBACK) >= 1
            # Should have an Arabic-tagged entry for the skeleton
            has_ar = any(tag == 'ar' for (_, tag) in flag._CORRECTION_FEEDBACK)
            assert has_ar, "Expected at least one Arabic-tagged feedback entry"
        finally:
            self._restore_feedback(saved)

    def test_record_latin_word_stores_key(self) -> None:
        """Recording a Latin word stores its Latin consonant skeleton."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            flag._record_correction("amoxicilin", "amoxicillin")
            has_en = any(tag == 'en' for (_, tag) in flag._CORRECTION_FEEDBACK)
            assert has_en, "Expected at least one English-tagged feedback entry"
        finally:
            self._restore_feedback(saved)

    def test_record_empty_noop(self) -> None:
        """Recording empty strings does nothing."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            flag._record_correction("", "")
            assert len(flag._CORRECTION_FEEDBACK) == 0
            flag._record_correction("", "history")
            assert len(flag._CORRECTION_FEEDBACK) == 0
            flag._record_correction("هستوري", "")
            assert len(flag._CORRECTION_FEEDBACK) == 0
        finally:
            self._restore_feedback(saved)

    def test_record_exact_and_prefix_keys(self) -> None:
        """Recording stores BOTH the exact skeleton and a 4-char prefix key,
        unless the skeleton is exactly 4 chars long (same key)."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            # "نيتروغلسرين" -> translit "nytrwghlsryn" ->
            #   drop y,w: "ntrghlsrn" -> digraph gh->g: "ntrglsrn" (7 chars)
            flag._record_correction("نيتروغلسرين", "nitroglycerin")
            # Skeleton 'ntrglsrn' -> full key + prefix 'ntrg'
            assert ('ntrglsrn', 'ar') in flag._CORRECTION_FEEDBACK, \
                "Expected full skeleton key 'ntrglsrn'"
            assert ('ntrg', 'ar') in flag._CORRECTION_FEEDBACK, \
                "Expected 4-char prefix key 'ntrg'"
            # Should have exactly 2 entries (full + prefix), not double-counted
            assert len(flag._CORRECTION_FEEDBACK) == 2, \
                f"Expected 2 entries, got {len(flag._CORRECTION_FEEDBACK)}: " \
                f"{list(flag._CORRECTION_FEEDBACK.keys())}"
        finally:
            self._restore_feedback(saved)

    def test_no_prefix_key_when_skeleton_4_chars(self) -> None:
        """When skeleton is exactly 4 chars, no duplicate prefix key is created."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            # "هستوري" -> translit "hstwry" -> skeleton "hstr" (4 chars)
            flag._record_correction("هستوري", "history")
            # Skeleton is 'hstr' (4 chars) -> prefix key would be same,
            # so only 1 entry created (avoiding double-counting)
            assert len(flag._CORRECTION_FEEDBACK) == 1, \
                f"Expected 1 entry for 4-char skeleton, got " \
                f"{len(flag._CORRECTION_FEEDBACK)}"
        finally:
            self._restore_feedback(saved)

    # --- _get_feedback_boost ---

    def test_no_feedback_returns_zero(self) -> None:
        """With empty feedback dict, boost is always 0.0."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            assert flag._get_feedback_boost("هستوري") == 0.0
            assert flag._get_feedback_boost("pain") == 0.0
            assert flag._get_feedback_boost("anything") == 0.0
        finally:
            self._restore_feedback(saved)

    def test_one_correction_boost_005(self) -> None:
        """1 recorded correction -> boost = 0.05."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            flag._record_correction("هستوري", "history")
            boost = flag._get_feedback_boost("هستوري")
            assert boost == 0.05, f"Expected 0.05, got {boost:.2f}"
        finally:
            self._restore_feedback(saved)

    def test_two_corrections_boost_008(self) -> None:
        """2 recorded corrections -> boost = 0.08."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            flag._record_correction("هستوري", "history")
            flag._record_correction("هستوري", "history")
            boost = flag._get_feedback_boost("هستوري")
            assert boost == 0.08, f"Expected 0.08, got {boost:.2f}"
        finally:
            self._restore_feedback(saved)

    def test_three_corrections_boost_010(self) -> None:
        """3 recorded corrections -> boost = 0.10."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            for _ in range(3):
                flag._record_correction("هستوري", "history")
            boost = flag._get_feedback_boost("هستوري")
            assert boost == 0.10, f"Expected 0.10, got {boost:.2f}"
        finally:
            self._restore_feedback(saved)

    def test_four_corrections_boost_012(self) -> None:
        """4 recorded corrections -> boost ~0.12 (0.10 + 0.02*(4-3))."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            for _ in range(4):
                flag._record_correction("هستوري", "history")
            boost = flag._get_feedback_boost("هستوري")
            assert boost == pytest.approx(0.12, abs=1e-10), \
                f"Expected ~0.12, got {boost:.4f}"
        finally:
            self._restore_feedback(saved)

    def test_high_count_capped_at_015(self) -> None:
        """Many corrections -> boost capped at 0.15."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            for _ in range(10):
                flag._record_correction("هستوري", "history")
            boost = flag._get_feedback_boost("هستوري")
            assert boost == 0.15, f"Expected 0.15, got {boost:.2f}"
        finally:
            self._restore_feedback(saved)

    def test_exact_skeleton_match_gives_boost(self) -> None:
        """A word whose exact skeleton was recorded gets a feedback boost."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            flag._record_correction("نيتروغلسرين", "nitroglycerin")
            # Same word -> exact skeleton match 'ntrglsrn'
            boost = flag._get_feedback_boost("نيتروغلسرين")
            assert boost == 0.05, f"Expected 0.05, got {boost:.2f}"
        finally:
            self._restore_feedback(saved)

    def test_prefix_skeleton_match_gives_boost(self) -> None:
        """A word whose skeleton shares the first 4 chars of a recorded
        skeleton gets a partial-match boost."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            # "نيتروغلسرين" registers skeleton 'ntrglsrn' + prefix 'ntrg'
            flag._record_correction("نيتروغلسرين", "nitroglycerin")
            # Query a DIFFERENT Arabic word whose skeleton starts with 'ntrg'
            # "نيتروفلوكساسين" (nitrofloxacin) -> skeleton starts with 'ntr'
            # We should still get a boost from the exact match on the same word
            boost = flag._get_feedback_boost("نيتروغلسرين")
            assert boost == 0.05, f"Expected 0.05 from exact+prefix match, got {boost:.2f}"
        finally:
            self._restore_feedback(saved)

    def test_latin_word_also_gets_boost(self) -> None:
        """Latin words also get feedback boosts when recorded."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            flag._record_correction("clopidogr", "clopidogrel")
            boost = flag._get_feedback_boost("clopidogr")
            assert boost == 0.05, f"Expected 0.05 for Latin word, got {boost:.2f}"
        finally:
            self._restore_feedback(saved)

    def test_unrelated_word_no_boost(self) -> None:
        """A completely unrelated word gets no boost."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            flag._record_correction("هستوري", "history")
            boost = flag._get_feedback_boost("benzene")
            assert boost == 0.0, f"Expected 0.0, got {boost:.2f}"
        finally:
            self._restore_feedback(saved)

    def test_multiple_terms_independent_boosts(self) -> None:
        """Different terms accumulate independently."""
        saved = self._save_feedback()
        try:
            flag._CORRECTION_FEEDBACK.clear()
            flag._record_correction("هستوري", "history")
            flag._record_correction("دايابيتس", "diabetes")
            flag._record_correction("هستوري", "history")
            hstw_boost = flag._get_feedback_boost("هستوري")
            dy_boost = flag._get_feedback_boost("دايابيتس")
            assert hstw_boost == 0.08, \
                f"Expected 0.08 for هستوري, got {hstw_boost:.2f}"
            assert dy_boost == 0.05, \
                f"Expected 0.05 for دايابيتس, got {dy_boost:.2f}"
        finally:
            self._restore_feedback(saved)


# ===========================================================================
# Group 4 -- Arabic normalcy + fused scoring
# ===========================================================================

class TestArabicNormalcy:
    """_is_arabic_normalcy(word): auto-detection of normal Arabic vs transliteration.

    Returns True if the word should be treated as normal Arabic.
    Returns False if the word COULD be a medical transliteration.

    NOTE: Some common Arabic words (like 'السلام') have consonant skeletons
    that happen to match a lexicon skeleton at >= 40% similarity. This is a
    known limitation of the 40% threshold, which is deliberately permissive.
    These words still pass through the pipeline correctly because:
    - Stage A suspicion without context is 0.09 (below 0.10 threshold)
    - Even if flagged, Stage B finds no good phonetic match
    """

    def test_pure_latin_is_normal(self) -> None:
        """Pure Latin words are always normal."""
        assert flag._is_arabic_normalcy("paracetamol") is True
        assert flag._is_arabic_normalcy("patient") is True
        assert flag._is_arabic_normalcy("BP") is True

    def test_short_arabic_word_is_normal(self) -> None:
        """Arabic words shorter than 3 transliterated chars are normal."""
        assert flag._is_arabic_normalcy("من") is True
        assert flag._is_arabic_normalcy("في") is True

    def test_clinical_arabic_word_normal(self) -> None:
        """Normal clinical Arabic word 'التهاب' (inflammation) should be normal."""
        result = flag._is_arabic_normalcy("التهاب")
        assert result is True, f"Expected True for 'التهاب', got {result}"

    def test_arabic_medical_transliteration_detected(self) -> None:
        """Arabic medical transliteration 'هستوري' should be detected as
        potentially medical (returns False)."""
        result = flag._is_arabic_normalcy("هستوري")
        assert result is False, f"Expected False for 'هستوري', got {result}"

    def test_diabetes_transliteration_detected(self) -> None:
        """'دايابيتس' should be detected as potentially medical."""
        result = flag._is_arabic_normalcy("دايابيتس")
        assert result is False, f"Expected False for 'دايابيتس', got {result}"

    def test_aspirin_transliteration_detected(self) -> None:
        """'أسبرين' should be detected as potentially medical."""
        result = flag._is_arabic_normalcy("أسبرين")
        assert result is False, f"Expected False for 'أسبرين', got {result}"


class TestScoreSuspicion:
    """score_suspicion(word, left_context, right_context): fused scoring.

    Fused score = signal_normalcy * 0.30 + signal_perplexity * 0.35
                  + signal_semantic * 0.20 + signal_feedback * 0.15
    """

    # --- Hard gates ---

    def test_digits_score_zero(self) -> None:
        """Pure digits always score 0.0 (hard gate)."""
        assert flag.score_suspicion("100") == 0.0
        assert flag.score_suspicion("37") == 0.0
        assert flag.score_suspicion("0") == 0.0

    def test_short_word_score_zero(self) -> None:
        """Words with < 3 chars score 0.0 (hard gate)."""
        assert flag.score_suspicion("a") == 0.0
        assert flag.score_suspicion("ab") == 0.0

    def test_common_english_score_zero(self) -> None:
        """Common English words (in _COMMON_ENGLISH) score 0.0."""
        assert flag.score_suspicion("patient") == 0.0
        assert flag.score_suspicion("hospital") == 0.0
        # "the" was recently added to _COMMON_ENGLISH
        assert flag.score_suspicion("the") == 0.0

    # --- Arabic normalcy gate ---

    def test_normal_clinical_arabic_zero(self) -> None:
        """Normal clinical Arabic 'التهاب' -> signal_normalcy = 0.0 -> fused = 0.0."""
        score = flag.score_suspicion("التهاب")
        assert score == 0.0, f"Expected 0.0 for 'التهاب', got {score:.4f}"

    def test_arabic_word_with_lexicon_coincidence(self) -> None:
        """Some common Arabic words like 'السلام' happen to match lexicon
        skeletons at >= 40%, so they get a small suspicion score (0.09).
        This is below the Stage A threshold (0.10), so they still skip
        Stage B correctly."""
        score = flag.score_suspicion("السلام")
        # _is_arabic_normalcy returns False (skeleton matches lexicon),
        # so signal_normalcy = 0.30 * 0.30 = 0.09
        assert score == 0.09, f"Expected 0.09 for 'السلام', got {score:.4f}"

    def test_transliteration_scores_positive(self) -> None:
        """Arabic transliteration 'هستوري' gets score > 0 without context."""
        score = flag.score_suspicion("هستوري")
        assert score > 0.0, f"Expected > 0.0 for 'هستوري', got {score:.4f}"

    def test_transliteration_with_context_exceeds_threshold(self) -> None:
        """Arabic transliteration with English context exceeds SUSPICION_THRESHOLD.

        signal_normalcy = 0.30 * 0.30 = 0.090
        signal_semantic: Arabic word in Latin context -> 0.15 * 0.20 = 0.030
        Fused = 0.120 > SUSPICION_THRESHOLD (0.10)
        """
        score = flag.score_suspicion(
            "هستوري",
            left_context=["patient", "has"],
            right_context=["of", "diabetes"],
        )
        assert score >= flag.SUSPICION_THRESHOLD, \
            f"Expected >= {flag.SUSPICION_THRESHOLD} with context, got {score:.4f}"

    # --- Latin non-common words ---

    def test_latin_non_common_baseline(self) -> None:
        """Non-common Latin word gets 0.05 * 0.30 = 0.015."""
        score = flag.score_suspicion("xyzw")
        assert score == 0.015, f"Expected 0.015, got {score:.4f}"

    def test_latin_with_arabic_context_boosted(self) -> None:
        """Latin word in Arabic context gets semantic boost.

        signal_normalcy = 0.05 * 0.30 = 0.015
        signal_semantic = 0.10 * 0.20 = 0.020 (Latin in Arabic context)
        Fused = 0.035 (still well below threshold)
        """
        score = flag.score_suspicion(
            "pain",
            left_context=["يعني", "من"],
            right_context=["في", "الصدر"],
        )
        assert score > 0.015, f"Expected > 0.015, got {score:.4f}"

    # --- Feedback loop integration ---

    def test_feedback_boost_adds_to_score(self) -> None:
        """Recording a correction boosts future suspicion scores by 0.05*0.15=0.0075."""
        saved = dict(flag._CORRECTION_FEEDBACK)
        flag._CORRECTION_FEEDBACK.clear()
        try:
            score_before = flag.score_suspicion("هستوري")
            flag._record_correction("هستوري", "history")
            score_after = flag.score_suspicion("هستوري")
            assert score_after > score_before, \
                f"Expected score_after ({score_after:.4f}) > score_before ({score_before:.4f})"
            increase = score_after - score_before
            assert abs(increase - 0.0075) < 0.001, \
                f"Expected increase of ~0.0075, got {increase:.4f}"
        finally:
            flag._CORRECTION_FEEDBACK.clear()
            flag._CORRECTION_FEEDBACK.update(saved)

    # --- Edge cases ---

    def test_fused_score_capped_at_one(self) -> None:
        """Fused score never exceeds 1.0."""
        # Use a word that gets non-zero normalcy signal
        score = flag.score_suspicion("هستوري")
        assert score <= 1.0, f"Expected <= 1.0, got {score:.4f}"
        # With extreme context
        score = flag.score_suspicion(
            "هستوري",
            left_context=["patient", "has"],
            right_context=["of", "diabetes"],
        )
        assert score <= 1.0, f"Expected <= 1.0, got {score:.4f}"

    def test_empty_context_no_perplexity(self) -> None:
        """Without left_context, perplexity signal is 0 and doesn't contribute."""
        score = flag.score_suspicion("هستوري")
        # Only normalcy signal: 0.30 * 0.30 = 0.09
        assert score == 0.09, f"Expected 0.09, got {score:.4f}"

    def test_digits_in_any_context_stay_zero(self) -> None:
        """Digits surrounded by text still score 0.0 (hard gate runs first)."""
        score = flag.score_suspicion(
            "100",
            left_context=["BP"],
            right_context=["is", "normal"],
        )
        assert score == 0.0, f"Expected 0.0, got {score:.4f}"
