from unittest.mock import Mock

from app.pipeline import scorer
from app.services import lexicon


# ── Fixtures ───────────────────────────────────────────────────────────────


def _force_heuristic_fallback(monkeypatch):
    """Force ``score_transcript`` to skip ModernBERT and use the heuristic
    fallback. Unit tests in this module were written against heuristic-specific
    suspicion thresholds and are not dependent on any ML model."""
    monkeypatch.setattr(scorer, "_try_modernbert_scorer", Mock(return_value=None))
    monkeypatch.setattr(scorer, "_init_mlm_pipeline", Mock())


def test_scorer_skips_stop_words_and_ranks_unknown_words_higher(tmp_path, monkeypatch):
    """Stop words get 0.0 suspicion; known canonical terms get low suspicion + in_lexicon;
    unknown words (not in lexicon, not common English) get medium suspicion."""
    _force_heuristic_fallback(monkeypatch)
    path = tmp_path / "medical_lexicon.jsonl"
    lexicon.add_term("Aspirin", type_="drug_generic", aliases=[], path=path)
    monkeypatch.setattr(lexicon, "DEFAULT_LEXICON_PATH", path)
    scorer.clear_caches()

    # "xylocarp" is unknown — not in lexicon, not common English, not similar to any canonical term
    words = scorer.score_transcript("the aspirin and xylocarp")

    assert [word.text for word in words] == ["the", "aspirin", "and", "xylocarp"]
    assert words[0].suspicion == 0.0  # stop word
    assert words[2].suspicion == 0.0  # stop word
    assert words[1].in_lexicon is True  # canonical match
    assert words[3].in_lexicon is False
    assert words[3].suspicion > words[1].suspicion  # unknown > known


def test_scorer_single_unknown_content_word_gets_medium_suspicion(tmp_path, monkeypatch):
    """A single unknown content word with no lexicon is uncertain (not automatically high)."""
    _force_heuristic_fallback(monkeypatch)
    path = tmp_path / "medical_lexicon.jsonl"
    monkeypatch.setattr(lexicon, "DEFAULT_LEXICON_PATH", path)
    scorer.clear_caches()

    # No lexicon entries at all — "xylocarp" is unknown and not common English
    words = scorer.score_transcript("xylocarp")

    assert len(words) == 1
    assert words[0].text == "xylocarp"
    assert words[0].in_lexicon is False
    # Unknown word with no similar canonical term -> medium (uncertain)
    assert words[0].suspicion == 0.45


def test_scorer_misspelling_similar_to_canonical_gets_high_suspicion(tmp_path, monkeypatch):
    """A misspelled word character-similar to a canonical term gets high suspicion."""
    _force_heuristic_fallback(monkeypatch)
    path = tmp_path / "medical_lexicon.jsonl"
    lexicon.add_term("sphygmomanometer", type_="device", aliases=[], path=path)
    monkeypatch.setattr(lexicon, "DEFAULT_LEXICON_PATH", path)
    scorer.clear_caches()

    words = scorer.score_transcript("sfigmomanometre")

    assert len(words) == 1
    assert words[0].in_lexicon is False
    # difflib match ratio for sfigmomanometre ↔ sphygmomanometer is ~0.77
    assert words[0].suspicion >= 0.7


def test_scorer_bigram_misspelling_detected(tmp_path, monkeypatch):
    """A two-word phrase matching a multi-word term's alias via bigram matching
    gets high suspicion on both words."""
    _force_heuristic_fallback(monkeypatch)
    path = tmp_path / "medical_lexicon.jsonl"
    lexicon.add_term("Doliprane", type_="drug", aliases=["dolly prahn", "dolipran"], path=path)
    monkeypatch.setattr(lexicon, "DEFAULT_LEXICON_PATH", path)
    scorer.clear_caches()

    words = scorer.score_transcript("take dolly prahn twice")

    dolly = [w for w in words if w.text == "dolly"]
    prahn = [w for w in words if w.text == "prahn"]
    assert dolly and prahn
    assert dolly[0].in_lexicon is False
    assert prahn[0].in_lexicon is False
    assert dolly[0].suspicion >= 0.8
    assert prahn[0].suspicion >= 0.8


def test_scorer_common_english_words_get_low_suspicion(tmp_path, monkeypatch):
    """Common English words like 'patient', 'fever', 'pressure' get low suspicion
    even with an empty lexicon."""
    _force_heuristic_fallback(monkeypatch)
    path = tmp_path / "medical_lexicon.jsonl"
    monkeypatch.setattr(lexicon, "DEFAULT_LEXICON_PATH", path)
    scorer.clear_caches()

    words = scorer.score_transcript("the patient has fever and high blood pressure")

    for w in words:
        if w.suspicion > 0.0:
            assert w.suspicion <= 0.10, f"{w.text} got suspicion {w.suspicion}"