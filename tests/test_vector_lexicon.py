"""Tests for app/services/vector_lexicon.py — multi-view n-gram term retrieval."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services import vector_lexicon


# Skip all FAISS-dependent tests if the library is not available.
# This is common on very new Python versions (e.g. 3.14) where
# faiss-cpu doesn't have a prebuilt wheel yet.
_has_faiss = False
try:
    import faiss  # noqa: F401
    _has_faiss = True
except ImportError:
    pass


_faiss_mark = pytest.mark.skipif(
    not _has_faiss,
    reason="faiss not installed (no wheel for this Python version)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Reset the VectorLexicon singleton before each test."""
    vector_lexicon._INSTANCE = None
    yield
    vector_lexicon._INSTANCE = None


@pytest.fixture
def sample_terms() -> list[dict]:
    """A small set of lexicon terms with aliases for testing."""
    return [
        {"term": "history", "type": "term", "aliases": ["hx", "h/o"]},
        {"term": "diabetes", "type": "disease", "aliases": ["dm", "diabetes mellitus"]},
        {"term": "hypertension", "type": "disease", "aliases": ["htn", "high blood pressure"]},
        {"term": "aspirin", "type": "drug", "aliases": ["asa", "acetyl salicylic acid"]},
        {"term": "paracetamol", "type": "drug", "aliases": ["acetaminophen", "panadol"]},
        {"term": "nitroglycerin", "type": "drug", "aliases": ["ntg", "glyceryl trinitrate"]},
        {"term": "insulin", "type": "drug", "aliases": []},
        {"term": "hyperglycemia", "type": "condition", "aliases": ["hyperglacymia"]},
        {"term": "blood sugar", "type": "term", "aliases": ["bs", "glucose"]},
        {"term": "shortness of breath", "type": "symptom", "aliases": ["sob", "dyspnea"]},
    ]


# ---------------------------------------------------------------------------
# _transliterate helper
# ---------------------------------------------------------------------------


class TestTransliterate:
    """_transliterate converts Arabic characters to Latin approximations."""

    def test_arabic_to_latin(self) -> None:
        assert vector_lexicon._transliterate("هستوري") == "hstwry"

    def test_arabic_contains(self) -> None:
        assert vector_lexicon._has_arabic("هستوري") is True
        assert vector_lexicon._has_arabic("history") is False
        assert vector_lexicon._has_arabic("mixed هستوري text") is True

    def test_transliterate_mixed(self) -> None:
        result = vector_lexicon._transliterate("هستوري history")
        assert "hstwry" in result
        assert "history" in result


# ---------------------------------------------------------------------------
# _text_views
# ---------------------------------------------------------------------------


class TestTextViews:
    """_text_views produces multiple character-level views for matching."""

    def test_english_word_has_two_views(self) -> None:
        views = vector_lexicon._text_views("history")
        assert len(views) >= 1
        assert "history" in views

    def test_arabic_word_has_transliteration_view(self) -> None:
        views = vector_lexicon._text_views("هستوري")
        # Should have: normalised, transliteration, possibly skeleton
        translit_views = [v for v in views if v.isascii()]
        assert len(translit_views) >= 1
        assert "hstwry" in translit_views

    def test_diacritics_stripped(self) -> None:
        views = vector_lexicon._text_views("الْتَهَاب")
        # Diacritics removed
        assert any("التهاب" in v for v in views)


# ---------------------------------------------------------------------------
# VectorLexicon build + query
# ---------------------------------------------------------------------------


@_faiss_mark
class TestVectorLexiconBuildAndQuery:
    """Build the n-gram index and query it."""

    def test_empty_terms_does_not_crash(self) -> None:
        vlex = vector_lexicon.VectorLexicon(backend="ngram")
        vlex.build([])  # empty list
        assert vlex._built is True
        assert len(vlex._entries) == 0

    def test_build_with_terms(self, sample_terms: list[dict]) -> None:
        vlex = vector_lexicon.VectorLexicon(backend="ngram")
        vlex.build(sample_terms)
        assert vlex._built is True
        assert len(vlex._entries) >= len(sample_terms)

    def test_query_returns_results(self, sample_terms: list[dict]) -> None:
        vlex = vector_lexicon.VectorLexicon(backend="ngram", similarity_threshold=0.15)
        vlex.build(sample_terms)
        results = vlex.query("history", top_k=3)
        assert len(results) >= 1
        assert results[0]["term"] == "history"

    def test_query_arabic_transliteration(self, sample_terms: list[dict]) -> None:
        """Arabic هستوري should match 'history' via transliteration bridge."""
        vlex = vector_lexicon.VectorLexicon(backend="ngram", similarity_threshold=0.10)
        vlex.build(sample_terms)
        results = vlex.query("هستوري", top_k=5)
        terms = [r["term"] for r in results]
        assert "history" in terms, f"Expected 'history' in terms, got {terms}"

    def test_query_diabetes_transliteration(self, sample_terms: list[dict]) -> None:
        """Arabic دايابيتس should match 'diabetes'."""
        vlex = vector_lexicon.VectorLexicon(backend="ngram", similarity_threshold=0.10)
        vlex.build(sample_terms)
        results = vlex.query("دايابيتس", top_k=5)
        terms = [r["term"] for r in results]
        assert "diabetes" in terms, f"Expected 'diabetes' in terms, got {terms}"

    def test_query_english_misspelling(self, sample_terms: list[dict]) -> None:
        """Misspelled 'hyperglacymia' should match 'hyperglycemia'."""
        vlex = vector_lexicon.VectorLexicon(backend="ngram", similarity_threshold=0.15)
        vlex.build(sample_terms)
        results = vlex.query("hyperglacymia", top_k=3)
        terms = [r["term"] for r in results]
        assert "hyperglycemia" in terms, f"Expected 'hyperglycemia' in terms, got {terms}"

    def test_query_normal_arabic_low_scores(self, sample_terms: list[dict]) -> None:
        """Normal Arabic words like حضر should not strongly match medical terms."""
        vlex = vector_lexicon.VectorLexicon(backend="ngram", similarity_threshold=0.30)
        vlex.build(sample_terms)
        results = vlex.query("حضر", top_k=3)
        # With threshold 0.30, normal words should return few or no results
        assert len(results) <= 2  # may have weak matches but shouldn't dominate

    def test_query_empty_string_returns_empty(self, sample_terms: list[dict]) -> None:
        vlex = vector_lexicon.VectorLexicon(backend="ngram")
        vlex.build(sample_terms)
        results = vlex.query("")
        assert results == []

    def test_query_short_word_returns_empty(self, sample_terms: list[dict]) -> None:
        vlex = vector_lexicon.VectorLexicon(backend="ngram")
        vlex.build(sample_terms)
        results = vlex.query("a")
        assert results == []

    def test_scores_descending(self, sample_terms: list[dict]) -> None:
        vlex = vector_lexicon.VectorLexicon(backend="ngram", similarity_threshold=0.0)
        vlex.build(sample_terms)
        results = vlex.query("aspirin", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_batch(self, sample_terms: list[dict]) -> None:
        vlex = vector_lexicon.VectorLexicon(backend="ngram", similarity_threshold=0.25)
        vlex.build(sample_terms)
        results = vlex.query_batch(["history", "aspirin"], top_k=2)
        assert "history" in results
        assert "aspirin" in results
        assert len(results["history"]) >= 1


@_faiss_mark
class TestVectorLexiconSingleton:
    """get_vector_lexicon() returns a singleton."""

    def test_singleton_returns_same_instance(self) -> None:
        a = vector_lexicon.get_vector_lexicon(backend="ngram")
        b = vector_lexicon.get_vector_lexicon(backend="ngram")
        assert a is b

    def test_singleton_built_once(self) -> None:
        vector_lexicon._INSTANCE = None
        vlex = vector_lexicon.get_vector_lexicon(backend="ngram")
        assert vlex._built is True
        assert len(vlex._entries) > 0

    def test_warm_up(self) -> None:
        """warm_up() should build the singleton without error."""
        vector_lexicon._INSTANCE = None
        vector_lexicon.warm_up()
        assert vector_lexicon._INSTANCE is not None
        assert vector_lexicon._INSTANCE._built is True


class TestLexiconEntry:
    """LexiconEntry dataclass."""

    def test_creation(self) -> None:
        entry = vector_lexicon.LexiconEntry("test", "drug", ["alias1", "alias2"])
        assert entry.term == "test"
        assert entry.term_type == "drug"
        assert "alias1" in entry.aliases

    def test_default_aliases(self) -> None:
        entry = vector_lexicon.LexiconEntry("test")
        assert entry.aliases == []

    def test_repr(self) -> None:
        entry = vector_lexicon.LexiconEntry("test", "drug")
        assert "test" in repr(entry)
        assert "drug" in repr(entry)
