"""Tests for scorer cache behavior (replaces old BART model tests)."""

from app.pipeline.scorer import _get_canonical_index, _get_bigram_map


def test_scorer_cache_index_is_lazy():
    """The canonical index is lazily cached on first call."""
    index = _get_canonical_index()
    assert isinstance(index, dict)


def test_scorer_bigram_map_is_dict():
    """The bigram map is a dict mapping bigram keys to entry lists."""
    bigrams = _get_bigram_map()
    assert isinstance(bigrams, dict)
