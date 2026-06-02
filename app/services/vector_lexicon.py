"""Vector-based lexicon retrieval for medical term matching.

Replaces skeleton matching with fast approximate nearest-neighbour search
using character n-gram features (primary) or transformer embeddings (optional).

Architecture:
  1. On startup, encode all lexicon entries into a FAISS index.
  2. query(noisy_word) returns top-k matches with similarity scores.
  3. Two backends: "ngram" (fast, no GPU) and "transformer" (semantic).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import lexicon

logger = logging.getLogger(__name__)

# ── Arabic-to-Latin transliteration bridge ───────────────────────────
# Maps Arabic characters to Latin approximations so that Arabic
# transliterations (e.g. هستوري) can match English terms (e.g. history)
# via shared n-grams in the Latin character space.

_ARABIC_TO_LATIN = {
    'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j',
    'ح': 'h', 'خ': 'kh', 'د': 'd', 'ذ': 'dh', 'ر': 'r',
    'ز': 'z', 'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'd',
    'ط': 't', 'ظ': 'z', 'ع': 'e', 'غ': 'gh', 'ف': 'f',
    'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n',
    'ه': 'h', 'و': 'w', 'ي': 'y', 'ى': 'a', 'ة': 'h',
    'ئ': 'y', 'ء': '',  'آ': 'a', 'أ': 'a', 'ؤ': 'w',
    'إ': 'a', 'ٱ': 'a', 'ٰ': '',
}


def _transliterate(text: str) -> str:
    """Convert Arabic characters in text to Latin approximations.

    Uses flag.py's battle-tested _translit function when available,
    which handles clitic stripping, digraphs (gh→gh, sh→sh, th→th),
    and tashkeel removal. Falls back to the simple character map.

    Non-Arabic characters pass through unchanged.
    """
    if not _has_arabic(text):
        return text.lower()
    try:
        from .flag import _translit as _flag_translit  # type: ignore
        return _flag_translit(text, strip_clitics=True)
    except ImportError:
        result = []
        for c in text:
            result.append(_ARABIC_TO_LATIN.get(c, c))
        return ''.join(result)


def _has_arabic(text: str) -> bool:
    """Check if text contains any Arabic characters."""
    return any('\u0600' <= c <= '\u06FF' for c in text)


def _normalise(text: str) -> str:
    """Normalise text: lowercase, strip diacritics, remove punctuation."""
    t = text.lower()
    t = re.sub(r"[_\s\-'\"،\.\,\;\:\(\)\[\]]+", "", t)
    t = re.sub(r"[\u064B-\u065F\u0670]", "", t)  # Arabic diacritics
    return t


def _text_views(text: str) -> List[str]:
    """Produce multiple character-level views of text for robust matching.

    Views produced:
      1. Normalised text (original script)
      2. Consonant skeleton (vowels removed) — bridges Arabic->English via
         shared consonants when vowel mappings differ
      3. Arabic transliteration (if text contains Arabic)
      4. Arabic transliteration skeleton

    Example for 'هستوري': views are ["هستوري", "hstwry", "hstwr"]
    Example for 'history':   views are ["history", "hstr"]
    Both share skeleton n-grams like 'hst' and 'str'.
    """
    normalised = _normalise(text)
    if not normalised:
        return []

    views = [normalised]

    # Consonant skeleton: remove vowels from Latin text
    stripped = re.sub(r'[aeiou]', '', normalised)
    if stripped and stripped != normalised and len(stripped) >= 2:
        views.append(stripped)

    # For Arabic text, add transliteration views
    if _has_arabic(normalised):
        translit = _transliterate(normalised)
        if translit:
            views.append(translit)
            # Transliteration skeleton (vowels removed from transliteration)
            t_stripped = re.sub(r'[aeiou]', '', translit)
            if t_stripped and t_stripped != translit and len(t_stripped) >= 2:
                views.append(t_stripped)

    # Deduplicate preserving order
    seen: set = set()
    unique: List[str] = []
    for v in views:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def _view_ngrams(views: List[str], n: int = 3) -> List[str]:
    """Generate character n-grams from all text views."""
    ngrams: List[str] = []
    for v in views:
        if len(v) < n:
            ngrams.append(v)
        else:
            ngrams.extend(v[i:i + n] for i in range(len(v) - n + 1))
    # Deduplicate
    return list(dict.fromkeys(ngrams))


def _ngram_vector_from_views(views: List[str], ngram_set: Dict[str, int], n: int = 3) -> np.ndarray:
    """Build a sparse-ish integer vector of n-gram counts from multiple text views."""
    vec = np.zeros(len(ngram_set), dtype=np.float32)
    for v in views:
        if len(v) < n:
            idx = ngram_set.get(v)
            if idx is not None:
                vec[idx] += 1.0
        else:
            for i in range(len(v) - n + 1):
                idx = ngram_set.get(v[i:i + n])
                if idx is not None:
                    vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── Embedding backend (transformer) ────────────────────────────────────

_EMBEDDING_MODEL: Any = None  # lazy loaded
_EMBEDDING_LOCK = threading.Lock()


def _load_embedding_model(model_name: str = "distilbert-base-multilingual-cased") -> Any:
    """Lazy-load the transformer model for embeddings (CPU only to avoid VRAM contention)."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    with _EMBEDDING_LOCK:
        if _EMBEDDING_MODEL is not None:
            return _EMBEDDING_MODEL
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            logger.info("Loading embedding model: %s (CPU)", model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            _EMBEDDING_MODEL = (tokenizer, model)
            logger.info("Embedding model loaded OK (%dM params)", model.num_parameters() // 1_000_000)
        except Exception as exc:
            logger.warning("Failed to load embedding model: %s. Falling back to ngram.", exc)
            _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


def _encode(texts: List[str], model_name: str = "distilbert-base-multilingual-cased") -> np.ndarray:
    """Encode texts with mean-pooled transformer embeddings."""
    loaded = _load_embedding_model(model_name)
    if loaded is None:
        raise RuntimeError("Embedding model not available")
    tokenizer, model = loaded
    import torch

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    attention_mask = inputs["attention_mask"]
    token_embs = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
    return (torch.sum(token_embs * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)).numpy()


# ── Lexicon entries with embeddings ────────────────────────────────────

class LexiconEntry:
    """A single lexicon entry with pre-computed features."""

    __slots__ = ("term", "term_type", "aliases", "features")

    def __init__(self, term: str, term_type: str = "", aliases: Optional[List[str]] = None):
        self.term = term
        self.term_type = term_type
        self.aliases = aliases or []

    def __repr__(self) -> str:
        return f"LexiconEntry({self.term!r}, type={self.term_type})"


# ── Vector Lexicon Index ──────────────────────────────────────────────

class VectorLexicon:
    """FAISS-based lexicon index for fast term similarity search.

    Two backends:
      - "ngram": character n-gram TF-IDF vectors (fast, CPU, deterministic)
      - "transformer": transformer sentence embeddings (semantic, slower)

    Typical usage:
        vlex = VectorLexicon()
        vlex.build()               # index all lexicon terms
        results = vlex.query("هستوري")  # returns [(LexiconEntry, score), ...]
    """

    def __init__(
        self,
        backend: str = "ngram",
        similarity_threshold: float = 0.35,
        embedding_model: str = "distilbert-base-multilingual-cased",
    ):
        self.backend = backend
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self._index: Any = None
        self._entries: List[LexiconEntry] = []
        self._ngram_set: Dict[str, int] = {}
        self._ngram_n: int = 3
        self._built = False
        self._lock = threading.Lock()

    # ── Build index ────────────────────────────────────────────────────

    def build(self, terms: Optional[List[Dict[str, Any]]] = None) -> None:
        """Build the FAISS index from lexicon terms.

        Args:
            terms: List of dicts with keys "term", "type", "aliases".
                   If None, loads from the project lexicon.
        """
        if terms is None:
            raw = lexicon.list_terms()
        else:
            raw = terms

        # Build LexiconEntry list
        entries: List[LexiconEntry] = []
        texts_for_index: List[str] = []
        for item in raw:
            term = item.get("term", "")
            if not term:
                continue
            term_type = item.get("type", item.get("term_type", ""))
            aliases = item.get("aliases", [])
            entry = LexiconEntry(term, term_type, aliases)
            entries.append(entry)
            texts_for_index.append(term)
            for alias in aliases:
                entries.append(LexiconEntry(alias, term_type))
                texts_for_index.append(alias)

        if not entries:
            logger.warning("VectorLexicon: no terms to index")
            self._entries = []
            self._built = True
            return

        if self.backend == "transformer":
            self._build_transformer(entries, texts_for_index)
        else:
            self._build_ngram(entries, texts_for_index)

    def _build_ngram(self, entries: List[LexiconEntry], texts: List[str]) -> None:
        """Build n-gram FAISS index using multi-view feature extraction."""
        import faiss

        # Collect all n-grams from all views of all texts
        ngram_set: Dict[str, int] = {}
        for t in texts:
            for ng in _view_ngrams(_text_views(t), self._ngram_n):
                if ng not in ngram_set:
                    ngram_set[ng] = len(ngram_set)

        if not ngram_set:
            logger.warning("VectorLexicon: no n-grams found across any text views")
            self._entries = entries
            self._ngram_set = {}
            self._built = True
            return

        # Build vectors
        vectors = np.zeros((len(texts), len(ngram_set)), dtype=np.float32)
        for i, t in enumerate(texts):
            vectors[i] = _ngram_vector_from_views(_text_views(t), ngram_set, self._ngram_n)

        # Normalise
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = np.where(norms > 0, vectors / norms, vectors)

        # Build FAISS index
        index = faiss.IndexFlatIP(len(ngram_set))
        index.add(vectors.astype(np.float32))

        self._index = index
        self._entries = entries
        self._ngram_set = ngram_set
        self._built = True
        logger.info(
            "VectorLexicon (ngram) built: %d entries, %d n-grams",
            len(entries), len(ngram_set),
        )

    def _build_transformer(self, entries: List[LexiconEntry], texts: List[str]) -> None:
        """Build transformer embedding FAISS index."""
        import faiss

        try:
            embs = _encode(texts, self.embedding_model)
        except Exception as exc:
            logger.warning("Transformer embedding failed: %s. Falling back to ngram.", exc)
            self.backend = "ngram"
            self._build_ngram(entries, texts)
            return

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = np.where(norms > 0, embs / norms, embs)

        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs.astype(np.float32))

        self._index = index
        self._entries = entries
        self._built = True
        logger.info(
            "VectorLexicon (transformer) built: %d entries, %d dims",
            len(entries), embs.shape[1],
        )

    # ── Query ──────────────────────────────────────────────────────────

    def query(
        self,
        word: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Find similar terms in the lexicon.

        Args:
            word: Noisy word to match (Arabic or English).
            top_k: Maximum number of candidates to return.
            threshold: Minimum similarity threshold (overrides instance default).

        Returns:
            List of dicts: [{"term": ..., "score": ..., "term_type": ...}, ...]
        """
        if not self._built or not self._entries:
            return []
        if not word or len(word) < 2:
            return []

        thresh = threshold if threshold is not None else self.similarity_threshold

        with self._lock:
            try:
                if self.backend == "transformer":
                    vec = _encode([word], self.embedding_model)
                else:
                    views = _text_views(word)
                    vec = _ngram_vector_from_views(views, self._ngram_set, self._ngram_n).reshape(1, -1)

                # Normalise
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

                scores_arr, indices_arr = self._index.search(vec.astype(np.float32), top_k)

                results: List[Dict[str, Any]] = []
                for score, idx in zip(scores_arr[0], indices_arr[0]):
                    if idx < 0 or idx >= len(self._entries):
                        continue
                    if score < thresh:
                        continue
                    entry = self._entries[idx]
                    results.append({
                        "term": entry.term,
                        "score": float(score),
                        "term_type": entry.term_type,
                    })
                return results
            except Exception as exc:
                logger.warning("VectorLexicon query failed: %s", exc)
                return []

    def retrieve_top_k(
        self,
        span_text: str,
        top_k: int = 20,
        threshold: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K lexicon candidates for a span, returning enough
        info for downstream scoring to pick the best match.

        This is the primary retrieval interface for the correction pipeline.
        It translates Arabic spans via flag.py, queries the n-gram index,
        and returns candidates with their canonical term and term_type.

        Args:
            span_text: The suspicious span (Arabic or English).
            top_k: Maximum candidates to retrieve.
            threshold: Minimum similarity score (0-1).

        Returns:
            List of dicts with keys "term", "score", "term_type".
        """
        if not self._built or not self._entries:
            return []
        if not span_text or len(span_text) < 2:
            return []

        # For Arabic spans, transliterate first
        query_text = span_text
        if _has_arabic(span_text):
            query_text = _transliterate(span_text)
            if not query_text or len(query_text) < 2:
                return []

        return self.query(query_text, top_k=top_k, threshold=threshold)

    def query_batch(
        self,
        words: List[str],
        top_k: int = 3,
        threshold: Optional[float] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Query multiple words and return a dict keyed by word."""
        return {w: self.query(w, top_k=top_k, threshold=threshold) for w in words}


# ── Singleton ──────────────────────────────────────────────────────────

_INSTANCE: Optional[VectorLexicon] = None
_INSTANCE_LOCK = threading.Lock()


def get_vector_lexicon(
    backend: str = "ngram",
    similarity_threshold: float = 0.35,
) -> VectorLexicon:
    """Get (or create) the singleton VectorLexicon."""
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            return _INSTANCE
        from .config import get_config

        cfg = get_config()
        vlex = VectorLexicon(
            backend=cfg.vector_backend if not backend else backend,
            similarity_threshold=cfg.vector_similarity_threshold if similarity_threshold == 0.35 else similarity_threshold,
            embedding_model=cfg.embedding_model_name,
        )
        vlex.build()
        _INSTANCE = vlex
        return _INSTANCE


def warm_up() -> None:
    """Pre-build the vector lexicon (call on startup)."""
    get_vector_lexicon()
