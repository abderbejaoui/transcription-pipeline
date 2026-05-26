"""Stage 1: score words for suspicion using ModernBERT-large + heuristic fallback.

The primary scorer uses a **hybrid** strategy that combines heuristic
pre-filtering with a local masked language model (MLM) loaded from the local HF
cache (``D:/HF_CACHE/``):

1. **Heuristic pre-filter** — for each word, first compute a heuristic
   suspicion score using edit distance against the medical lexicon, bigram
   matching, and **SUBTLEX-US word frequency norms** (74K words at the 70th
   frequency percentile). Function words are identified via **spaCy POS
   tagging** (``{DET, ADP, CONJ, CCONJ, AUX, PART, PRON}``) instead of a
   hardcoded stop-word list. Words the heuristic considers *not suspicious*
   (< ``SIMILARITY_MIN``) take the heuristic score directly and skip the MLM.
   This prevents false positives on common English words ("patient", "fever",
   "daily") which may have many plausible alternatives in context.
2. **ModernBERT refinement** — only for words the heuristic flags as
   potentially misspelled (>= 0.40), run the ModernBERT masked LM to refine
   the suspicion: build a masked version of the sentence where that word is
   replaced with ``[MASK]``, then ask ModernBERT's fill-mask pipeline to
   predict the probability of the original word.
   ``suspicion = 1.0 - p(original_word | context)``.
3. The final suspicion is the **maximum** of the heuristic and MLM scores,
   so a word must look bad by *both* criteria to score high.

Fallback chain: ModernBERT → pure heuristic.

The caller (``runner.py``) can check ``last_scoring_used_modernbert()`` to
report the approach used.
"""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from Levenshtein import distance as _lev_distance

from app.services import lexicon

from .models import ScoredWord, Token


# -- ModernBERT model path (local HF cache) --------------------------------

MODERNBERT_MODEL_PATH = "D:/HF_CACHE/models/answerdotai/ModernBERT-large"
MODERNBERT_TOP_K = 50
"""Number of top predictions to retrieve from ModernBERT per word."""

MLM_REFINE_THRESHOLD = 0.40
"""Only call the MLM for words the heuristic scores >= this threshold."""


# -- Stage 0: Tokenisation (str.split, not regex) -------------------------

PUNCT_SET = frozenset('.,;:!?"\'()[]{}<>-')


def tokenize_stage0(transcript: str) -> List[Token]:
    """Split *transcript* on whitespace, preserving original casing and
    trailing punctuation per ``Token`` fields.
    """
    tokens: List[Token] = []
    for idx, word in enumerate(transcript.split()):
        punct = ""
        body = word
        while body and body[-1] in PUNCT_SET:
            punct = body[-1] + punct
            body = body[:-1]
        tokens.append(Token(
            index=idx,
            text=body.lower(),
            original=body,
            punct=punct,
        ))
    return tokens


# -- Legacy regex tokenisation (kept for runner's char-offset logic) --

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+(?:\.\d+)?")


def tokenize_transcript(transcript: str) -> List[Tuple[str, int, int]]:
    """Legacy regex-based tokeniser returning ``(text, start, end)`` tuples.
    Kept for backward compatibility with ``runner._span_char_offsets``.
    """
    return [(match.group(), match.start(), match.end())
            for match in WORD_RE.finditer(transcript)]


SIMILARITY_MIN = 0.55


# -- Helpers -------------------------------------------------


# -- SUBTLEX-US word frequency data (lazy-loaded) ---------------------------

_SUBTLEX_HIGH_FREQ: Set[str] = set()
"""Set of words above the 70th frequency percentile."""

_SUBTLEX_LOADED: bool = False


def _load_subtlex_us() -> None:
    global _SUBTLEX_HIGH_FREQ, _SUBTLEX_LOADED
    if _SUBTLEX_LOADED:
        return

    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "subtlex_us.json"
    if not data_path.is_file():
        print(f"[Stage 1] SUBTLEX-US not found at {data_path}, high-frequency detection disabled")
        _SUBTLEX_LOADED = True
        return

    import json
    with open(data_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # Sort by frequency descending
    sorted_entries = sorted(entries, key=lambda x: x["count"], reverse=True)
    total = len(sorted_entries)

    # 70th percentile = top 30% most frequent words
    cutoff_idx = int(total * 0.7)
    cutoff_count = sorted_entries[cutoff_idx]["count"]

    _SUBTLEX_HIGH_FREQ = {
        e["word"].lower() for e in sorted_entries if e["count"] >= cutoff_count
    }
    _SUBTLEX_LOADED = True
    print(f"[Stage 1] SUBTLEX-US loaded: {len(_SUBTLEX_HIGH_FREQ):,} high-frequency words"
          f" (cutoff: count >= {cutoff_count})")


def _is_high_frequency(word: str) -> bool:
    """Return ``True`` if *word* is above the 70th frequency percentile."""
    if not _SUBTLEX_LOADED:
        _load_subtlex_us()
    return word.lower() in _SUBTLEX_HIGH_FREQ


# -- pyenchant spell-check (lazy-loaded) -----------------------------------

_enchant_dict: Any = None
"""pyenchant en_US dictionary, loaded lazily for medical word validity check."""


def _init_enchant() -> None:
    global _enchant_dict
    if _enchant_dict is not None:
        return
    try:
        import enchant
        _enchant_dict = enchant.Dict("en_US")
    except Exception:
        _enchant_dict = None  # enchant unavailable on this system


def _is_valid_medical_word(word: str) -> bool:
    """Check if *word* is a valid English or medical term.

    Three-tier check (in order):
    1. SUBTLEX-US high-frequency word → valid.
    2. Medical lexicon canonical match → valid.
    3. pyenchant en_US spell-check → valid.

    Only when ALL THREE fail is the word considered potentially misspelled.
    """
    w = word.strip().lower()
    if not w:
        return True

    # Tier 1: SUBTLEX-US high-frequency (common English words)
    if _is_high_frequency(w):
        return True

    # Tier 2: Medical lexicon canonical match
    if _lexicon_entry(w) is not None:
        return True

    # Tier 3: pyenchant en_US spell-check
    _init_enchant()
    if _enchant_dict is not None:
        try:
            if _enchant_dict.check(w):
                return True
        except Exception:
            pass

    return False


# -- Medical wordlist for Levenshtein distance checks (lazy-loaded) ---------

_MEDICAL_WORDLIST: List[str] = []
"""Combined set of known medical terms (lexicon + medical_terms.txt)."""
_MEDICAL_WORDLIST_LOADED: bool = False


def _load_medical_wordlist() -> None:
    global _MEDICAL_WORDLIST, _MEDICAL_WORDLIST_LOADED
    if _MEDICAL_WORDLIST_LOADED:
        return

    known: set[str] = set()

    # Source 1: medical lexicon canonical forms
    for entry in lexicon.load_lexicon():
        cf = entry.canonical_form.strip().lower()
        if cf:
            known.add(cf)
            for alias in entry.aliases:
                alias_norm = alias.strip().lower()
                if alias_norm:
                    known.add(alias_norm)

    # Source 2: legacy/medical_terms.txt
    terms_path = Path(__file__).resolve().parent.parent.parent / "legacy" / "medical_terms.txt"
    if terms_path.is_file():
        with open(terms_path, "r", encoding="utf-8") as f:
            for line in f:
                term = line.strip().lower()
                if term and not term.startswith("#"):
                    known.add(term)

    _MEDICAL_WORDLIST = sorted(known)
    _MEDICAL_WORDLIST_LOADED = True
    print(f"[Stage 1] Medical wordlist loaded: {len(_MEDICAL_WORDLIST):,} terms")


def _get_medical_wordlist() -> List[str]:
    if not _MEDICAL_WORDLIST_LOADED:
        _load_medical_wordlist()
    return _MEDICAL_WORDLIST


def _has_close_dictionary_match(word: str) -> bool:
    """Return True if *word* is within Levenshtein distance 1-3 of any known
    medical term (lexicon + medical_terms.txt).

    Only meaningful for words with suspicion > 0.50.
    """
    w = word.strip().lower()
    if not w or len(w) < 3:
        return False
    wordlist = _get_medical_wordlist()
    for term in wordlist:
        if abs(len(term) - len(w)) > 3:
            continue  # skip terms that differ in length by more than 3
        try:
            if _lev_distance(w, term) <= 3:
                return True
        except Exception:
            continue
    return False


# -- spaCy POS tagging (lazy-loaded) ---------------------------------------

_nlp: Any = None
"""spaCy English model, loaded lazily for POS tagging."""

FUNCTION_WORD_POS_TAGS = frozenset({"DET", "ADP", "CONJ", "CCONJ", "AUX", "PART", "PRON"})
"""Universal POS tags that identify function words (closed-class parts of speech).

- DET: determiner ("the", "a", "this")
- ADP: adposition ("in", "on", "of")
- CONJ / CCONJ: conjunction ("and", "but", "or")
- AUX: auxiliary ("is", "have", "will")
- PART: particle ("not", "to")
- PRON: pronoun ("he", "she", "it")
"""


def _init_spacy() -> None:
    global _nlp
    if _nlp is not None:
        return
    import spacy
    _nlp = spacy.load("en_core_web_sm")


def _compute_function_words(transcript: str) -> Set[str]:
    """POS-tag *transcript* with spaCy and return the set of function words.

    A word is considered a function word if its universal POS tag is one of
    ``FUNCTION_WORD_POS_TAGS`` (DET, ADP, CONJ, CCONJ, AUX, PART, PRON).
    """
    _init_spacy()
    doc = _nlp(transcript)
    function_words: Set[str] = set()
    for token in doc:
        if token.pos_ in FUNCTION_WORD_POS_TAGS:
            function_words.add(token.text.lower())
    return function_words


def is_function_word(word: str) -> bool:
    """Check if a single word is a function word via spaCy POS tagging.

    A word is a function word if its universal POS tag is one of
    ``FUNCTION_WORD_POS_TAGS`` (DET, ADP, CONJ, CCONJ, AUX, PART, PRON).

    This is a convenience wrapper for callers (e.g. ``flagger.py``) that
    need to check individual words without processing a full transcript.
    """
    _init_spacy()
    doc = _nlp(word)
    for token in doc:
        return token.pos_ in FUNCTION_WORD_POS_TAGS
    return False


def _canonical_form(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _char_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _lexicon_entry(token: str) -> Optional[Dict[str, Any]]:
    entry = lexicon.find_by_canonical(token)
    if entry is None:
        return None
    return {
        "term": entry.term,
        "type": entry.term_type,
        "canonical_form": _canonical_form(entry.term),
        "aliases": {_canonical_form(a) for a in entry.aliases if a.strip()},
    }


# -- Canonical index helpers (cached) --------------------------------

_canonical_index: Dict[str, Dict[str, Any]] = {}
_bigram_map: Dict[str, List[Dict[str, Any]]] = {}
_cache_dirty: bool = True


def clear_caches() -> None:
    global _cache_dirty, _canonical_index, _bigram_map
    _cache_dirty = True
    _canonical_index = {}
    _bigram_map = {}


def _build_canonical_index() -> Dict[str, Dict[str, Any]]:
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


def _find_bigram_matches(tokens: List[Tuple[int, str]], index: int) -> bool:
    """Check if token at ``index`` is part of a bigram matching a lexicon entry.
    ``tokens`` is a list of ``(index, text)`` pairs.
    """
    bg_map = _get_bigram_map()
    if index + 1 < len(tokens):
        bg = _canonical_form(f"{tokens[index][1]} {tokens[index+1][1]}")
        if bg in bg_map:
            return True
    if index - 1 >= 0:
        bg = _canonical_form(f"{tokens[index-1][1]} {tokens[index][1]}")
        if bg in bg_map:
            return True
    return False


# -- ModernBERT MLM pipeline (lazy-loaded singleton) -----------------------

_mlm_pipeline: Any = None  # transformers.Pipeline
"""Module-level singleton holding the fill-mask pipeline for ModernBERT-large.

Initialised once by ``_init_mlm_pipeline()`` on first use.
"""


def _init_mlm_pipeline() -> None:
    global _mlm_pipeline
    if _mlm_pipeline is not None:
        return

    import torch
    from transformers import pipeline as hf_pipeline

    model_path = Path(MODERNBERT_MODEL_PATH)
    resolved = model_path.resolve()

    if not resolved.is_dir():
        raise FileNotFoundError(
            f"ModernBERT model not found at {resolved}. "
            f"Expected answerdotai/ModernBERT-large in HF cache."
        )

    print(f"[Stage 1] Loading ModernBERT-large from {resolved}...")

    device = -1  # CPU
    if torch.cuda.is_available():
        device = 0

    _mlm_pipeline = hf_pipeline(
        "fill-mask",
        model=str(resolved),
        tokenizer=str(resolved),
        device=device,
        top_k=MODERNBERT_TOP_K,
    )
    print("[Stage 1] ModernBERT-large loaded successfully")


# -- ModernBERT scoring tracking -------------------------------------------

_last_used_modernbert: bool = False


def last_scoring_used_modernbert() -> bool:
    """Return ``True`` if the last ``score_transcript`` call used ModernBERT."""
    return _last_used_modernbert


def reset_modernbert_flag() -> None:
    global _last_used_modernbert
    _last_used_modernbert = False


# -- ModernBERT masked scorer ----------------------------------------------


def score_word(word: str, sentence: str) -> float:
    """Score a single word using ModernBERT fill-mask.

    Replaces *word* in *sentence* with ``[MASK]`` once, runs the ModernBERT
    fill-mask pipeline, and checks if the original word appears in the top-50
    predictions.

    Returns a suspicion score in [0.0, 1.0] where higher = more likely an error.
    If the original word is not in the top-50 predictions at all, returns 0.95.
    """
    masked = sentence.replace(word, "[MASK]", 1)
    try:
        results = _mlm_pipeline(masked)
    except Exception as exc:
        print(f"[Stage 1] ModernBERT fill-mask failed for '{word}': {exc}")
        return 0.45

    original_lower = word.strip().lower()
    for r in results:
        if r["token_str"].strip().lower() == original_lower:
            # High probability assigned by the model = low suspicion
            return 1.0 - r["score"]

    # Word not in top-50 predictions = very suspicious
    return 0.95


# -- Heuristic scoring (fallback) ------------------------------


def _best_canonical_similarity(token: str) -> float:
    cf = _canonical_form(token)
    if not cf:
        return 0.0
    best = 0.0
    for canon, entry in _get_canonical_index().items():
        s = _char_similarity(cf, canon)
        if s > best:
            best = s
        for alias in entry["aliases"]:
            s = _char_similarity(cf, alias)
            if s > best:
                best = s
    return best


def _score_token(token: str, token_pairs: List[Tuple[int, str]],
                 index: int) -> float:
    """Heuristic suspicion score for a single token."""
    cf = _canonical_form(token)
    if not cf:
        return 0.0

    if _find_bigram_matches(token_pairs, index):
        return 0.85

    sim = _best_canonical_similarity(token)

    # High-frequency words (SUBTLEX-US 70th percentile) are almost certainly
    # spelled correctly — assign zero suspicion.
    if _is_high_frequency(token):
        return 0.0

    if sim >= SIMILARITY_MIN:
        score = 0.60 + (sim - SIMILARITY_MIN) / (1.0 - SIMILARITY_MIN) * 0.35
        return min(0.95, max(0.60, score))

    if len(token) < 4:
        return 0.25

    return 0.45


def _score_transcript_heuristic(transcript: str) -> List[ScoredWord]:
    """Score words using edit-distance + bigram heuristics.

    Uses Stage 0 tokenisation so ``original`` and ``punct`` are populated.
    """
    tokens = tokenize_stage0(transcript)
    if not tokens:
        return []

    # Compute function words via spaCy POS tagging once for the entire transcript.
    function_words = _compute_function_words(transcript)
    token_pairs = [(t.index, t.text) for t in tokens]

    scored: List[ScoredWord] = []
    for token in tokens:
        is_function_word = token.text in function_words
        entry = _lexicon_entry(token.text)

        if is_function_word:
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=0.0, in_lexicon=(entry is not None),
                score_source="zero", has_close_dictionary_match=False,
            ))
            continue

        if entry is not None:
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=0.05, in_lexicon=True,
                score_source="heuristic", has_close_dictionary_match=True,
            ))
            continue

        # Bigram check — if the word is part of a multi-word alias, skip the
        # medical word validity gate so that _score_token can flag it (it
        # returns 0.85 for bigram matches like "dolly prahn" → Doliprane).
        bigram_match = _find_bigram_matches(token_pairs, token.index)

        # Medical word validity check — only for non-bigram words.
        if not bigram_match and _is_valid_medical_word(token.text):
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=0.10, in_lexicon=False,
                score_source="heuristic", has_close_dictionary_match=True,
            ))
            continue

        suspicion = _score_token(token.text, token_pairs, token.index)
        hcd = _has_close_dictionary_match(token.text) if suspicion > 0.50 else False
        scored.append(ScoredWord(
            index=token.index, text=token.text,
            original=token.original, punct=token.punct,
            suspicion=suspicion, in_lexicon=False,
            score_source="heuristic", has_close_dictionary_match=hcd,
        ))

    return scored


# -- Main entry point -----------------------------------------


def score_transcript(transcript: str) -> List[ScoredWord]:
    """Score each word in the transcript.

    Uses ModernBERT-large in fill-mask mode as the primary scorer.
    Falls back to character-level heuristic if ModernBERT is unavailable.
    """
    global _last_used_modernbert

    tokens = tokenize_stage0(transcript)
    if not tokens:
        return []

    # ── Primary: ModernBERT (fill-mask scoring) ─────────────────────────
    try:
        _init_mlm_pipeline()
        mlm_results = _try_modernbert_scorer(transcript)
        if mlm_results is not None:
            _last_used_modernbert = True
            return mlm_results
    except Exception as exc:
        print(f"[Stage 1] ModernBERT scoring failed: {exc}")

    # ── Fallback: Heuristic (edit-distance only) ────────────────────────
    _last_used_modernbert = False
    print("[Stage 1] ModernBERT unavailable, using heuristic fallback")
    return _score_transcript_heuristic(transcript)


def _try_modernbert_scorer(transcript: str) -> Optional[List[ScoredWord]]:
    """Score words using hybrid heuristic + ModernBERT masked language model.

    For each word:
    1. Stop words → 0.0 suspicion (skip MLM).
    2. Canonical lexicon entry → 0.05 suspicion (skip MLM).
    3. Compute heuristic ``_score_token`` suspicion. If < ``MLM_REFINE_THRESHOLD``,
       use the heuristic score directly (skip MLM).
    4. If heuristic suspects the word (>= ``MLM_REFINE_THRESHOLD``), also run
       ModernBERT fill-mask and take **max(heuristic, modernbert)** as the final score.

    Returns ``None`` if ModernBERT cannot be loaded or fails entirely.
    """
    tokens = tokenize_stage0(transcript)
    if not tokens:
        return None

    # Compute function words via spaCy POS tagging once for the entire transcript.
    function_words = _compute_function_words(transcript)
    token_pairs = [(t.index, t.text) for t in tokens]

    scored: List[ScoredWord] = []
    for token in tokens:
        # Function words (determiners, prepositions, conjunctions, etc.) → 0.0 suspicion, skip MLM.
        if token.text in function_words:
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=0.0, in_lexicon=False,
                score_source="zero", has_close_dictionary_match=False,
            ))
            continue

        # Canonical lexicon match → low suspicion, skip MLM.
        entry = _lexicon_entry(token.text)
        if entry is not None:
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=0.05, in_lexicon=True,
                score_source="heuristic", has_close_dictionary_match=True,
            ))
            continue

        # Heuristic pre-filter.
        heuristic_score = _score_token(token.text, token_pairs, token.index)

        # Bigram check — if the word is part of a multi-word alias, it needs
        # MLM refinement even if it passes the spell-check individually
        # (e.g. "dolly" in "dolly prahn" → Doliprane). Bigram matches always
        # score >= 0.85, so they'll always proceed to MLM.
        bigram_match = _find_bigram_matches(token_pairs, token.index)

        # Medical word validity check — only for non-bigram words.
        # Prevents false positives on valid-but-rare medical terms like
        # "nebulization" without blocking bigram-detected misspellings.
        if not bigram_match and _is_valid_medical_word(token.text):
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=0.10, in_lexicon=False,
                score_source="heuristic", has_close_dictionary_match=True,
            ))
            continue

        # If heuristic says not suspicious, skip MLM entirely.
        if heuristic_score < MLM_REFINE_THRESHOLD:
            hcd = _has_close_dictionary_match(token.text) if heuristic_score > 0.50 else False
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=heuristic_score, in_lexicon=False,
                score_source="heuristic", has_close_dictionary_match=hcd,
            ))
            continue

        # Heuristic flagged the word; refine with ModernBERT context.
        # score_word() handles masking internally via sentence.replace(word, "[MASK]", 1).
        # Use token.original (not token.text) to match the casing in the transcript.
        mlm_score = score_word(token.original, transcript)

        # Take the max of heuristic and ModernBERT.
        suspicion = max(heuristic_score, mlm_score)
        suspicion_rounded = round(max(0.0, min(1.0, suspicion)), 6)

        hcd = _has_close_dictionary_match(token.text) if suspicion_rounded > 0.50 else False

        scored.append(ScoredWord(
            index=token.index, text=token.text,
            original=token.original, punct=token.punct,
            suspicion=suspicion_rounded, in_lexicon=False,
            score_source="modernbert", has_close_dictionary_match=hcd,
        ))

    return scored
