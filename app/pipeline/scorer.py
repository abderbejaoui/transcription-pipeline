"""Stage 1: score words for suspicion using BART / Qwen + heuristic fallback.

The primary scorer uses a **hybrid** strategy that combines heuristic
pre-filtering with a local language model (LM) loaded from the local HF
cache (``D:/HF_CACHE/``):

1. **Heuristic pre-filter** — for each word, first compute a heuristic
   suspicion score using edit distance against the medical lexicon, bigram
   matching, and a common-English-word list. Words the heuristic considers
   *not suspicious* (< ``SIMILARITY_MIN``) take the heuristic score directly
   and skip the LM. This prevents false positives on common English words
   ("patient", "fever", "daily") which may have many plausible alternatives
   in context.
2. **LM refinement** — only for words the heuristic flags as potentially
   misspelled, run the contextual LM to refine the suspicion:

   * **Qwen** (``Qwen2.5-1.5B-Instruct``, default): the transcript is
     prepended with a medical-reviewer system prompt, then run through
     a single causal-LM forward pass. Per-token log-probabilities are
     extracted and averaged per word to produce a suspicion score. The
     system prompt conditions the model's hidden states on a medical
     reviewer context, improving misspelling detection.
   * **BART** (``facebook/bart-large``, fallback): build a masked version
     of the sentence where that word is replaced with ``<mask>``, then ask
     BART's fill-mask pipeline to predict the probability of the original
     word. ``suspicion = 1.0 - p(original_word | context)``.
3. The final suspicion is the **maximum** of the heuristic and LM scores,
   so a word must look bad by *both* criteria to score high.

Fallback chain: Qwen → BART → pure heuristic.

The caller (``runner.py``) can check ``last_scoring_used_qwen()`` or
``last_scoring_used_bart()`` to report the approach used.
"""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import threading

from app.services import lexicon

from .models import ScoredWord, Token


# -- BART model path (local HF cache) ------------------------------------

BART_MODEL_PATH = (
    "D:/HF_CACHE/models--facebook--bart-large/snapshots/"
    "cb48c1365bd826bd521f650dc2e0940aee54720c"
)
BART_FILL_MASK_TOP_K = 5
"""Number of top predictions to retrieve from BART per word."""


# -- Qwen model path (local HF cache) -------------------------------------

QWEN_MODEL_PATH = (
    "D:/HF_CACHE/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)

QWEN_SYSTEM_PROMPT = (
    "As a medical transcription reviewer, evaluate each word for correctness. "
)


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


STOP_WORDS: Set[str] = {
    "a", "also", "an", "and", "are", "as", "at", "be", "by", "for", "from", "had",
    "has", "have", "he", "her", "him", "his", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "she", "so", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "to", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your", "should",
}

COMMON_ENGLISH: Set[str] = {
    "patient", "daily", "day", "days", "week", "weeks", "month", "months",
    "year", "years", "take", "takes", "taking", "taken", "took", "dose",
    "doses", "dosage", "mg", "ml", "cc", "hour", "hours", "minute", "minutes",
    "time", "times", "once", "twice", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "every", "per", "oral", "iv", "intravenous",
    "intramuscular", "subcutaneous", "topical", "inhaled", "given",
    "administered", "prescribed", "recommended", "started", "continued",
    "discontinued", "stopped", "increased", "decreased", "adjusted",
    "monitored", "checked", "tested", "showed", "revealed", "indicated",
    "demonstrated", "reported", "complained", "presented", "admitted",
    "discharged", "transferred", "seen", "evaluated", "assessed", "examined",
    "measured", "observed", "noted", "noticed", "developed", "experienced",
    "suffered", "improved", "worsened", "resolved", "fever", "pain", "cough",
    "sputum", "dyspnea", "shortness", "breath", "wheeze", "wheezing",
    "crackles", "rhonchi", "chest", "lung", "lungs", "heart", "cardiac",
    "blood", "pressure", "rate", "rhythm", "pulse", "oxygen", "saturation",
    "temperature", "weight", "height", "bmi", "headache", "nausea", "vomiting",
    "diarrhea", "constipation", "abdomen", "abdominal", "back", "neck",
    "throat", "nose", "ear", "eyes", "skin", "rash", "lesion", "ulcer",
    "wound", "infection", "fracture", "trauma", "surgery", "surgical",
    "procedure", "biopsy", "scan", "xray", "x-ray", "mri", "ct", "ultrasound",
    "ekg", "ecg", "lab", "labs", "test", "tests", "results", "normal",
    "abnormal", "positive", "negative", "elevated", "decreased", "within",
    "without", "history", "past", "family", "social", "allergies",
    "medications", "treatment", "plan", "follow", "followup", "follow-up",
    "next", "return", "clinic", "primary", "care", "emergency", "room",
    "hospital", "ward", "icu", "nursing", "home", "rehabilitation",
    "physical", "therapy", "occupational", "speech", "diet", "nutrition",
    "fluid", "fluids", "electrolytes", "potassium", "sodium", "calcium",
    "magnesium", "phosphorus", "glucose", "sugar", "hemoglobin",
    "hematocrit", "platelet", "platelets", "white", "red", "cell", "cells",
    "wbc", "rbc", "hgb", "hct", "bun", "creatinine", "liver", "kidney",
    "renal", "hepatic", "cardiac", "pulmonary", "neurologic",
    "musculoskeletal", "skin", "soft", "tissue", "bone", "joint", "joints",
    "muscle", "muscles", "numbness", "tingling", "weakness", "fatigue",
    "dizziness", "syncope", "seizure", "seizures", "confusion",
    "disorientation", "lethargy", "somnolence", "coma", "unconscious",
    "unresponsive", "awake", "alert", "oriented", "person", "place",
    "situation", "secondary", "presents", "presenting", "alongside", "using",
    "attending", "attends", "attended", "high", "low", "range", "mild",
    "moderate", "severe", "acute", "chronic", "recurrent",
    # Additional common medical words (manually curated, missing from the original set)
    "diabetes", "inflammation", "hypertension", "continue",
    "needs", "since", "because", "during", "without",
}

SIMILARITY_MIN = 0.55
COMMON_ENGLISH_SIM_CAP = 0.90


# -- Helpers -------------------------------------------------


def _is_stop_word(token: str) -> bool:
    return token.lower() in STOP_WORDS


def _is_common_english(token: str) -> bool:
    return token.lower() in COMMON_ENGLISH


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


# -- BART pipeline (lazy-loaded) -----------------------------------------

_bart_pipeline: Any = None  # transformers.Pipeline
"""Module-level singleton holding the fill-mask pipeline for BART.

Initialised once by ``_init_bart_pipeline()`` on first use.
"""


def _init_bart_pipeline() -> None:
    global _bart_pipeline
    if _bart_pipeline is not None:
        return

    import torch
    from transformers import pipeline as hf_pipeline

    model_path = Path(BART_MODEL_PATH)
    resolved = model_path.resolve()

    if not resolved.is_dir():
        # Try to find the latest snapshot in the HF cache parent directory.
        parent = model_path.parents[2]  # up to models--facebook--bart-large/
        snapshots_dir = parent / "snapshots"
        if snapshots_dir.is_dir():
            snaps = sorted(snapshots_dir.iterdir())
            if snaps:
                resolved = snaps[-1]  # most recent
                print(f"[Stage 1] Primary BART path not found; using fallback: {resolved}")

    if not resolved.is_dir():
        raise FileNotFoundError(
            f"BART model not found at {resolved}. "
            f"Expected facebook/bart-large in HF cache."
        )

    print(f"[Stage 1] Loading BART from {resolved}...")

    device = -1  # CPU
    if torch.cuda.is_available():
        device = 0

    _bart_pipeline = hf_pipeline(
        "fill-mask",
        model=str(resolved),
        tokenizer=str(resolved),
        device=device,
    )
    print("[Stage 1] BART loaded successfully")


# -- BART scoring tracking ------------------------------------------------

_last_used_bart: bool = False


def last_scoring_used_bart() -> bool:
    """Return ``True`` if the last ``score_transcript`` call used BART."""
    return _last_used_bart


def reset_bart_flag() -> None:
    global _last_used_bart
    _last_used_bart = False


# -- BART masked scorer ---------------------------------------------------


def _mask_word_in_transcript(transcript: str, word_index: int,
                             mask_token: str) -> str:
    """Replace the word at ``word_index`` with ``mask_token``."""
    parts = transcript.split()
    if 0 <= word_index < len(parts):
        parts[word_index] = mask_token
    return " ".join(parts)


def _bart_score_word(pipe: Any, masked_sentence: str,
                     original_text: str) -> float:
    """Run BART fill-mask and compute suspicion for *original_text*.

    Returns a suspicion score in [0.0, 1.0] where higher = more likely an error.
    """
    try:
        predictions = pipe(masked_sentence, top_k=BART_FILL_MASK_TOP_K)
    except Exception as exc:
        print(f"[Stage 1] BART fill-mask failed for '{original_text}': {exc}")
        return 0.45

    original_lower = original_text.strip().lower()
    best_score = 0.0

    for pred in predictions:
        pred_str = pred.get("token_str", "")
        pred_text = pred_str.strip().lower()
        # Check exact match or high character similarity (handles
        # BPE-subword differences between BART's tokeniser and our
        # whitespace split — e.g. "amoxicilin" vs "amoxicillin").
        match = (
            original_lower == pred_text
            or (len(original_lower) > 3
                and _char_similarity(original_lower, pred_text) >= 0.80)
        )
        if match:
            score = pred.get("score", 0.0)
            if score > best_score:
                best_score = score

    if best_score > 0.0:
        return 1.0 - best_score

    # BART did not predict the original word at all — likely an error.
    return 0.75


# -- Qwen pipeline (lazy-loaded) -----------------------------------------

_qwen_model: Any = None
"""Module-level singleton holding the Qwen2.5-1.5B-Instruct model."""

_qwen_tokenizer: Any = None
"""Module-level singleton holding the Qwen2.5-1.5B-Instruct tokenizer."""


_last_used_qwen: bool = False

# Thread lock to prevent concurrent Qwen loading (e.g. prewarm vs API request).
_qwen_load_lock = threading.Lock()


def last_scoring_used_qwen() -> bool:
    """Return ``True`` if the last ``score_transcript`` call used Qwen."""
    return _last_used_qwen


def reset_qwen_flag() -> None:
    global _last_used_qwen
    _last_used_qwen = False


def _init_qwen_pipeline() -> None:
    """Load Qwen2.5-1.5B-Instruct from local HF cache.

    Uses FP16 on GPU if available (fits in ~3 GB on a 4 GB card),
    falls back to FP32 on CPU.

    Thread-safe: uses a module-level lock to prevent concurrent loading
    (e.g. prewarm + first API request). Safe to call multiple times —
    the model is loaded once and reused.
    """
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is not None and _qwen_tokenizer is not None:
        return

    if not _qwen_load_lock.acquire(blocking=False):
        # Another thread is already loading; wait for it.
        _qwen_load_lock.acquire(blocking=True)
        _qwen_load_lock.release()
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(QWEN_MODEL_PATH)
    resolved = model_path.resolve()

    if not resolved.is_dir():
        parent = model_path.parents[2]
        snapshots_dir = parent / "snapshots"
        if snapshots_dir.is_dir():
            snaps = sorted(snapshots_dir.iterdir())
            if snaps:
                resolved = snaps[-1]
                print(f"[Stage 1] Primary Qwen path not found; using fallback: {resolved}")

    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Qwen model not found at {resolved}. "
            f"Expected Qwen2.5-1.5B-Instruct in HF cache."
        )

    print(f"[Stage 1] Loading Qwen from {resolved}...")

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # Use torch_dtype + device_map='auto' for reliable GPU placement.
        # Using device_map='auto' lets accelerate/safetensors handle sharding
        # and avoids hanging that can occur with manual .to("cuda:0") in
        # multi-threaded server processes.
        _qwen_tokenizer = AutoTokenizer.from_pretrained(
            str(resolved), cache_dir=None,
        )
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            str(resolved),
            cache_dir=None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        _qwen_model.eval()
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**3
            print(f"[Stage 1] Qwen loaded on GPU. VRAM used: {vram:.1f} GB")
        else:
            print("[Stage 1] Qwen loaded on CPU.")
    finally:
        _qwen_load_lock.release()


def qwen_available() -> bool:
    """Return ``True`` if the Qwen pipeline is loaded and ready."""
    return _qwen_model is not None and _qwen_tokenizer is not None


# -- Qwen log-probability scorer -----------------------------------------


def _qwen_build_token_alignment(tokenizer: Any,
                                input_ids: Any,
                                transcript_token_offset: int = 0) -> List[Tuple[int, int, int]]:
    """Map Qwen token positions back to whitespace word indices.

    *transcript_token_offset* is the number of prefix tokens (system prompt)
    before the actual transcript begins. Words are indexed from 0 starting
    at the transcript portion.

    Returns a list of ``(word_index, start_token_pos, end_token_pos)``
    for each whitespace word found in the transcript portion of the input.

    Relies on Qwen's BPE convention that subword tokens continue within
    a word unless their decoded form starts with a space (``' token'``).
    """
    seq_len = input_ids.shape[1]
    boundaries: List[int] = [transcript_token_offset]

    for pos in range(transcript_token_offset + 1, seq_len):
        tok_str = tokenizer.decode(
            input_ids[0, pos].item(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if tok_str and tok_str[0] == ' ':
            boundaries.append(pos)

    boundaries.append(seq_len)  # sentinel

    aligned: List[Tuple[int, int, int]] = []
    for w_idx in range(len(boundaries) - 1):
        aligned.append((w_idx, boundaries[w_idx], boundaries[w_idx + 1] - 1))

    return aligned


def _qwen_compute_word_log_probs(transcript: str) -> Optional[Dict[int, float]]:
    """Run a single Qwen forward pass and compute avg log-prob per word.

    The transcript is **prepended with a system prompt** (``QWEN_SYSTEM_PROMPT``)
    so the model's hidden states are conditioned on a medical-reviewer context.
    This improves the separation between correct and misspelled words compared
    to raw log-probs on the bare transcript.

    Returns ``{word_index: avg_log_prob, ...}`` or ``None`` on failure.

    Each word's average log-probability is the arithmetic mean of the
    log-probabilities of its constituent subword tokens, where each
    token's log-prob is conditioned on all previous tokens in the sentence.
    """
    import torch

    try:
        _init_qwen_pipeline()
    except Exception as exc:
        print(f"[Stage 1] Failed to initialise Qwen pipeline: {exc}")
        return None

    model = _qwen_model
    tokenizer = _qwen_tokenizer

    # Build system-prompt conditioned input.
    # The prompt prefix activates the model's medical-reviewer knowledge
    # and influences the hidden states of every subsequent token.
    prompt_text = QWEN_SYSTEM_PROMPT
    combined_text = prompt_text + transcript

    inputs = tokenizer(combined_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    # Determine how many tokens the prompt prefix occupies.
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")
    prompt_len = prompt_ids["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    log_probs_all = torch.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]

    # Build alignment, skipping the system-prompt tokens
    alignment = _qwen_build_token_alignment(tokenizer, input_ids,
                                            transcript_token_offset=prompt_len)

    # For each word, extract its token log-probs and average them
    word_log_probs: Dict[int, float] = {}
    for w_idx, start_tok, end_tok in alignment:
        word_lps: List[float] = []
        for tok_pos in range(start_tok, end_tok + 1):
            if tok_pos == 0:
                continue
            target_id = input_ids[0, tok_pos].item()
            lp = log_probs_all[0, tok_pos - 1, target_id].item()
            word_lps.append(lp)

        if word_lps:
            word_log_probs[w_idx] = sum(word_lps) / len(word_lps)
        else:
            word_log_probs[w_idx] = 0.0

    return word_log_probs


def _qwen_logprob_to_suspicion(avg_log_prob: float) -> float:
    """Map an average log-probability to a suspicion score in [0, 1].

    Log-probs near 0 (very predictable) → low suspicion.
    Log-probs below -15 (very surprising) → high suspicion.

    Mapping (linear in the middle):
        -3 or higher  → 0.0
        -15 or lower  → 1.0
        [-3, -15]     → linear from 0.0 to 1.0
    """
    if avg_log_prob >= -3.0:
        return 0.0
    if avg_log_prob <= -15.0:
        return 1.0
    return round((-avg_log_prob - 3.0) / 12.0, 6)


def _try_qwen_scorer(transcript: str) -> Optional[List[ScoredWord]]:
    """Score words using hybrid heuristic + Qwen log-probability.

    Architecture mirrors ``_try_bart_scorer``:
    1. Heuristic pre-filter on all words.
    2. Only run the Qwen forward pass for words the heuristic flags
       as >= ``SIMILARITY_MIN``.
    3. Final score = max(heuristic, qwen_suspicion).

    Returns ``None`` if Qwen cannot be loaded or fails entirely.
    """
    tokens = tokenize_stage0(transcript)
    if not tokens:
        return None

    # Run one forward pass to get per-word log-probs
    word_log_probs = _qwen_compute_word_log_probs(transcript)
    if word_log_probs is None:
        return None

    token_pairs = [(t.index, t.text) for t in tokens]

    scored: List[ScoredWord] = []
    for token in tokens:
        # Stop words → 0.0 suspicion, skip Qwen.
        if _is_stop_word(token.text):
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=0.0, in_lexicon=False,
            ))
            continue

        # Canonical lexicon match → low suspicion, skip Qwen.
        entry = _lexicon_entry(token.text)
        if entry is not None:
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=0.05, in_lexicon=True,
            ))
            continue

        # Heuristic pre-filter.
        heuristic_score = _score_token(token.text, token_pairs, token.index)

        # If heuristic says not suspicious, skip Qwen entirely.
        if heuristic_score < SIMILARITY_MIN:
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=heuristic_score, in_lexicon=False,
            ))
            continue

        # Heuristic flagged the word; refine with Qwen log-prob.
        avg_lp = word_log_probs.get(token.index, -10.0)
        qwen_suspicion = _qwen_logprob_to_suspicion(avg_lp)

        suspicion = max(heuristic_score, qwen_suspicion)

        scored.append(ScoredWord(
            index=token.index, text=token.text,
            original=token.original, punct=token.punct,
            suspicion=round(max(0.0, min(1.0, suspicion)), 6),
            in_lexicon=False,
        ))

    return scored


def _try_bart_scorer(transcript: str) -> Optional[List[ScoredWord]]:
    """Score words using hybrid heuristic + BART masked language model.

    For each word:
    1. Stop words → 0.0 suspicion (skip BART).
    2. Canonical lexicon entry → 0.05 suspicion (skip BART).
    3. Compute heuristic ``_score_token`` suspicion. If < 0.55 (not
       suspicious), use the heuristic score directly (skip BART). This
       prevents BART from generating false positives on common English
       words like "patient", "fever", "daily" that have many plausible
       alternatives in context.
    4. If heuristic suspects the word (>= 0.55), also run BART fill-mask
       and take **max(heuristic, bart)** as the final score.

    Returns ``None`` if BART cannot be loaded or fails entirely.
    """
    tokens = tokenize_stage0(transcript)
    if not tokens:
        return None

    try:
        _init_bart_pipeline()
    except Exception as exc:
        print(f"[Stage 1] Failed to initialise BART pipeline: {exc}")
        return None

    pipe = _bart_pipeline
    mask_tok = pipe.tokenizer.mask_token or "<mask>"
    token_pairs = [(t.index, t.text) for t in tokens]

    scored: List[ScoredWord] = []
    for token in tokens:
        # Stop words → 0.0 suspicion, skip BART.
        if _is_stop_word(token.text):
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=0.0, in_lexicon=False,
            ))
            continue

        # Canonical lexicon match → low suspicion, skip BART.
        entry = _lexicon_entry(token.text)
        if entry is not None:
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=0.05, in_lexicon=True,
            ))
            continue

        # Heuristic pre-filter.
        heuristic_score = _score_token(token.text, token_pairs, token.index)

        # If heuristic says not suspicious, skip BART entirely.
        # This avoids false positives on common English words.
        if heuristic_score < SIMILARITY_MIN:
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=heuristic_score, in_lexicon=False,
            ))
            continue

        # Heuristic flagged the word; refine with BART context.
        masked = _mask_word_in_transcript(transcript, token.index, mask_tok)
        bart_score = _bart_score_word(pipe, masked, token.text)

        # Take the max of heuristic and BART.
        suspicion = max(heuristic_score, bart_score)

        scored.append(ScoredWord(
            index=token.index, text=token.text,
            original=token.original, punct=token.punct,
            suspicion=round(max(0.0, min(1.0, suspicion)), 6),
            in_lexicon=False,
        ))

    return scored


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

    if _is_common_english(token):
        if sim >= COMMON_ENGLISH_SIM_CAP:
            # If the token is an exact alias match (sim == 1.0), it is already
            # stored in the lexicon as a known variant — treat it as low suspicion.
            # Without this check, a common English word stored as an alias (e.g.
            # "mg" stored as an alias of "milligram") would be flagged highly.
            if sim >= 1.0 - 1e-6:
                return 0.05
            return 0.60 + (sim - COMMON_ENGLISH_SIM_CAP) / (1.0 - COMMON_ENGLISH_SIM_CAP) * 0.30
        return 0.05

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

    token_pairs = [(t.index, t.text) for t in tokens]

    scored: List[ScoredWord] = []
    for token in tokens:
        is_stop = _is_stop_word(token.text)
        entry = _lexicon_entry(token.text)

        if is_stop:
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=0.0, in_lexicon=(entry is not None),
            ))
            continue

        if entry is not None:
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=0.05, in_lexicon=True,
            ))
            continue

        suspicion = _score_token(token.text, token_pairs, token.index)
        scored.append(ScoredWord(
            index=token.index, text=token.text,
            original=token.original, punct=token.punct,
            suspicion=suspicion, in_lexicon=False,
        ))

    return scored


# -- Main entry point -----------------------------------------


def score_transcript(transcript: str) -> List[ScoredWord]:
    """Score each word in the transcript.

    Uses the configured scorer model (``config.SCORER_MODEL``) — BART
    (``"bart"``, default) or Qwen2.5-1.5B-Instruct (``"qwen"``) — as
    the primary scorer. Falls back to character-level heuristic if the
    LM scorer is unavailable.
    """
    global _last_used_bart, _last_used_qwen

    tokens = tokenize_stage0(transcript)
    if not tokens:
        return []

    from . import config

    selected_model = getattr(config, "SCORER_MODEL", "qwen")

    # ── Primary: Qwen (system-prompt-conditioned log-prob scoring) ───
    if selected_model == "qwen":
        try:
            qwen_results = _try_qwen_scorer(transcript)
            if qwen_results is not None:
                _last_used_qwen = True
                _last_used_bart = False
                return qwen_results
        except Exception as exc:
            print(f"[Stage 1] Qwen scoring failed: {exc}")

    # ── Fallback 1: BART (fill-mask scoring) ─────────────────────────
    try:
        bart_results = _try_bart_scorer(transcript)
        if bart_results is not None:
            _last_used_bart = True
            _last_used_qwen = False
            return bart_results
    except Exception as exc:
        print(f"[Stage 1] BART scoring failed: {exc}")

    # ── Fallback 2: Heuristic (edit-distance only) ───────────────────
    _last_used_bart = False
    _last_used_qwen = False
    print("[Stage 1] Both LM scorers unavailable, using heuristic fallback")
    return _score_transcript_heuristic(transcript)
