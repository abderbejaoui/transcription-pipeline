"""Suspicious-word detector.

For each word coming out of Whisper we decide whether it is worth running
through the audio + LLM correction pipeline. We deliberately do NOT call
the LLM here. Three signals:

  1. Whisper word confidence is low.
  2. The word is not a common English word AND not already a known medical
     term (term + alias text matches across the lexicon).
  3. The word has high fuzzy/phonetic similarity to a medical term but is
     not exactly that term — i.e. it looks like an ASR mishearing of
     something medical.

Public API
----------
detect(words, lexicon_terms) -> List[dict]
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import jellyfish
from rapidfuzz import fuzz


# Tiny English word list — common function/glue/everyday words. We only need
# to cover what doctors actually say in dictation around the medical terms.
COMMON_ENGLISH: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "of", "on", "in", "at", "to",
    "for", "with", "without", "from", "by", "as", "is", "are", "was", "were",
    "be", "been", "being", "am", "do", "does", "did", "doing", "have", "has",
    "had", "having", "will", "would", "should", "could", "can", "may", "might",
    "must", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "their", "our", "this", "that", "these",
    "those", "no", "not", "yes", "ok", "okay",

    # everyday verbs/nouns common in dictation
    "patient", "patients", "doctor", "doctors", "nurse", "nurses",
    "clinic", "hospital", "discharge", "admission", "morning", "evening",
    "night", "today", "yesterday", "tomorrow", "now", "later", "next",
    "last", "year", "years", "month", "months", "week", "weeks", "day",
    "days", "hour", "hours", "minute", "minutes",
    "take", "takes", "taking", "took", "taken", "give", "gives", "given",
    "giving", "gave", "start", "started", "starting", "stop", "stopped",
    "stopping", "stops", "continue", "continues", "continuing",
    "prescribe", "prescribed", "prescribes", "prescribing", "use", "used",
    "uses", "using", "need", "needs", "needed", "see", "saw", "seen",
    "show", "showed", "shown", "feel", "feels", "felt", "say", "said",
    "says", "talk", "talked", "talks", "go", "goes", "going", "went",
    "come", "came", "coming", "comes", "want", "wanted", "wants",

    # frequency / dose words
    "twice", "once", "thrice", "daily", "weekly", "monthly", "every", "per",
    "mg", "mcg", "ml", "cc", "g", "kg", "lb", "lbs", "tablet", "tablets",
    "capsule", "capsules", "pill", "pills", "dose", "doses", "drop", "drops",

    # generic descriptive
    "good", "bad", "well", "very", "much", "more", "less", "many", "few",
    "some", "any", "all", "every", "each", "other", "another", "same",
    "different", "new", "old", "high", "low", "big", "small", "long",
    "short", "early", "late", "fast", "slow",

    # body / symptom basics that are not specifically medical-suspicious
    "pain", "fever", "cough", "cold", "head", "back", "chest", "leg",
    "arm", "stomach", "throat", "heart", "lung", "lungs", "blood",
    "skin", "ear", "ears", "eye", "eyes", "nose", "mouth", "tooth",
    "teeth", "hand", "hands", "foot", "feet", "knee", "knees",
    "shoulder", "shoulders", "hip", "hips",

    # connectors
    "because", "since", "while", "until", "after", "before", "during",
    "however", "also", "too", "just", "still", "already", "only", "even",
    "really", "quite",

    "hello", "hi", "thanks", "thank", "please", "welcome", "sir", "madam",
    "mr", "mrs", "ms", "dr",
}


_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?")


def _norm(s: str) -> str:
    return s.lower().strip()


def _word_only(s: str) -> bool:
    return bool(s) and s[0].isalpha()


def detect(
    words: Sequence[Dict[str, Any]],
    lexicon_terms: Sequence[str],
    *,
    confidence_threshold: float = 0.6,
    high_confidence_threshold: float = 0.85,
    fuzzy_floor: int = 78,
    phonetic_floor: int = 82,
    min_chars: int = 4,
) -> List[Dict[str, Any]]:
    """Decide which words are worth correcting.

    Args
    ----
    words: list of dicts with keys {word, start, end, probability}
        — output of faster-whisper word_timestamps=True.
    lexicon_terms: every canonical term + alias from the medical lexicon,
        plus the canonical terms of any voice-index entries.

    Returns
    -------
    A list of dicts (one per suspicious word) with:
        index    -> position in `words`
        text     -> raw word string
        start, end, probability   -> from input
        reason   -> "low_confidence" | "oov" | "near_medical"
    """
    # Pre-process lexicon for fuzzy/phonetic comparisons.
    known_norms: Set[str] = set()
    metaphones: Set[str] = set()
    for term in lexicon_terms:
        for tok in _TOKEN_PATTERN.findall(term):
            n = _norm(tok)
            if n:
                known_norms.add(n)
                meta = jellyfish.metaphone(n)
                if meta:
                    metaphones.add(meta)

    def _is_known(word_norm: str) -> bool:
        if word_norm in COMMON_ENGLISH:
            return True
        if word_norm in known_norms:
            return True
        return False

    out: List[Dict[str, Any]] = []
    for i, w in enumerate(words):
        text = (w.get("word") or "").strip()
        if not text or not _word_only(text):
            continue
        # ASR token text often has leading whitespace/punctuation; clean it.
        clean = re.sub(r"^[\s\W_]+|[\s\W_]+$", "", text)
        if not clean or len(clean) < min_chars:
            continue
        norm = _norm(clean)

        prob = w.get("probability")
        prob_val = float(prob) if isinstance(prob, (int, float)) else 1.0
# Common English / known medical / number-only words: never flag.
        if _is_known(norm):
            continue

        # Score similarity to medical terms once.
        best_fuzzy = 0
        best_phon = 0
        meta = jellyfish.metaphone(norm) or ""
        for kn in known_norms:
            f = fuzz.ratio(norm, kn)
            if f > best_fuzzy:
                best_fuzzy = f
            if meta:
                km = jellyfish.metaphone(kn) or ""
                if km:
                    p = fuzz.ratio(meta, km)
                    if p > best_phon:
                        best_phon = p

        looks_medical = best_fuzzy >= fuzzy_floor or best_phon >= phonetic_floor

        reason: Optional[str] = None

        if prob_val >= high_confidence_threshold:
            # Whisper is confident. Only flag if the word ALSO has strong
            # similarity to a medical term — otherwise it's a normal
            # English word we don't know.
            if looks_medical:
                reason = "near_medical"
        elif prob_val < confidence_threshold:
            # Low confidence: flag whether or not it looks medical.
            reason = "low_confidence"
        else:
            # Mid confidence: only flag if it looks medical.
            if looks_medical:
                reason = "near_medical"

        if reason is None:
            continue

        out.append(
            {
                "index": i,
                "text": clean,
                "start": float(w.get("start") or 0.0),
                "end": float(w.get("end") or 0.0),
                "probability": prob_val,
                "reason": reason,
            }
        )
    return out


def merge_adjacent(
    suspicious: Sequence[Dict[str, Any]],
    *,
    max_gap_s: float = 0.10,
) -> List[Dict[str, Any]]:
    """Merge consecutive suspicious words whose timestamps are within
    `max_gap_s` of each other. Catches Whisper splitting one spoken word
    into multiple tokens (e.g. "Selim o alakal").
    """
    if not suspicious:
        return []
    merged: List[Dict[str, Any]] = []
    cur = dict(suspicious[0])
    cur["indices"] = [cur["index"]]
    cur["text_parts"] = [cur["text"]]
    for s in suspicious[1:]:
        gap = s["start"] - cur["end"]
        if gap <= max_gap_s and s["index"] == cur["indices"][-1] + 1:
            cur["end"] = s["end"]
            cur["indices"].append(s["index"])
            cur["text_parts"].append(s["text"])
            cur["probability"] = min(cur["probability"], s["probability"])
            cur["reason"] = cur["reason"]  # keep first reason
        else:
            cur["text"] = " ".join(cur["text_parts"])
            cur["index_first"] = cur["indices"][0]
            cur["index_last"] = cur["indices"][-1]
            merged.append(cur)
            cur = dict(s)
            cur["indices"] = [cur["index"]]
            cur["text_parts"] = [cur["text"]]
    cur["text"] = " ".join(cur["text_parts"])
    cur["index_first"] = cur["indices"][0]
    cur["index_last"] = cur["indices"][-1]
    merged.append(cur)
    return merged
