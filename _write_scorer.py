"""Write the new scorer.py content with \r\n line endings."""
import pathlib

content = '''"""Stage 1: score words for suspicion using Groq API + heuristic fallback.

The primary scorer now calls the Groq API (llama-3.3-70b-versatile) with the
full transcript in one shot -- no per-word calls, no local model loading.

Pipeline
--------
1. Tokenize the transcript via ``str.split()`` (Stage 0) producing ``Token``
   objects with ``index``, ``text`` (lowercased), ``original``, ``punct``.
2. Send the full transcript to Groq in a single API call.
3. Map returned scores back to tokens; words not in the response get 0.02.
4. If Groq is unavailable, fall back to character-level edit-distance
   heuristics (the original approach).

The caller (``runner.py``) can check ``last_scoring_used_groq()`` to
report the approach used.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

from app.services import lexicon

from .models import ScoredWord, Token


# -- Stage 0: Tokenisation (str.split, not regex) -------------------------

PUNCT_SET = frozenset(".,;:!?\\"'()[]{}<>-")


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

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*|\\d+(?:\\.\\d+)?")


def tokenize_transcript(transcript: str) -> List[Tuple[str, int, int]]:
    """Legacy regex-based tokeniser returning ``(text, start, end)`` tuples.
    Kept for backward compatibility with ``runner._span_char_offsets``.
    """
    return [(match.group(), match.start(), match.end())
            for match in WORD_RE.finditer(transcript)]


STOP_WORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "had",
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


# -- Groq vs heuristic tracking ------------------------------------

_last_used_groq: bool = False


def last_scoring_used_groq() -> bool:
    """Return ``True`` if the last ``score_transcript`` call used Groq."""
    return _last_used_groq


def reset_groq_flag() -> None:
    global _last_used_groq
    _last_used_groq = False


# -- Groq API call -------------------------------------------

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


def _groq_prompt(transcript: str) -> str:
    return (
        "You are analyzing a medical transcript for speech-to-text errors.\\n\\n"
        f'Transcript: "{transcript}"\\n\\n'
        "For each word, assign a suspicion score from 0.0 to 1.0:\\n"
        "- 1.0 = almost certainly a speech-to-text error (not real English, "
        "not a real medical term, sounds like a mishearing)\\n"
        "- 0.0 = definitely correct (common English word, fits context perfectly)\\n"
        "- Skip stop words entirely (the, and, a, an, is, for, with, of, "
        "to, in, by, at, on, using, was, were, should)\\n\\n"
        "Return ONLY a JSON array, no explanation, no markdown:\\n"
        "[\\n"
        '  {"word": "dolly", "index": 8, "suspicion": 0.87},\\n'
        '  {"word": "prahn", "index": 9, "suspicion": 0.92}\\n'
        "]\\n"
        "Only include content words with suspicion > 0.05.\\n"
        "Stop words get score 0.0 and should be omitted entirely."
    )


def _parse_groq_response(text: str) -> List[Dict[str, Any]]:
    """Extract a JSON array from Groq's response text (handles fences)."""
    raw = text.strip()
    for fence in ("```json", "```"):
        if fence in raw:
            parts = raw.split(fence)
            for p in parts[1:]:
                p = p.strip()
                if p.startswith("["):
                    raw = p
                    break

    m = re.search(r"\\[.*\\]", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    return []


def _try_groq_scorer(transcript: str) -> Optional[List[Dict[str, Any]]]:
    """Call Groq API to score the transcript.

    Returns a list of ``{"word": ..., "index": ..., "suspicion": ...}``
    or ``None`` if the API is unavailable.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        print("[Stage 1] No GROQ_API_KEY set, skipping Groq scorer")
        return None

    prompt = _groq_prompt(transcript)
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        req = urllib.request.Request(
            GROQ_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        response_text = data["choices"][0]["message"]["content"].strip()
        items = _parse_groq_response(response_text)

        out: List[Dict[str, Any]] = []
        for item in items:
            word = str(item.get("word", "")).strip().lower()
            idx = item.get("index")
            if not word or idx is None:
                continue
            suspicion = max(0.0, min(1.0, float(item.get("suspicion", 0.5))))
            out.append({"word": word, "index": int(idx), "suspicion": suspicion})
        return out
    except Exception as exc:
        print(f"[Stage 1] Groq API call failed: {exc}")
        return None


# -- Merge Groq results into ScoredWord list -----------------------

def _merge_groq_scores(tokens: List[Token],
                       groq_results: List[Dict[str, Any]]) -> List[ScoredWord]:
    """Merge Groq-flagged words into ScoredWord objects.

    * Words flagged by Groq get the Groq suspicion score.
    * Canonical lexicon matches get low suspicion (lexicon overrides).
    * All other words get a low default score (0.02).
    """
    groq_scores: Dict[int, float] = {}
    for item in groq_results:
        idx = item.get("index")
        score = max(0.0, min(1.0, float(item.get("suspicion", 0.5))))
        if idx is not None:
            groq_scores[int(idx)] = score

    scored: List[ScoredWord] = []
    for token in tokens:
        entry = _lexicon_entry(token.text)

        if entry is not None:
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=0.05, in_lexicon=True,
            ))
            continue

        groq_score = groq_scores.get(token.index)
        if groq_score is not None and groq_score >= 0.3:
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=groq_score, in_lexicon=False,
            ))
        else:
            scored.append(ScoredWord(
                index=token.index, text=token.text,
                original=token.original, punct=token.punct,
                suspicion=0.02, in_lexicon=False,
            ))

    return scored


# -- Heuristic scoring (fallback) -------------------------------

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

    Uses Groq API as primary scorer; falls back to heuristic if Groq
    is unavailable or returns an error.
    """
    global _last_used_groq

    tokens = tokenize_stage0(transcript)
    if not tokens:
        return []

    try:
        groq_results = _try_groq_scorer(transcript)
        if groq_results is not None:
            _last_used_groq = True
            return _merge_groq_scores(tokens, groq_results)
    except Exception as exc:
        print(f"[Stage 1] Groq scoring failed: {exc}")

    _last_used_groq = False
    print("[Stage 1] Groq unavailable, using heuristic fallback")
    return _score_transcript_heuristic(transcript)
'''

target = pathlib.Path("app/pipeline/scorer.py")
target.write_text(content.replace("\\n", "\n").replace("\\\"", "\""), encoding="utf-8")
print("Written", len(content), "chars to", target)
