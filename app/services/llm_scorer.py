"""LLM-based suspicion scorer for Arabic medical transcripts.

Scores ALL words in a transcript in a SINGLE API call so we respect the
~40 req/min rate limit.  The LLM is asked to return indices of words
that could be ASR mishearings of medical terms (transliterations of
English drug/disease names, misspelled English medical words).

When the API call succeeds, the result is cached per-transcript and used
as a high-weight signal in Stage A suspicion scoring.  When it fails
(timeout / rate-limit / network), the caller falls back to the existing
algorithmic signals.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

from .llm_config import (
    get_llm_headers,
    get_llm_model,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)

# ---------------------------------------------------------------------------
# Cache: {transcript_md5 -> {word_index: suspicion_float}}
# ---------------------------------------------------------------------------
_LLM_CACHE: Dict[str, Dict[int, float]] = {}
_LLM_CACHE_TTL: float = 300.0  # 5 minutes
_LLM_CACHE_TIME: Dict[str, float] = {}


def _cache_key(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _get_cached(key: str) -> Optional[Dict[int, float]]:
    now = time.time()
    entry_time = _LLM_CACHE_TIME.get(key)
    if entry_time is None:
        return None
    if now - entry_time > _LLM_CACHE_TTL:
        _LLM_CACHE.pop(key, None)
        _LLM_CACHE_TIME.pop(key, None)
        return None
    return _LLM_CACHE.get(key)


def _set_cache(key: str, scores: Dict[int, float]) -> None:
    _LLM_CACHE[key] = scores
    _LLM_CACHE_TIME[key] = time.time()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a medical ASR auditor specializing in Gulf Arabic clinical transcripts with code-switched English. Your job: identify words that could be ASR mishearings of medical terms.

Rules:
1. Return STRICT JSON only — no markdown, no explanation outside the JSON. The response MUST parse with json.loads().
2. The "words" array below has the transcript split into tokens. Use 0-based indices.
3. In the "suspicious" array, include ONLY words that are:
   (a) Arabic-script words that are TRANSLITERATIONS of English medical terms
       (e.g., هستوري→history, دايابيتس→diabetes, هايبرتنشن→hypertension,
        شورتنس→shortness, بريث→breath, دزي→dizzy, شيفر→shiver,
        ناوسيا→nausea, انتيبلاتلت→antiplatelet, كارداك→cardiac,
        انزايمز→enzymes, نيتروغلسرين→nitroglycerin, اسبرين→aspirin,
        بلاد شوجر→blood sugar (two words), ساتوريشن→saturation,
        تمبرتشر→temperature, فيتل→vital, هارت→heart, بلد برشر→blood pressure,
        ريزلتس→results)
   (b) English words that are clearly misspelled medical terms
       (e.g., "bain"→pain, "wheezeng"→wheezing, "creptations"→crepitations,
        "possble"→possible, "chenges"→changes, "troponen"→troponin,
        "hyperglacymia"→hyperglycemia, "breth"→breath)
4. For each suspicious word, include a brief "reason" explaining what medical term it likely represents.
5. DO NOT flag:
   - Normal Arabic words (greetings, prepositions, verbs, pronouns, numbers)
   - Arabic anatomy words (قلب, رأس, صدر, معدة, كبد, etc.)
   - Words that are already correct English medical terms
   - Punctuation, digits, measurements
   - Common Arabic clinical context words (مريض, فحص, تحليل, تاريخ, etc.)
   - Arabic filler words (في, من, على, مع, و, etc.)

Output format:
{"suspicious": [{"index": <int>, "reason": "<brief reason>"}, ...], "non_suspicious_count": <int>}"""


def _build_user_prompt(words: List[str]) -> str:
    """Build the user message with the pre-tokenized word list."""
    return json.dumps(
        {"task": "Identify words that could be ASR mishearings of medical terms.",
         "words": words,
         "total_words": len(words)},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_llm(
    prompt: str,
    model: Optional[str] = None,
    timeout: float = 20.0,
) -> Optional[str]:
    """Call the LLM and return the raw response text, or None on failure."""
    provider = get_llm_provider()
    payload = {
        "model": model or get_llm_model(provider),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": min(4000, max(500, 500 + len(prompt) // 2)),
        "temperature": 0.0,
        "stream": False,
    }

    try:
        req = urllib.request.Request(
            get_llm_url(provider),
            data=json.dumps(payload).encode("utf-8"),
            headers=get_llm_headers(provider),
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return parse_chat_content(data, provider)
    except Exception as exc:
        print(f"[llm_scorer] API call failed: {exc!r}")
        return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(raw: str) -> List[Dict[str, object]]:
    """Parse the LLM response into a list of {index, reason} dicts."""
    m = _JSON_RE.search(raw)
    if not m:
        print(f"[llm_scorer] No JSON found in response: {raw[:200]}")
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        print(f"[llm_scorer] JSON parse error: {exc}")
        return []
    suspicious = obj.get("suspicious") or []
    if not isinstance(suspicious, list):
        return []
    return [
        {"index": int(entry["index"]), "reason": str(entry.get("reason", ""))}
        for entry in suspicious
        if isinstance(entry, dict) and "index" in entry
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_words(
    words: List[str],
    *,
    transcript: Optional[str] = None,
    timeout: float = 20.0,
    model: Optional[str] = None,
    skip_cache: bool = False,
) -> Optional[Dict[int, float]]:
    """Score ALL words in the transcript in a single LLM call.

    Returns a dict mapping 0-based word index → suspicion score (0.0 or 1.0),
    or None if the API call fails (caller should fall back to algorithmic).

    The result is cached by transcript text for 5 minutes.
    """
    if not words:
        return {}

    # Check cache
    key = _cache_key(" ".join(words))
    if not skip_cache:
        cached = _get_cached(key)
        if cached is not None:
            return cached

    # Build prompt
    prompt = _build_user_prompt(words)

    # Call LLM
    raw = _call_llm(prompt, model=model, timeout=timeout)
    if raw is None:
        return None

    # Parse response
    entries = _parse_response(raw)
    if not entries:
        # LLM returned empty list — all words are clean
        result: Dict[int, float] = {}
        _set_cache(key, result)
        return result

    # Build result dict: suspicious words get score 1.0
    result = {}
    for entry in entries:
        idx = entry["index"]
        if 0 <= idx < len(words):
            result[idx] = 1.0

    _set_cache(key, result)
    return result


def clear_cache() -> None:
    """Clear the LLM suspicion cache."""
    _LLM_CACHE.clear()
    _LLM_CACHE_TIME.clear()
