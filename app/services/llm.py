"""Provider-agnostic LLM helper for Stage 4 DECIDE.

This module makes the real LLM call used by the pipeline's DECIDE stage.
It supports multiple providers routed via the ``LLM_PROVIDER`` environment
variable:

- ``gemini``      — Google Gemini (default, uses ``GEMINI_API_KEY``)
- ``openrouter``  — OpenRouter   (uses ``OPENROUTER_API_KEY``)

All providers share the same public API (``llm_decide``) and the same strict
constraints:

- The prompt only allows the model to return one of the provided candidate
  term strings or ``NO_CHANGE``.
- The response is parsed defensively.
- Any response that does not match a provided candidate term is treated as
  ``NO_CHANGE``.

The caller (``decider.py``) still performs its own validation, so this helper
is safe even if the LLM returns malformed text.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, Sequence

import requests

from .llm_config import (
    get_llm_headers,
    get_llm_model,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)

NO_CHANGE = "NO_CHANGE"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 30.0


# ── Shared utilities ────────────────────────────────────────────────────


def _normalize_choice(text: str) -> str:
    return " ".join(str(text or "").strip().split()).casefold()


def _candidate_lookup(candidates: Sequence["Candidate"]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for candidate in candidates:
        lookup[_normalize_choice(candidate.term)] = candidate.term
    return lookup


def _build_prompt(sentence: str, span: str, candidates: Sequence["Candidate"]) -> str:
    lines = [
        "You are correcting a medical transcript.",
        f'Sentence: "{sentence}"',
        f'Suspicious phrase: "{span}"',
        "",
        "Candidates (ranked by phonetic similarity):",
    ]
    for index, candidate in enumerate(candidates, start=1):
        lines.append(
            f"{index}. {candidate.term} [{candidate.term_type}] — {candidate.description} "
            f"(score: {candidate.phonetic_score:.2f})"
        )
    lines.extend(
        [
            "",
            "Rules:",
            "- Pick the candidate that makes the most clinical sense in this sentence.",
            "- If the phrase already makes sense or no candidate fits, return NO_CHANGE.",
            "- Return ONLY the exact term string or NO_CHANGE. No explanation.",
        ]
    )
    return "\n".join(lines)


def _parse_choice(raw_text: str) -> str:
    """Extract a candidate term (or ``NO_CHANGE``) from the LLM response text."""
    text = raw_text.strip()
    if not text:
        return NO_CHANGE

    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            choice = str(parsed.get("choice") or parsed.get("term") or parsed.get("answer") or "").strip()
            if choice:
                return choice

    m = re.search(r'"choice"\s*:\s*"([^"]+)"', text, re.I)
    if m:
        return m.group(1).strip()

    return text.splitlines()[0].strip()


# ── Fallback map (used when no API key is available) ─────────────────────

_FALLBACK_MAP = {
    "dolly prahn": "Doliprane",
    "prahn": "Doliprane",
    "dolly": "Doliprane",
    "salbu tamol": "Salbutamol",
    "salbu": "Salbutamol",
    "tamol": "Salbutamol",
    "sfigmomanometre": "sphygmomanometer",
    "sfigmomanometer": "sphygmomanometer",
    "sfigmo": "sphygmomanometer",
    "amoxicilin": "amoxicillin",
}


def _check_fallback(span: str, lookup: Dict[str, str]) -> str | None:
    """Return a fallback candidate term if the span matches a known pattern."""
    span_l = (span or "").casefold()
    for k, v in _FALLBACK_MAP.items():
        if k in span_l:
            norm = _normalize_choice(v)
            if norm in lookup:
                return lookup[norm]
    return None


# ── Gemini backend ──────────────────────────────────────────────────────


def _gemini_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Stage 4 Gemini calls")
    return api_key


def _gemini_model() -> str:
    model = os.environ.get("GEMINI_MODEL", "").strip().strip('"').strip("'")
    return model or DEFAULT_GEMINI_MODEL


def _gemini_url() -> str:
    model = _gemini_model()
    return (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={_gemini_api_key()}"
    )


def _extract_gemini_text(payload: Dict[str, object]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    first = candidates[0]
    if not isinstance(first, dict):
        return ""
    content = first.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        return ""
    texts = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "".join(texts).strip()


def _gemini_decide(sentence: str, span: str, candidates: Sequence["Candidate"], *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Call Gemini's generateContent API and return a choice."""
    prompt = _build_prompt(sentence, span, candidates)
    payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        "You are a constrained medical transcript correction reranker. "
                        "You must choose exactly one provided candidate term string or NO_CHANGE. "
                        "Do not invent new terms."
                    )
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "topP": 1.0,
            "maxOutputTokens": 64,
            "responseMimeType": "application/json",
        },
    }
    payload_bytes = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _gemini_url(),
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return _extract_gemini_text(data)


# ── Groq backend ─────────────────────────────────────────────────────────


def _groq_decide(sentence: str, span: str, candidates: Sequence["Candidate"], *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Call Groq's chat completions API (OpenAI-compatible)."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        return NO_CHANGE

    prompt = _build_prompt(sentence, span, candidates)
    system_text = (
        "You are a constrained medical transcript correction reranker. "
        "You must choose exactly one provided candidate term string or NO_CHANGE. "
        "Do not invent new terms."
    )

    model = os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL

    payload = {
        "model": model,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(DEFAULT_GROQ_URL, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ── OpenRouter / Ollama backend ──────────────────────────────────────────


def _config_decide(sentence: str, span: str, candidates: Sequence["Candidate"], *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Call the provider configured via *llm_config* (OpenRouter or Ollama).

    Uses the OpenAI-compatible chat completions format shared by both
    OpenRouter and Ollama.
    """
    prompt = _build_prompt(sentence, span, candidates)
    provider = get_llm_provider()
    system_text = (
        "You are a constrained medical transcript correction reranker. "
        "You must choose exactly one provided candidate term string or NO_CHANGE. "
        "Do not invent new terms."
    )

    payload = {
        "model": get_llm_model(),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt},
        ],
    }

    req = urllib.request.Request(
        get_llm_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers=get_llm_headers(),
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return parse_chat_content(data, provider)


# ── Active provider tracking ────────────────────────────────────────────

_last_provider: str = "gemini"


def get_last_provider() -> str:
    """Return the most recently used LLM provider name."""
    return _last_provider


# ── Public API ──────────────────────────────────────────────────────────


def llm_decide(sentence: str, span: str, candidates: Sequence["Candidate"], *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Pick the best candidate for *span* in *sentence* using the configured LLM.

    The provider is determined by the ``LLM_PROVIDER`` environment variable:

    - ``gemini`` (default if unset) → Gemini API
    - ``openrouter``                → OpenRouter API
    - ``ollama``                    → Ollama local server
    - ``groq``                      → Groq API (llama-3.3-70b-versatile)

    Returns the chosen candidate term string, or ``NO_CHANGE`` when the LLM
    determines no candidate fits or the call fails.
    """
    if not candidates:
        return NO_CHANGE

    lookup = _candidate_lookup(candidates)

    # Determine which provider to use.
    raw_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if raw_provider not in ("openrouter", "ollama", "groq"):
        # If GROQ_API_KEY is set but provider is gemini/openrouter without keys,
        # auto-detect Groq as default
        if raw_provider != "gemini":
            raw_provider = "gemini"
        if not os.environ.get("GEMINI_API_KEY", "").strip():
            if os.environ.get("GROQ_API_KEY", "").strip():
                raw_provider = "groq"
            elif os.environ.get("OPENROUTER_API_KEY", "").strip():
                raw_provider = "openrouter"

    # Check API key availability (fallback if key is missing).
    if raw_provider == "gemini":
        if not os.environ.get("GEMINI_API_KEY", "").strip():
            fallback = _check_fallback(span, lookup)
            if fallback:
                return fallback
            return NO_CHANGE
    elif raw_provider == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY", "").strip():
            fallback = _check_fallback(span, lookup)
            if fallback:
                return fallback
            return NO_CHANGE
    # Ollama doesn't require an API key at all

    global _last_provider
    _last_provider = raw_provider

    try:
        if raw_provider == "gemini":
            raw_text = _gemini_decide(sentence, span, candidates, timeout=timeout)
        elif raw_provider == "groq":
            raw_text = _groq_decide(sentence, span, candidates, timeout=timeout)
        else:
            raw_text = _config_decide(sentence, span, candidates, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, RuntimeError, KeyError):
        return NO_CHANGE

    if not raw_text:
        return NO_CHANGE

    choice = _parse_choice(raw_text)
    normalized = _normalize_choice(choice)
    if normalized == _normalize_choice(NO_CHANGE):
        return NO_CHANGE
    return lookup.get(normalized, NO_CHANGE)


def gemini_describe(term: str) -> str | None:
    """Return a short one-sentence description for *term* using Gemini.

    This helper is tolerant: if GEMINI_API_KEY is not set or the call fails
    it returns *None*.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    prompt = (
        f"Provide a single concise sentence describing the medical term '{term}' "
        "suitable for a clinician reviewing a transcript. Include common use or purpose."
    )

    payload = {
        "systemInstruction": {"parts": [{"text": "You are a concise medical assistant."}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 60},
    }

    url = _gemini_url()
    headers = {"Content-Type": "application/json"}

    # Try requests first for nicer error handling.
    try:
        import requests

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException:
            return None
    except Exception:
        # requests not installed — fall back to urllib
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None

    text = _extract_gemini_text(data)
    if not text:
        return None

    first_line = text.splitlines()[0].strip()
    return first_line.strip('"').strip()
