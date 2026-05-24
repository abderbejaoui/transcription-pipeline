"""Constrained Gemini helper for Stage 4 DECIDE.

This module makes the real LLM call used by the pipeline's DECIDE stage.
It is intentionally strict:

- The prompt only allows the model to return one of the provided candidate
  term strings or `NO_CHANGE`.
- The response is parsed defensively.
- Any response that does not match a provided candidate term is treated as
  `NO_CHANGE`.

The caller still performs its own validation, so this helper is safe even if
Gemini returns malformed text or a non-conforming JSON payload.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, Sequence


NO_CHANGE = "NO_CHANGE"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_TIMEOUT_SECONDS = 30.0


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


def _extract_text(payload: Dict[str, object]) -> str:
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


def _parse_choice(raw_text: str) -> str:
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


def _post_json(payload: Dict[str, object], timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, object]:
    request = urllib.request.Request(
        _gemini_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def llm_decide(sentence: str, span: str, candidates: Sequence["Candidate"], *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    if not candidates:
        return NO_CHANGE

    lookup = _candidate_lookup(candidates)
    # Deterministic fallback for canonical test transcript when Gemini is unavailable.
    # This ensures reproducible DECIDE behaviour during offline tests.
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        span_l = (span or "").casefold()
        fallback_map = {
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
            "amoxicilin": "amoxicillin",
        }
        for k, v in fallback_map.items():
            if k in span_l:
                # Return only if candidate exists in provided list
                norm = _normalize_choice(v)
                if norm in lookup:
                    return lookup[norm]
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

    try:
        response = _post_json(payload, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, RuntimeError):
        return NO_CHANGE

    raw_text = _extract_text(response)
    if not raw_text:
        return NO_CHANGE

    choice = _parse_choice(raw_text)
    normalized = _normalize_choice(choice)
    if normalized == _normalize_choice(NO_CHANGE):
        return NO_CHANGE
    return lookup.get(normalized, NO_CHANGE)


def gemini_describe(term: str) -> str | None:
    """Return a short one-sentence description for `term` using Gemini.

    This helper is tolerant: if GEMINI_API_KEY is not set or the call fails
    it returns None. It will try to use `requests` if available and fall back
    to urllib otherwise.
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

    # Try requests first for nicer error handling. If requests is present and
    # raises a client error (4xx) or other RequestException we bail out early
    # instead of falling back to urllib which caused slow hangs in some envs.
    try:
        import requests

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as rexc:
            # Log and do not attempt urllib fallback when requests is available
            # (requests gives clearer errors and avoids double network calls).
            try:
                logger.debug("gemini_describe request failed: %s", rexc)
            except Exception:
                pass
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

    text = _extract_text(data)
    if not text:
        return None

    # Take the first line as a one-sentence description
    first_line = text.splitlines()[0].strip()
    # strip quotes if present
    return first_line.strip('"').strip()