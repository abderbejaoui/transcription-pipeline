"""LLM-based coherence verification between raw and corrected transcripts."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Any, Dict, Optional

from .llm_config import (
    get_llm_headers,
    get_llm_model,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)
from .llm_runtime import get_model_id


DEFAULT_TIMEOUT = float(os.environ.get("LLM_VERIFY_TIMEOUT", "90"))


def _llm_provider() -> str:
    return get_llm_provider()


def _llm_url() -> str:
    return get_llm_url(_llm_provider())


def _llm_model() -> str:
    model_id = get_model_id("verify")
    return model_id or get_llm_model(_llm_provider())


_SYSTEM = (
    "You are a clinical coherence checker. "
    "Compare the raw transcript and the corrected transcript. "
    "Return JSON with a confidence score in [0,1] indicating whether the "
    "corrections are clinically plausible and preserve meaning. "
    "If uncertain, return a low confidence. Output strict JSON only."
)


def _build_user(raw_text: str, corrected_text: str) -> str:
    return json.dumps(
        {
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "output_schema": {
                "type": "object",
                "required": ["confidence"],
                "properties": {
                    "confidence": {"type": "number"},
                    "issues": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        ensure_ascii=False,
    )


def _post(user: str, timeout: float) -> str:
    payload = {
        "model": _llm_model(),
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
    }
    last_exc: Optional[BaseException] = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                _llm_url(),
                data=json.dumps(payload).encode("utf-8"),
                headers=get_llm_headers(_llm_provider()),
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return parse_chat_content(data, _llm_provider())
        except Exception as exc:
            last_exc = exc
            wait = 1.0 * (2 ** attempt)
            print(f"[llm_verify] LLM call failed (attempt {attempt+1}/4): {exc!r}; retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"llm_verify: all retries failed: {last_exc!r}")


def _parse(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if not (text.startswith("{") and text.endswith("}")):
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {"confidence": 0.0, "issues": ["parse_error"]}
    try:
        confidence = float(obj.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    issues = obj.get("issues")
    if not isinstance(issues, list):
        issues = []
    issues = [str(i) for i in issues if str(i).strip()]
    return {"confidence": max(0.0, min(1.0, confidence)), "issues": issues}


def verify(
    raw_text: str,
    corrected_text: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    if not raw_text.strip() or not corrected_text.strip():
        return {"confidence": 1.0, "issues": []}
    user = _build_user(raw_text, corrected_text)
    raw = _post(user, timeout)
    return _parse(raw)
