"""LLM-based correction with deterministic JSON output."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Any, Dict, Optional

from .llm_config import (
    build_chat_payload,
    describe_http_error,
    get_llm_headers,
    get_llm_model,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)
from .llm_runtime import get_model_id


DEFAULT_TIMEOUT = float(os.environ.get("LLM_CORRECT_TIMEOUT", "90"))


def _llm_provider() -> str:
    return get_llm_provider()


def _llm_url() -> str:
    return get_llm_url(_llm_provider())


def _llm_model(kind: str) -> str:
    return get_model_id(kind) or get_llm_model(_llm_provider())


_GENERAL_SYSTEM = (
    "You are a transcript correction assistant focused on general language. "
    "You receive a sentence and a low-confidence word or phrase from ASR. "
    "Return a JSON object with a corrected replacement if you are confident. "
    "If you are not confident, return an empty replacement and confidence 0. "
    "Never invent medical terms. Output strict JSON only."
)

_MEDICAL_SYSTEM = (
    "You are a medical transcript correction assistant. "
    "You receive a sentence and a low-confidence word or phrase from ASR. "
    "Treat the span as a phonetic ASR misrecognition first: prefer the closest-sounding correction that fits the sentence. "
    "Only propose a correction if it is clearly a medical entity (drug, symptom, diagnosis, procedure, or disease). "
    "In most cases the ambiguous span will be the name of a drug, symptom, or disease. "
    "If you are not confident, return an empty replacement and confidence 0. "
    "Output strict JSON only."
)


def _build_user(span: str, sentence: str) -> str:
    return json.dumps(
        {
            "sentence": sentence,
            "span": span,
            "output_schema": {
                "type": "object",
                "required": ["replacement", "confidence"],
                "properties": {
                    "replacement": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
            },
        },
        ensure_ascii=False,
    )


def _post(system: str, user: str, model: str, timeout: float) -> str:
    payload = build_chat_payload(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        json_mode=True,
        temperature=0.0,
    )
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
            detail = describe_http_error(exc)
            print(f"[llm_correct] LLM call failed (attempt {attempt+1}/4): {detail}; retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"llm_correct: all retries failed: {last_exc!r}")


def _parse(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if not (text.startswith("{") and text.endswith("}")):
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {"replacement": "", "confidence": 0.0, "reason": "parse_error"}

    replacement = str(obj.get("replacement") or "").strip()
    confidence = obj.get("confidence")
    try:
        confidence_val = float(confidence)
    except (TypeError, ValueError):
        confidence_val = 0.0
    reason = str(obj.get("reason") or "").strip()
    if replacement.upper() == "NO_CHANGE":
        replacement = ""
    return {
        "replacement": replacement,
        "confidence": max(0.0, min(1.0, confidence_val)),
        "reason": reason,
    }


def _correct(span: str, sentence: str, system: str, kind: str, timeout: float) -> Dict[str, Any]:
    if not span.strip() or not sentence.strip():
        return {"replacement": "", "confidence": 0.0, "reason": "empty"}
    user = _build_user(span, sentence)
    raw = _post(system, user, _llm_model(kind), timeout)
    return _parse(raw)


def correct_general(
    span: str,
    sentence: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    return _correct(span, sentence, _GENERAL_SYSTEM, "general", timeout)


def correct_medical(
    span: str,
    sentence: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    return _correct(span, sentence, _MEDICAL_SYSTEM, "medical", timeout)
