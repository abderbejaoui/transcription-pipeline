"""LLM-driven medical correction picker.

Given the full transcript and, per suspicious span, a small list of
candidates (each with a sound-similarity score and a short description),
pick exactly ONE candidate per span — or NO_CHANGE — based on whether the
candidate's meaning fits the patient's clinical context.

Public API
----------
decide(transcript, items) -> list[{id, choice, confidence}]
    items = [{
        id: str,
        span: str,
        candidates: [
            {"term": str, "similarity": float, "description": str|None},
            ...
        ],
    }]
    choice = a candidate term, or None when NO_CHANGE.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from .llm_config import (
    build_chat_payload,
    get_llm_headers,
    get_llm_model,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)


def _llm_provider() -> str:
    return get_llm_provider()


def _llm_url() -> str:
    return get_llm_url(_llm_provider())


def _llm_model() -> str:
    return get_llm_model(_llm_provider())


NO_CHANGE = "NO_CHANGE"


_SYSTEM = (
    "You are a constrained medical transcript correction reranker. "
    "You receive the transcript and, for each suspicious span, a list of "
    "candidate medical terms with their sound similarity to the audio and "
    "a short description. "
    "Your job: for each span pick the ONE candidate whose description fits "
    "the patient's clinical context in the transcript, or return "
    "\"NO_CHANGE\" if none clearly fits. "
    "Strict rules: "
    "1. Output strict JSON only. No prose. "
    "2. Default to \"NO_CHANGE\" when uncertain. "
    "3. Prefer candidates whose description matches the patient's apparent "
    "diagnosis or symptoms (e.g. metformin for diabetes, ceftriaxone for "
    "pneumonia). In most cases the ambiguous span is the name of a drug, "
    "symptom, or disease, so use that as the default medical bias. "
    "4. Sound similarity matters but is NOT decisive — meaning matters "
    "more. A high-similarity candidate that does not fit medically should "
    "still be \"NO_CHANGE\". "
    "5. Never return a term that was not in the candidate list."
)


def _build_user(transcript: str, items: Sequence[Dict[str, Any]]) -> str:
    return json.dumps(
        {
            "transcript": transcript,
            "spans": [
                {
                    "id": item["id"],
                    "span": item["span"],
                    "candidates": [
                        {
                            "term": c["term"],
                            "similarity": round(float(c.get("similarity", 0.0)), 4),
                            "description": (c.get("description") or "")[:300],
                        }
                        for c in item["candidates"]
                    ]
                    + [{"term": NO_CHANGE, "similarity": 0.0, "description": "leave the span unchanged"}],
                }
                for item in items
            ],
            "output_schema": {
                "type": "object",
                "required": ["choices"],
                "properties": {
                    "choices": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "choice"],
                            "properties": {
                                "id": {"type": "string"},
                                "choice": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                        },
                    }
                },
            },
        },
        ensure_ascii=False,
    )


def _post(content: str, timeout: float) -> str:
    payload = build_chat_payload(
        _llm_model(),
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": content},
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
            print(f"[llm_decide] LLM call failed (attempt {attempt+1}/4): {exc!r}; retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"llm_decide: all retries failed: {last_exc!r}")


def _parse(raw: str) -> Dict[str, Dict[str, str]]:
    text = raw.strip()
    if not (text.startswith("{") and text.endswith("}")):
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for c in obj.get("choices") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        if not cid:
            continue
        out[cid] = {
            "choice": str(c.get("choice") or "").strip(),
            "reason": str(c.get("reason") or "").strip(),
        }
    return out


def decide(
    transcript: str,
    items: Sequence[Dict[str, Any]],
    *,
    timeout: float = 120.0,
) -> List[Dict[str, Any]]:
    from . import tracing

    if not items:
        return []
    user_payload = _build_user(transcript, items)
    tracing.emit("decide.request", {
        "system": _SYSTEM,
        "user": json.loads(user_payload),
        "model": _llm_model(),
        "url": _llm_url(),
        "provider": _llm_provider(),
    })
    try:
        content = _post(user_payload, timeout=timeout)
        tracing.emit("decide.response", {"raw": content})
        parsed = _parse(content)
    except Exception as exc:
        print(f"[llm_decide] LLM call failed: {exc!r}")
        tracing.emit("decide.error", {"error": repr(exc)})
        parsed = {}

    out: List[Dict[str, Any]] = []
    for item in items:
        cid = item["id"]
        cand_terms = [c["term"] for c in item["candidates"]]
        info = parsed.get(cid) or {}
        choice_str = info.get("choice", "")
        reason = info.get("reason", "")
        if not choice_str or choice_str.upper() == NO_CHANGE:
            out.append({"id": cid, "choice": None, "reason": reason})
            continue
        match = None
        for t in cand_terms:
            if t == choice_str:
                match = t
                break
        if match is None:
            for t in cand_terms:
                if t.lower() == choice_str.lower():
                    match = t
                    break
        out.append({"id": cid, "choice": match, "reason": reason})
    return out
