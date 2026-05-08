"""LLM-driven suspicious-word detector.

Given Whisper's word-level transcript, the LLM returns the index ranges of
spans that look like ASR mishearings of medical terms (drugs, diseases,
procedures, lab tests). No hardcoded English word lists, no fuzzy/phonetic
rules — purely the LLM's medical judgement.

Public API
----------
detect(words: list[{word, start, end, probability}]) -> list[Span]
    Span = {"index_start": int, "index_end": int (exclusive),
            "text": str, "start_s": float, "end_s": float,
            "probability_min": float, "reason": str}
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from .llm_config import (
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


_SYSTEM = (
    "You are a medical transcript auditor. The transcript was produced by "
    "an automatic speech-to-text system that often misspells drug names, "
    "disease names, and other clinical terms. "
    "Your job: identify the WORD INDEX ranges of spans that are likely "
    "such mishearings — words that LOOK or SOUND like a medical term but "
    "are not standard medical spellings. "
    "Strict rules: "
    "1. Output strict JSON only. No prose. "
    "2. Index ranges are zero-based, half-open: [start, end). "
    "3. Do NOT flag normal English words (numbers, articles, verbs, dates, "
    "common body words). "
    "4. Do NOT flag medical terms that are ALREADY spelled correctly. "
    "5. If nothing is suspicious, return an empty list. "
    "6. A span may cover multiple consecutive tokens if Whisper split one "
    "spoken word into pieces."
)


def _build_user(words: Sequence[Dict[str, Any]]) -> str:
    enumerated = [
        {"i": i, "word": (w.get("word") or "").strip(), "p": w.get("probability")}
        for i, w in enumerate(words)
    ]
    return json.dumps(
        {
            "task": (
                "List suspicious medical-term spans in this token stream. "
                "Each entry: {\"index_start\": int, \"index_end\": int, "
                "\"reason\": short string}. Empty list if no suspicions."
            ),
            "tokens": enumerated,
            "output_schema": {
                "type": "object",
                "required": ["spans"],
                "properties": {
                    "spans": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["index_start", "index_end"],
                            "properties": {
                                "index_start": {"type": "integer"},
                                "index_end": {"type": "integer"},
                                "reason": {"type": "string"},
                            },
                        },
                    }
                },
            },
        },
        ensure_ascii=False,
    )


def _post(content: str, timeout: float) -> Dict[str, Any]:
    payload = {
        "model": _llm_model(),
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": content},
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
            content = parse_chat_content(data, _llm_provider())
            return {"content": content}
        except Exception as exc:
            last_exc = exc
            wait = 1.0 * (2 ** attempt)
            print(f"[llm_detect] LLM call failed (attempt {attempt+1}/4): {exc!r}; retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"llm_detect: all retries failed: {last_exc!r}")


def _parse(raw: str) -> List[Dict[str, Any]]:
    text = raw.strip()
    if not (text.startswith("{") and text.endswith("}")):
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    spans = obj.get("spans") or []
    return [s for s in spans if isinstance(s, dict)]


def detect(
    words: Sequence[Dict[str, Any]], *, timeout: float = 120.0
) -> List[Dict[str, Any]]:
    """Returns validated spans with their actual text + audio timestamps."""
    from . import tracing

    if not words:
        return []
    user_payload = _build_user(words)
    tracing.emit("detect.request", {
        "system": _SYSTEM,
        "user": json.loads(user_payload),
        "model": _llm_model(),
        "url": _llm_url(),
        "provider": _llm_provider(),
    })
    try:
        msg = _post(user_payload, timeout=timeout)
        raw = msg.get("content") or ""
        tracing.emit("detect.response", {"raw": raw})
        spans = _parse(raw)
    except Exception as exc:
        print(f"[llm_detect] LLM call failed: {exc!r}")
        tracing.emit("detect.error", {"error": repr(exc)})
        return []

    out: List[Dict[str, Any]] = []
    n = len(words)
    for s in spans:
        try:
            i0 = int(s.get("index_start"))
            i1 = int(s.get("index_end"))
        except (TypeError, ValueError):
            continue
        if not (0 <= i0 < i1 <= n):
            continue
        toks = words[i0:i1]
        text = " ".join((w.get("word") or "").strip() for w in toks).strip()
        if not text:
            continue
        starts = [w.get("start") for w in toks if isinstance(w.get("start"), (int, float))]
        ends = [w.get("end") for w in toks if isinstance(w.get("end"), (int, float))]
        if not starts or not ends:
            continue
        probs = [w.get("probability") for w in toks if isinstance(w.get("probability"), (int, float))]
        out.append(
            {
                "index_start": i0,
                "index_end": i1,
                "text": text,
                "start_s": float(min(starts)),
                "end_s": float(max(ends)),
                "probability_min": float(min(probs)) if probs else 1.0,
                "reason": str(s.get("reason") or "near_medical")[:128],
            }
        )
    # Merge adjacent / overlapping spans. Whisper often splits one spoken
    # word into multiple tokens (e.g. "Doliprane" -> "doly", "prems"); the
    # LLM may correctly flag both but we must treat them as ONE acoustic
    # unit when slicing the audio for fingerprinting.
    out.sort(key=lambda s: (s["index_start"], s["index_end"]))
    merged: List[Dict[str, Any]] = []
    for s in out:
        if merged and s["index_start"] <= merged[-1]["index_end"]:
            prev = merged[-1]
            prev["index_end"] = max(prev["index_end"], s["index_end"])
            prev["end_s"] = max(prev["end_s"], s["end_s"])
            prev["start_s"] = min(prev["start_s"], s["start_s"])
            prev["text"] = (prev["text"] + " " + s["text"]).strip()
            prev["probability_min"] = min(prev["probability_min"], s["probability_min"])
        else:
            merged.append(dict(s))
    return merged
