"""Batched constrained reranker against an Ollama endpoint.

For each suspicious span we send the LLM a candidate list. The LLM picks
ONE or returns "NO_CHANGE". One HTTP call per transcript.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", "http://100.68.87.28:11434/api/chat"
)
DEFAULT_OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL", "hf.co/bartowski/calme-3.2-instruct-78b-GGUF:IQ4_XS"
)
NO_CHANGE = "NO_CHANGE"


_SYSTEM = (
    "You are a constrained medical transcript correction reranker. "
    "For each suspicious span, decide whether ANY candidate is a clearly "
    "correct medical term in the transcript context. "
    "Strict rules: "
    "1. Output strict JSON only. No prose. "
    "2. Default to \"NO_CHANGE\". Only pick a candidate when you are "
    "confident the spoken word is that medical term and the sentence "
    "makes medical sense with it. "
    "3. If the suspicious span is a common English word (numbers, articles, "
    "verbs, body parts in everyday language), return \"NO_CHANGE\". "
    "4. Never pick a candidate just because it is the only option."
)


def _build_user_payload(transcript: str, items: Sequence[Dict[str, Any]]) -> str:
    return json.dumps(
        {
            "task": (
                "For each suspicious span, decide whether the spoken word is a "
                "specific medical term from its candidate list. "
                "If yes AND it makes medical sense in the transcript, return "
                "that exact candidate. Otherwise return \"NO_CHANGE\". "
                "Common English words must always be \"NO_CHANGE\"."
            ),
            "transcript": transcript,
            "spans": [
                {
                    "id": item["id"],
                    "span": item["span"],
                    "candidates": list(item["candidates"]) + [NO_CHANGE],
                }
                for item in items
            ],
            "examples": [
                {"span": "four", "candidates": ["Doliprane"], "choice": "NO_CHANGE"},
                {"span": "the", "candidates": ["Doliprane"], "choice": "NO_CHANGE"},
                {"span": "patient", "candidates": ["pyelonephritis"], "choice": "NO_CHANGE"},
                {"span": "Dali pran", "candidates": ["Doliprane", "diltiazem"], "choice": "Doliprane"},
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
                            },
                        },
                    }
                },
            },
        },
        ensure_ascii=False,
    )


def _post_chat(url: str, model: str, system: str, user: str, timeout: float) -> str:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["message"]["content"]


def _parse_choices(content: str) -> Dict[str, str]:
    text = content.strip()
    if not (text.startswith("{") and text.endswith("}")):
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {}
    out: Dict[str, str] = {}
    for c in obj.get("choices") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id", "")).strip()
        choice = str(c.get("choice", "")).strip()
        if cid:
            out[cid] = choice
    return out


def rerank(
    transcript: str,
    items: Sequence[Dict[str, Any]],
    *,
    url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
    timeout: float = 60.0,
) -> List[Dict[str, Optional[str]]]:
    """Returns [{id, choice}] where choice is one of the provided candidates
    (case-sensitive) or None if NO_CHANGE / unknown / parsing failed."""
    if not items:
        return []
    user = _build_user_payload(transcript, items)
    try:
        content = _post_chat(url, model, _SYSTEM, user, timeout)
        parsed = _parse_choices(content)
    except Exception as exc:
        print(f"[llm_rerank] LLM call failed: {exc!r}; defaulting all to NO_CHANGE")
        parsed = {}

    out: List[Dict[str, Optional[str]]] = []
    for item in items:
        cid = item["id"]
        cand_list = list(item["candidates"])
        choice_str = parsed.get(cid, "")
        if not choice_str or choice_str.upper() == NO_CHANGE:
            out.append({"id": cid, "choice": None})
            continue
        match = None
        for c in cand_list:
            if c == choice_str:
                match = c
                break
        if match is None:
            for c in cand_list:
                if c.lower() == choice_str.lower():
                    match = c
                    break
        if match is None:
            out.append({"id": cid, "choice": None})
        else:
            out.append({"id": cid, "choice": match})
    return out
