"""Cached LLM-generated medical descriptions.

Each medical term gets a one- or two-sentence description that explains:
  - what it is (drug class, disease type, ...)
  - what condition it treats / is associated with

Used by the DECIDE step so the LLM can reason about which candidate fits
the patient's clinical context (e.g. "diabetes" -> prefer metformin over
ceftriaxone). Cached on disk in `data/descriptions.jsonl`; we never call
the LLM twice for the same term.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from .llm_config import (
    get_llm_headers,
    get_llm_model,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DESCRIPTIONS_PATH = DATA_DIR / "descriptions.jsonl"


def _ollama_url() -> str:
    return get_llm_url(get_llm_provider())


def _ollama_model() -> str:
    return get_llm_model(get_llm_provider())


_lock = threading.Lock()
_cache: Dict[str, str] = {}
_loaded = False


import time as _time


def _post_with_retry(
    payload: Dict, *, timeout: float, label: str, attempts: int = 4, backoff: float = 1.0
) -> Optional[str]:
    """POST to Ollama with retries on network errors. Returns content string
    on success, None after all attempts fail."""
    last_exc: Optional[BaseException] = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(
                _ollama_url(),
                data=json.dumps(payload).encode("utf-8"),
                headers=get_llm_headers(get_llm_provider()),
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return parse_chat_content(data, get_llm_provider()).strip()
        except Exception as exc:
            last_exc = exc
            wait = backoff * (2 ** i)
            print(f"[{label}] LLM call failed (attempt {i+1}/{attempts}): {exc!r}; retrying in {wait:.1f}s")
            _time.sleep(wait)
    print(f"[{label}] all {attempts} attempts failed: {last_exc!r}")
    return None


def _key(term: str) -> str:
    return term.strip().lower()


def _load() -> None:
    global _loaded
    with _lock:
        if _loaded:
            return
        if DESCRIPTIONS_PATH.exists():
            with DESCRIPTIONS_PATH.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    term = row.get("term") or ""
                    desc = row.get("description") or ""
                    if term and desc:
                        _cache[_key(term)] = desc
        _loaded = True


def _append(term: str, description: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with DESCRIPTIONS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"term": term, "description": description}, ensure_ascii=False)
            + "\n"
        )


def get(term: str) -> Optional[str]:
    """Return cached description (no LLM call)."""
    _load()
    return _cache.get(_key(term))


def save(term: str, description: str) -> None:
    """Persist a manually-provided description, bypassing LLM generation."""
    _load()
    k = _key(term)
    with _lock:
        if k not in _cache:
            _cache[k] = description
            _append(term, description)


def get_or_generate(
    term: str,
    *,
    type_hint: Optional[str] = None,
    timeout: float = 60.0,
) -> Optional[str]:
    """Return cached description, or call the LLM once and cache the result."""
    _load()
    k = _key(term)
    if k in _cache:
        return _cache[k]
    desc = _generate(term, type_hint=type_hint, timeout=timeout)
    if desc:
        with _lock:
            _cache[k] = desc
        _append(term, desc)
    return desc


def _generate(term: str, *, type_hint: Optional[str] = None, timeout: float = 120.0) -> Optional[str]:
    """Single LLM call: ask for a short medical description in strict JSON."""
    type_str = f"({type_hint})" if type_hint else ""
    user = json.dumps(
        {
            "task": (
                "Provide a concise medical description of the term below. "
                "If it is a drug, include drug class and the main conditions "
                "it is used to treat. If it is a disease, include type and "
                "common signs/treatments. One or two sentences. Output JSON only."
            ),
            "term": term,
            "term_type": type_hint or "unknown",
            "output_schema": {
                "type": "object",
                "required": ["description"],
                "properties": {"description": {"type": "string"}},
            },
        },
        ensure_ascii=False,
    )
    payload = {
        "model": _ollama_model(),
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.0},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical reference summariser. "
                    "Return ONLY strict JSON with a single key `description`. "
                    "No prose."
                ),
            },
            {"role": "user", "content": user},
        ],
    }
    content = _post_with_retry(payload, timeout=timeout, label=f"descriptions[{term!r}]")
    if not content:
        return None

    if not (content.startswith("{") and content.endswith("}")):
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            content = m.group(0)
    try:
        obj = json.loads(content)
        desc = str(obj.get("description") or "").strip()
        return desc or None
    except json.JSONDecodeError:
        return None


def all_descriptions() -> List[Dict[str, str]]:
    _load()
    return [{"term": k, "description": v} for k, v in _cache.items()]
