"""Shared LLM configuration and warm-up helpers."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Optional

from .llm_config import (
    build_chat_payload,
    get_llm_headers,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)


DEFAULT_SHARED_MODEL = os.environ.get(
    "LLM_MODEL_SHARED", "MaziyarPanahi/Calme-7B-Instruct-v0.2"
)


def get_model_id(kind: str) -> str:
    key = f"LLM_MODEL_{kind.upper()}"
    override = os.environ.get(key, "").strip()
    if override:
        return override
    return DEFAULT_SHARED_MODEL


def warm_up(*, timeout: float = 30.0) -> None:
    if os.environ.get("LLM_WARMUP", "1") != "1":
        return
    model_id = get_model_id("general")
    payload = build_chat_payload(
        model_id,
        [
            {
                "role": "system",
                "content": "Return strict JSON: {\"ok\": true}.",
            },
            {"role": "user", "content": "warmup"},
        ],
        json_mode=True,
        temperature=0.0,
    )
    try:
        req = urllib.request.Request(
            get_llm_url(get_llm_provider()),
            data=json.dumps(payload).encode("utf-8"),
            headers=get_llm_headers(get_llm_provider()),
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        parse_chat_content(data, get_llm_provider())
    except Exception as exc:
        print(f"[llm_runtime] warm-up failed: {exc!r}")
        time.sleep(0.05)
