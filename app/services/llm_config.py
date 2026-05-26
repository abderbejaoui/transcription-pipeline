"""Shared LLM configuration for Ollama or OpenRouter."""

from __future__ import annotations

import os
import urllib.error
from typing import Dict, List

DEFAULT_OLLAMA_URL = "http://100.68.87.28:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "MaziyarPanahi/Calme-7B-Instruct-v0.2"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"


def get_llm_provider() -> str:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider in {"ollama", "openrouter"}:
        return provider
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    return "ollama"


def get_llm_url(provider: str | None = None) -> str:
    provider = provider or get_llm_provider()
    if provider == "openrouter":
        return os.environ.get("OPENROUTER_URL", DEFAULT_OPENROUTER_URL)
    return os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)


def get_llm_model(provider: str | None = None) -> str:
    provider = provider or get_llm_provider()
    if provider == "openrouter":
        return os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def get_llm_headers(provider: str | None = None) -> Dict[str, str]:
    provider = provider or get_llm_provider()
    headers = {"Content-Type": "application/json"}
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter"
            )
        headers["Authorization"] = f"Bearer {api_key}"
        referer = os.environ.get("OPENROUTER_APP_URL", "").strip()
        title = os.environ.get("OPENROUTER_APP_NAME", "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
    return headers


def build_chat_payload(
    model: str,
    messages: List[Dict[str, str]],
    *,
    json_mode: bool = False,
    temperature: float = 0.0,
) -> Dict[str, object]:
    provider = get_llm_provider()
    max_tokens_env = os.environ.get("LLM_MAX_TOKENS", "").strip()
    max_tokens = int(max_tokens_env) if max_tokens_env.isdigit() else None
    payload: Dict[str, object] = {
        "model": model,
        "stream": False,
        "messages": messages,
    }
    if provider == "openrouter":
        payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
    else:
        if json_mode:
            payload["format"] = "json"
        payload["think"] = False
        options: Dict[str, object] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload["options"] = options
    return payload


def describe_http_error(exc: BaseException, *, max_chars: int = 800) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        body = ""
        try:
            body = exc.read().decode("utf-8", "ignore")
        except Exception:
            body = ""
        if body:
            return f"{exc} body={body[:max_chars]}"
        return str(exc)
    return repr(exc)


def parse_chat_content(data: Dict[str, object], provider: str | None = None) -> str:
    provider = provider or get_llm_provider()
    if provider == "openrouter":
        choices = data.get("choices") or []
        if not isinstance(choices, list) or not choices:
            raise KeyError("OpenRouter response missing choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise KeyError("OpenRouter response missing message")
        content = message.get("content")
        if content is None:
            raise KeyError("OpenRouter response missing message content")
        return str(content)
    message = data.get("message")
    if not isinstance(message, dict):
        raise KeyError("Ollama response missing message")
    content = message.get("content")
    if content is None:
        raise KeyError("Ollama response missing message content")
    return str(content)

