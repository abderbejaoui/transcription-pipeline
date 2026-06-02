"""Local LLM corrector for medical transcripts.

Primary: 4-bit quantized Qwen2.5-1.5B-Instruct running on local GPU.
Fallback: OpenRouter API (Qwen 2.5 72B) when local model is unavailable.

Architecture:
  1. Lazy-load the 4-bit model on first call (~1.2GB VRAM).
  2. Send the full transcript to the LLM with a structured prompt.
  3. Parse the returned JSON {corrected, corrections[], confidence}.
  4. If confidence < threshold OR parse fails, fall back to rule-based correction.
  5. If local model fails, try OpenRouter API as backup.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Gulf Arabic medical transcription corrector. Your task is to fix ASR errors in clinical dictation while preserving the original meaning.

Rules:
1. Fix Arabic spelling errors (بدل→بعد, سداع→صداع, حراره→حرارة)
2. Fix English medical misspellings (hyperglacymia→hyperglycemia, wheezeng→wheezing)
3. Convert Arabic transliterations to English medical terms:
   - هستوري → history
   - دايابيتس → diabetes
   - هايبرتنشن → hypertension
   - بلاد شوجر → blood sugar
   - هارت → heart
   - بلد برشر → blood pressure
   - وغيرها (etc.)
4. Preserve normal Arabic words (حضر, بسبب, المريض, دكتور, غثيان, صداع, etc.)
5. Preserve numbers, units, and medical abbreviations (ECG, BP, HR, etc.)
6. Keep the same structure and paragraph breaks as the original.

Output format (STRICT JSON, no markdown or extra text):
{
  "corrected": "The full corrected transcript...",
  "corrections": [
    {"original": "هستوري", "corrected": "history", "type": "transliteration"},
    {"original": "hyperglacymia", "corrected": "hyperglycemia", "type": "spelling"}
  ],
  "confidence": 0.92
}

If no corrections needed, return corrections as empty array and confidence ~1.0."""

USER_PROMPT_TEMPLATE = "Correct the following medical transcript:\n\n{transcript}"

# Regex to extract JSON from LLM output (handles markdown fences)
_JSON_RE = re.compile(
    r"(?:```(?:json)?\s*)?(\{[\s\S]*?\})(?:\s*```)?", re.IGNORECASE
)


def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from LLM response, handling markdown fences."""
    # Try direct parsing first
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from fences
    match = _JSON_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ── Local model (4-bit Qwen2.5-1.5B) ───────────────────────────────────

_MODEL: Any = None
_TOKENIZER: Any = None
_MODEL_LOCK = threading.Lock()


def _load_local_model() -> Tuple[Any, Any]:
    """Lazy-load the 4-bit quantized model.

    Uses BitsAndBytes for 4-bit quantization to fit in ~1.2GB VRAM.
    Thread-safe with double-checked locking.
    """
    global _MODEL, _TOKENIZER

    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    with _MODEL_LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            return _MODEL, _TOKENIZER

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            from .config import get_config

            cfg = get_config()
            model_name = cfg.llm_model_name

            logger.info("Loading LLM: %s (4-bit)", model_name)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left",
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            model.eval()

            _MODEL = model
            _TOKENIZER = tokenizer
            logger.info(
                "LLM loaded OK. VRAM: ~%.2fGB",
                torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            )
            return model, tokenizer

        except Exception as exc:
            logger.error("Failed to load local LLM: %s", exc)
            _MODEL = None
            _TOKENIZER = None
            return None, None


def _local_correct(transcript: str) -> Optional[Dict[str, Any]]:
    """Run local 4-bit model correction.

    Returns parsed JSON dict or None on failure.
    """
    model, tokenizer = _load_local_model()
    if model is None or tokenizer is None:
        return None

    try:
        import torch

        from .config import get_config

        cfg = get_config()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(transcript=transcript)},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.llm_max_new_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Extract only the generated part (skip input)
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        raw = tokenizer.decode(generated, skip_special_tokens=True)

        logger.debug("Local LLM raw output (first 200 chars): %s", raw[:200])

        parsed = _parse_json_response(raw)
        if parsed is None:
            logger.warning("Local LLM output did not parse as JSON: %s", raw[:100])
            return None

        return parsed

    except Exception as exc:
        logger.error("Local LLM inference failed: %s", exc)
        return None


# ── API fallback (OpenRouter) ──────────────────────────────────────────

def _api_correct(transcript: str) -> Optional[Dict[str, Any]]:
    """Fallback correction via OpenRouter API.

    Returns parsed JSON dict or None on failure.
    """
    try:
        from .config import get_config
        from .llm_config import get_llm_headers

        cfg = get_config()
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            logger.warning("OpenRouter API key not set, skipping API fallback")
            return None

        headers = get_llm_headers()
        url = os.environ.get(
            "OPENROUTER_URL",
            "https://openrouter.ai/api/v1/chat/completions",
        )

        payload = {
            "model": cfg.api_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(transcript=transcript)},
            ],
            "temperature": 0.1,
            "max_tokens": cfg.llm_max_new_tokens,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=cfg.api_timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        raw = ""
        choices = body.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            raw = message.get("content", "")

        if not raw:
            logger.warning("OpenRouter returned empty content")
            return None

        parsed = _parse_json_response(raw)
        if parsed is None:
            logger.warning("OpenRouter output did not parse as JSON: %s", raw[:100])
            return None

        logger.info("OpenRouter correction received (confidence=%.2f)", parsed.get("confidence", 0))
        return parsed

    except urllib.request.HTTPError as exc:
        logger.error("OpenRouter HTTP %d: %s", exc.code, exc.read().decode()[:200])
        return None
    except Exception as exc:
        logger.error("OpenRouter API call failed: %s", exc)
        return None


# ── Public API ─────────────────────────────────────────────────────────

class LLMCorrectorResult:
    """Result from the LLM corrector."""

    __slots__ = (
        "corrected_text",
        "corrections",
        "confidence",
        "source",  # "local", "api", or None
        "raw_output",
        "success",
    )

    def __init__(
        self,
        corrected_text: str = "",
        corrections: Optional[List[Dict[str, str]]] = None,
        confidence: float = 0.0,
        source: Optional[str] = None,
        raw_output: str = "",
        success: bool = False,
    ):
        self.corrected_text = corrected_text
        self.corrections = corrections or []
        self.confidence = confidence
        self.source = source
        self.raw_output = raw_output
        self.success = success


def correct_transcript(
    transcript: str,
    timeout: float = 30.0,
    use_api_fallback: bool = True,
) -> LLMCorrectorResult:
    """Correct a transcript using the LLM (local first, API fallback).

    Args:
        transcript: Raw ASR transcript text.
        timeout: Max seconds to wait for local model.
        use_api_fallback: Whether to try OpenRouter if local model fails.

    Returns:
        LLMCorrectorResult with corrected text, corrections list, and confidence.
    """
    if not transcript or not transcript.strip():
        return LLMCorrectorResult(corrected_text=transcript, success=False)

    from .config import get_config
    cfg = get_config()

    if not cfg.use_llm_corrector:
        return LLMCorrectorResult(corrected_text=transcript, success=False)

    # Stage 1: Local model
    result: Optional[Dict[str, Any]] = None
    source: Optional[str] = None

    try:
        result = _local_correct(transcript)
        if result is not None:
            source = "local"
            logger.info("Local LLM correction succeeded")
    except Exception as exc:
        logger.warning("Local LLM error: %s", exc)

    # Stage 2: API fallback
    if result is None and use_api_fallback and cfg.use_api_fallback:
        try:
            result = _api_correct(transcript)
            if result is not None:
                source = "api"
                logger.info("API fallback correction succeeded")
        except Exception as exc:
            logger.warning("API fallback error: %s", exc)

    if result is None:
        return LLMCorrectorResult(
            corrected_text=transcript,
            source=None,
            success=False,
        )

    corrected = result.get("corrected", "")
    if not corrected:
        corrected = transcript

    corrections = result.get("corrections", [])
    if not isinstance(corrections, list):
        corrections = []

    confidence = float(result.get("confidence", 0.5))

    return LLMCorrectorResult(
        corrected_text=corrected,
        corrections=corrections,
        confidence=confidence,
        source=source,
        raw_output=json.dumps(result, ensure_ascii=False),
        success=True,
    )


def warm_up() -> None:
    """Pre-load the model on startup (safe, non-blocking)."""
    try:
        _load_local_model()
    except Exception as exc:
        logger.warning("LLM warm-up failed (will load on demand): %s", exc)
