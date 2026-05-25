"""DEPRECATED: Stage 1 scorer powered by a local LLM (Qwen2.5-1.5B-Instruct).

This module is no longer imported by the pipeline. The scorer now uses the
Groq API (``app/pipeline/scorer.py``).

Kept here for reference only. All new development should use the Groq-based
scorer in ``app.pipeline.scorer``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_device = None
_loaded_model_id: Optional[str] = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
    return _device


def is_loaded() -> bool:
    """Return True if the model is currently loaded in memory."""
    return _model is not None


def _patch_safetensors() -> None:
    """Monkey-patch safetensors.safe_open to bypass Windows mmap issues.

    On Windows, ``safe_open`` uses memory-mapped I/O for safetensors files.
    For 3 GB+ models this can fail with ``OSError 1455`` (paging file too
    small) even when the page file is large, because Windows cannot create
    a file mapping for such a large contiguous range.

    The patch replaces ``safe_open`` with a wrapper that reads each file
    into memory via ``safetensors.torch.load_file()``, which avoids mmap
    entirely.
    """
    import safetensors as _st
    import safetensors.torch as _st_torch

    if hasattr(_st, "_patched"):
        return  # already patched

    _original = _st.safe_open

    class _InMemorySafeOpen:
        """Dict-like wrapper around the result of ``load_file``."""
        def __init__(self, filename: str, **kwargs):
            self._data = _st_torch.load_file(filename, device="cpu")

        def get_tensor(self, name: str):
            return self._data[name]

        def keys(self):
            return self._data.keys()

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

        def __contains__(self, name):
            return name in self._data

    def _patched_safe_open(filename, framework="pt", device="cpu"):
        return _InMemorySafeOpen(filename, framework=framework, device=device)

    _st.safe_open = _patched_safe_open
    _st._patched = True
    print("[llm_scorer] safetensors mmap patched (in-memory loading)")


def load_model() -> None:
    """Load Qwen2.5-1.5B-Instruct on GPU.

    Strategy:
    1. GPU FP16 directly (needs ~3 GB VRAM on a 4 GB GPU).
    2. CPU fallback if CUDA is unavailable.

    The model is loaded once and kept in memory across pipeline runs.

    Calling this when the model is already loaded is a no-op.
    """
    global _model, _tokenizer, _loaded_model_id
    if _model is not None and _loaded_model_id is not None:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    cache_dir = os.environ.get("HF_CACHE_DIR", "D:/HF_CACHE")
    device = _get_device()

    # Patch safetensors BEFORE any from_pretrained call
    _patch_safetensors()

    # Try GPU FP16 first
    if device.type == "cuda":
        try:
            print(f"[llm_scorer] Loading {model_id} on GPU (FP16)...")
            _tokenizer = AutoTokenizer.from_pretrained(
                model_id, cache_dir=cache_dir,
            )
            _model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            _model.eval()
            _loaded_model_id = model_id
            vram_gb = torch.cuda.memory_allocated() / 1024**3
            print(f"[llm_scorer] GPU load OK. VRAM used: {vram_gb:.1f} GB")
            return
        except Exception as exc:
            print(f"[llm_scorer] GPU load failed: {exc}")
            _model = None
            _tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # CPU fallback
    print(f"[llm_scorer] Loading {model_id} on CPU...")
    _tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir,
    )
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    _model.eval()
    _loaded_model_id = model_id
    print(f"[llm_scorer] CPU load OK.")


def unload_model() -> None:
    """Free GPU memory by deleting the model."""
    global _model, _tokenizer, _loaded_model_id
    _model = None
    _tokenizer = None
    _loaded_model_id = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[llm_scorer] Model unloaded, VRAM freed.")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a medical transcript quality checker. "
    "Identify words in the sentence that appear to be misspelled medical terms.\n\n"
    "Rules:\n"
    "- Do NOT flag common English words (the, patient, has, with, and, for, of, etc.)\n"
    "- Do NOT flag correctly spelled medical terms (myocardial, infarction, hypertension, etc.)\n"
    "- Only flag words that look like misspellings of medical terms\n"
    "- If no words are suspicious, return an empty array\n\n"
    'Respond with ONLY valid JSON in this format:\n'
    '{"suspicious_words": [{"word": "myokardial", "suspicion": 0.85}]}'
)


def _build_prompt(sentence: str) -> str:
    return f"Sentence: {sentence}\n\nIdentify any misspelled medical terms."


def _parse_response(text: str) -> List[Dict[str, Any]]:
    """Extract and parse the JSON list of suspicious words from the model response."""
    raw = text.strip()
    # Strip markdown code fences
    for fence in ("```json", "```"):
        if fence in raw:
            parts = raw.split(fence)
            # Take content after the opening fence, before the closing fence
            for p in parts[1:]:
                p = p.strip()
                if p.startswith("{"):
                    raw = p
                    break

    # Try full JSON object
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, dict) and "suspicious_words" in data:
                return data["suspicious_words"]
            # Try top-level as a list
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # Try the entire response as a JSON array
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("suspicious_words", "words", "flagged", "results"):
                val = data.get(key, [])
                if isinstance(val, list):
                    return val
            return []
    except json.JSONDecodeError:
        pass

    return []


# ---------------------------------------------------------------------------
# Scoring call
# ---------------------------------------------------------------------------

def score_transcript_llm(transcript: str) -> Optional[List[Dict[str, Any]]]:
    """Score words in *transcript* using the local LLM.

    Returns a list of dicts::

        [{"word": "myokardial", "suspicion": 0.85, "reason": "..."}, ...]

    Returns ``None`` if the model is not available or the call fails
    (caller should fall back to heuristic scoring).
    """
    load_model()

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(transcript)},
    ]

    raw = _generate(messages, max_new_tokens=256)
    if not raw:
        return None

    items = _parse_response(raw)
    if not items:
        return []

    # Normalise and clamp suspicion scores
    out: List[Dict[str, Any]] = []
    for item in items:
        word = str(item.get("word", "")).strip().lower()
        if not word:
            continue
        suspicion = float(item.get("suspicion", item.get("score", 0.5)))
        suspicion = max(0.0, min(1.0, suspicion))
        reason = str(item.get("reason", ""))
        out.append({"word": word, "suspicion": suspicion, "reason": reason})

    return out


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(messages: list, max_new_tokens: int = 256) -> Optional[str]:
    """Generate a response from the local LLM."""
    try:
        if _tokenizer is None or _model is None:
            return None

        text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = _tokenizer([text], return_tensors="pt").to(_model.device)

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return _tokenizer.decode(generated, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"[llm_scorer] Generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Module initialisation (optional — call ``load_model()`` eagerly instead)
# ---------------------------------------------------------------------------

def warm_up() -> None:
    """Eagerly load the model so the first user request is fast.

    Call this at application startup.
    """
    load_model()
