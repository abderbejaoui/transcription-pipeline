"""Speech-to-text service.

Wraps faster-whisper if available. faster-whisper is heavy and optional, so
this module is lazy: the model is only loaded the first time `transcribe` is
called. If the package is not installed, the API surfaces a clear error
instead of crashing on import.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


_MODEL = None
_MODEL_NAME: Optional[str] = None


def _load_model(model_size: str = "small"):
    global _MODEL, _MODEL_NAME
    if _MODEL is not None and _MODEL_NAME == model_size:
        return _MODEL
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "faster-whisper is not installed. "
            "Run: pip install faster-whisper"
        ) from exc

    # Use MPS (Apple Silicon) when available, fall back to CPU.
    # compute_type must be "int8" for CPU; "float16" for GPU/MPS if supported.
    default_device = "auto"
    default_compute = "int8"
    device = os.environ.get("WHISPER_DEVICE", default_device)
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", default_compute)
    _MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
    _MODEL_NAME = model_size
    return _MODEL


def transcribe(audio_path: str | Path, model_size: str = "small", language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe a wav/webm/mp3 file. Returns the transcript and per-word info."""
    model = _load_model(model_size)
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        word_timestamps=True,
    )

    full_text_parts: list[str] = []
    words: list[Dict[str, Any]] = []
    for seg in segments:
        full_text_parts.append(seg.text)
        if seg.words:
            for w in seg.words:
                words.append(
                    {
                        "word": w.word,
                        "start": float(w.start) if w.start is not None else None,
                        "end": float(w.end) if w.end is not None else None,
                        "probability": float(w.probability) if w.probability is not None else None,
                    }
                )

    return {
        "text": "".join(full_text_parts).strip(),
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration": float(info.duration),
        "words": words,
    }
