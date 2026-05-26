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
_MODEL_BACKEND: Optional[str] = None


def _backend() -> str:
    return os.environ.get("WHISPER_BACKEND", "faster").strip().lower()


def _load_model(model_size: str = "large-v3"):
    global _MODEL, _MODEL_NAME, _MODEL_BACKEND
    backend = _backend()
    if _MODEL is not None and _MODEL_NAME == model_size and _MODEL_BACKEND == backend:
        return _MODEL

    device = os.environ.get("WHISPER_DEVICE", "cpu")

    if backend in {"openai", "openai-whisper", "whisper"}:
        try:
            import whisper
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "openai-whisper is not installed. "
                "Run: pip install openai-whisper"
            ) from exc
        _MODEL = whisper.load_model(model_size, device=device)
        _MODEL_NAME = model_size
        _MODEL_BACKEND = backend
        return _MODEL

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "faster-whisper is not installed. "
            "Run: pip install faster-whisper"
        ) from exc

    # CPU + int8 keeps the demo runnable on a laptop without a GPU. On a GPU
    # box, switch to compute_type="float16".
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
    _MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
    _MODEL_NAME = model_size
    _MODEL_BACKEND = backend
    return _MODEL


def transcribe(audio_path: str | Path, model_size: str = "small", language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe a wav/webm/mp3 file. Returns the transcript and per-word info."""
    model = _load_model(model_size)
    backend = _backend()

    if backend in {"openai", "openai-whisper", "whisper"}:
        kwargs: Dict[str, Any] = {
            "language": language,
            "word_timestamps": True,
            "verbose": False,
        }
        try:
            result = model.transcribe(str(audio_path), **kwargs)
        except TypeError:
            kwargs.pop("word_timestamps", None)
            result = model.transcribe(str(audio_path), **kwargs)

        text = str(result.get("text") or "").strip()
        segments = result.get("segments") or []
        duration = result.get("duration")
        if duration is None:
            last_end = 0.0
            for seg in segments:
                if isinstance(seg, dict) and seg.get("end") is not None:
                    last_end = max(last_end, float(seg.get("end")))
            duration = last_end
        words: list[Dict[str, Any]] = []
        for seg in segments:
            seg_words = seg.get("words") if isinstance(seg, dict) else None
            if not seg_words:
                continue
            for w in seg_words:
                if not isinstance(w, dict):
                    continue
                words.append(
                    {
                        "word": w.get("word") or "",
                        "start": float(w.get("start")) if w.get("start") is not None else None,
                        "end": float(w.get("end")) if w.get("end") is not None else None,
                        "probability": float(w.get("probability")) if w.get("probability") is not None else None,
                    }
                )

        return {
            "text": text,
            "language": result.get("language") or (language or ""),
            "language_probability": float(result.get("language_probability") or 0.0),
            "duration": float(duration or 0.0),
            "words": words,
        }

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
