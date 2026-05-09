"""Speech-to-text service (faster-whisper) + forced-prefix scoring.

This module exposes two operations:

  transcribe(audio_path, model_size, language, hotwords=...)
      - Runs Whisper with optional `hotwords` biasing. Returns the
        transcript and per-word info.

  score_candidate(audio_path, candidate_text, *, start_s=None, end_s=None)
      - Forces Whisper to emit `candidate_text` as a prefix on the audio
        (or audio slice). Returns Whisper's avg_logprob for that segment.
        Higher = better acoustic-language fit. Used for ranking lexicon
        candidates against a suspect span (Whisper-twice architecture).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


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
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc

    default_device = "auto"
    default_compute = "int8"
    device = os.environ.get("WHISPER_DEVICE", default_device)
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", default_compute)
    _MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
    _MODEL_NAME = model_size
    return _MODEL


# ---------------------------------------------------------------------------
# Free transcription with optional biasing
# ---------------------------------------------------------------------------


def transcribe(
    audio_path: str | Path,
    model_size: str = "small",
    language: Optional[str] = None,
    hotwords: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe a wav/webm/mp3 file. Returns transcript + per-word info.

    `hotwords` is a comma-separated string of preferred terms — Whisper's
    decoder gets a soft bias toward them. This catches OOV medical names
    that Whisper would otherwise hallucinate as common English.
    """
    model = _load_model(model_size)
    kwargs: Dict[str, Any] = dict(
        language=language,
        vad_filter=True,
        word_timestamps=True,
    )
    if hotwords:
        kwargs["hotwords"] = hotwords
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    segments, info = model.transcribe(str(audio_path), **kwargs)

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


# ---------------------------------------------------------------------------
# Forced-prefix scoring (the second pass)
# ---------------------------------------------------------------------------


def _slice_to_temp_wav(audio_path: str | Path, start_s: float, end_s: float, pad_s: float = 0.10) -> str:
    """Cut the audio in [start_s-pad_s, end_s+pad_s] into a temporary WAV."""
    import subprocess
    import tempfile

    ts0 = max(0.0, float(start_s) - pad_s)
    dur = max(0.20, float(end_s) - float(start_s) + 2.0 * pad_s)
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="seg_")
    os.close(fd)
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y",
        "-ss", f"{ts0:.3f}",
        "-i", str(audio_path),
        "-t", f"{dur:.3f}",
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        path,
    ]
    subprocess.run(cmd, check=True)
    return path


def score_candidate(
    audio_path: str | Path,
    candidate_text: str,
    *,
    model_size: str = "small",
    language: Optional[str] = "en",
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Score how well `candidate_text` fits the audio (or audio slice).

    Implementation: feed `candidate_text` as the `prefix` to Whisper.
    Whisper is forced to emit those tokens at the start of the segment,
    then continues freely. Whisper reports `avg_logprob` for the segment —
    higher (less negative) = better acoustic-language match for the
    candidate.

    Returns:
        {
          "avg_logprob": float,      # the score (higher is better)
          "text": str,               # what Whisper actually emitted
          "candidate": str,
        }
    """
    model = _load_model(model_size)
    if start_s is not None and end_s is not None:
        audio_input = _slice_to_temp_wav(audio_path, float(start_s), float(end_s))
        cleanup = audio_input
    else:
        audio_input = str(audio_path)
        cleanup = None

    try:
        segments, _info = model.transcribe(
            audio_input,
            language=language,
            beam_size=1,
            prefix=candidate_text,
            without_timestamps=True,
            condition_on_previous_text=False,
            temperature=0.0,
            vad_filter=False,
            word_timestamps=False,
        )
        segs = list(segments)
        if not segs:
            return {
                "avg_logprob": float("-inf"),
                "text": "",
                "candidate": candidate_text,
            }
        avg = float(np.mean([s.avg_logprob for s in segs]))
        text = "".join(s.text for s in segs).strip()
        return {
            "avg_logprob": avg,
            "text": text,
            "candidate": candidate_text,
        }
    finally:
        if cleanup:
            try:
                os.unlink(cleanup)
            except OSError:
                pass
