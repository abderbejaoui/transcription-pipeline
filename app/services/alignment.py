"""Word-level forced alignment for Arabic ASR transcripts.

Why this rewrite
----------------
The previous attempt used facebook/wav2vec2-base-960h (English-only) to
"align" Arabic audio via Latin transliteration. That gave 0.00 confidence
on every word because the model has no idea what Arabic phonemes sound
like — its frame-level output is meaningless for our purposes.

Correct approach
----------------
Use Whisper-small (multilingual) which **natively** produces per-word
timestamps for Arabic. We discard Whisper's transcript and use only its
timing data, then realign those timings to OUR (Qwen3-LoRA) transcript
via a SequenceMatcher diff.

This pattern is what every production forced-aligner does: pick a model
that actually speaks the language, take its timing, ignore everything
else.

Public API
----------
align_words(audio_path, target_transcript) -> list[{
    "word": str,
    "start_s": float | None,
    "end_s": float | None,
    "confidence": float,
}]
align_span(audio_path, transcript, span_text) -> {start_s, end_s, ...}
"""

from __future__ import annotations

import difflib
import os
import re
import subprocess
import tempfile
import threading
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Lazy Whisper loader (multilingual, knows Arabic)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_whisper_model = None
_whisper_lib = None
_whisper_loaded_name = None


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_whisper(model_size: str = "small"):
    """Load openai-whisper (the original lib, NOT faster-whisper).

    openai-whisper gives us per-word timestamps via word_timestamps=True
    without needing CTranslate2, which keeps it working on the DGX ARM
    build where CT2 has no CUDA support.
    """
    global _whisper_model, _whisper_lib, _whisper_loaded_name
    target = os.environ.get("ALIGN_WHISPER_SIZE", model_size)
    if _whisper_model is not None and _whisper_loaded_name == target:
        return _whisper_lib, _whisper_model
    with _lock:
        if _whisper_model is not None and _whisper_loaded_name == target:
            return _whisper_lib, _whisper_model
        import whisper  # type: ignore  (openai-whisper)
        _whisper_lib = whisper
        device = "cuda" if _cuda_available() else "cpu"
        print(f"[align] loading whisper-{target} on {device}")
        _whisper_model = whisper.load_model(target, device=device)
        _whisper_loaded_name = target
        return _whisper_lib, _whisper_model


# ---------------------------------------------------------------------------
# Audio normalisation — any format -> 16 kHz mono wav
# ---------------------------------------------------------------------------


def _to_wav16(audio_path: Path) -> Path:
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path),
         "-ar", "16000", "-ac", "1", "-f", "wav", str(tmp)],
        check=True, capture_output=True,
    )
    return tmp


# ---------------------------------------------------------------------------
# Text normalisation — for matching whisper words to our transcript words
# ---------------------------------------------------------------------------

_TASHKEEL_RE = re.compile(r"[\u064b-\u0652\u0670\u0640]")
_PUNCT_RE = re.compile(r"[^\w\s\u0621-\u064a]", flags=re.UNICODE)

# Arabic letter unification: collapses alef variants, hamza carriers,
# ta marbuta, etc. so two ASR systems can agree on Arabic words even
# when they spell them with slightly different diacritics or carriers.
_AR_UNIFY = {
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي", "ء": "",
}


def _normalise(word: str) -> str:
    s = unicodedata.normalize("NFKC", word)
    s = _TASHKEEL_RE.sub("", s)
    s = "".join(_AR_UNIFY.get(c, c) for c in s)
    s = _PUNCT_RE.sub("", s)
    return s.lower().strip()


# ---------------------------------------------------------------------------
# Core: get word-level timestamps from Whisper
# ---------------------------------------------------------------------------


def _whisper_word_times(audio_path: Path) -> List[Dict[str, Any]]:
    """Returns list of {word, start, end, prob} in audio time."""
    _whisper, model = _load_whisper()
    wav_path = _to_wav16(audio_path)
    tmp = wav_path if wav_path != audio_path else None
    try:
        result = model.transcribe(
            str(wav_path),
            word_timestamps=True,
            language="ar",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            fp16=_cuda_available(),
        )
    finally:
        if tmp and tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    words: List[Dict[str, Any]] = []
    for seg in result.get("segments") or []:
        for w in seg.get("words") or []:
            text = (w.get("word") or "").strip()
            if not text:
                continue
            words.append({
                "word": text,
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "prob": float(w.get("probability", 0.0)),
            })
    return words


# ---------------------------------------------------------------------------
# Public: align target transcript words to audio time via Whisper
# ---------------------------------------------------------------------------


def align_words(
    audio_path: str | Path,
    target_transcript: str,
    *,
    pad_s: float = 0.05,
) -> List[Dict[str, Any]]:
    """Return one record per word in `target_transcript`, mapped to its
    audio interval (start_s, end_s) using Whisper as the timing reference.

    Words our transcript has that Whisper didn't pick up keep start_s=None.
    Confidence is 1.0 for an exact-match alignment, 0.5 for a fuzzy
    neighbour match, 0.0 when unaligned.
    """
    if not target_transcript or not target_transcript.strip():
        return []

    audio_path = Path(audio_path)
    try:
        whisper_words = _whisper_word_times(audio_path)
    except Exception as exc:
        print(f"[align] whisper failed: {exc!r}")
        return [
            {"word": w, "start_s": None, "end_s": None, "confidence": 0.0}
            for w in re.split(r"\s+", target_transcript.strip()) if w
        ]

    target_words = [w for w in re.split(r"\s+", target_transcript.strip()) if w]
    out: List[Dict[str, Any]] = [
        {"word": w, "start_s": None, "end_s": None, "confidence": 0.0}
        for w in target_words
    ]
    if not whisper_words:
        return out

    target_norm = [_normalise(w) for w in target_words]
    whisper_norm = [_normalise(w["word"]) for w in whisper_words]

    matcher = difflib.SequenceMatcher(a=target_norm, b=whisper_norm, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                ti = i1 + off
                wj = j1 + off
                wt = whisper_words[wj]
                out[ti]["start_s"] = max(0.0, wt["start"] - pad_s)
                out[ti]["end_s"] = wt["end"] + pad_s
                out[ti]["confidence"] = 1.0
        elif tag in ("replace", "delete", "insert"):
            # The block target[i1:i2] doesn't exactly match whisper[j1:j2].
            # If both blocks are non-empty, distribute the target words
            # proportionally over the whisper time span — gives a usable
            # approximate interval per word (better than nothing).
            n_tgt = i2 - i1
            n_wh = j2 - j1
            if n_wh == 0 or n_tgt == 0:
                continue
            blk_start = whisper_words[j1]["start"]
            blk_end = whisper_words[j2 - 1]["end"]
            if blk_end <= blk_start:
                continue
            step = (blk_end - blk_start) / n_tgt
            for off in range(n_tgt):
                ti = i1 + off
                s = blk_start + off * step
                e = blk_start + (off + 1) * step
                out[ti]["start_s"] = max(0.0, s - pad_s)
                out[ti]["end_s"] = e + pad_s
                out[ti]["confidence"] = 0.5

    return out


def align_span(
    audio_path: str | Path, transcript: str, span_text: str
) -> Optional[Dict[str, Any]]:
    """Find the (start_s, end_s) covering a multi-word span in the transcript."""
    aligned = align_words(audio_path, transcript)
    if not aligned:
        return None
    span_words = [w for w in re.split(r"\s+", span_text.strip()) if w]
    if not span_words:
        return None
    n = len(span_words)
    span_norm = [_normalise(w) for w in span_words]
    for i in range(0, len(aligned) - n + 1):
        window = [_normalise(aligned[i + k]["word"]) for k in range(n)]
        if window == span_norm:
            usable = [aligned[i + k] for k in range(n)
                      if aligned[i + k]["start_s"] is not None]
            if not usable:
                return None
            return {
                "span": span_text,
                "start_s": min(a["start_s"] for a in usable),
                "end_s": max(a["end_s"] for a in usable),
                "confidence": sum(a["confidence"] for a in usable) / len(usable),
            }
    return None
