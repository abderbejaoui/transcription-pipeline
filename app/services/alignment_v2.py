"""Production-grade forced alignment using ctc-forced-aligner (MMS).

Why a v2
--------
v1 used Whisper's `word_timestamps=True`. That's a cross-attention
approximation — accurate to roughly +/- 150-300 ms on Arabic. For our
"play this exact word" UI that is NOT good enough: the user reported
the slice for 'البرسيتامول' started during 'من' (early by ~200 ms) and
ended before the final 'ل' (late by ~150 ms).

ctc-forced-aligner uses Meta's MMS (Massively Multilingual Speech) model
to compute frame-level CTC emissions, then Viterbi-aligns each character
of our target transcript against those frames. Accuracy: 20-50 ms on
clean speech, which is well below human perceptibility.

Install
-------
    pip install ctc-forced-aligner

This pulls Meta's `mms-300m-1130-forced-aligner` checkpoint (~1.2 GB)
on first use.

Public API (same as alignment.py so it's a drop-in)
---------------------------------------------------
align_words(audio_path, target_transcript) -> list[{
    "word": str, "start_s": float | None,
    "end_s": float | None, "confidence": float,
}]

Fallback
--------
If `ctc-forced-aligner` is not installed (or its model isn't downloaded
yet), we fall back to the Whisper-based v1 aligner. So the pipeline
keeps working; it just loses accuracy until the new package is in.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import alignment as _v1  # the Whisper-based fallback


# ---------------------------------------------------------------------------
# Lazy MMS aligner loader
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_aligner = None
_aligner_unavailable_reason: Optional[str] = None


def _ensure_aligner():
    """Load Meta's MMS forced-aligner once. Returns the module-level
    helpers we need. Sets _aligner_unavailable_reason on failure so we
    don't retry import storms on every request."""
    global _aligner, _aligner_unavailable_reason
    if _aligner is not None:
        return _aligner
    if _aligner_unavailable_reason is not None:
        return None
    with _lock:
        if _aligner is not None:
            return _aligner
        if _aligner_unavailable_reason is not None:
            return None
        try:
            import torch
            from ctc_forced_aligner import (
                load_alignment_model,
                generate_emissions,
                preprocess_text,
                get_alignments,
                get_spans,
                postprocess_results,
            )
        except Exception as exc:
            _aligner_unavailable_reason = (
                f"ctc-forced-aligner not available ({exc!r}). "
                "Run: pip install ctc-forced-aligner"
            )
            print(f"[align_v2] {_aligner_unavailable_reason}")
            return None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"[align_v2] loading MMS forced-aligner on {device} ({dtype})")
        try:
            model, tokenizer = load_alignment_model(
                device=device, dtype=dtype,
            )
        except Exception as exc:
            _aligner_unavailable_reason = (
                f"MMS aligner load failed: {exc!r}. "
                "First run downloads ~1.2 GB; check network / disk."
            )
            print(f"[align_v2] {_aligner_unavailable_reason}")
            return None

        _aligner = {
            "torch": torch,
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "dtype": dtype,
            "generate_emissions": generate_emissions,
            "preprocess_text": preprocess_text,
            "get_alignments": get_alignments,
            "get_spans": get_spans,
            "postprocess_results": postprocess_results,
        }
        return _aligner


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def align_words(
    audio_path: str | Path,
    target_transcript: str,
    *,
    pad_s: float = 0.02,
) -> List[Dict[str, Any]]:
    """Force-align each whitespace-separated word in `target_transcript`
    to an interval in the audio. Uses MMS forced alignment for character-
    level accuracy. Falls back to the Whisper-based v1 aligner if the
    MMS package is not installed.

    `pad_s` is intentionally small (20 ms) — MMS is accurate enough that
    we don't need the generous padding the Whisper aligner used.
    """
    if not target_transcript or not target_transcript.strip():
        return []

    agg = _ensure_aligner()
    if agg is None:
        # Fallback to the Whisper-based aligner.
        return _v1.align_words(audio_path, target_transcript)

    audio_path = Path(audio_path)
    wav_path = _v1._to_wav16(audio_path)
    tmp = wav_path if wav_path != audio_path else None

    try:
        emissions, stride = agg["generate_emissions"](
            agg["model"],
            str(wav_path),
            batch_size=1,
        )
        tokens_starred, text_starred = agg["preprocess_text"](
            target_transcript,
            romanize=True,
            language="ara",  # ISO 639-3 for Arabic
        )
        segments, scores, blank_token = agg["get_alignments"](
            emissions, tokens_starred, agg["tokenizer"],
        )
        spans = agg["get_spans"](tokens_starred, segments, blank_token)
        word_timestamps = agg["postprocess_results"](
            text_starred, spans, stride, scores,
        )
    except Exception as exc:
        print(f"[align_v2] MMS alignment failed: {exc!r} — falling back to v1")
        if tmp and tmp.exists():
            try: tmp.unlink()
            except OSError: pass
        return _v1.align_words(audio_path, target_transcript)
    finally:
        if tmp and tmp.exists():
            try: tmp.unlink()
            except OSError: pass

    # Map MMS output back to our target word list. MMS already returns
    # one record per whitespace-separated word, so the mapping is 1:1.
    out: List[Dict[str, Any]] = []
    for rec in word_timestamps:
        start = float(rec.get("start", 0.0))
        end = float(rec.get("end", 0.0))
        score = float(rec.get("score", 0.0))
        out.append({
            "word": rec.get("text", ""),
            "start_s": max(0.0, start - pad_s),
            "end_s": end + pad_s,
            "confidence": score,
        })
    return out
